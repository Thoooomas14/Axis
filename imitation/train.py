import os
import sys
import gc
import logging

# Suppress TF INFO/WARNING logs to reduce noise (e.g. OUT_OF_RANGE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from datetime import datetime
import pypose as pp
import tensorflow as tf
from collections import deque
import time
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.axis import AxisModel
from imitation.data.local_loader import LocalDataLoader
from imitation.utils.scheduler import CosineAnnealingWarmupRestarts
from imitation.utils.logger import TrainingLogger
from imitation.utils.visualizer import Visualizer
from imitation.utils.ema import EMA


# Force TensorFlow to use CPU only (prevents VRAM fighting with PyTorch and CUDA errors in workers)
tf.config.set_visible_devices([], "GPU")

# Set PyTorch matmul precision to high for better performance on Ampere+ GPUs (can be overridden by user args if needed)
torch.set_float32_matmul_precision("high")


# --- Logging Setup ---
def setup_logging(verbose: int = 1):
    """Configure logging with verbosity levels. 0=WARNING, 1=INFO, 2=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbose, logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


log = logging.getLogger(__name__)
CHUNK_SIZE = 10


class ThroughputMonitor:
    """
    Tracks wall-clock throughput using a sliding window.
    Accounts for I/O pauses better than standard EMA.
    """

    def __init__(self, window_size=100, total_steps=None):
        self.window = deque(maxlen=window_size)
        self.total_steps = total_steps
        self.start_time = time.time()
        self.window.append((self.start_time, 0))  # time, step_count

    def update(self, current_step):
        now = time.time()
        self.window.append((now, current_step))

    def get_stats(self):
        if len(self.window) < 2:
            return 0.0, "N/A"

        t_start, s_start = self.window[0]
        t_end, s_end = self.window[-1]

        duration = t_end - t_start
        steps = s_end - s_start

        if duration < 1e-6:
            return 0.0, "N/A"

        rate = steps / duration

        eta_str = "N/A"
        if self.total_steps and rate > 0:
            remaining = self.total_steps - s_end
            if self.total_steps == float("inf"):  # Infinite epochs
                eta_str = "??"
            else:
                remaining_sec = remaining / rate
                eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))

        return rate, eta_str


def train(args):
    # Setup logging based on verbosity
    global log
    log = setup_logging(args.verbose)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.debug("Script started. Initializing...")

    # --- Pre-process Checkpoint Arguments ---
    # Must be done BEFORE TrainingLogger/Visualizer initialization
    if args.checkpoint_dir.endswith(".pt"):
        load_checkpoint_path = args.checkpoint_dir
        # Ensure dir arg is actually a dir for other usages (Logger, Visualizer)
        args.checkpoint_dir = os.path.dirname(args.checkpoint_dir)
    else:
        load_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")

    # Always save to standard name, regardless of input resume file
    save_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")

    config = "AxisV3"

    # --- Endpoint Chordal Loss (task-oriented, trig-free) ---
    def orthonormalize_rotation(R):
        """
        Orthonormalize rotation matrices using SVD.
        Supports arbitrary batch dimensions (e.g., (B, 3, 3) or (B, W, 3, 3)).
        """
        U, S, V = torch.svd(R)
        with torch.no_grad():
            det = torch.det(U @ V.transpose(-2, -1))
            diag = torch.ones_like(S)
            diag[..., -1] = det  # Ellipsis supports N-dimensional batches

        return U @ torch.diag_embed(diag) @ V.transpose(-2, -1)

    def endpoint_chordal_loss(
        pred_twists,
        start_poses,
        target_poses,
        horizon,
        discount=0.8,
        omega_rot=1.0,
        omega_trans=1.0,
    ):
        """
        Compute SE(3) chordal loss at EVERY step using Normalized Geometric Weighting.

        Weights w_t = discount^t / sum(discount^i)
        Sum of weights is exactly 1.0.

        Args:
            pred_twists: (B, W, 7) predicted twists [ω, v, gripper_delta] per step
            start_poses: (B, 13) starting pose [R_flat, pos, gripper]
            target_poses: (B, H, 13) target poses for each step t=1...H
            horizon: Number of twist steps to apply
            discount: Geometric discount factor (0 < gamma <= 1)
            omega_rot: Weight for rotational component
            omega_trans: Weight for translational component

        Returns:
            Scalar loss: Weighted sum of step losses
        """
        # Extract arbitrary batch dimensions (e.g., (B,) or (B, W,))
        batch_shape = pred_twists.shape[:-2]
        device = pred_twists.device
        dtype = pred_twists.dtype

        # 1. Build Predicted SE(3) Transform
        R_start = start_poses[..., :9].reshape(*batch_shape, 3, 3)
        p_start = start_poses[..., 9:12]

        # Initialize Identity matrix with correct N-dimensional shape
        T_pred = (
            torch.eye(4, device=device, dtype=dtype)
            .view(*(1,) * len(batch_shape), 4, 4)
            .expand(*batch_shape, 4, 4)
            .clone()
        )

        R_start_clean = R_start
        T_pred[..., :3, :3] = R_start_clean
        T_pred[..., :3, 3] = p_start

        T_pred_pp = pp.from_matrix(T_pred, ltype=pp.SE3_type, check=False)

        gripper_curr = start_poses[..., 12:13]

        steps = torch.arange(horizon, device=device, dtype=dtype)
        raw_weights = discount**steps
        weights = raw_weights / raw_weights.sum()

        total_loss = torch.zeros(batch_shape, device=device, dtype=dtype)

        # Pre-process targets to avoid reshaping inside the loop
        target_Rs = target_poses[..., :9].reshape(*batch_shape, horizon, 3, 3)
        target_ps = target_poses[..., 9:12]
        target_gs = target_poses[..., 12:13]

        # Rollout & Loss Accumulation
        for t in range(horizon):
            twist_6d = pred_twists[..., t, :6]
            gripper_delta = pred_twists[..., t, 6:7]

            T_delta_pp = pp.Exp(pp.se3(twist_6d))
            T_pred_pp = T_pred_pp @ T_delta_pp

            gripper_curr = torch.clamp(gripper_curr + gripper_delta, 0.0, 1.0)

            T_step = T_pred_pp.matrix()
            R_pred = T_step[..., :3, :3]
            p_pred = T_step[..., :3, 3]

            # Rotation (Chordal) - Unreduced!
            rot_diff = R_pred - target_Rs[..., t, :, :]
            rot_loss = torch.sum(rot_diff**2, dim=(-2, -1))

            # Translation - Unreduced!
            trans_diff = p_pred - target_ps[..., t, :]
            trans_loss = torch.sum(trans_diff**2, dim=-1)

            # Gripper (MSE) - Unreduced!
            grip_loss = ((gripper_curr - target_gs[..., t, :]) ** 2).squeeze(-1)

            step_loss = omega_rot * rot_loss + omega_trans * trans_loss + grip_loss
            total_loss += weights[t] * step_loss

        # Returns shape identical to batch_shape (e.g., (B, W))
        return total_loss

    # --- Utilities ---
    training_logger = TrainingLogger(args.checkpoint_dir, resume=args.resume)
    training_logger.log_hparams(vars(args))
    visualizer = Visualizer(args.checkpoint_dir)

    # --- Model ---
    model = AxisModel(config).to(device)
    config = model.config  # Get config from model (handles defaults)
    log.info("Model initialized.")

    # Dataset

    # === Dynamic Memory Parameters (can be reduced on OOM) ===
    effective_batch_size = args.batch_size
    effective_num_workers = args.num_workers

    def create_dataloader(
        batch_size,
        num_workers,
        split="train",
        split_start=0.0,
        split_end=1.0,
        mix_episodes=8
    ) -> tuple[torch.utils.data.DataLoader, LocalDataLoader]:
        """Factory function to create/recreate dataloader with specified params."""

        # Use local loader if local_data_path is provided
        
        log.info(
            f"Creating LOCAL dataloader: batch_size={batch_size}, "
            f"split=[{split_start:.0%}-{split_end:.0%}], "
            f"max_episodes={args.max_episodes}"
        )
        stream = LocalDataLoader(
            data_path=args.local_data_path,
            window_size=config.get("window_size", 10),
            loss_horizon=args.loss_horizon,
            # FORCE shuffle=False if filtering by max_episodes to ensure train/eval consistency
            shuffle=(
                ("train" in str(split_start) or split_start == 0.0)
                and args.max_episodes == 0
            ),
            repeat=(args.epochs == 0),
            split_start=split_start,
            split_end=split_end,
            max_episodes=args.max_episodes,
            mix_episodes=mix_episodes,
            ram_usage_limit=args.ram_limit,
        )
        # Local loader supports multi-worker
        effective_workers = num_workers
        
        loader = torch.utils.data.DataLoader(
            stream,
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=True,  # Enable pinning for speed
            prefetch_factor=4 if effective_workers > 0 else None,
            persistent_workers=True if effective_workers > 0 else False,
        )
        return loader, stream

    # Create initial dataloader
    # For streaming: use TFDS split syntax; for local: use split_start/split_end
    train_split = f"train[:{int(args.train_split_pct * 100)}%]"
    val_split = f"train[{int(args.train_split_pct * 100)}%:]"

    dataloader, train_stream = create_dataloader(
        effective_batch_size,
        effective_num_workers,
        split=train_split,
        split_start=0.0,
        split_end=args.train_split_pct,
        mix_episodes=args.mix_episodes,
    )
    if args.dataset == "droid":
        avg_episode_length = 250
    else:
        avg_episode_length = 100
    # Estimate total windows for progress tracking (if method exists)
    if hasattr(train_stream, "estimate_total_windows"):
        estimated_windows = (
            train_stream.estimate_total_windows()
            if isinstance(train_stream, LocalDataLoader)
            else train_stream.estimate_total_windows(
                avg_episode_length=avg_episode_length
            )
        )
        estimated_batches = (
            estimated_windows // effective_batch_size if estimated_windows else None
        )
        if estimated_batches:
            log.info(
                f"Estimated ~{estimated_windows:,} windows ({estimated_batches:,} batches) per epoch"
            )
    else:
        estimated_batches = None
        log.info("Progress estimation not available for subprocess loader")

    # Validation Dataloader (smaller batch size to save memory if needed)
    if args.val_batch_size == 0:
        args.val_batch_size = effective_batch_size
    val_dataloader, _ = create_dataloader(
        args.val_batch_size,
        0,
        0,
        split=val_split,
        split_start=args.train_split_pct,
        split_end=1.0,
        mix_episodes=1,
    )

    need_dataloader_rebuild = False  # Flag to trigger rebuild from outer loop

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.steps,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=args.warmup_steps,
    )

    # --- GradScaler (AMP) ---
    scaler = torch.cuda.amp.GradScaler()

    # --- EMA ---
    ema = EMA(model, decay=0.9999)
    log.info("EMA initialized.")

    # --- Augmentation ---
    # Applied on GPU
    augmentations = nn.Sequential(
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        # Add slight blur occasionally
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
    ).to(device)

    # Confidence Loss Function (Option C: MSELoss with soft labels)
    # Using MSE allows the model to output confidence values [0, 1] directly
    # while avoiding the double-sigmoid issue of BCEWithLogitsLoss + Sigmoid layer
    confidence_criterion = nn.MSELoss()

    # --- Training Loop ---
    model.train()

    # --- Resume from Checkpoint ---
    start_step = 0
    start_epoch = 0

    if args.resume and os.path.exists(load_checkpoint_path):
        log.info(f"Resuming from checkpoint: {load_checkpoint_path}")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)

        # Filter state dict
        state_dict = checkpoint["model_state_dict"]
        model_state = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            log.info("Loaded EMA state.")

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            log.info("Loaded Scaler state.")

        start_step = checkpoint["step"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]

            # Scheduler Load Logic
            log.warning(
                "No scheduler state in checkpoint. Fast-forwarding scheduler to step %d...",
                start_step,
            )
            scheduler.step(start_step)

        # Force override LR if args.lr differs from saved/initialized state
        # This ensures user can change LR on resume
        if scheduler.base_max_lr != args.lr:
            log.info(
                f"Overriding scheduler saved LR ({scheduler.base_max_lr}) with new LR ({args.lr})"
            )
            scheduler.base_max_lr = args.lr
            # Recompute max_lr for current cycle
            scheduler.max_lr = scheduler.base_max_lr * (
                scheduler.gamma**scheduler.cycle
            )

        log.info(f"Resumed at step {start_step}, epoch {start_epoch}")
    elif os.path.exists(load_checkpoint_path):
        log.warning(
            f"Checkpoint found at {load_checkpoint_path} but --resume was not provided."
        )
        log.warning("Starting from scratch. Use --resume to continue training.")
    else:
        log.info("No checkpoint found. Starting from scratch.")

    # Determine total steps/epochs
    if args.epochs > 0:
        log.info(f"Training for {args.epochs} epochs.")
    else:
        log.info(f"Training for {args.steps} steps.")

    # Calculate target step (relative to start)
    if args.epochs == 0:
        target_step = start_step + args.steps
    else:
        target_step = float("inf")

    log.info("Starting training...")

    import time

    start_time = time.time()
    step = start_step

    # === Dynamic Memory Management State ===
    consecutive_cuda_oom = 0
    consecutive_ram_oom = 0
    max_consecutive_ooms = 3
    successful_batches_since_oom = 0

    # Minimum values before graceful exit
    MIN_BATCH_SIZE = 1
    MIN_SHUFFLE_BUFFER = 1
    MIN_NUM_WORKERS = 0
    microBatchSize = args.batch_size

    # Loss average for confidence target calculation
    loss_avg_pool = []
    LOSS_AVG_WINDOW = 1000
    loss_avg = 100.0

    try:
        log.debug("Entering training loop...")

        # Loop structure
        num_epochs = args.epochs if args.epochs > 0 else 1

        # Global pbar for step-based training
        if args.epochs == 0:
            start_epoch = 0
            pbar = tqdm(
                total=target_step,
                initial=start_step,
                desc="Training Steps",
                smoothing=0.0,
            )  # Disable tqdm smoothing
            monitor = ThroughputMonitor(window_size=100, total_steps=target_step)
            monitor.update(start_step)

        steps_per_epoch_actual = None

        for epoch in range(start_epoch, num_epochs):
            steps_in_current_epoch = 0  # Initialize for both modes

            if args.epochs > 0:
                log.info(f"--- Epoch {epoch + 1}/{num_epochs} ---")

                # Dynamic Progress Bar with estimation
                # Use estimated_batches for first epoch, then actual count from previous epochs
                # Use estimated_batches if we haven't completed an epoch yet (e.g. start or resume)
                if steps_per_epoch_actual is None:
                    epoch_total = (
                        estimated_batches  # Use estimate for ETA (may be None)
                    )
                else:
                    epoch_total = (
                        steps_per_epoch_actual  # Actual count from previous epoch
                    )

                pbar = tqdm(
                    total=epoch_total,
                    initial=start_step,
                    desc=f"Epoch {epoch + 1}",
                    smoothing=0.0,
                )
                monitor = ThroughputMonitor(window_size=100, total_steps=epoch_total)

            for full_batch in dataloader:
                if args.epochs == 0 and step >= target_step:
                    break

                # Check time limit
                if args.time_limit_min > 0:
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= args.time_limit_min:
                        log.info(
                            f"Time limit of {args.time_limit_min} minutes reached. Stopping."
                        )
                        break

                # === OOM Protection: Wrap batch processing in try/except ===
                success = False
                while not success:
                    try:
                        # CRITICAL: Must zero grads at the start of the while loop so
                        # failed OOM attempts don't leave partial ghost gradients behind
                        optimizer.zero_grad()

                        full_batch = dict(full_batch)
                        B_full = full_batch["proprio"].shape[0]
                        W = full_batch["proprio"].shape[1]
                        H = args.loss_horizon

                        # Pre-stitch targets for the full batch to save compute
                        full_trajectory = torch.cat(
                            [
                                full_batch["proprio"].to(device),
                                full_batch["target_poses"].to(device),
                            ],
                            dim=1,
                        )
                        rolling_targets_full = torch.zeros(
                            (B_full, W, H, 13), device=device
                        )
                        for i in range(W):
                            rolling_targets_full[:, i, :, :] = full_trajectory[
                                :, i + 1 : i + 1 + H, :
                            ]

                        # Pre-calculate Context Weights
                        context_discount = 0.8
                        token_indices = torch.arange(
                            W - 1, -1, -1, device=device, dtype=torch.float32
                        )
                        raw_context_weights = context_discount**token_indices
                        context_weights = (
                            raw_context_weights / raw_context_weights.sum()
                        )

                        should_log_tb = (
                            step % args.tb_histogram_interval == 0 and step != 0
                        )
                        # === Dynamic Micro-Batch Loop ===
                        for mbCount in range(0, B_full, microBatchSize):
                            # 1. SLICE TENSORS (Not the dictionary)
                            images = full_batch["images"][
                                mbCount : mbCount + microBatchSize
                            ].to(device)
                            proprio = full_batch["proprio"][
                                mbCount : mbCount + microBatchSize
                            ].to(device)
                            goal_embs = full_batch["goal"][
                                mbCount : mbCount + microBatchSize
                            ].to(device)
                            mb_targets = rolling_targets_full[
                                mbCount : mbCount + microBatchSize
                            ]

                            mb_B = images.shape[0]  # Actual size of this micro-batch

                            # 2. Data Augmentation
                            if model.training:
                                _, _, C, ImgH, ImgW = images.shape
                                fl_imgs = images.view(-1, 3, ImgH, ImgW)
                                aug_imgs = augmentations(fl_imgs)
                                images = aug_imgs.view(mb_B, W, 3, ImgH, ImgW)

                            # 3. Forward Pass
                            with torch.autocast(device_type=device.type):
                                # Only log TB for the first micro-batch chunk to avoid spam

                                if should_log_tb:
                                    pred_action, requery_pred, attn_weights = model(
                                        images,
                                        proprio,
                                        goal_embs,
                                        return_attn_weights=True,
                                    )
                                else:
                                    pred_action, requery_pred = model(
                                        images, proprio, goal_embs
                                    )
                                    attn_weights = None

                                # 4. Endpoint Loss
                                with torch.autocast(
                                    device_type=device.type, enabled=False
                                ):
                                    raw_action_loss = endpoint_chordal_loss(
                                        pred_twists=pred_action.float(),
                                        start_poses=proprio.float(),
                                        target_poses=mb_targets.float(),
                                        horizon=H,
                                        discount=args.loss_discount,
                                        omega_rot=args.omega_rot,
                                        omega_trans=args.omega_trans,
                                    )

                                # 5. GRADIENT ACCUMULATION SCALING
                                weighted_action_loss = raw_action_loss * context_weights
                                # Divide sum by B_full to perfectly replicate .mean() over the whole batch
                                action_loss = weighted_action_loss.sum() / B_full


                                loss_avg_pool.append(action_loss.item())
                                if len(loss_avg_pool) > LOSS_AVG_WINDOW:
                                    loss_avg_pool.pop(0)
                                loss_avg = sum(loss_avg_pool) / len(loss_avg_pool)
                                # Confidence Loss
                                per_token_loss = raw_action_loss.detach()
                                confidence_target = torch.exp(
                                    -per_token_loss / loss_avg
                                ).unsqueeze(-1)

                                mb_requery_loss = confidence_criterion(
                                    requery_pred.float(), confidence_target
                                )
                                # Scale requery loss by the ratio of the micro-batch to the full batch
                                requery_loss = mb_requery_loss * (mb_B / B_full)
                                mb_confidence_loss = confidence_criterion(confidence_pred.float(), confidence_target)
                                # Scale confidence loss by the ratio of the micro-batch to the full batch
                                confidence_loss = mb_confidence_loss * (mb_B / B_full)

                                # Weighted Loss & Accumulate Gradients
                                loss = action_loss + (
                                    requery_loss * args.requery_weight
                                )
                                scaler.scale(loss).backward()

                        # === Backward with Scaler (Executes once per FULL batch) ===
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        # === EMA Update ===
                        ema.update(model)

                        scheduler.step()

                        # Logging
                        training_logger.log_step(
                            step,
                            epoch,
                            loss.item(),
                            action_loss.item(),
                            confidence_loss.item(),
                        )

                        # TensorBoard Logging
                        if step == start_step:
                            try:
                                # Log computation graph once
                                training_logger.log_graph(
                                    model, (images, proprio, goal_embs)
                                )
                            except Exception as e:
                                log.warning(f"Could not log graph: {e}")

                        if step % args.tb_scalar_interval == 0:
                            training_logger.log_scalars(
                                {
                                    "Loss/train": loss.item(),
                                    "Loss/action": action_loss.item(),
                                    "Loss/requery": requery_loss.item(),
                                    "Hyperparameters/learning_rate": optimizer.param_groups[
                                        0
                                    ]["lr"],
                                    "Loss/confidence": confidence_loss.item(),
                                    "Hyperparameters/learning_rate": optimizer.param_groups[0]["lr"],
                                },
                                step,
                            )

                        if should_log_tb:
                            training_logger.log_histograms(model, step)
                            import torchvision.utils as vutils

                            # Images (Batch 0, middle of window)
                            mid_idx = W // 2
                            grid = vutils.make_grid(
                                images[0, mid_idx].unsqueeze(0), normalize=True
                            )
                            training_logger.log_images("Images/Input_Data", grid, step)

                            # Attention Maps (if available)
                            if attn_weights is not None and len(attn_weights) > 0:
                                # Loop through every layer's attention map
                                for layer_idx, layer_attn in enumerate(attn_weights):
                                    # Average the attention heads for the first item in the batch
                                    mean_attn = layer_attn[0].mean(dim=0, keepdim=True)
                                    attn_map_grid = vutils.make_grid(
                                        mean_attn.unsqueeze(0), normalize=True
                                    )
                                    # Log each layer dynamically (e.g., Images/Attention_Map_Layer_0, Layer_1, etc.)
                                    training_logger.log_images(
                                        f"Images/Attention_Map_Layer_{layer_idx}",
                                        attn_map_grid,
                                        step,
                                    )

                        step += 1
                        steps_in_current_epoch += 1

                        # Update Monitor
                        monitor.update(
                            step if args.epochs == 0 else steps_in_current_epoch
                        )
                        rate, eta = monitor.get_stats()

                        desc = f"L:{loss.item():.4f} A:{action_loss.item():.4f} R:{requery_loss.item():.4f} | "
                        desc += f"{rate:.2f}it/s | ETA: {eta}"
                        pbar.set_description(desc)
                        pbar.update(1)  # Increment progress bar counter

                        # === Persistent Memory Fix ===
                        # Python doesn't always release memory to OS. We force it periodically.
                        if step % 1000 == 0:
                            gc.collect()

                        # === Validation Loop ===
                        if step % args.val_interval == 0:
                            # Force GC to clear any transient training memory
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            model.eval()
                            val_loss_total = 0.0
                            val_batches = 0
                            log.info("Running Validation...")
                            with torch.no_grad():
                                for val_full_batch in val_dataloader:
                                    val_full_batch = dict(val_full_batch)
                                    B_val_full = val_full_batch["proprio"].shape[0]

                                    for mbCount in range(0, B_val_full, microBatchSize):
                                        if val_batches >= args.val_batches:
                                            break

                                        v_imgs = val_full_batch["images"][
                                            mbCount : mbCount + microBatchSize
                                        ].to(device)
                                        v_props = val_full_batch["proprio"][
                                            mbCount : mbCount + microBatchSize
                                        ].to(device)
                                        v_goals = val_full_batch["goal"][
                                            mbCount : mbCount + microBatchSize
                                        ].to(device)
                                        v_target_poses = val_full_batch["target_poses"][
                                            mbCount : mbCount + microBatchSize
                                        ].to(device)

                                        mb_B_val = v_imgs.shape[0]
                                        H_val = args.loss_horizon

                                        with torch.autocast(device_type=device.type):
                                            v_pred, v_requery_pred = model(
                                                v_imgs, v_props, v_goals
                                            )

                                        with torch.autocast(
                                            device_type=device.type, enabled=False
                                        ):
                                            raw_v_a_loss = endpoint_chordal_loss(
                                                pred_twists=v_pred.float(),
                                                start_poses=v_props[:, -1, :].float(),
                                                target_poses=v_target_poses.float(),
                                                horizon=H_val,
                                                discount=args.loss_discount,
                                                omega_rot=args.omega_rot,
                                                omega_trans=args.omega_trans,
                                            )

                                            v_confidence_target = torch.exp(
                                                -raw_v_a_loss.detach()
                                                / loss_avg
                                            ).unsqueeze(-1)

                                            v_r_loss = requery_criterion(
                                                v_requery_pred.float(),
                                                v_confidence_target,
                                            )

                                            # Use the exact same gradient-accumulation scaling math to track metrics
                                            v_a_loss_scaled = (
                                                raw_v_a_loss.sum() / B_val_full
                                            )
                                            v_r_loss_scaled = v_r_loss * (
                                                mb_B_val / B_val_full
                                            )

                                            v_loss = v_a_loss_scaled + (
                                                v_r_loss_scaled * args.requery_weight
                                            )
                                            val_loss_total += v_loss.item()

                                    val_batches += 1  # Only increment after finishing the full batch

                            avg_val_loss = val_loss_total / max(1, val_batches)
                            log.info(f"Validation Loss: {avg_val_loss:.4f}")

                            # Log validation step
                            training_logger.log_step(
                                step, epoch, None, None, None, val_loss=avg_val_loss
                            )
                            training_logger.log_scalars(
                                {"Loss/val": avg_val_loss}, step
                            )
                            model.train()

                            # Cleanup validation variables
                            del val_loss_total, val_batches
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # Checkpointing (Step-based)
                        if step % args.save_interval == 0:
                            # Define checkpoint data
                            ckpt_data = {
                                "step": step,
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "ema_state_dict": ema.state_dict(),  # Save EMA
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),  # Save Scheduler
                                "scaler_state_dict": scaler.state_dict(),  # Save Scaler
                                "loss": loss.item()
                                if "loss" in locals()
                                else float("inf"),
                            }

                            try:
                                # Try primary save (latest)
                                os.makedirs(
                                    os.path.dirname(save_checkpoint_path), exist_ok=True
                                )
                                torch.save(ckpt_data, save_checkpoint_path)
                            except Exception as e:
                                log.error(
                                    f"Failed to save primary checkpoint '{save_checkpoint_path}': {e}"
                                )

                                # Fallback save (timestamped/step-based)
                                try:
                                    fallback_name = f"checkpoint_step_{step}.pt"
                                    fallback_path = os.path.join(
                                        args.checkpoint_dir, fallback_name
                                    )
                                    log.info(
                                        f"Attempting fallback save to '{fallback_path}'..."
                                    )
                                    torch.save(ckpt_data, fallback_path)
                                    log.info("Fallback save successful!")
                                except Exception as e2:
                                    log.error(
                                        f"CRITICAL: Fallback save also failed: {e2}"
                                    )

                            training_logger.plot_progress()

                        # Visualization (Step-based)
                        if args.viz and step % args.viz_interval == 0:
                            visualizer.visualize_batch(step, full_batch, pred_action)

                        success = True

                    except torch.cuda.OutOfMemoryError:
                        # CUDA OOM: Reduce batch size
                        consecutive_cuda_oom += 1
                        successful_batches_since_oom = 0

                        log.warning(
                            f"[CUDA OOM] Out of memory at step {step}. (OOM #{consecutive_cuda_oom})"
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)

                        model.train()

                        if consecutive_cuda_oom >= max_consecutive_ooms:
                            if microBatchSize > MIN_BATCH_SIZE:
                                microBatchSize = max(
                                    MIN_BATCH_SIZE, microBatchSize // 2
                                )
                                consecutive_cuda_oom = 0
                                log.warning(
                                    f"[CUDA OOM] Reducing microbatch size to {microBatchSize}"
                                )
                                continue  # Retry with smaller batch size

                            log.error(
                                f"[CUDA OOM] Microbatch size already at minimum ({MIN_BATCH_SIZE})."
                                + " Cannot reduce further."
                            )
                            log.error(
                                "[CUDA OOM] Saving checkpoint and exiting gracefully..."
                            )
                            raise SystemExit(
                                "CUDA OOM: Cannot reduce batch size further"
                            ) from None

                        continue

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Treat as CUDA OOM
                            consecutive_cuda_oom += 1
                            successful_batches_since_oom = 0

                            log.warning(
                                f"[CUDA OOM] Runtime memory error at step {step}. (OOM #{consecutive_cuda_oom})"
                            )
                            gc.collect()
                            torch.cuda.empty_cache()
                            optimizer.zero_grad(set_to_none=True)

                            model.train()

                            if consecutive_cuda_oom >= max_consecutive_ooms:
                                if microBatchSize > MIN_BATCH_SIZE:
                                    microBatchSize = max(
                                        MIN_BATCH_SIZE, microBatchSize // 2
                                    )
                                    consecutive_cuda_oom = 0
                                    log.warning(
                                        f"[CUDA OOM] Reducing microbatch size to {microBatchSize}"
                                    )
                                    continue  # Retry with smaller batch size
                                log.error(
                                    f"[CUDA OOM] Microbatch size already at minimum ({MIN_BATCH_SIZE})."
                                    + " Cannot reduce further."
                                )
                                raise SystemExit(
                                    "CUDA OOM: Cannot reduce microbatch size further"
                                ) from e

                            continue
                        raise

                    except MemoryError:
                        # System RAM OOM: Reduce shuffle_buffer first, then num_workers
                        consecutive_ram_oom += 1
                        successful_batches_since_oom = 0

                        log.warning(
                            f"[RAM OOM] System memory exhausted at step {step}. (OOM #{consecutive_ram_oom})"
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        optimizer.zero_grad(set_to_none=True)

                        model.train()

                        if consecutive_ram_oom >= max_consecutive_ooms:
                            # First try reducing workers
                            if effective_num_workers > MIN_NUM_WORKERS:
                                effective_num_workers = max(
                                    MIN_NUM_WORKERS, effective_num_workers - 1
                                )
                                consecutive_ram_oom = 0
                                need_dataloader_rebuild = True
                                log.warning(
                                    f"[RAM OOM] Reducing num_workers to {effective_num_workers}"
                                )
                                break

                            log.error(
                                "[RAM OOM] All parameters at minimum. Cannot reduce further."
                            )
                            log.error(
                                "[RAM OOM] Saving checkpoint and exiting gracefully..."
                            )
                            raise SystemExit(
                                "RAM OOM: Cannot reduce memory parameters further"
                            ) from None

                        continue

                # Successful batch - reset OOM counters
                consecutive_cuda_oom = 0
                consecutive_ram_oom = 0
                successful_batches_since_oom += 1

                # Auto-recover AR steps after sustained success (disabled for now)
                # if ar_reduction_active and successful_batches_since_oom >= 500:
                #     ...

            # Check if we need to rebuild dataloader (from OOM reduction)
            if need_dataloader_rebuild:
                log.info("[Memory] Rebuilding dataloader with new parameters...")
                dataloader, _ = create_dataloader(
                    effective_batch_size,
                    effective_num_workers,
                    split=train_split,
                    split_start=0.0,
                    split_end=args.train_split_pct,
                )
                continue

            # Close epoch pbar
            if args.epochs > 0:
                pbar.close()
                steps_per_epoch_actual = steps_in_current_epoch

            # End of Epoch Actions (Epoch-based)
            if args.epochs > 0:
                log.info(f"Saving checkpoint at end of epoch {epoch + 1}")
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch + 1,  # Save as next epoch start
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "loss": loss.item() if "loss" in locals() else float("inf"),
                    },
                    save_checkpoint_path,
                )
                training_logger.plot_progress()

                if args.viz:
                    log.info(f"Visualizing at end of epoch {epoch + 1}")
                    # Use the last batch for visualization
                    visualizer.visualize_batch(step, full_batch, pred_action)

            # End of inner loop
            if args.epochs == 0 and step >= target_step:
                break

            # Time limit check break outer
            if args.time_limit_min > 0:
                elapsed_min = (time.time() - start_time) / 60.0
                if elapsed_min >= args.time_limit_min:
                    break

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Log Projector Embeddings
        try:
            log.info("Computing embeddings projector data...")
            model.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    v_imgs = val_batch["images"].to(device)
                    v_props = val_batch["proprio"].to(device)
                    v_goals = val_batch["goal"].to(device)

                    _, _, tokens = model(v_imgs, v_props, v_goals, return_tokens=True)
                    last_tokens = tokens[:, -1, :]  # (B, 512)

                    meta = [str(g.cpu().numpy()[:5]) for g in v_goals]
                    label_img = v_imgs[:, -1, ...]  # (B, 3, 224, 224)

                    training_logger.log_embeddings(
                        last_tokens, metadata=meta, label_img=label_img, step=step
                    )
                    break
        except Exception as e:
            log.warning(f"Embedding projector failed: {e}")

        # Archival: Always save a fresh checkpoint to avoid losing data if 'latest' is locked
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}_{args.dataset}_steps{step}_epoch{epoch}.pt"
            run_path = os.path.join(args.checkpoint_dir, run_name)

            # Save fresh - do NOT rely on copying 'checkpoint_latest.pt'
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": loss.item() if "loss" in locals() else float("inf"),
                },
                run_path,
            )

            log.info(f"Run saved to: {run_path}")
            training_logger.plot_progress()
        except Exception as e:
            log.error(f"Failed to save final checkpoint: {e}")

        log.info("Training complete.")


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()

    # === Data Source Args ===
    # Load from preprocessed local file
    parser.add_argument(
        "--local_data_path",
        type=str,
        default=None,
        help="Path to preprocessed HDF5 file",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Max episodes to use (0 = all). Set to 1 for overfit testing.",
    )

    # === Training Args ===
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of training steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=0, help="Number of epochs (0 = use steps instead)"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="LR warmup steps"
    )
    parser.add_argument(
        "--window_size", type=int, default=10, help="Sliding window size"
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--ram_limit",
        type=float,
        default=12.0,
        help="RAM usage limit in GB for data workers",
    )
    parser.add_argument(
        "--mix_episodes", type=int, default=8, help="Number of episodes to mix in RAM"
    )

    # === Checkpointing ===
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Steps between checkpoints"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--time_limit_min",
        type=float,
        default=0.0,
        help="Stop after N minutes (0=no limit)",
    )

    # === Visualization ===
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--save_gif", action="store_true", help="Enable GIF generation")
    parser.add_argument(
        "--viz_interval", type=int, default=1000, help="Steps between visualizations"
    )

    # === Loss Function ===
    parser.add_argument(
        "--loss_horizon",
        type=int,
        default=10,
        help="Twist steps to rollout before loss (1 <= horizon <= window_size(10))",
    )
    parser.add_argument(
        "--loss_discount",
        type=float,
        default=0.8,
        help="Geometric discount factor for horizon steps",
    )
    parser.add_argument(
        "--omega_rot", type=float, default=1.0, help="Rotation loss weight"
    )
    parser.add_argument(
        "--omega_trans", type=float, default=1.0, help="Translation loss weight"
    )
    parser.add_argument(
        "--confidence_weight", type=float, default=1.0, help="Confidence loss weight"
    )

    # === Validation ===
    parser.add_argument(
        "--train_split_pct", type=float, default=0.95, help="Train split percentage"
    )
    parser.add_argument(
        "--val_interval", type=int, default=5000, help="Steps between validation"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Validation batch size"
    )
    parser.add_argument(
        "--val_batches", type=int, default=50, help="Batches per validation run"
    )

    # === Logging ===
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity: 0=WARNING, 1=INFO, 2=DEBUG"
    )
    parser.add_argument(
        "--tb_scalar_interval",
        type=int,
        default=100,
        help="Steps between logging scalars",
    )
    parser.add_argument(
        "--tb_histogram_interval",
        type=int,
        default=5000,
        help="Steps between logging histograms/images",
    )

    args = parser.parse_args()

    # === Arg Validation & Conflict Warnings ===
    # Convert no_subprocess to use_subprocess (inverted logic)
    args.use_subprocess = not args.no_subprocess

    # Warn about conflicting/ignored args when using local data
    if args.local_data_path:
        ignored_args = []

        if args.dataset != "fractal20220817_data":  # Non-default
            ignored_args.append(f"--dataset={args.dataset}")
        if args.data_dir != "gs://gresearch/robotics":  # Non-default
            ignored_args.append(f"--data_dir={args.data_dir}")
        if args.image_key is not None:
            ignored_args.append(f"--image_key={args.image_key}")
        if args.no_subprocess:
            ignored_args.append("--no_subprocess")
        if args.shuffle_buffer_size != 10:  # Non-default
            ignored_args.append(f"--shuffle_buffer_size={args.shuffle_buffer_size}")

        if ignored_args:
            import warnings

            warnings.warn(
                f"Using --local_data_path, the following streaming args are IGNORED: {', '.join(ignored_args)}",
                stacklevel=2,
            )

    # Validate loss_horizon
    if args.loss_horizon < 1 or args.loss_horizon > CHUNK_SIZE:
        parser.error(
            f"--loss_horizon must be between 1 and chunk_size ({CHUNK_SIZE}), got {args.loss_horizon}"
        )

    log.debug(f"Args parsed. Viz: {args.viz}, Viz Interval: {args.viz_interval}")
    train(args)
