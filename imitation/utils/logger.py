import logging
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that routes logs through tqdm.write()"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            # Use tqdm.write instead of standard print/sys.stdout
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class TrainingLogger:
    def __init__(self, log_dir, resume=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "training_log.csv")
        self.plot_path = os.path.join(log_dir, "training_plot.png")

        # Reset if not resuming (and file exists)
        if not resume:
            if os.path.exists(self.log_path):
                try:
                    os.remove(self.log_path)
                    logging.info(f"[Logger] Removed old log file: {self.log_path}")
                except Exception as e:
                    logging.error(f"[Logger] Failed to remove old log: {e}")

            if os.path.exists(self.plot_path):
                try:
                    os.remove(self.plot_path)
                    logging.info(f"[Logger] Removed old plot file: {self.plot_path}")
                except Exception as e:
                    logging.error(f"[Logger] Failed to remove old plot: {e}")

        # Initialize CSV with fallback for permission errors
        try:
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "step",
                            "epoch",
                            "loss",
                            "action_loss",
                            "confidence_loss",
                            "val_loss",
                        ]
                    )

            # Verify writable
            with open(self.log_path, "a", newline="") as f:
                pass
        except PermissionError:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"training_log_{ts}.csv")
            logging.info(
                f"[Logger] Permission denied on main log. Falling back to: {self.log_path}"
            )
            # Create fallback
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "epoch",
                        "loss",
                        "action_loss",
                        "confidence_loss",
                        "val_loss",
                    ]
                )

        # TensorBoard integration
        writer_dir = os.path.join("runs", os.path.basename(log_dir))
        self.tb_writer = SummaryWriter(writer_dir)

    def log_scalars(self, tag_scalar_dict, step):
        for tag, scalar_value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(tag, scalar_value, step)

    def log_histograms(self, model, step):
        for name, param in model.named_parameters():
            try:
                self.tb_writer.add_histogram(f"Weights/{name}", param.data, step)
                if param.grad is not None:
                    self.tb_writer.add_histogram(f"Gradients/{name}", param.grad, step)
            except ValueError as e:
                # Safely ignore empty histograms (e.g., NaN gradients during AMP scaler warmups)
                if "The histogram is empty" not in str(e):
                    raise e

    def log_images(self, tag, img_tensor, step):
        # Expects img_tensor to be a grid or valid tensor for add_image
        self.tb_writer.add_image(tag, img_tensor, step)

    def log_graph(self, model, input_to_model):
        self.tb_writer.add_graph(model, input_to_model)

    def log_embeddings(self, mat, metadata, label_img, step):
        # We will use this at the end or occasionally
        self.tb_writer.add_embedding(
            mat, metadata=metadata, label_img=label_img, global_step=step
        )

    def log_hparams(self, hparam_dict, metric_dict=None):
        if metric_dict is None:
            metric_dict = {}
        # Preprocess hparam_dict so types are compatible with add_hparams (e.g. no lists or dicts)
        clean_hparam_dict = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                clean_hparam_dict[k] = v
            else:
                clean_hparam_dict[k] = str(v)
        self.tb_writer.add_hparams(clean_hparam_dict, metric_dict)

    def close(self):
        self.tb_writer.close()

    def log_step(self, step, epoch, loss, action_loss, confidence_loss, val_loss=None):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, loss, action_loss, confidence_loss, val_loss])

    def plot_progress(self):
        try:
            df = pd.read_csv(self.log_path, on_bad_lines="skip")

            if len(df) < 2:
                return

            plt.figure(figsize=(12, 6))

            # Plot Total Loss with Validation
            plt.subplot(1, 2, 1)
            plt.plot(
                df["step"], df["loss"], label="Train Loss", color="blue", alpha=0.7
            )
            if "val_loss" in df.columns:
                # Plot validation points only where they exist (not NaNs)
                val_data = df[df["val_loss"].notna()]
                if len(val_data) > 0:
                    plt.plot(
                        val_data["step"],
                        val_data["val_loss"],
                        label="Val Loss",
                        color="orange",
                        marker="o",
                        linestyle="--",
                    )

            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Plot Components
            plt.subplot(1, 2, 2)
            plt.plot(
                df["step"],
                df["action_loss"],
                label="Action Loss",
                color="green",
                alpha=0.7,
            )
            plt.plot(
                df["step"],
                df["confidence_loss"],
                label="Confidence Loss",
                color="red",
                alpha=0.7,
            )
            plt.xlabel("Step")
            plt.title("Loss Components")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.plot_path)
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting progress: {e}")
