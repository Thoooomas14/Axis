# Training Process (V2)

This document outlines how to train the Axis V2 model.

## Training Script: `imitation/train.py`

The core training logic handles:
1. Model initialization with 13D SE(3) poses and 7D twist actions
2. Dataset loading: streaming from GCS **or** local preprocessed HDF5
3. Action chunking - model predicts W future actions simultaneously
4. Self-supervised confidence training (requery signal)
5. Dynamic OOM recovery

## Configuration

Key parameters in the config dictionary:

| Parameter | Description | Default |
|:---|:---|:---|
| `goal_dim` | Dimension of input goal embeddings | 38 |
| `embed_dim` | Token embedding dimension (vision+proprio+goal) | 1024 |
| `num_heads` | Number of attention heads | 16 |
| `num_layers` | Number of transformer layers | 8 |
| `proprio_dim` | Dimension of SE(3) full matrix pose | **13** |
| `action_dim` | Dimension of twist output | **7** |
| `window_size` | Sliding window of historical observations | 8 |
| `chunk_size` | Model parameter: number of future actions to predict | **10** |

## Loss Functions

### Chordal Loss (Actions) - Trig-Free!

Axis V2 uses a **chordal loss** for SE(3) that directly compares rotation matrices using the Frobenius norm. During rollout, we integrate predicted twists to compute future end-effector poses:

```python
import pypose as pp
import torch

def endpoint_chordal_loss(pred_twists, start_poses, target_poses, horizon, omega_rot=1.0, omega_trans=1.0):
    # start_poses: (B, 13) -> [R(9), p(3), g(1)]
    # pred_twists: (B, Chunk, 7) -> [ω(3), v(3), Δg(1)]
    
    # 1. Convert 13D to SE3
    T_curr = pp.mat2SE3(build_matrix_from_13d(start_poses))
    
    total_loss = 0
    for t in range(horizon):
        # 2. Integrate twist
        T_delta = pp.Exp(pp.se3(pred_twists[:, t, :6]))
        T_curr = T_curr @ T_delta
        
        # 3. Compare to ground truth target pose
        T_target = pp.mat2SE3(build_matrix_from_13d(target_poses[:, t]))
        
        # Chordal: ||R_curr - R_target||_F + ||p_curr - p_target||_2
        rot_loss = torch.norm(T_curr.rotation().matrix() - T_target.rotation().matrix(), p='fro', dim=(-2, -1))
        trans_loss = torch.norm(T_curr.translation() - T_target.translation(), p=2, dim=-1)
        
        total_loss += (omega_rot * rot_loss + omega_trans * trans_loss).mean()
        
    return total_loss / horizon
```

**Why Chordal Loss?**
- No trigonometric functions (no `sin`, `cos`, `acos`, `atan2`)
- Numerically stable gradients
- Direct matrix comparison

### Endpoint Loss (Task-Oriented)

For task-oriented training, **endpoint loss** focuses on where the robot ends up rather than following the exact trajectory:

```python
# Apply predicted twists to reach endpoint using PyPose
for t in range(loss_horizon):
    T_curr = (pp.mat2SE3(T_curr) @ pp.Exp(pp.se3(pred_twist[t]))).matrix()

# Compare to target endpoint (chordal distance)
loss = chordal_distance(final_pose, target_endpoint)
```

This is controlled by `--loss_horizon`:
- `--loss_horizon 1`: Immediate next-step loss (fine-grained)
- `--loss_horizon 8`: Full chunk endpoint (task completion)

### Confidence Loss (Requery)

The requery signal is trained as a **self-supervised confidence measure** using MSELoss:

```python
# Per-sample loss → Confidence target
per_sample_loss = mean((pred - target)**2, dim=[-1, -2])  # (B,)
confidence_target = exp(-per_sample_loss / temperature)    # (B, 1)

# High accuracy predictions → High confidence target
# Low accuracy predictions → Low confidence target
requery_loss = MSELoss(requery_pred, confidence_target)
```

This teaches the model to predict its own accuracy - when to ask for help vs. proceed confidently.

## Data Loading

### Option 1: Streaming (default)

**RTXStreamLoader** streams data from Google Cloud Storage in RLDS format:
- **Pose Format**: Converts raw 8D poses (pos + quaternion + gripper) to 13D full SE(3)
- **Twist Computation**: Calculates ground truth twists using PyPose `Log(T_curr⁻¹ · T_next)`
- **Continuous Episodes**: Processes full episodes without subtask splitting
- **Dynamic Goals**: Goal vector updates when gripper state changes

### Option 2: Local Preprocessed (recommended for large datasets)

**LocalDataLoader** loads from preprocessed HDF5 files:

```bash
# 1. Preprocess once (supports external drives)
python -m imitation.data.preprocessor --output E:/data/droid.h5 --dataset droid

# 2. Train from local
python imitation/train.py --local_data_path E:/data/droid.h5
```

> When `--local_data_path` is set, streaming args are ignored (with warning).

### GCS Authentication (streaming only)

```bash
# Local machine with browser
gcloud auth application-default login

# Remote/headless machine
gcloud auth application-default login --no-launch-browser
```

## Running Training

```bash
python imitation/train.py \
    --dataset fractal20220817_data \
    --batch_size 32 \
    --lr 1e-4 \
    --steps 10000 \
    --window_size 8 \
    --loss_horizon 1 \
    --omega_rot 1.0 \
    --omega_trans 1.0 \
    --confidence_temperature 1.0 \
    --verbose 1
```

## CLI Arguments

### Data Source
| Argument | Description | Default |
|:---|:---|:---|
| `--dataset` | [Streaming] TFDS dataset name | `fractal20220817_data` |
| `--data_dir` | [Streaming] Data directory (GCS) | `gs://gresearch/robotics` |
| `--image_key` | [Streaming] Image key to use | auto |
| `--local_data_path` | Path to preprocessed HDF5 (overrides streaming) | None |
| `--no_subprocess` | [Streaming] Disable subprocess isolation | False |
| `--shuffle_buffer_size` | [Streaming] TFDS shuffle buffer | 10 |

### Core Training
| Argument | Description | Default |
|:---|:---|:---|
| `--steps` | Training steps (if epochs=0) | 10000 |
| `--epochs` | Training epochs (overrides steps if >0) | 0 |
| `--batch_size` | Batch size | 1 |
| `--lr` | Learning rate | 1e-4 |
| `--warmup_steps` | LR warmup steps | 1000 |
| `--window_size` | Size of the observation history window | 10 |
| `--loss_horizon` | Future steps for endpoint loss (**must be <= chunk_size**) | 1 |

### Checkpointing & Validation
| Argument | Description | Default |
|:---|:---|:---|
| `--checkpoint_dir` | Directory for checkpoints | `checkpoints` |
| `--save_interval` | Steps between checkpoints | 1000 |
| `--resume` | Resume from latest checkpoint | False |
| `--val_interval` | Steps between validation | 5000 |
| `--val_batch_size` | Validation batch size | 32 |
| `--val_batches` | Number of validation batches | 50 |
| `--train_split_pct` | Train/val split percentage | 0.95 |

### Memory & Performance
| Argument | Description | Default |
|:---|:---|:---|
| `--num_workers` | Dataloader workers | 0 |
| `--shuffle_buffer_size` | TFDS shuffle buffer (reduce if OOM) | 10 |
| `--time_limit_min` | Stop after N minutes (0=disabled) | 0 |

### Visualization & Logging
| Argument | Description | Default |
|:---|:---|:---|
| `--viz` | Enable visualization | False |
| `--viz_interval` | Steps between visualizations | 1000 |
| `--verbose` | Logging verbosity: 0=WARN, 1=INFO, 2=DEBUG | 1 |
| `--tb_scalar_interval`| Steps between logging scalars | 50 |
| `--tb_histogram_interval`| Steps between logging histograms/images | 500 |

## Tracking with TensorBoard

The training loop natively integrates with TensorBoard, capturing scalars, distributions (weights and gradients), images, attention maps, computational graphs, and high-dimensional embeddings.

It creates a `runs/` directory at the project root where it saves all tracking data.

### 1. Local Environment
If you are running training on your localhost (e.g. your personal machine), launch the tensorboard server in a separate terminal:
```bash
tensorboard --logdir runs/
```
You can view the dashboard by opening `http://localhost:6006` in your web browser.

### 2. Docker on Remote Host
If your training is running inside a Docker container on a remote GPU server, you need to expose port 6006 and bind TensorBoard to all network interfaces (`0.0.0.0`) instead of localhost so that the port binding works correctly.

Inside the remote Docker container, start TensorBoard with:
```bash
tensorboard --logdir runs/ --host 0.0.0.0 --port 6006
```
*(Make sure your docker run command exposes the port: `docker run ... -p 6006:6006 ...`)*

**To view the dashboard, you have two options:**
- **Direct IP (if firewall allows):** Open `http://<remote_ip>:6006` in your browser.
- **SSH Port Forwarding (more secure):** From your local machine, tunnel into the remote host:
  ```bash
  ssh -L 6006:localhost:6006 your_user@remote_host
  ```
  Then simply go to `http://localhost:6006` on your local machine.

## Training Flow

```mermaid
graph TD
    A[Load Batch] --> B[Data Augmentation]
    B --> C[Forward Pass]
    C --> D[Compute Action Loss<br/>geodesic_loss]
    D --> E[Compute Per-Sample Loss]
    E --> F[Confidence Target<br/>exp-loss/temp]
    F --> G[Compute Requery Loss<br/>MSE]
    D --> H[Total Loss]
    G --> H
    H --> I[Backward + Optimizer Step]
    I --> J[EMA Update]
    J --> K[Log & Checkpoint]
```

## Memory Management

The training script includes automatic OOM recovery:

### CUDA OOM
- Automatically halves `batch_size`
- Rebuilds dataloader with new parameters
- Continues training

### RAM OOM
- First reduces `shuffle_buffer_size`
- Then reduces `num_workers`
- Graceful exit if at minimum

### Platform-Specific
- **Linux**: Calls `malloc_trim()` periodically to release memory
- **Windows**: Falls back to `gc.collect()`

## Checkpointing

- **During Training**: Only `checkpoint_latest.pt` is kept
- **On Completion**: Archived as `run_<timestamp>_<dataset>_steps<N>.pt`
- **Logs**: `training_log.csv` and `training_plot.png`

## Action Chunking

Unlike autoregressive training, Axis V2 uses **action chunking** from the current state:

```python
# Model extracts LAST token and predicts chunk
pred_action, requery_pred = model(images, proprio, goals)
# pred_action: (B, ChunkSize, 7) - Future trajectory
# requery_pred: (B, 1) - Confidence score for this chunk

# Loss computed over the trajectory up to loss_horizon
action_loss = chordal_loss(pred_action, target_actions, horizon=loss_horizon)
```

Benefits:
- **Efficiency**: Single forward pass predicts long-horizon trajectory.
- **Independence**: Decouples observation window (history) from prediction chunk (future).
- **Consistency**: High-quality trajectories without autoregressive drift.
