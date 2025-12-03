# Training Process

This document outlines how to train the Axis model using the provided training script.

## Training Script: `training/train.py`

The core training logic is contained in `training/train.py`. This script handles:
1.  Model initialization.
2.  Dataset loading (Streaming from GCS).
3.  The main training loop (forward pass, loss calculation, backpropagation).
4.  Checkpointing, Logging, and Visualization.

## Configuration

Configuration is currently defined in a dictionary within `train.py`. Key parameters include:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `goal_dim` | Dimension of input goal embeddings | 64 |
| `cond_dim` | Dimension of conditioning vector | 256 |
| `vision_feature_dim` | Channels in vision encoder output | 256 |
| `num_vision_tokens` | Number of tokens from Token Learner | 8 |
| `latent_dim` | Dimension of latent memory state | 256 |
| `queue_size` | Length of latent history window | 10 |
| `embed_dim` | Transformer embedding dimension | 256 |
| `num_heads` | Number of attention heads | 4 |
| `num_layers` | Number of transformer layers | 4 |
| `proprio_dim` | Dimension of EE Pose (Pos+Quat+Grip) | 8 |
| `action_dim` | Dimension of EE Pose (Pos+Quat+Grip) | 8 |

## Data Loading & GCS Authentication

Axis uses `RTXStreamLoader` (in `training/data/rtx_stream_loader.py`) to stream data directly from Google Cloud Storage (GCS). This avoids the need to download massive datasets locally.

### 1. Prerequisite: GCS Authentication

To stream data from `gs://` buckets, you must authenticate your machine with Google Cloud.

#### Option A: Local Machine (With Browser)

If you are running on a machine with a desktop environment:

1.  **Install Google Cloud SDK**: [Installation Guide](https://cloud.google.com/sdk/docs/install)
2.  **Login**:
    ```bash
    gcloud auth application-default login
    ```
    This opens a browser window. Log in with your Google account.

#### Option B: Remote Machine / SSH (Headless)

If you are connected to a remote Linux server via SSH, you cannot open a browser locally. Follow these steps:

1.  **Install Google Cloud SDK (Linux)**:
    ```bash
    # Update and install dependencies
    sudo apt-get update
    sudo apt-get install apt-transport-https ca-certificates gnupg curl

    # Import the Google Cloud public key
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

    # Add the gcloud CLI distribution URI as a package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

    # Update and install the gcloud CLI
    sudo apt-get update && sudo apt-get install google-cloud-cli
    ```

2.  **Login with `--no-launch-browser`**:
    Run the following command on the **remote server**:
    ```bash
    gcloud auth application-default login --no-launch-browser
    ```

3.  **Authenticate**:
    -   The command will print a long URL starting with `https://accounts.google.com/...`.
    -   **Copy** this URL and paste it into a browser on your **local computer**.
    -   Log in with your Google account and allow access.
    -   You will be given a verification code string.
    -   **Copy** the verification code.
    -   **Paste** it back into the terminal on your **remote server** and press Enter.

4.  **Verify**:
    You should see a message confirming that credentials have been saved to `/home/<user>/.config/gcloud/application_default_credentials.json`.

### 2. Dataset Format

The loader expects datasets in the RLDS (Reinforcement Learning Datasets) format, hosted on GCS or locally.
-   **Input**: The loader yields a window of 8 steps: `(images, proprio, goal, action, requery)`.
-   **Goal Generation**: The `GoalOracle` automatically generates 64D semantic embeddings based on the task type (Pick/Place/Move) and subtask endpoints.

## Optimization

-   **Optimizer**: AdamW
-   **Scheduler**: Cosine Annealing with Warmup (`CosineAnnealingWarmupRestarts`).
-   **Loss Function**:
    -   **Action**: MSE Loss (Mean Squared Error).
    -   **Requery**: BCE Loss (Binary Cross Entropy).

## Running the Training

```bash
python training/train.py \
    --dataset fractal20220817_data \
    --batch_size 32 \
    --lr 1e-4 \
    --steps 10000 \
    --checkpoint_dir checkpoints \
    --viz \
    --viz_interval 1000
```

### Arguments

-   `--dataset`: Name of the TFDS dataset to load (e.g., `fractal20220817_data`).
-   `--data_dir`: Directory to cache/store dataset (optional).
-   `--batch_size`: Batch size.
-   `--lr`: Learning rate.
-   `--steps`: Total training steps.
-   `--warmup_steps`: Steps for linear warmup.
-   `--save_interval`: How often to save checkpoints.
-   `--checkpoint_dir`: Directory to save `.pt` files and logs.
-   `--time_limit_min`: Stop training after N minutes (0 to disable).
-   `--use_latent_memory`: Enable recurrent latent memory updates during training.
-   `--queue_size`: Size of the latent memory queue (default: 10).
-   `--viz`: Enable visualization (saves images to checkpoint dir).
-   `--viz_interval`: Step interval for visualization.

## Checkpointing & Logging

-   **Checkpoints**: Only `checkpoint_latest.pt` is kept during training to save space.
-   **Archival**: On completion or interruption, the latest checkpoint is renamed to `run_<timestamp>_<dataset>_steps<N>.pt`.
-   **Logs**: Metrics are saved to `training_log.csv` and plotted in `training_plot.png`.
-   **Visuals**: Comparison images are saved as `viz_step_<N>.png`.
