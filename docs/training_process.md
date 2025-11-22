# Training Process

This document outlines how to train the Axis model using the provided training script.

## Training Script: `training/train.py`

The core training logic is contained in `training/train.py`. This script handles:
1.  Model initialization.
2.  Dataset loading (RTX/TFDS).
3.  The main training loop (forward pass, loss calculation, backpropagation).
4.  Checkpointing.

## Configuration

Configuration is currently defined in a dictionary within `train.py` (lines 23-39). Key parameters include:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `goal_dim` | Dimension of input goal embeddings | 768 |
| `cond_dim` | Dimension of conditioning vector | 256 |
| `vision_feature_dim` | Channels in vision encoder output | 256 |
| `num_vision_tokens` | Number of tokens from Token Learner | 8 |
| `latent_dim` | Dimension of latent memory state | 256 |
| `queue_size` | Length of latent history window | 10 |
| `embed_dim` | Transformer embedding dimension | 256 |
| `num_heads` | Number of attention heads | 4 |
| `num_layers` | Number of transformer layers | 4 |

## Data Loading

Axis uses `RTXDataset` (in `training/data/rtx_loader.py`) to load data. This is designed to work with **TensorFlow Datasets (TFDS)**, specifically the RTX (Open X-Embodiment) datasets.

- **Input**: The dataset is expected to yield `(image, proprio, action, instruction)`.
- **Goal Embeddings**: The training script expects a pickle file containing pre-computed embeddings for the natural language instructions found in the dataset.
    - Argument: `--embeddings_path`
    - If not found, it falls back to random embeddings (useful for debugging, but not for convergence).

## Optimization

- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing with Warmup (`CosineAnnealingWarmupRestarts`).
    - Helps stabilize training early on and allows for restarting the learning rate to escape local minima.
- **Loss Function**: MSE Loss (Mean Squared Error) between predicted action and ground truth action.

## Running the Training

```bash
python training/train.py \
    --dataset fractal20220817_data \
    --batch_size 32 \
    --lr 1e-4 \
    --steps 10000 \
    --checkpoint_dir checkpoints
```

### Arguments

- `--dataset`: Name of the TFDS dataset to load.
- `--batch_size`: Batch size.
- `--lr`: Learning rate.
- `--steps`: Total training steps.
- `--warmup_steps`: Steps for linear warmup.
- `--save_interval`: How often to save checkpoints.
- `--checkpoint_dir`: Directory to save `.pt` files.
- `--embeddings_path`: Path to instruction embeddings pickle.
