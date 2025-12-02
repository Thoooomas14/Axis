# Axis: Adaptive Robot Control Policy

Axis is a PyTorch-based robot learning framework designed for real-time, adaptive motion control. It integrates vision, semantic goals, and latent memory into a unified transformer-based policy, capable of controlling multiple robot embodiments (e.g., UR3e, WidowX).

## Features

- **Multi-Modal Input**: Consumes RGB images, proprioceptive state, and semantic goal embeddings.
- **Latent Memory**: Utilizes a sliding-window latent queue to maintain temporal context without the cost of full trajectory attention.
- **Transformer Backbone**: A flexible transformer architecture for sensor fusion and policy prediction.
- **Multi-Robot Support**: Modular "Robot Adapters" allow training a single policy across different robot hardware.
- **Requery Mechanism**: Predicts when a sub-task is complete or requires re-evaluation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Thoooomas14/Axis.git
   cd Axis
   ```

For detailed instructions, see the [Installation Guide](docs/installation.md).

### Quick Setup (Conda)

```bash
conda env create -f environment.yml
conda activate axis_env
```

## Quick Start

### 1. Data Preprocessing (Optional but Recommended)

To speed up training, you can preprocess the TFDS dataset into `.pt` files:

```bash
python scripts/preprocess_dataset.py \
    --dataset fractal20220817_data \
    --num_workers 4
```

### 2. Training

To start training (using raw TFDS):
```bash
python training/train.py --dataset fractal20220817_data --batch_size 32
```

To train using preprocessed data (faster):
```bash
python training/train.py --use_processed_data --data_dir data/processed/fractal_v1
```

### 3. Evaluation

To visualize model predictions on a test episode:

```bash
python scripts/evaluate_sequence.py \
    --checkpoint_dir checkpoints \
    --use_processed_data
```

For more configuration options, run:
```bash
python training/train.py --help
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Model Design](docs/model_design.md): Architecture details, components, and data flow.
- [Training Process](docs/training_process.md): Explanation of the training loop, configuration, and data loading.
- [References](docs/references.md): Related papers and concepts.

## Project Structure

- `training/`: Core training logic and model definitions.
  - `model/`: Neural network architecture (AxisModel, Transformer, Adapters).
  - `data/`: Data loading utilities (RTXDataset).
- `simulation/`: Simulation environments (if applicable).
- `scripts/`: Utility scripts.
