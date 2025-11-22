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

2. **Create the environment:**
   ```bash
   # Using Conda (recommended)
   conda env create -f environment.yml
   conda activate axis
   
   # OR using pip
   pip install -r requirements.txt
   ```

## Quick Start

To start training the model with default settings:

```bash
python training/train.py --dataset fractal20220817_data --batch_size 32
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
