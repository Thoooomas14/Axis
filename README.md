# Axis: Adaptive Robot Control Policy

Axis is a PyTorch-based robot learning framework designed for real-time, adaptive motion control. It integrates vision, semantic goals, and latent memory into a unified transformer-based policy, capable of controlling multiple robot embodiments (e.g., UR3e, WidowX).

## Features

-   **Multi-Modal Input**: Consumes RGB images, Proprioception (7D EE Pose), and Semantic Goal Embeddings (64D).
-   **Window-Based Control**: Processes a sliding window of observations ($W=8$) to capture temporal dynamics.
-   **GCS Streaming**: Streams large datasets directly from Google Cloud Storage, enabling training on massive datasets without local storage overhead.
-   **Semantic Goal Oracle**: Deterministically encodes task types (Pick/Place/Move) and POIs into dense goal vectors.
-   **Latent Memory**: Utilizes a latent queue to maintain long-term context (optional).
-   **Requery Mechanism**: Predicts when a sub-task is complete or requires re-evaluation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Thoooomas14/Axis.git
    cd Axis
    ```

2.  **Quick Setup (Conda)**:
    ```bash
    conda env create -f environment.yml
    conda activate axis_env
    ```

For detailed instructions, see the [Installation Guide](docs/installation.md).

## Quick Start

### 1. GCS Authentication (Required)

To stream data, you must authenticate with Google Cloud.
-   **Local**: `gcloud auth application-default login`
-   **Remote/SSH**: `gcloud auth application-default login --no-launch-browser`

See [Training Process](docs/training_process.md) for a full walkthrough.

### 2. Training

To start training (streaming `fractal20220817_data` from GCS):

```bash
python training/train.py \
    --dataset fractal20220817_data \
    --batch_size 32 \
    --viz \
    --viz_interval 1000
```

### 3. Evaluation

To visualize model predictions on a test episode:

```bash
python scripts/evaluate_sequence.py \
    --checkpoint_dir checkpoints
```

For more configuration options, run:
```bash
python training/train.py --help
```

## Documentation

Detailed documentation is available in the `docs/` directory:

-   [Model Design](docs/model_design.md): Architecture details (64D Goal, 7D Proprio).
-   [Training Process](docs/training_process.md): GCS Auth, Configuration, and Checkpointing.
-   [Data Pipeline](docs/data_pipeline.md): Explanation of `RTXStreamLoader` and `GoalOracle`.
-   [References](docs/references.md): Related papers and concepts.

## Project Structure

-   `training/`: Core training logic and model definitions.
    -   `model/`: Neural network architecture (AxisModel, Transformer).
    -   `data/`: Data loading utilities (RTXStreamLoader, GoalOracle).
    -   `utils/`: Logging and Visualization tools.
-   `scripts/`: Utility scripts (Verification, Evaluation).
-   `docs/`: Project documentation.
