# Installation Guide

This guide provides detailed instructions for setting up the Axis development environment.

## Prerequisites

- **Anaconda** or **Miniconda** (Recommended): [Download Here](https://docs.conda.io/en/latest/miniconda.html)
- **Git**: [Download Here](https://git-scm.com/downloads)
- **CUDA-capable GPU** (Recommended for training)

## Option 1: Conda Environment (Recommended)

We provide an `environment.yml` file that creates a Conda environment with all necessary dependencies, including Python 3.11, PyTorch, and TensorFlow.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Thoooomas14/Axis.git
    cd Axis
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
    *Note: This may take a few minutes as it downloads and installs packages.*

3.  **Activate the environment:**
    ```bash
    conda activate axis_env
    ```
    *(Note: The environment name is defined as `axis_env` in the YAML file. If you want a different name, edit the first line of `environment.yml` before creating it.)*

4.  **Verify Installation:**
    ```bash
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
    ```

## Option 2: Pip Installation

If you prefer using `pip` directly (e.g., in a virtualenv):

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Troubleshooting

### Using Windows
The Repo was designed for training and runing on linux (Ubuntu). Windows is not activly supported so continue at your own risk.
If possible install Ubuntu 24.04 on WSL2 (Windows Subsystem for Linux).

### PyTorch CUDA
The `environment.yml` specifies PyTorch with CUDA 12.1 support (`cu121`). If your driver supports a different version, you may need to modify the installation command or install PyTorch manually from [pytorch.org](https://pytorch.org/get-started/locally/).
