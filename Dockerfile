# Use a base image with Miniconda installed
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are often required for OpenCV/rendering tasks
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY environment.yml .

# Create the Conda environment
# We use the name 'axis_env' as defined in the yaml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "axis_env", "/bin/bash", "-c"]

# Verify installation (Optional but recommended)
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Copy the rest of the application code
COPY . .

# Set the default shell to ensure the environment is activated when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "axis_env"]

# Default command (can be overridden)
CMD ["/bin/bash"]