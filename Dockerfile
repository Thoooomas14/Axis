# Use a base image with Miniconda installed
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
# COMBINED: All apt packages in one layer, followed immediately by the cleanup
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY environment.yml .

# Create the Conda environment and IMMEDIATELY wipe the cache
# - conda clean -afy removes cached tarballs and unused packages
# - rm -rf /root/.cache/pip clears the pip cache if your yaml uses pip dependencies
RUN conda env create -f environment.yml \
    && conda clean -afy \
    && rm -rf /root/.cache/pip

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "axis_env", "/bin/bash", "-c"]

# Verify installation (Optional but recommended)
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Copy the rest of the application code
COPY . .

# Set the default shell to ensure the environment is activated when the container starts
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "axis_env"]

# Default command (can be overridden)
CMD ["/bin/bash"]