#!/bin/bash
# Autoregressive Training Curriculum Script
# Runs each phase sequentially - if one fails, the next still runs

set -e  # Exit on error (remove if you want all phases to attempt)

# Common args
COMMON_ARGS="--dataset fractal20220817_data \
    --data_dir gs://gresearch/robotics \
    --batch_size 128 \
    --resume \
    --num_workers 1 \
    --shuffle_buffer_size 5 \
    --viz \
    --viz_interval 10000"

echo "=========================================="
echo "Phase 1: Standard Training (AR=1)"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 40000 \
    --autoregressive_steps 1

echo "=========================================="
echo "Phase 2: Autoregressive Training (AR=2)"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 20000 \
    --autoregressive_steps 2

echo "=========================================="
echo "Phase 3: Autoregressive Training (AR=4)"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 15000 \
    --autoregressive_steps 4

echo "=========================================="
echo "Phase 4: Autoregressive Training (AR=8)"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 10000 \
    --autoregressive_steps 8

echo "=========================================="
echo "Training Curriculum Complete!"
echo "=========================================="
