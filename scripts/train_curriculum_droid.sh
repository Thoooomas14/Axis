#!/bin/bash
# Endpoint Loss Horizon Curriculum Training Script (DROID Dataset)
# Gradually increases loss_horizon from immediate to full chunk
# This trains the model to focus on increasingly long-horizon task completion

set -e  # Exit on error

# Common args (lower shuffle_buffer for DROID's larger episodes)
COMMON_ARGS="--dataset droid \
    --data_dir gs://gresearch/robotics \
    --batch_size 16 \
    --resume \
    --num_workers 0 \
    --shuffle_buffer_size 1 \
    --window_size 10 \
    --omega_rot 100.0 \
    --omega_trans 100.0 \
    --confidence_temperature 0.1"

echo "=========================================="
echo "Phase 1: Immediate Loss (loss_horizon=1)"
echo "Focus: Learn basic twist predictions"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 40000 \
    --loss_horizon 1

echo "=========================================="
echo "Phase 2: Short Horizon (loss_horizon=2)"
echo "Focus: 2-step endpoint accuracy"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 20000 \
    --loss_horizon 2

echo "=========================================="
echo "Phase 3: Medium Horizon (loss_horizon=4)"
echo "Focus: Half-window endpoint accuracy"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 15000 \
    --loss_horizon 4

echo "=========================================="
echo "Phase 4: Full Chunk (loss_horizon=8)"
echo "Focus: Complete task endpoint accuracy"
echo "=========================================="
docker run --gpus all --rm --shm-size=16g \
    -v $(pwd):/app \
    -v ~/.config/gcloud:/root/.config/gcloud \
    axis-training \
    python imitation/train.py \
    $COMMON_ARGS \
    --steps 10000 \
    --loss_horizon 8

echo "=========================================="
echo "Training Curriculum Complete!"
echo "=========================================="
