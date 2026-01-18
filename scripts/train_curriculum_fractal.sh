#!/bin/bash
# Endpoint Loss Horizon Curriculum Training Script
# Gradually increases loss_horizon from immediate to full chunk
# This trains the model to focus on increasingly long-horizon task completion

set -e  # Exit on error

# Common args
COMMON_ARGS="--dataset fractal20220817_data \
    --data_dir gs://gresearch/robotics \
    --batch_size 128 \
    --resume \
    --num_workers 1 \
    --shuffle_buffer_size 5 \
    --window_size 10 \
    --omega_rot 50.0 \
    --omega_trans 50.0 \
    --confidence_temperature 0.2 \
    --viz \
    --viz_interval 10000"

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
