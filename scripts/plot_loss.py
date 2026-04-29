import os
import torch
import matplotlib.pyplot as plt
import argparse


def plot_loss(args):
    if not os.path.exists(args.checkpoint_dir):
        print(f"Checkpoint directory {args.checkpoint_dir} not found.")
        return

    checkpoints = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        print("No checkpoints found.")
        return

    steps = []
    losses = []

    print(f"Found {len(checkpoints)} checkpoints. Extracting data...")

    for cp_name in checkpoints:
        cp_path = os.path.join(args.checkpoint_dir, cp_name)
        try:
            # Map to CPU to avoid GPU OOM if training is running
            checkpoint = torch.load(cp_path, map_location="cpu")

            if "step" in checkpoint and "loss" in checkpoint:
                steps.append(checkpoint["step"])
                losses.append(checkpoint["loss"])
            else:
                print(f"Skipping {cp_name}: missing step or loss key.")
        except Exception as e:
            print(f"Error loading {cp_name}: {e}")

    # Sort by step
    data = sorted(zip(steps, losses))
    steps, losses = zip(*data)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker="o", linestyle="-", color="b", label="Training Loss")

    plt.title("Training Loss over Time")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_path = "loss_plot.png"
    plt.savefig(output_path)
    print(f"Saved loss plot to {output_path}")

    # Print raw data for quick view
    print("\n--- Loss History ---")
    for s, loss in zip(steps, losses):
        print(f"Step {s}: {loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    plot_loss(args)
