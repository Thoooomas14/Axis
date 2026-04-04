import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.axis import AxisModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cpu")  # Use CPU for counting

    # Config matching V2 defaults
    config = {
        "device": device,
        "goal_dim": 64,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "proprio_dim": 10,  # 3 Pos + 6 Rot + 1 Gripper
        "action_dim": 10,
        "use_rope": True,
    }

    print("Initializing AxisModel with config:")
    print(config)

    try:
        model = AxisModel(config).to(device)
        num_params = count_parameters(model)
        print(f"\nTotal Trainable Parameters: {num_params:,}")

        # Optional: Breakdown by component
        print("\nBreakdown:")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name}: {params:,}")

    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
