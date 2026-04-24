# ruff: noqa: T201
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.axis import AxisModel

def count_parameters(model):
    """Returns a tuple of (trainable_params, frozen_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen

def main():
    device = torch.device("cpu")  # Use CPU for counting

    # Config matching V2 defaults
    config = "AxisV2"

    print(f"Initializing AxisModel with config: {config}")

    try:
        model = AxisModel(config).to(device)
        trainable_params, frozen_params = count_parameters(model)
        total_params = trainable_params + frozen_params

        print(f"\n{'='*55}")
        print(f"Total Parameters:     {total_params:>15,}")
        print(f"Trainable Parameters: {trainable_params:>15,}")
        print(f"Frozen Parameters:    {frozen_params:>15,}")
        print(f"{'='*55}\n")

        print("Architectural Breakdown:")
        print(f"{'Module Name':<25} | {'Trainable':<12} | {'Frozen':<12}")
        print("-" * 55)

        # Breakdown by top-level component
        for name, module in model.named_children():
            m_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            m_frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)

            # Only display modules that actually contain parameters
            if m_trainable > 0 or m_frozen > 0:
                print(f"{name:<25} | {m_trainable:<12,} | {m_frozen:<12,}")

        print("-" * 55)

    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
