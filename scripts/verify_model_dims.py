import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.axis import AxisModel


def verify_model():
    print("Verifying Axis Model Dimensions (SE(3) Update)...")

    # Config matching new requirements
    config = "AxisV2"  # This will trigger the default config with 7D proprio and 64D goal

    model = AxisModel(config)
    model.eval()

    # Dummy Inputs
    B = 2
    W = 8  # Window size 8
    C, H, W_img = 3, 224, 224

    images = torch.randn(B, W, C, H, W_img)
    proprio = torch.randn(B, W, 7)  # 7D Input
    goal = torch.randn(B, 64)  # 64D Goal

    print("Input Shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Proprio: {proprio.shape}")
    print(f"  Goal: {goal.shape}")

    # Forward Pass
    try:
        # Returns: pred_action, requery_logit
        action, requery = model(images, proprio, goal)

        print("\nOutput Shapes:")
        print(f"  Action: {action.shape}")
        print(f"  Requery: {requery.shape}")

        # Assertions
        # Action should be (B, W, 7) as per chunking logic in model.forward
        expected_action_shape = (B, W, 7)
        assert action.shape == expected_action_shape, (
            f"Expected Action {expected_action_shape}, got {action.shape}"
        )

        assert requery.shape == (B, W, 1) or requery.shape == (B, 1), (
            f"Unexpected Requery shape {requery.shape}"
        )

        # Test Safety Clamps
        # Pass a huge input through action decoder directly to test clamping
        # Construct large fake embeddings
        print("\nTesting Safety Clamps...")
        large_embeds = torch.randn(B, W, 512) * 1000
        safe_actions = model.action_decoder(large_embeds)

        max_angular = model.action_decoder.max_rot_delta
        max_linear = model.action_decoder.max_pos_delta

        # Check Angular (0:3)
        assert (safe_actions[..., 0:3].abs() <= max_angular + 1e-4).all(), (
            "Angular velocity not clamped!"
        )
        # Check Linear (3:6)
        assert (safe_actions[..., 3:6].abs() <= max_linear + 1e-4).all(), (
            "Linear velocity not clamped!"
        )

        print(
            f"  Max Angular: {safe_actions[..., 0:3].abs().max().item()} (Limit: {max_angular})"
        )
        print(
            f"  Max Linear: {safe_actions[..., 3:6].abs().max().item()} (Limit: {max_linear})"
        )

        print(
            "\n✅ Verification Successful: Dimensions and Safety Clamps match requirements."
        )

    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_model()
