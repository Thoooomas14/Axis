import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_rtx_loader_imports():
    print("Testing RTXDataset imports...")
    try:
        from training.data.rtx_loader import RTXDataset
        from training.data.mock_loader import MockDataset
        print("Successfully imported RTXDataset.")
        
        # Test MockDataset shapes
        print("Testing MockDataset shapes...")
        ds = MockDataset(batch_size=2, image_size=(128, 128), length=10)
        
        batch = next(iter(ds))
        images, proprio, action, goal, mask = batch
        
        print(f"Images: {images.shape}")
        print(f"Proprio: {proprio.shape}")
        print(f"Action: {action.shape}")
        print(f"Goal: {goal.shape}")
        print(f"Mask: {mask.shape}")
        
        # Verify dimensions (B, T, ...)
        assert images.ndim == 5, "Images should be (B, T, C, H, W)"
        assert proprio.ndim == 3, "Proprio should be (B, T, D)"
        assert mask.ndim == 2, "Mask should be (B, T)"
        
    except ImportError as e:
        print(f"Failed to import RTXDataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print("Test Passed!")

if __name__ == "__main__":
    test_rtx_loader_imports()
