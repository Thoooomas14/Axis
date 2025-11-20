import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_rtx_loader_imports():
    print("Testing RTXDataset imports...")
    try:
        from training.data.rtx_loader import RTXDataset
        print("Successfully imported RTXDataset.")
    except ImportError as e:
        print(f"Failed to import RTXDataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print("Test Passed!")

if __name__ == "__main__":
    test_rtx_loader_imports()
