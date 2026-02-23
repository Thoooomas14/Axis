import os
import pytest
import torch
from imitation.data.local_loader import LocalDataLoader
from imitation.data.rtx_stream_loader import RTXStreamLoader

# Helper to check if we have the local dummy data available
HAS_LOCAL_DATA = os.path.exists('data/debug_processed.h5')

@pytest.mark.skipif(not HAS_LOCAL_DATA, reason="Requires data/debug_processed.h5")
def test_local_dataloader(capfd):
    """Test LocalDataLoader with a known HDF5 file."""
    loader = LocalDataLoader(
        data_path='data/debug_processed.h5',
        window_size=8,
        loss_horizon=1,
        mix_episodes=2,
        ram_usage_limit=32.0,  # Ensure no silent blocking
        shuffle=False
    )
    
    iterator = iter(loader)
    
    # We only test fetching the first batch to ensure pipeline connects
    try:
        batch = next(iterator)
    except StopIteration:
        pytest.fail("LocalDataLoader raised StopIteration immediately without yielding.")
        
    expected_keys = {'images', 'proprio', 'goal', 'actions', 'target_poses', 'subtask_end_pose', 'object_props'}
    assert set(batch.keys()) == expected_keys, f"Batch keys mismatched. Expected {expected_keys}, got {set(batch.keys())}"
    
    # Check that shapes conform to expectations (window_size=8, loss_horizon=1)
    assert batch['images'].shape[0] == 8
    assert batch['proprio'].shape[0] == 8
    assert batch['actions'].shape[0] == 1  # loss_horizon
    assert batch['goal'].shape[0] == 8
    assert batch['subtask_end_pose'].shape[0] == 8

def test_rtx_stream_loader(capfd):
    """
    Test RTXStreamLoader instantiation and basic logic.
    We don't actually fetch a batch unless we have TFDS data available.
    """
    loader = RTXStreamLoader(
        dataset_name='fractal20220817_data',
        split='train',
        window_size=8,
        loss_horizon=1,
        mix_episodes=2,
        ram_usage_limit=32.0,
        use_subprocess=False # Keep it simple for unit test instantiation
    )
    
    # Just assert the interface works and it registers variables properly
    assert loader.dataset_name == 'fractal20220817_data'
    assert loader.window_size == 8
    assert loader.loss_horizon == 1
    assert loader.mix_episodes == 2

@pytest.mark.skipif(not HAS_LOCAL_DATA, reason="Requires data/debug_processed.h5")
def test_oom_protection_local_loader():
    """Verify LocalDataLoader raises MemoryError when ram_usage_limit is natively exceeded."""
    loader = LocalDataLoader(
        data_path='data/debug_processed.h5',
        window_size=8,
        loss_horizon=1,
        mix_episodes=2,
        ram_usage_limit=0.0001,  # Impossibly small limit to trigger OOM fail-safe
        shuffle=False
    )
    
    with pytest.raises(RuntimeError, match="MemoryError"):
        iterator = iter(loader)
        next(iterator)
