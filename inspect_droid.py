
import os
import torch
import numpy as np
import tensorflow as tf
from imitation.data.rtx_stream_loader import RTXStreamLoader
from src.utils.rotation_utils import se3_log

def inspect_data():
    print("Initializing DROID dataloader...")
    # Use the user's settings
    loader = RTXStreamLoader(
        dataset_name='droid',
        data_dir='gs://gresearch/robotics',
        batch_size=16,
        window_size=2,
        loss_horizon=1,
        image_key='exterior_image_1_left',
        shuffle_buffer_size=1
    )
    
    
    # RTXStreamLoader is an IterableDataset, so we can iterate it directly
    # However, it expects to be wrapped in a DataLoader for multi-worker support usually,
    # but for single process inspection we can iterate directly or wrap it.
    ds = loader # It *is* the dataset
    iterator = iter(ds)
    
    print("\n=== Sampling DROID Data (Loss Horizon = 1) ===")
    
    pos_deltas = []
    rot_deltas = []
    
    for i in range(10):  # Check 10 batches
        try:
            batch = next(iterator)
        except StopIteration:
            break
            
        # Props: (B, W, 7)
        # We want the stats between last prop frame and target (horizon=1)
        proprio = batch['proprio']
        target_poses = batch['target_poses']
        
        start_pose = proprio[:, -1, :] # (B, 7)
        end_pose = target_poses[:, 0, :] # (B, 7) -> horizon=1
        
        # Position Delta
        p_start = start_pose[:, 3:6]
        p_end = end_pose[:, 3:6]
        pos_diff = torch.norm(p_end - p_start, dim=1)
        pos_deltas.append(pos_diff)
        
        # Rotation Delta
        # Calculate relative rotation magnitude (angle of rotation)
        # se3_log of (R_start^T @ R_end) gives the relative rotation vector
        r_start = se3_exp(torch.cat([start_pose[:, :3], torch.zeros_like(p_start)], dim=1))[:, :3, :3]
        r_end = se3_exp(torch.cat([end_pose[:, :3], torch.zeros_like(p_end)], dim=1))[:, :3, :3]
        r_rel = torch.bmm(r_start.transpose(1, 2), r_end)
        trace = r_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.acos(cos_theta)
        rot_deltas.append(angle)
        
        print(f"Batch {i}: Mean Pos Delta = {pos_diff.mean().item()*1000:.2f} mm | Mean Rot Delta = {angle.mean().item():.4f} rad")

    all_pos = torch.cat(pos_deltas)
    all_rot = torch.cat(rot_deltas)
    print(f"\nOVERALL Mean Position Delta: {all_pos.mean().item()*1000:.4f} mm")
    print(f"OVERALL Mean Rotation Delta: {all_rot.mean().item():.4f} rad")
    
    if all_pos.mean().item() < 0.001 and all_rot.mean().item() < 0.01:
        print("\nCONCLUSION: Both translational and rotational deltas are tiny.")
        print("DIAGNOSIS: Raw 15Hz control yields tiny steps -> tiny loss.")
        print("ROOT FIX: Scale up loss weights (omega_rot/trans) by ~100x-1000x to meaningful gradient scale.")

if __name__ == "__main__":
    inspect_data()
