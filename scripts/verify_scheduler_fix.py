
import sys
import os
import torch
import torch.optim as optim
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imitation.utils.scheduler import CosineAnnealingWarmupRestarts

def verify_scheduler_fix():
    print("--- Verifying Scheduler Fix ---")
    
    # Setup
    model = torch.nn.Linear(10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 1. Simulate "Original" Run
    steps = 1000
    warmup = 100
    scheduler_orig = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=steps, 
        max_lr=1e-4, 
        min_lr=1e-6, 
        warmup_steps=warmup
    )
    
    # Step forward 500 steps
    target_step = 500
    for i in range(target_step):
        scheduler_orig.step()
        
    expected_lr = scheduler_orig.get_lr()[0]
    print(f"Step {target_step}: Expected LR = {expected_lr:.8f}")
    
    # 2. Simulate "Legacy Resume" (Fresh scheduler, no state loaded)
    optimizer_new = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler_new = CosineAnnealingWarmupRestarts(
        optimizer_new, 
        first_cycle_steps=steps, 
        max_lr=1e-4, 
        min_lr=1e-6, 
        warmup_steps=warmup
    )
    
    # At this point, scheduler_new is at step 0 (or -1)
    # This is what happens currently on resume -> Desync
    print(f"Fresh Scheduler LR (Start): {scheduler_new.get_lr()[0]:.8f}")
    
    # 3. Apply Fix: Fast-forward
    print(f"Applying fix: calling scheduler.step({target_step})...")
    scheduler_new.step(target_step)
    
    restored_lr = scheduler_new.get_lr()[0]
    print(f"Restored Scheduler LR: {restored_lr:.8f}")
    
    # 4. Compare
    if math.isclose(expected_lr, restored_lr, rel_tol=1e-6):
        print("SUCCESS: Scheduler restored correctly!")
    else:
        print(f"FAILURE: LR Mismatch! Expected {expected_lr:.8f}, Got {restored_lr:.8f}")

    # 5. Verify LR Override
    print("\n--- Verifying LR Override ---")
    new_user_lr = 2e-4 # Doubling the LR
    
    # Simulate loading state (restoring old max_lr)
    # In train.py, we modify base_max_lr and max_lr directly
    scheduler_new.base_max_lr = new_user_lr
    scheduler_new.max_lr = scheduler_new.base_max_lr * (scheduler_new.gamma**scheduler_new.cycle)
    
    overridden_lr = scheduler_new.get_lr()[0]
    expected_overridden = expected_lr * 2.0 # Should be exactly double since we doubled max_lr (linear scaling in cosine?)
    # Cosine schedule is: base + (max - base) * cos(...)
    # If base (min_lr) is small, it scales roughly linearly with max_lr.
    # Let's calculate exact expectation:
    # LR = min_lr + (max_lr - min_lr) * factor
    # If we double max_lr, we change the slope and peak.
    # The factor depends only on steps, which are preserved.
    # So new_lr = min + (2*max - min) * factor
    # old_lr = min + (max - min) * factor
    # So it won't be EXACTLY 2.0x if min_lr > 0, but since min_lr=1e-6 and max=1e-4, it is very close.
    
    print(f"Old LR: {restored_lr:.8f}")
    # Re-calculate expected manually for precision
    # factor = (restored_lr - min_lr) / (old_max - min_lr)
    min_lr = 1e-6
    factor = (restored_lr - min_lr) / (1e-4 - min_lr)
    exact_expected = min_lr + (new_user_lr - min_lr) * factor
    
    print(f"New LR Target (calc): {exact_expected:.8f}")
    print(f"Actual New LR: {overridden_lr:.8f}")
    
    if math.isclose(exact_expected, overridden_lr, rel_tol=1e-6):
        print("SUCCESS: LR Override works!")
    else:
        print("FAILURE: LR Override failed.")

if __name__ == "__main__":
    verify_scheduler_fix()
