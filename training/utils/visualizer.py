import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.action_labels = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'g']

    def visualize_batch(self, step, batch, pred_action, save_prefix='viz'):
        """
        Visualizes the first episode in the batch.
        Args:
            step: Current training step
            batch: Dict from RTXStreamLoader
            pred_action: (B, 7) - Prediction for the LAST step in the window.
                         Wait, during training we only predict the last step.
                         To visualize a sequence, we need to run inference on the whole window?
                         Or just visualize the single step prediction?
                         
                         User asked for "10 timesteps over an episode".
                         Since the training loop processes random windows, we can't easily reconstruct a full episode
                         unless we specifically fetch one or accumulate windows.
                         
                         However, the batch contains a WINDOW of 8 steps.
                         We can visualize these 8 steps.
                         But we only have prediction for the LAST step (t=7).
                         
                         Option A: Run inference on all steps in the window (0..7).
                         Option B: Just visualize the last step.
                         
                         Let's do Option A: Run inference on the window steps to show trajectory.
        """
        try:
            # Unpack
            images = batch['images'] # (B, 8, 3, 128, 128)
            target_action = batch['action'] # (B, 7) -> This is the target for the LAST step?
            # Wait, RTXStreamLoader yields:
            # 'action': target_pose (next step)
            # It yields ONE target per window.
            
            # If we want to visualize a sequence, we need to look at the window inputs.
            # The window inputs contain 'proprio' which is the state at each step.
            # We can compare the proprio trajectory?
            # Or we can run the model on the window to predict actions for each step?
            
            # Let's visualize the 8 steps in the window.
            # We will plot the Image for each step.
            # And we will plot the Ground Truth Action (from proprio next step?) vs Predicted Action.
            
            # Actually, 'target_action' in the batch is only for the *last* step of the window.
            # We don't have ground truth actions for the earlier steps in the window readily available in the batch dictionary
            # UNLESS we change the loader to yield them.
            # But the loader yields (Window, Target).
            
            # Compromise: Visualize the 8 images in the window.
            # And visualize the prediction vs target for the LAST step.
            
            # User asked: "shows 10 timesteps over an episode... and where the robot moved compared to what my model predicted"
            # Since we are training on shuffled windows, we don't have a full episode.
            # We only have a window of 8.
            
            # Let's visualize the 8 steps of the window.
            # And for the last step, show the bar chart comparison.
            
            # Search for a sample with significant movement (action norm > threshold)
            # to avoid visualizing boring "pause" frames.
            best_idx = 0
            max_norm = -1.0
            
            # Check up to batch_size samples
            B = images.shape[0]
            for b in range(B):
                # Calculate movement norm (excluding gripper)
                act = target_action[b].cpu().numpy()
                movement_norm = np.linalg.norm(act[:6]) # x,y,z,rx,ry,rz
                
                if movement_norm > max_norm:
                    max_norm = movement_norm
                    best_idx = b
                    
                # If we find a good one, stop early
                if movement_norm > 0.01:
                    break
            
            # Take the best sample found
            imgs = images[best_idx].cpu().permute(0, 2, 3, 1).numpy() # (8, 128, 128, 3)
            tgt_act = target_action[best_idx].cpu().numpy() # (7,)
            pred_act = pred_action[best_idx].detach().cpu().numpy() # (7,)
            
            W = imgs.shape[0]
            
            # Create Figure
            # Row 1: Images
            # Row 2: Action Comparison (Only for last step)
            
            fig = plt.figure(figsize=(16, 6))
            
            # Plot Images
            for i in range(W):
                ax = fig.add_subplot(2, W, i + 1)
                ax.imshow(imgs[i])
                ax.set_title(f"t-{W-1-i}")
                ax.axis('off')
                
            # Plot Action Comparison (Spanning bottom row)
            ax_act = fig.add_subplot(2, 1, 2)
            x = np.arange(7)
            width = 0.35
            
            ax_act.bar(x - width/2, tgt_act, width, label='Ground Truth', color='green', alpha=0.7)
            ax_act.bar(x + width/2, pred_act, width, label='Prediction', color='red', alpha=0.7)
            
            ax_act.set_ylabel('Value')
            ax_act.set_title(f'Action Prediction at Step {step} (Last Frame)')
            ax_act.set_xticks(x)
            ax_act.set_xticklabels(self.action_labels)
            ax_act.legend()
            ax_act.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.png")
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing batch: {e}")
