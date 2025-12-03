import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    def __init__(self, save_dir):
        plt.switch_backend('Agg') # Essential for headless VMs
        self.save_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(self.save_dir, exist_ok=True)
        self.action_labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'g']

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
                movement_norm = np.linalg.norm(act[:7]) # x,y,z,qx,qy,qz,qw
                
                if movement_norm > max_norm:
                    max_norm = movement_norm
                    best_idx = b
            
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
            x = np.arange(8)
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

    def create_gif(self, step, images, target_actions, pred_actions, requery_preds, save_prefix='episode'):
        """
        Creates a GIF visualizing an entire episode.
        Args:
            step: Training step
            images: (T, 3, 128, 128) numpy array or tensor
            target_actions: (T, 7) numpy array or tensor
            pred_actions: (T, 7) numpy array or tensor
            requery_preds: (T, 1) numpy array or tensor
        """
        try:
            import matplotlib.animation as animation
            
            # Convert to numpy if needed
            if isinstance(images, torch.Tensor): images = images.cpu().permute(0, 2, 3, 1).numpy()
            if isinstance(target_actions, torch.Tensor): target_actions = target_actions.cpu().numpy()
            if isinstance(pred_actions, torch.Tensor): pred_actions = pred_actions.cpu().numpy()
            if isinstance(requery_preds, torch.Tensor): requery_preds = requery_preds.cpu().numpy()
            
            T = images.shape[0]
            fig = plt.figure(figsize=(12, 6))
            
            # Layout: Image on Left, Action Bar Chart on Right
            ax_img = fig.add_subplot(1, 2, 1)
            ax_act = fig.add_subplot(1, 2, 2)
            
            def update(t):
                ax_img.clear()
                ax_act.clear()
                
                # 1. Image
                ax_img.imshow(images[t])
                ax_img.set_title(f"Step {t}/{T}")
                ax_img.axis('off')
                
                # 2. Action Bar Chart
                x = np.arange(8)
                width = 0.35
                
                tgt = target_actions[t]
                pred = pred_actions[t]
                req = requery_preds[t].item()
                
                ax_act.bar(x - width/2, tgt, width, label='Ground Truth', color='green', alpha=0.7)
                ax_act.bar(x + width/2, pred, width, label='Prediction', color='red', alpha=0.7)
                
                ax_act.set_ylim(-1.0, 1.0) # Assume normalized actions
                ax_act.set_xticks(x)
                ax_act.set_xticklabels(self.action_labels)
                ax_act.legend(loc='upper right')
                ax_act.grid(True, alpha=0.3)
                
                # Requery Indicator
                req_color = 'red' if req > 0.5 else 'gray'
                ax_act.text(0.5, 1.05, f"Requery: {req:.2f}", transform=ax_act.transAxes, 
                            ha='center', fontsize=12, color=req_color, weight='bold')
                
            ani = animation.FuncAnimation(fig, update, frames=T, interval=200)
            
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.gif")
            ani.save(save_path, writer='pillow')
            plt.close()
            print(f"Saved GIF to {save_path}", flush=True)
            
        except Exception as e:
            print(f"Error creating GIF: {e}", flush=True)
