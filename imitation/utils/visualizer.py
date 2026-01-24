import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import sys

# Try to import pypose, but don't fail if not present (fallback or just warn)
try:
    import pypose as pp
except ImportError:
    pp = None
    print("Warning: PyPose not found. Trajectory integration will be limited.")

class Visualizer:
    def __init__(self, save_dir):
        plt.switch_backend('Agg') # Essential for headless VMs
        self.save_dir = os.path.join(save_dir, 'visualizations')
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except PermissionError:
            print(f"Warning: Permission denied creating {self.save_dir}. Falling back to ./visualizations")
            self.save_dir = './visualizations'
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Twist Labels: Linear Velocity (3), Angular Velocity (3), Gripper (1)
        self.action_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'g']

    def integrate_twist(self, start_pose_13d, twists_7d):
        """
        Integrate a sequence of twists starting from a 13D pose.
        
        Args:
            start_pose_13d: (13,) [R_flat(9), pos(3), grip(1)]
            twists_7d: (H, 7) [xi(6), grip_delta(1)]
            
        Returns:
            trajectory_poses: (H, 13) integrated poses
        """
        if pp is None:
            return np.tile(start_pose_13d, (len(twists_7d), 1))

        # 1. Convert start pose to SE(3) matrix
        start_rot = start_pose_13d[:9].reshape(3, 3)
        start_pos = start_pose_13d[9:12]
        start_grip = start_pose_13d[12]
        
        T_curr = np.eye(4)
        T_curr[:3, :3] = start_rot
        T_curr[:3, 3] = start_pos
        
        # Convert to PyPose SE3
        T_curr_pp = pp.mat2SE3(torch.tensor(T_curr, dtype=torch.float32).unsqueeze(0)) # (1, 7)
        
        # 2. Integrate Twists
        integrated_poses_pp = []
        integrated_grippers = []
        
        curr_grip = start_grip
        
        # Process twists
        # Ensure twists are tensor
        twists_tensor = torch.tensor(twists_7d, dtype=torch.float32)
        
        for i in range(len(twists_tensor)):
            twist = twists_tensor[i]
            xi = twist[:6] # (6,)
            g_delta = twist[6]
            
            # Exp map: T_next = T_curr * Exp(xi)
            # PyPose Exp operates on LieTensor
            # Note: PyPose se3 is (6,)
            Xi = pp.se3(xi.unsqueeze(0)) # (1, 6)
            T_inc = pp.Exp(Xi) # (1, 7)
            
            T_next_pp = T_curr_pp @ T_inc
            
            integrated_poses_pp.append(T_next_pp)
            
            curr_grip += g_delta.item()
            integrated_grippers.append(curr_grip)
            
            T_curr_pp = T_next_pp

        # 3. Convert back to 13D format
        # Stack PyPose objects
        if not integrated_poses_pp:
             return np.array([])
             
        # Use torch.cat explicitly on the underlying tensor data if needed, but pypose might handle list
        # Safest: process one by one or stack logic
        integrated_poses_13d = []
        
        for i, T_pp in enumerate(integrated_poses_pp):
            # Extract Matrix
            mat = T_pp.matrix().squeeze(0).numpy() # (4, 4)
            rot_flat = mat[:3, :3].flatten()
            pos = mat[:3, 3]
            grip = integrated_grippers[i]
            
            pose_13d = np.concatenate([rot_flat, pos, [grip]])
            integrated_poses_13d.append(pose_13d)
            
        return np.array(integrated_poses_13d, dtype=np.float32)

    def visualize_batch(self, step, batch, pred_action, save_prefix='viz'):
        """
        Visualizes the first episode in the batch.
        
        Args:
            step: Current training step
            batch: Dict with 'images', 'proprio', 'actions', 'target_poses'
            pred_action: (B, 7) tensor - predicted twist for LAST step in window
            save_prefix: Filename prefix
        """
        try:
            # We visualize the LAST step of the window for the first batch element
            b = 0
            
            # 1. Images
            images = batch['images'][b] # (T, 3, H, W)
            # Take last frame
            img = images[-1].cpu().permute(1, 2, 0).numpy() # (H, W, 3)
            
            # 2. Twist Comparison
            gt_twist = batch['actions'][b][0].cpu().numpy() # (7,) - assuming loss_horizon=1
            pred_twist = pred_action[b].detach().cpu().numpy() # (7,)
            
            # 3. Trajectory Integration (1 step)
            # Start Pose: Last proprio in window
            start_pose = batch['proprio'][b][-1].cpu().numpy() # (13,)
            
            # Integrate GT
            # We have pre-computed target poses in batch, let's use them directly for GT
            if 'target_poses' in batch:
                gt_next_pose = batch['target_poses'][b][0].cpu().numpy() # (13,)
            else:
                # Fallback integration
                gt_next_pose = self.integrate_twist(start_pose, gt_twist[None, :])[0]
                
            # Integrate Pred
            pred_next_pose = self.integrate_twist(start_pose, pred_twist[None, :])[0]
            
            # --- Plotting ---
            fig = plt.figure(figsize=(16, 8))
            
            # A. Image
            ax_img = fig.add_subplot(2, 2, 1)
            ax_img.imshow(img)
            ax_img.set_title(f"Step {step} (Input)")
            ax_img.axis('off')
            
            # B. Twist Bar Chart
            ax_twist = fig.add_subplot(2, 2, 2)
            x = np.arange(7)
            width = 0.35
            ax_twist.bar(x - width/2, gt_twist, width, label='GT Twist', color='green', alpha=0.7)
            ax_twist.bar(x + width/2, pred_twist, width, label='Pred Twist', color='red', alpha=0.7)
            ax_twist.set_xticks(x)
            ax_twist.set_xticklabels(self.action_labels)
            ax_twist.set_title("Twist Prediction (Velocities)")
            ax_twist.legend()
            ax_twist.grid(True, alpha=0.3)
            
            # C. Trajectory / Next Pose Comparison (Position)
            ax_pos = fig.add_subplot(2, 2, 3)
            # Compare XYZ + Gripper
            # GT
            gt_pos = gt_next_pose[9:12]
            gt_g = gt_next_pose[12]
            gt_vec = np.concatenate([gt_pos, [gt_g]])
            
            # Pred
            pred_pos = pred_next_pose[9:12]
            pred_g = pred_next_pose[12]
            pred_vec = np.concatenate([pred_pos, [pred_g]])
            
            labels = ['x', 'y', 'z', 'g']
            x_pos = np.arange(4)
            
            ax_pos.bar(x_pos - width/2, gt_vec, width, label='GT Pose', color='blue', alpha=0.7)
            ax_pos.bar(x_pos + width/2, pred_vec, width, label='Pred Pose', color='orange', alpha=0.7)
            ax_pos.set_xticks(x_pos)
            ax_pos.set_xticklabels(labels)
            ax_pos.set_title("Next Step Pose (Position + Gripper)")
            ax_pos.legend()
            
            # D. Text Info
            ax_text = fig.add_subplot(2, 2, 4)
            ax_text.axis('off')
            info = f"Movement Norm (GT): {np.linalg.norm(gt_twist[:6]):.4f}\n"
            info += f"Prediction Norm: {np.linalg.norm(pred_twist[:6]):.4f}\n"
            info += f"Pos Error: {np.linalg.norm(gt_pos - pred_pos):.4f}"
            ax_text.text(0.1, 0.5, info, fontsize=12, va='center')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.png")
            plt.savefig(save_path)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error visualizing batch: {e}")
            import traceback
            traceback.print_exc()

    def create_gif(self, step, images, target_actions, pred_actions, requery_preds=None, inference_times=None, save_prefix='episode', dpi=50):
        """
        Creates a GIF visualizing an entire episode, comparing Trajectories.
        Assumes inputs are full episode sequences.
        """
        try:
            import matplotlib.animation as animation
            
            # Convert to numpy
            if isinstance(images, torch.Tensor): images = images.cpu().permute(0, 2, 3, 1).numpy()
            if isinstance(target_actions, torch.Tensor): target_actions = target_actions.cpu().numpy()
            if isinstance(pred_actions, torch.Tensor): pred_actions = pred_actions.cpu().numpy()
            
            T = images.shape[0]
            
            # Limit frames
            if T > 200:
                print(f"Truncating GIF from {T} to 200 frames.")
                T = 200
            
            fig = plt.figure(figsize=(14, 6))
            ax_img = fig.add_subplot(1, 2, 1)
            ax_twist = fig.add_subplot(1, 2, 2)
            
            def update(t):
                ax_img.clear()
                ax_twist.clear()
                
                # 1. Image
                ax_img.imshow(images[t])
                ax_img.set_title(f"Step {t}")
                ax_img.axis('off')
                
                # 2. Twist
                gt = target_actions[t]
                pred = pred_actions[t]
                
                x = np.arange(7)
                width = 0.35
                
                ax_twist.bar(x - width/2, gt, width, color='green', alpha=0.7, label='GT')
                ax_twist.bar(x + width/2, pred, width, color='red', alpha=0.7, label='Pred')
                
                ax_twist.set_ylim(-1.0, 1.0) 
                ax_twist.set_xticks(x)
                ax_twist.set_xticklabels(self.action_labels)
                ax_twist.legend(loc='upper right')
                ax_twist.set_title("Twist")
                ax_twist.grid(True, alpha=0.3)
                
            ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.gif")
            ani.save(save_path, writer='pillow', dpi=dpi)
            plt.close(fig)
            print(f"Saved GIF to {save_path}")
            
        except Exception as e:
            print(f"Error creating GIF: {e}")

