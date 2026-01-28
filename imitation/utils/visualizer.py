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
        
    def check_pypose(self):
        if pp is None:
            print("ERROR: PyPose not found! Trajectory integration will fail (stuck line).")
            print("Please install pypose: pip install pypose")

    def orthonormalize_rotation(self, R):
        """
        Orthonormalize rotation matrices using SVD.
        R: (..., 3, 3)
        Returns: (..., 3, 3) valid rotation matrices
        """
        U, S, V = torch.svd(torch.tensor(R, dtype=torch.float32))
        with torch.no_grad():
             det = torch.det(U @ V.transpose(-2, -1))
             diag = torch.ones_like(S)
             diag[..., -1] = det
             R_new = U @ torch.diag_embed(diag) @ V.transpose(-2, -1)
        return R_new.numpy()

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
            self.check_pypose()
            return np.tile(start_pose_13d, (len(twists_7d), 1))

        # 1. Convert start pose to SE(3) matrix
        start_rot = start_pose_13d[:9].reshape(3, 3)
        
        # Sanitize Rotation! (PyPose is strict)
        start_rot = self.orthonormalize_rotation(start_rot)
        
        start_pos = start_pose_13d[9:12]
        start_grip = start_pose_13d[12]
        
        T_curr = np.eye(4)
        T_curr[:3, :3] = start_rot
        T_curr[:3, 3] = start_pos
        
        # Convert to PyPose SE3
        # Ensure input to mat2SE3 is clean
        T_curr_pp = pp.mat2SE3(torch.tensor(T_curr, dtype=torch.float32).unsqueeze(0)) # (1, 7)
        
        # 2. Integrate Twists
        integrated_poses_pp = []
        integrated_grippers = []
        
        curr_grip = start_grip
        
        # Process twists
        # Process twists
        # Ensure twists are tensor
        if isinstance(twists_7d, torch.Tensor):
            twists_tensor = twists_7d.clone().detach().to(dtype=torch.float32)
        else:
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
            pred_action: (B, H, 7) tensor - predicted twist sequence
            save_prefix: Filename prefix
        """
        try:
            # We visualize the LAST step of the window for the first batch element
            b = 0
            
            # Helper to ensure shape (H, 7)
            if pred_action.dim() == 2: # (B, 7) -> (B, 1, 7)
                pred_action = pred_action.unsqueeze(1)
                
            horizon = pred_action.shape[1]
            
            # 1. Images
            images = batch['images'][b] # (T, 3, H, W)
            # Take last frame
            img = images[-1].cpu().permute(1, 2, 0).numpy() # (H, W, 3)
            
            # 2. Twist Comparison (Comparing average or first step?)
            # Let's verify we have enough GT steps
            gt_action_seq = batch['actions'][b].cpu().numpy() # (GT_H, 7)
            
            # Truncate to min horizon
            common_horizon = min(horizon, gt_action_seq.shape[0])
            
            gt_twist_seq = gt_action_seq[:common_horizon]
            pred_twist_seq = pred_action[b, :common_horizon].detach().cpu().numpy()
            
            # Plot the FIRST step twist for bar chart clarity (or average?)
            gt_twist_0 = gt_twist_seq[0]
            pred_twist_0 = pred_twist_seq[0]
            
            # 3. Trajectory Integration (Horizon steps)
            # Start Pose: Last proprio in window
            start_pose = batch['proprio'][b][-1].cpu().numpy() # (13,)
            
            # Integrate GT (Full sequence)
            # Note: target_poses usually contains the N-step future poses.
            # If we have target_poses, we can just grab the H-th one.
            if 'target_poses' in batch and batch['target_poses'][b].shape[0] >= common_horizon:
                 gt_next_pose = batch['target_poses'][b][common_horizon-1].cpu().numpy()
            else:
                 # Fallback: Integrate GT twists
                 gt_poses = self.integrate_twist(start_pose, gt_twist_seq)
                 gt_next_pose = gt_poses[-1]
                 
            # Integrate Pred
            pred_poses = self.integrate_twist(start_pose, pred_twist_seq)
            pred_next_pose = pred_poses[-1]
            
            # --- Plotting ---
            fig = plt.figure(figsize=(16, 8))
            
            # A. Image
            ax_img = fig.add_subplot(2, 2, 1)
            ax_img.imshow(img)
            ax_img.set_title(f"Step {step} (Input) | Horizon {common_horizon}")
            ax_img.axis('off')
            
            # B. Twist Bar Chart (Step 0)
            ax_twist = fig.add_subplot(2, 2, 2)
            x = np.arange(7)
            width = 0.35
            ax_twist.bar(x - width/2, gt_twist_0, width, label='GT Twist (t=0)', color='green', alpha=0.7)
            ax_twist.bar(x + width/2, pred_twist_0, width, label='Pred Twist (t=0)', color='red', alpha=0.7)
            ax_twist.set_xticks(x)
            ax_twist.set_xticklabels(self.action_labels)
            ax_twist.set_title("Twist Prediction (Step 0 Only)")
            ax_twist.legend()
            ax_twist.grid(True, alpha=0.3)
            
            # C. Trajectory / Next Pose Comparison (Final Pose at Horizon)
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
            
            ax_pos.bar(x_pos - width/2, gt_vec, width, label=f'GT Pose (t+{common_horizon})', color='blue', alpha=0.7)
            ax_pos.bar(x_pos + width/2, pred_vec, width, label=f'Pred Pose (t+{common_horizon})', color='orange', alpha=0.7)
            ax_pos.set_xticks(x_pos)
            ax_pos.set_xticklabels(labels)
            ax_pos.set_title(f"Pose at Horizon t+{common_horizon} (Pos+Grip)")
            ax_pos.legend()
            
            # D. Text Info
            ax_text = fig.add_subplot(2, 2, 4)
            ax_text.axis('off')
            info = f"Movement Norm (GT Steps 0-{common_horizon}): {np.linalg.norm(gt_twist_seq):.4f}\n"
            info += f"Prediction Norm (Steps 0-{common_horizon}): {np.linalg.norm(pred_twist_seq):.4f}\n"
            info += f"Pos Error (at t+{common_horizon}): {np.linalg.norm(gt_pos - pred_pos):.4f}"
            ax_text.text(0.1, 0.5, info, fontsize=12, va='center')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.png")
            plt.savefig(save_path)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error visualizing batch: {e}")
            import traceback
            traceback.print_exc()

    def create_gif(self, step, images, target_actions, pred_actions, gt_poses=None, pred_poses=None, requery_preds=None, inference_times=None, save_prefix='episode', dpi=50):
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
            if gt_poses is not None and isinstance(gt_poses, torch.Tensor): gt_poses = gt_poses.cpu().numpy()
            if pred_poses is not None and isinstance(pred_poses, torch.Tensor): pred_poses = pred_poses.cpu().numpy()
            
            T = images.shape[0]
            
            # Limit frames
            if T > 200:
                print(f"Truncating GIF from {T} to 200 frames.")
                T = 200
            
            # Layout: 2x2 grid (Image, XY, XZ, ZY)
            fig = plt.figure(figsize=(12, 10))
            
            ax_img = fig.add_subplot(2, 2, 1)
            ax_xy = fig.add_subplot(2, 2, 2)
            ax_xz = fig.add_subplot(2, 2, 3)
            ax_zy = fig.add_subplot(2, 2, 4)
            
            # Pre-calculate limits for all plots if poses exist
            pass_poses = (gt_poses is not None and pred_poses is not None)
            if pass_poses:
                margin = 0.05
                all_x = np.concatenate([gt_poses[:, 9], pred_poses[:, 9]])
                all_y = np.concatenate([gt_poses[:, 10], pred_poses[:, 10]])
                all_z = np.concatenate([gt_poses[:, 11], pred_poses[:, 11]])
                
                xlim = (all_x.min()-margin, all_x.max()+margin)
                ylim = (all_y.min()-margin, all_y.max()+margin)
                zlim = (all_z.min()-margin, all_z.max()+margin)

            def update(t):
                ax_img.clear()
                ax_xy.clear()
                ax_xz.clear()
                ax_zy.clear()
                
                # 1. Image
                ax_img.imshow(images[t])
                ax_img.set_title(f"Step {t}")
                ax_img.axis('off')
                
                if pass_poses:
                    # Common plot helper
                    def plot_projection(ax, dim1_idx, dim2_idx, label1, label2, title, limits):
                        # GT
                        ax.plot(gt_poses[:t+1, dim1_idx], gt_poses[:t+1, dim2_idx], label='GT', color='green')
                        # Pred
                        ax.plot(pred_poses[:t+1, dim1_idx], pred_poses[:t+1, dim2_idx], label='Pred', color='red', linestyle='--')
                        
                        # Current Point
                        ax.scatter(gt_poses[t, dim1_idx], gt_poses[t, dim2_idx], c='green', s=50)
                        ax.scatter(pred_poses[t, dim1_idx], pred_poses[t, dim2_idx], c='red', s=50)
                        
                        ax.set_title(title)
                        ax.set_xlabel(label1)
                        ax.set_ylabel(label2)
                        ax.grid(True)
                        
                        # Set limits
                        if limits[0] is not None: ax.set_xlim(limits[0])
                        if limits[1] is not None: ax.set_ylim(limits[1])
                        
                        if t == 0: ax.legend() # Only legend on first frame to save clutter/time? Or just always.

                    start_t = 0
                    
                    # 2. XY (Top View)
                    plot_projection(ax_xy, 9, 10, "X (mm)", "Y (mm)", "XY Projection (Top)", (xlim, ylim))
                    
                    # 3. XZ (Front View)
                    plot_projection(ax_xz, 9, 11, "X (mm)", "Z (mm)", "XZ Projection (Front)", (xlim, zlim))
                    
                    # 4. ZY (Side View - Z vs Y) - as requested
                    plot_projection(ax_zy, 11, 10, "Z (mm)", "Y (mm)", "ZY Projection", (zlim, ylim))
                else:
                    ax_xy.text(0.5, 0.5, "Poses not provided", ha='center')
                
            ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
            save_path = os.path.join(self.save_dir, f"{save_prefix}_step_{step}.gif")
            ani.save(save_path, writer='pillow', dpi=dpi)
            plt.close(fig)
            print(f"Saved GIF to {save_path}")
            
        except Exception as e:
            print(f"Error creating GIF: {e}")

