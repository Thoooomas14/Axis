import numpy as np
import torch

class MockGeminiPlanner:
    """
    Mocks the output of the Gemini Planner.
    Generates a 77-dimensional vector containing structured goal information.
    
    Structure:
    - target_pose_w (7): [x,y,z,qx,qy,qz,qw] (Start Pose)
    - goal_pose_w (7): [x,y,z,qx,qy,qz,qw] (End Pose)
    - action_type (8): One-hot
    - action_params (10): Vector
    - roi_box (4): [cx,cy,w,h]
    - roi_mask_id (1): Int
    - object_id (1): Int
    - tcp_hint (6): [approach_dir, up_dir]
    - keepout_zones (12): 2x AABB
    - constraints (6): Limits
    - horizon_steps (1): Int
    - confidence (6): Vector
    - flags (8): Bitset
    Total: 77
    """
    
    def __init__(self):
        self.goal_dim = 77
        
    def encode_goal(self, start_pose, end_pose, instruction=""):
        """
        Args:
            start_pose: (7,) np.array or list [x,y,z,qx,qy,qz,qw]
            end_pose: (7,) np.array or list
            instruction: str
        Returns:
            goal_vector: (77,) np.array
        """
        vec = np.zeros(self.goal_dim, dtype=np.float32)
        offset = 0
        
        # 1. Target Pose (Start) - 7
        if start_pose is not None:
            vec[offset:offset+7] = start_pose
        offset += 7
        
        # 2. Goal Pose (End) - 7
        if end_pose is not None:
            vec[offset:offset+7] = end_pose
        offset += 7
        
        # 3. Action Type (8) - One-hot
        # Heuristic: "pick" -> 0, "place" -> 1, "open" -> 5, etc.
        action_idx = 0 # Default 'pick'
        if "place" in instruction: action_idx = 1
        elif "push" in instruction: action_idx = 2
        elif "pull" in instruction: action_idx = 3
        elif "rotate" in instruction: action_idx = 4
        elif "open" in instruction: action_idx = 5
        elif "close" in instruction: action_idx = 6
        elif "idle" in instruction: action_idx = 7
        
        vec[offset + action_idx] = 1.0
        offset += 8
        
        # 4. Action Params (10)
        # Mock values
        vec[offset] = 0.08 # grasp_width
        vec[offset+1] = 0.1 # preferred_height_offset
        vec[offset+2:offset+5] = [0, 0, -1] # push_dir_w (down)
        vec[offset+5] = 0.2 # push_distance
        vec[offset+6] = 0.05 # place_clearance
        vec[offset+7] = 1.0 # speed_scale
        vec[offset+8] = 10.0 # force_limit
        vec[offset+9] = 1.0 # priority
        offset += 10
        
        # 5. ROI Box (4)
        vec[offset:offset+4] = [0.5, 0.5, 0.1, 0.1] # Center
        offset += 4
        
        # 6. ROI Mask ID (1)
        vec[offset] = 1.0
        offset += 1
        
        # 7. Object ID (1)
        vec[offset] = 1.0
        offset += 1
        
        # 8. TCP Hint (6)
        vec[offset:offset+3] = [0, 0, -1] # Approach down
        vec[offset+3:offset+6] = [1, 0, 0] # Up X
        offset += 6
        
        # 9. Keepout Zones (12) - 2 boxes
        offset += 12
        
        # 10. Constraints (6)
        vec[offset] = 0.0 # yaw_lock
        vec[offset+1] = 1.57 # roll_pitch_max
        vec[offset+2] = 20.0 # tcp_force_max
        vec[offset+3] = 0.5 # tcp_vel_max
        vec[offset+4] = 0.1 # joint_margin_min
        vec[offset+5] = 1.0 # keepout_active
        offset += 6
        
        # 11. Horizon Steps (1)
        vec[offset] = 50.0
        offset += 1
        
        # 12. Confidence (6)
        vec[offset:offset+6] = 1.0
        offset += 6
        
        # 13. Flags (8)
        vec[offset] = 1.0 # Valid
        offset += 8
        
        assert offset == self.goal_dim, f"Offset {offset} != Goal Dim {self.goal_dim}"
        
        return vec
