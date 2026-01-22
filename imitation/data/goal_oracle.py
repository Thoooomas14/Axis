import torch
import numpy as np


class GoalOracle:
    """
    Generates standardized 64D goal embeddings for robot manipulation tasks.
    
    Goal Vector Layout (64D) - Updated for 13D SE(3) poses:
    | Indices | Component                              |
    |---------|----------------------------------------|
    | 0-2     | Task type one-hot [Move, Pick, Place]  |
    | 3-15    | Start pose (13D): [R_flat, pos, grip]  |
    | 16-28   | Target pose (13D): [R_flat, pos, grip] |
    | 29-37   | Object properties (size, color, shape) |
    | 38-55   | Reserved (zeros)                       |
    | 56-63   | Random noise                           |
    """
    
    def __init__(self, output_dim: int = 64, noise_scale: float = 0.1):
        assert output_dim == 64, "Goal vector must be 64D"
        self.output_dim = output_dim
        self.noise_scale = noise_scale

    def encode_goal(
        self,
        task_type: int,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        object_props: dict = None
    ) -> torch.Tensor:
        """
        Construct standardized 64D goal vector.
        
        Args:
            task_type: 0=Move, 1=Pick, 2=Place
            start_pose: 13D current pose [R_flat(9), pos(3), gripper(1)]
            end_pose: 13D target pose [R_flat(9), pos(3), gripper(1)]
            object_props: Optional dict with 'size', 'color', 'shape'
                - size: (3,) array [width, height, depth]
                - color: (3,) array [r, g, b] normalized to [0, 1]
                - shape: (3,) one-hot array [cube, cylinder, sphere]
        
        Returns:
            (64,) float32 tensor
        """
        goal = np.zeros(64, dtype=np.float32)
        
        # Task type one-hot (indices 0-2)
        if 0 <= task_type < 3:
            goal[task_type] = 1.0
        
        # Start pose (indices 3-15) - 13D SE(3) + gripper
        goal[3:16] = start_pose[:13]
        
        # Target pose (indices 16-28) - 13D SE(3) + gripper
        goal[16:29] = end_pose[:13]
        
        # Object properties (indices 29-37) - optional
        if object_props:
            if 'size' in object_props:
                goal[29:32] = object_props['size']
            if 'color' in object_props:
                goal[32:35] = object_props['color']
            if 'shape' in object_props:  # one-hot [cube, cylinder, sphere]
                goal[35:38] = object_props['shape']
        
        # Reserved (indices 38-55) - zeros
        
        # Random noise for regularization (indices 56-63)
        goal[56:64] = np.random.randn(8) * self.noise_scale
        
        return torch.tensor(goal, dtype=torch.float32)

