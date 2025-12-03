import torch
import numpy as np
import hashlib

class GoalOracle:
    """
    Deterministically generates 'Gemini-like' goal embeddings from instruction and poses.
    Replaces the need for a live Gemini instance during training.
    """
    def __init__(self, output_dim=64):
        self.output_dim = output_dim
        # Fixed random projection matrix for stability
        # We use a fixed seed so it's deterministic across runs
        rng = np.random.RandomState(42)
        self.proj = rng.randn(7 + 7 + 32, output_dim).astype(np.float32) # Start(7) + End(7) + InstrHash(32)

    def encode_goal(self, task_type: int, start_pose: np.ndarray, end_pose: np.ndarray) -> torch.Tensor:
        """
        Args:
            task_type: int (0=Move, 1=Pick, 2=Place)
            start_pose: (7,)
            end_pose: (7,)
        Returns:
            (output_dim,) tensor
        """
        # 1. One-hot encode task type
        # We assume 3 types for now
        type_vec = np.zeros(3, dtype=np.float32)
        if 0 <= task_type < 3:
            type_vec[task_type] = 1.0
            
        # 2. Concatenate inputs
        # Ensure poses are 7D
        if start_pose.shape[0] < 7: start_pose = np.pad(start_pose, (0, 7 - start_pose.shape[0]))
        if end_pose.shape[0] < 7: end_pose = np.pad(end_pose, (0, 7 - end_pose.shape[0]))
        
        # Input vector: [Type(3) + Start(7) + End(7)] = 17
        input_vec = np.concatenate([type_vec, start_pose, end_pose]) 
        
        # 3. Project
        # Resize projection matrix if needed or just slice it
        # Original was 46. New is 17.
        # We can just use the first 17 rows of the existing random matrix (if it was large enough)
        # or re-initialize. Since we re-init every time the class is created, we can just change init.
        
        embedding = np.dot(input_vec, self.proj[:input_vec.shape[0]])
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        return torch.tensor(embedding, dtype=torch.float32)
