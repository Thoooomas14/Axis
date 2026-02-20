import torch
import numpy as np
import re


# --- Text Parsing for Object Properties ---

# Color mapping: keyword -> normalized RGB [0, 1]
COLOR_MAP = {
    # Basic colors
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'blue': [0.0, 0.0, 1.0],
    'yellow': [1.0, 1.0, 0.0],
    'orange': [1.0, 0.5, 0.0],
    'purple': [0.5, 0.0, 0.5],
    'pink': [1.0, 0.75, 0.8],
    'white': [1.0, 1.0, 1.0],
    'black': [0.0, 0.0, 0.0],
    'gray': [0.5, 0.5, 0.5],
    'grey': [0.5, 0.5, 0.5],
    'brown': [0.6, 0.3, 0.0],
    # Metallic / household colors
    'silver': [0.75, 0.75, 0.75],
    'gold': [1.0, 0.84, 0.0],
    'bronze': [0.8, 0.5, 0.2],
    'copper': [0.72, 0.45, 0.2],
    'metallic': [0.7, 0.7, 0.7],
    'stainless': [0.8, 0.8, 0.8],
    'chrome': [0.85, 0.85, 0.85],
    'wooden': [0.55, 0.35, 0.15],
    'wood': [0.55, 0.35, 0.15],
    'beige': [0.96, 0.96, 0.86],
    'cream': [1.0, 0.99, 0.82],
    'tan': [0.82, 0.71, 0.55],
    'clear': [0.9, 0.9, 0.95],
    'transparent': [0.9, 0.9, 0.95],
}

# Shape mapping: keyword -> one-hot [cube, cylinder, sphere]
SHAPE_MAP = {
    # Cube-like (index 0)
    'cube': [1.0, 0.0, 0.0],
    'block': [1.0, 0.0, 0.0],
    'box': [1.0, 0.0, 0.0],
    'drawer': [1.0, 0.0, 0.0],
    'container': [1.0, 0.0, 0.0],
    'tray': [1.0, 0.0, 0.0],
    'plate': [1.0, 0.0, 0.0],
    'dish': [1.0, 0.0, 0.0],
    'book': [1.0, 0.0, 0.0],
    'phone': [1.0, 0.0, 0.0],
    'remote': [1.0, 0.0, 0.0],
    'sponge': [1.0, 0.0, 0.0],
    'cloth': [1.0, 0.0, 0.0],
    'towel': [1.0, 0.0, 0.0],
    # Cylinder-like (index 1)
    'cylinder': [0.0, 1.0, 0.0],
    'can': [0.0, 1.0, 0.0],
    'bottle': [0.0, 1.0, 0.0],
    'cup': [0.0, 1.0, 0.0],
    'mug': [0.0, 1.0, 0.0],
    'glass': [0.0, 1.0, 0.0],
    'jar': [0.0, 1.0, 0.0],
    'pot': [0.0, 1.0, 0.0],
    'pan': [0.0, 1.0, 0.0],
    'bowl': [0.0, 1.0, 0.0],
    'bucket': [0.0, 1.0, 0.0],
    'vase': [0.0, 1.0, 0.0],
    'tube': [0.0, 1.0, 0.0],
    'roll': [0.0, 1.0, 0.0],
    'handle': [0.0, 1.0, 0.0],
    # Sphere-like (index 2)
    'sphere': [0.0, 0.0, 1.0],
    'ball': [0.0, 0.0, 1.0],
    'apple': [0.0, 0.0, 1.0],
    'orange': [0.0, 0.0, 1.0],  # fruit
    'egg': [0.0, 0.0, 1.0],
    'tomato': [0.0, 0.0, 1.0],
    'lemon': [0.0, 0.0, 1.0],
}

# Size mapping: keyword -> normalized [width, height, depth] (heuristic)
SIZE_MAP = {
    'tiny': [0.1, 0.1, 0.1],
    'small': [0.25, 0.25, 0.25],
    'medium': [0.5, 0.5, 0.5],
    'large': [0.75, 0.75, 0.75],
    'big': [0.75, 0.75, 0.75],
    'huge': [1.0, 1.0, 1.0],
}

# Default values when no keywords match
DEFAULT_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # black
DEFAULT_SHAPE = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # cube
DEFAULT_SIZE = np.array([0.5, 0.5, 0.5], dtype=np.float32)   # medium


def extract_object_properties(instruction: str) -> dict:
    """
    Extract object properties from a language instruction string.
    
    Args:
        instruction: Natural language instruction (e.g., "pick up the red block")
        
    Returns:
        dict with 'size', 'color', 'shape' arrays (each 3D).
        Defaults to black/cube/medium if instruction is blank or no keywords found.
    """
    # Start with defaults
    result = {
        'size': DEFAULT_SIZE.copy(),
        'color': DEFAULT_COLOR.copy(),
        'shape': DEFAULT_SHAPE.copy(),
    }
    
    if not instruction or not instruction.strip():
        return result
    
    # Lowercase for matching
    text = instruction.lower()
    
    # Extract color
    for keyword, rgb in COLOR_MAP.items():
        if re.search(r'\b' + keyword + r'\b', text):
            result['color'] = np.array(rgb, dtype=np.float32)
            break
    
    # Extract shape
    for keyword, one_hot in SHAPE_MAP.items():
        if re.search(r'\b' + keyword + r'\b', text):
            result['shape'] = np.array(one_hot, dtype=np.float32)
            break
    
    # Extract size
    for keyword, size_vec in SIZE_MAP.items():
        if re.search(r'\b' + keyword + r'\b', text):
            result['size'] = np.array(size_vec, dtype=np.float32)
            break
    
    return result


class GoalOracle:
    """
    Generates standardized 38D goal embeddings for robot manipulation tasks.
    
    Goal Vector Layout (38D):
    | Indices | Component                              |
    |---------|----------------------------------------|
    | 0-2     | Task type one-hot [Move, Pick, Place]  |
    | 3-15    | Start pose (13D): [R_flat, pos, grip]  |
    | 16-28   | Target pose (13D): [R_flat, pos, grip] |
    | 29-31   | Object size [width, height, depth]     |
    | 32-34   | Object color [R, G, B] normalized      |
    | 35-37   | Object shape one-hot [cube, cyl, sph]  |
    """
    
    def __init__(self, output_dim: int = 38):
        assert output_dim == 38, "Goal vector must be 38D"
        self.output_dim = output_dim

    def encode_goal(
        self,
        task_type: int,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        object_props: dict = None
    ) -> torch.Tensor:
        """
        Construct standardized 38D goal vector.
        
        Args:
            task_type: 0=Move, 1=Pick, 2=Place
            start_pose: 13D current pose [R_flat(9), pos(3), gripper(1)]
            end_pose: 13D target pose [R_flat(9), pos(3), gripper(1)]
            object_props: Optional dict with 'size', 'color', 'shape'
                - size: (3,) array [width, height, depth]
                - color: (3,) array [r, g, b] normalized to [0, 1]
                - shape: (3,) one-hot array [cube, cylinder, sphere]
        
        Returns:
            (38,) float32 tensor
        """
        goal = np.zeros(38, dtype=np.float32)
        
        # Task type one-hot (indices 0-2)
        if 0 <= task_type < 3:
            goal[task_type] = 1.0
        
        # Start pose (indices 3-15) - 13D SE(3) + gripper
        goal[3:16] = start_pose[:13]
        
        # Target pose (indices 16-28) - 13D SE(3) + gripper
        goal[16:29] = end_pose[:13]
        
        # Object properties (indices 29-37)
        if object_props:
            if 'size' in object_props and object_props['size'] is not None:
                goal[29:32] = object_props['size']
            if 'color' in object_props and object_props['color'] is not None:
                goal[32:35] = object_props['color']
            if 'shape' in object_props and object_props['shape'] is not None:
                goal[35:38] = object_props['shape']
        
        return goal


if __name__ == '__main__':
    # Quick test
    oracle = GoalOracle(output_dim=38)
    
    # Test with properties
    props = extract_object_properties("pick up the red block")
    print(f"Extracted from 'pick up the red block': {props}")
    
    start = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0.4, 0.0, 0.3, 0.0], dtype=np.float32)
    end = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.0, 0.1, 1.0], dtype=np.float32)
    
    goal = oracle.encode_goal(1, start, end, props)
    print(f"Goal shape: {goal.shape}")
    print(f"Goal vector: {goal}")
    
    # Test with blank instruction
    props_blank = extract_object_properties("")
    print(f"\nExtracted from '': {props_blank}")
    goal_blank = oracle.encode_goal(0, start, end, props_blank)
    print(f"Goal (blank props): {goal_blank}")
