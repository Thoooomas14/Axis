# Goal Vector Specification (Axis V2)

This document specifies the **64-dimensional goal vector** used by Axis V2 to encode the semantic intent of each robot subtask.

---

## Overview

The Axis goal vector is a dense 64D embedding containing:

- **Determinism**: Same inputs always produce the same embedding
- **Interpretability**: Each dimension has defined semantic meaning
- **Efficiency**: No LLM inference required during training or deployment
- **Completeness**: Contains pose, task type, and object context

---

## Vector Layout

| Indices | Dims | Component | Description |
|---------|------|-----------|-------------|
| **0-2** | 3 | Task Type | One-hot: `[Move, Pick, Place]` |
| **3-9** | 7 | Start Pose | 7D SE(3) minimal: `[RotVec(3), Pos(3), Gripper(1)]` |
| **10-16** | 7 | Target Pose | 7D SE(3) minimal: `[RotVec(3), Pos(3), Gripper(1)]` |
| **17-19** | 3 | Object Size | Normalized `[width, height, depth]` |
| **20-22** | 3 | Object Color | RGB normalized to `[0, 1]` |
| **23-25** | 3 | Object Shape | One-hot: `[Cube, Cylinder, Sphere]` |
| **26-33** | 8 | Spatial Relations | One-hot spatial target |
| **34-55** | 22 | Reserved | Future expansion (zeros) |
| **56-63** | 8 | Noise | Regularization noise (σ configurable) |

---

## Component Details

### Task Type (Indices 0-2)

| Index | Task | Trigger |
|-------|------|---------|
| 0 | **Move** | No gripper change |
| 1 | **Pick** | Gripper Open → Closed |
| 2 | **Place** | Gripper Closed → Open |

### Poses (Indices 3-16)

Axis V2 uses **7D minimal SE(3)** representation:

```
[φ_x, φ_y, φ_z, p_x, p_y, p_z, gripper]
 └─ Rotation Vector ─┘  └─ Translation ─┘   └ Gripper
```

The rotation vector `φ = θ * axis` encodes rotation about an axis by angle θ (Rodrigues form).

| Sub-Index | Dims | Component |
|-----------|------|-----------|
| 0-2 | 3 | Rotation vector (axis-angle) |
| 3-5 | 3 | Position (x, y, z) |
| 6 | 1 | Gripper state (0=open, 1=closed) |

### Object Properties (Indices 17-25)

| Indices | Component | Encoding |
|---------|-----------|----------|
| 17-19 | Size | `[width, height, depth]` normalized |
| 20-22 | Color | `[R, G, B]` in `[0, 1]` |
| 23-25 | Shape | One-hot `[Cube, Cylinder, Sphere]` |

### Spatial Relations (Indices 26-33)

| Index | Relation |
|-------|----------|
| 26 | Above |
| 27 | Below |
| 28 | Left |
| 29 | Right |
| 30 | Front |
| 31 | Back |
| 32 | Inside |
| 33 | On |

### Noise (Indices 56-63)

Configurable regularization noise:

```python
oracle = GoalOracle(output_dim=64, noise_scale=0.1)
# goal[56:64] = np.random.randn(8) * noise_scale
```

---

## Example: Constructing Goal Manually

```python
import numpy as np
from scipy.spatial.transform import Rotation

def quaternion_to_rotvec(quat):
    """Convert quaternion [x,y,z,w] to rotation vector."""
    return Rotation.from_quat(quat).as_rotvec()

def construct_goal_vector(
    task_type: int,
    start_pose: dict,
    target_pose: dict,
    object_props: dict = None,
    noise_scale: float = 0.1
) -> np.ndarray:
    """
    Construct 64D goal vector.
    
    Args:
        task_type: 0=Move, 1=Pick, 2=Place
        start_pose: dict with 'position'(3), 'quaternion'(4), 'gripper'(float)
        target_pose: dict with 'position'(3), 'quaternion'(4), 'gripper'(float)
    """
    goal = np.zeros(64, dtype=np.float32)
    
    # Task Type (0-2)
    goal[task_type] = 1.0
    
    # Start Pose (3-9): 7D SE(3) minimal
    goal[3:6] = quaternion_to_rotvec(start_pose['quaternion'])
    goal[6:9] = start_pose['position']
    goal[9] = start_pose['gripper']
    
    # Target Pose (10-16): 7D SE(3) minimal
    goal[10:13] = quaternion_to_rotvec(target_pose['quaternion'])
    goal[13:16] = target_pose['position']
    goal[16] = target_pose['gripper']
    
    # Object Properties (17-25)
    if object_props:
        if 'size' in object_props:
            goal[17:20] = object_props['size']
        if 'color' in object_props:
            goal[20:23] = object_props['color']
        if 'shape' in object_props:
            goal[23:26] = object_props['shape']
    
    # Noise (56-63)
    goal[56:64] = np.random.randn(8) * noise_scale
    
    return goal
```

---

## Example: Using GoalOracle

```python
from imitation.data.goal_oracle import GoalOracle
import numpy as np

oracle = GoalOracle(output_dim=64, noise_scale=0.1)

# 7D minimal SE(3) poses
start_pose = np.array([0, 0, 0, 0.4, 0.0, 0.3, 0.0])  # [rotvec, pos, gripper]
end_pose = np.array([0, 0, 0, 0.5, 0.0, 0.1, 1.0])

goal = oracle.encode_goal(
    task_type=1,  # Pick
    start_pose=start_pose,
    end_pose=end_pose
)

print(f"Goal shape: {goal.shape}")  # (64,)
```

---

## Inference Integration

```python
from src.inference import AxisInference

inference = AxisInference('checkpoint.pt', config)

# Generate goal
goal = oracle.encode_goal(task_type=1, start_pose=current, end_pose=target)

# Run inference
result = inference.predict(images, proprio, goal.numpy())

# Check confidence
if result['requery'] < 0.5:
    print("Model uncertain - request new goal")
else:
    print(f"Confident action: {result['action_twist']}")
```

---

## Requery (Confidence Signal)

In V2, the requery output represents **model confidence**:

| Confidence | Meaning | Action |
|------------|---------|--------|
| > 0.7 | High | Proceed with predicted action |
| 0.3 - 0.7 | Medium | Proceed with caution |
| < 0.3 | Low | Request new goal from planner |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.1** | 2026-01 | 7D SE(3) poses, configurable noise, confidence requery |
| 2.0 | 2026-01 | Standardized 64D format |
| 1.0 | 2025-06 | Random projection (deprecated) |
