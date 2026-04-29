# Goal Vector Specification (Axis V2)

This document specifies the **38-dimensional goal vector** used by Axis V2 to encode the semantic intent of each robot subtask.

---

## Overview

The Axis goal vector is a dense 38D embedding containing:

- **Determinism**: Same inputs always produce the same embedding
- **Interpretability**: Each dimension has defined semantic meaning
- **Efficiency**: No LLM inference required during training or deployment
- **Completeness**: Contains pose, task type, and object context

---

## Vector Layout

| Indices | Dims | Component | Description |
|---------|------|-----------|-------------|
| **0-2** | 3 | Task Type | One-hot: `[Move, Pick, Place]` |
| **3-15** | 13 | Start Pose | 13D SE(3): `[R_flat(9), Pos(3), Gripper(1)]` |
| **16-28** | 13 | Target Pose | 13D SE(3): `[R_flat(9), Pos(3), Gripper(1)]` |
| **29-31** | 3 | Object Size | Normalized `[width, height, depth]` (from language instruction) |
| **32-34** | 3 | Object Color | RGB normalized to `[0, 1]` (from language instruction) |
| **35-37** | 3 | Object Shape | One-hot: `[Cube, Cylinder, Sphere]` (from language instruction) |

> [!NOTE]
> Object properties are inferred from language instructions using keyword matching.
> If no keywords are found or the instruction is blank, these fields default to:
> - **Size**: Medium `[0.5, 0.5, 0.5]`
> - **Color**: Black `[0.0, 0.0, 0.0]`
> - **Shape**: Cube `[1.0, 0.0, 0.0]`

---

## Component Details

### Task Type (Indices 0-2)

| Index | Task | Trigger |
|-------|------|---------|
| 0 | **Move** | No gripper change |
| 1 | **Pick** | Gripper Open → Closed |
| 2 | **Place** | Gripper Closed → Open |

### Poses (Indices 3-28)

Axis V2 uses **13D full SE(3)** representation:

```
[R_00, R_10, R_20, R_01, R_11, R_21, R_02, R_12, R_22, p_x, p_y, p_z, gripper]
 └──────────── Flattened 3x3 Rotation Matrix ────────────┘  └─ Translation ─┘   └─ Gripper
```

Using the full rotation matrix avoids trigonometric conversions during training.

| Sub-Index | Dims | Component |
|-----------|------|-----------|
| 0-8 | 9 | Flattened rotation matrix (column-major) |
| 9-11 | 3 | Position (x, y, z) |
| 12 | 1 | Gripper state (0=open, 1=closed) |

### Object Properties (Indices 29-37)

| Indices | Component | Encoding |
|---------|-----------|----------|
| 29-31 | Size | `[width, height, depth]` normalized |
| 32-34 | Color | `[R, G, B]` in `[0, 1]` |
| 35-37 | Shape | One-hot `[Cube, Cylinder, Sphere]` |

---

## Example: Constructing Goal Manually

```python
import numpy as np
import pypose as pp
import torch

def construct_goal_vector(
    task_type: int,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    object_props: dict = None
) -> np.ndarray:
    """
    Construct 38D goal vector.
    
    Args:
        task_type: 0=Move, 1=Pick, 2=Place
        start_pose: 13D SE(3) array
        target_pose: 13D SE(3) array
    """
    goal = np.zeros(38, dtype=np.float32)
    
    # Task Type (0-2)
    goal[task_type] = 1.0
    
    # Start Pose (3-15): 13D SE(3)
    goal[3:16] = start_pose
    
    # Target Pose (16-28): 13D SE(3)
    goal[16:29] = target_pose
    
    # Object Properties (29-37)
    if object_props:
        if 'size' in object_props:
            goal[29:32] = object_props['size']
        if 'color' in object_props:
            goal[32:35] = object_props['color']
        if 'shape' in object_props:
            goal[35:38] = object_props['shape']
    
    return goal
```

---

## Example: Using GoalOracle

```python
from imitation.data.goal_oracle import GoalOracle, extract_object_properties
import numpy as np

oracle = GoalOracle(output_dim=38)

# 13D full SE(3) poses: [R_flat(9), pos(3), gripper(1)]
# Identity rotation
start_pose = np.concatenate([np.eye(3).flatten(), [0.4, 0.0, 0.3, 0.0]])
end_pose = np.concatenate([np.eye(3).flatten(), [0.5, 0.0, 0.1, 1.0]])

# Extract properties from instruction
object_props = extract_object_properties("pick up the red block")

goal = oracle.encode_goal(
    task_type=1,  # Pick
    start_pose=start_pose,
    target_pose=end_pose,
    object_props=object_props
)

print(f"Goal shape: {goal.shape}")  # (38,)
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
| **2.3** | 2026-01 | 38D refactor: removed noise/zeros, added text parsing for obj props |
| 2.2 | 2026-01 | 13D full SE(3) poses (flattened rotation matrix), chordal loss |
| 2.1 | 2026-01 | 7D SE(3) poses, configurable noise, confidence requery |
| 2.0 | 2026-01 | Standardized 64D format |
| 1.0 | 2025-06 | Random projection (deprecated) |
