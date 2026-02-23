# Data Pipeline (V2)

Axis V2 supports two data loading modes:
1. **Streaming**: Direct from GCS via RTXStreamLoader (default)
2. **Local**: From preprocessed HDF5 via LocalDataLoader (faster, lower RAM)

## Components

### 1. RTXStreamLoader (`imitation/data/rtx_stream_loader.py`)

The core data loader wrapping `tensorflow_datasets` (TFDS) for streaming.

#### Key Features

- **Streaming**: Uses `tfds.load(..., streaming=True)` to avoid downloading entire datasets
- **Continuous Episodes**: Processes full episodes without subtask splitting
- **Dynamic Goals**: Goal vector updates automatically when gripper state changes
- **13D SE(3) Conversion**: Converts raw 8D poses (pos + quaternion + gripper) to 13D full matrix format
- **Twist Computation**: Calculates ground truth 7D twist actions using PyPose Log map
- **Platform-Aware**: Windows-compatible memory management

#### Episode Processing

Episodes are processed as continuous sequences. The goal vector updates dynamically when the gripper state changes (subtask transitions):

```python
# Gripper change detection
is_closed = grippers > 0.5
gripper_changes = np.where(is_closed[1:] != is_closed[:-1])[0] + 1

# Goal updates at each transition
# Pick: open → closed
# Place: closed → open
# Move: no change
```

#### Twist Computation

Ground truth twists computed via PyPose SE(3) log map:
```python
import pypose as pp

# T_curr, T_next are 4x4 SE(3) matrices
T_curr_pp = pp.mat2SE3(T_curr)
T_next_pp = pp.mat2SE3(T_next)
twist = pp.Log(T_curr_pp.Inv() @ T_next_pp).tensor()
# twist = [ω_x, ω_y, ω_z, v_x, v_y, v_z]  (6D)

# With gripper delta (7D)
twist_7d = [twist, gripper_next - gripper_curr]
```

#### Subtask Types

| Type | Trigger | Description |
|------|---------|-------------|
| **Pick** | Gripper Open → Closed | Approach and grasp object |
| **Place** | Gripper Closed → Open | Move to location and release |
| **Move** | No significant change | Navigate end-effector |

### 2. Goal Oracle (`imitation/data/goal_oracle.py`)

Generates deterministic 38D semantic goal embeddings during training.

#### Goal Vector Layout (38D)

| Indices | Dims | Component |
|---------|------|-----------|
| 0-2 | 3 | Task type one-hot `[Move, Pick, Place]` |
| 3-15 | 13 | Start pose (13D SE(3) full matrix) |
| 16-28 | 13 | End pose (13D SE(3) full matrix) |
| 29-37 | 9 | Object properties (size, color, shape) |

### 3. Output Format

Each batch contains (where H = `loss_horizon` and W = `window_size`):

| Key | Shape | Description |
|-----|-------|--------------|
| `images` | `(B, W, 3, 128, 128)` | RGB images (observation window) |
| `proprio` | `(B, W, 13)` | 13D SE(3) poses (observation window) |
| `goal` | `(B, 38)` | Dynamic goal embedding |
| `actions` | `(B, H, 7)` | 7D twist targets for future H steps |
| `target_poses` | `(B, H, 13)` | Direct future poses for endpoint loss |
| `subtask_end_pose`| `(B, W, 13)`| Target end pose of current subtask |
| `object_props` | `(B, W, 9)` | Numeric object properties if matched |

## Data Flow (Memory-Mapped Architecture)

Both loaders use an Index-Based memory-mapped approach combined with background `multiprocessing` workers:

1. **Episode Buffer**: `num_workers` fetch full episodes into memory simultaneously up to `mix_episodes` (e.g., 8).
2. **Pointer Generation**: Calculates all valid `window_size` + `loss_horizon` start markers.
3. **Global Shuffle**: Pointers are perfectly shuffled across all `mix_episodes` yielding highly IID distributions.
4. **Dynamic Slice**: `(B, W, ...)` batches are sliced efficiently upon iterating, drastically lowering memory footprint vs duplicate buffers.

## Memory Management

The loaders implement proactive out-of-memory protections through `ram_usage_limit`. 
If a background worker detects the system RAM exceeding the limit, it pauses fetching. If a single payload exceeds this natively, the loader throws a fatal `MemoryError` traceback.

## Usage

```python
from imitation.data.rtx_stream_loader import RTXStreamLoader

loader = RTXStreamLoader(
    dataset_name='fractal20220817_data',
    split='train',
    window_size=8,
    mix_episodes=8,
    ram_usage_limit=32.0, # Blocks fetching if usage exceeds 32GB
    use_subprocess=True # Isolate memory
)

# Iterator dynamically slices batches of size 1 (loader output is (W, ...))
iterator = iter(loader)
for batch in iterator:
    images = batch['images']      # (W, 3, 128, 128)
    proprio = batch['proprio']    # (W, 13)
```

---

## Local Data Loading (Recommended for Large Datasets)

For large datasets like DROID, preprocessing once and loading locally is recommended.

### 1. Preprocessor (`imitation/data/preprocessor.py`)

One-time script to extract and save minimal episode data:

```bash
# Preprocess DROID
python -m imitation.data.preprocessor --output E:/data/droid.h5 --dataset droid

# Preprocess Fractal
python -m imitation.data.preprocessor --output E:/data/fractal.h5 \
    --dataset fractal20220817_data --data_dir gs://gresearch/robotics

# Test with subset
python -m imitation.data.preprocessor --output ./test.h5 --max_episodes 10
```

**Stored per episode:**
| Field | Shape | Dtype |
|-------|-------|-------|
| images | (T, 3, 128, 128) | uint8 |
| proprio | (T, 13) | float32 |

### 2. LocalDataLoader (`imitation/data/local_loader.py`)

Loads preprocessed data and computes windowing, goals, twists at load time:

```python
from imitation.data.local_loader import LocalDataLoader

loader = LocalDataLoader(
    data_path='E:/data/droid.h5',
    window_size=8,
    loss_horizon=1,
)

for batch in loader:
    # Same format as RTXStreamLoader
    images = batch['images']      # (B, W, 3, 128, 128)
    proprio = batch['proprio']    # (B, W, 13)
    goal = batch['goal']          # (B, 64)
```

### 3. Train from Local Data

```bash
python imitation/train.py --local_data_path E:/data/droid.h5 --steps 10000
```

> **Note**: When `--local_data_path` is set, streaming args (`--dataset`, `--data_dir`, etc.) are ignored.
