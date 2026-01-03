# Isaac Lab Integration Guide

This repository now supports Reinforcement Learning (RL) and Evaluation using [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

## 1. Directory Structure

The repository is organized to separate the core model from the training backends:

- **`src/models/`**: Contains the shared `AxisModel` and transformer architecture. This is used by both Imitation Learning and Isaac Lab.
- **`imitation/`**: Contains the legacy Imitation Learning training code (formerly `training/`).
- **`isaac_sim/`**: Contains the **Isaac Lab Extension** (`my_robot_ext`). This is a standalone Python package.
- **`scripts/`**: Contains utility scripts, including `eval_isaac.py`.

## 2. The Isaac Lab Extension (`my_robot_ext`)

The code in `isaac_sim/` is structured as a pip-installable package named `my_robot_ext`.

### Components
- **`config/robots.py`**: Defines robot configurations (e.g., `FrankaCfg`, `GoogleRobotCfg`).
- **`tasks/eval_env.py`**: The `AxisEvalEnv` class, which manages the simulation scene, robot spawning, and basic physics stepping.
- **`wrappers/axis_wrapper.py`**: The `AxisObservationWrapper` adapts the environment's single-step observations into the sliding-window format `(B, Window, ...)` required by the Transformer model.

## 3. Installation & Linking

To use this code with your local Isaac Sim / Isaac Lab installation, you must install the `my_robot_ext` package into the Python environment used by Isaac Lab.

### Prerequisites
- **Isaac Sim**: Installed at `c:\isaac-sim` (or similar).
- **Isaac Lab**: Installed at `c:\isaac-lab`.

### Steps to Link
The most reliable way to make Isaac Lab "see" this folder is to install it in **editable mode** using the python executable provided by Isaac Lab.

1.  **Open a Terminal** (PowerShell or Command Prompt).
2.  **Navigate to your Isaac Lab directory**:
    ```powershell
    cd c:\isaac-lab
    ```
3.  **Run the Install Command**:
    Use the `isaaclab.bat` (or `isaaclab.sh` on Linux) wrapper to access the correct python environment. Point it to the `isaac_sim` directory in this repository.

    ```powershell
    # Replace PATH_TO_REPO with the actual path to this Axis repository
    .\isaaclab.bat -p -m pip install -e "c:\Users\thoma\AI Projects\Axis\isaac_sim"
    ```

    **Explanation**:
    - `-p`: Runs the python executable associated with Isaac Lab.
    - `-m pip install -e ...`: Installs the package in "editable" mode. Changes you make to the code will be immediately reflected without reinstalling.

4.  **Verify Installation**:
    Create a file named `test_install.py` with the following content:
    ```python
    import my_robot_ext
    print('Success! installed at:', my_robot_ext.__file__)
    ```
    Then run it:
    ```powershell
    .\isaaclab.bat -p test_install.py
    ```
    *Alternatively, enter the python shell:*
    ```powershell
    .\isaaclab.bat -p
    >>> import my_robot_ext
    >>> print(my_robot_ext.__file__)
    ```

## 4. Running Evaluation

Once linked, you can run the evaluation script. This script loads your trained model and runs it inside the simulation.

1.  **Run the script** using the Isaac Lab python wrapper:

    ```powershell
    cd c:\isaac-lab
    .\isaaclab.bat -p "c:\Users\thoma\AI Projects\Axis\scripts\eval_isaac.py" --checkpoint "c:\Users\thoma\AI Projects\Axis\checkpoints\checkpoint_latest.pt" --robot franka --enable_cameras --video --steps 1000000 --model_refresh 15
    ```

### Configuration
- **Selecting a Robot**: Use `--robot franka` or `--robot google`.
    - *Note*: For the Google Robot, ensure you have updated the USD path in `isaac_sim/my_robot_ext/config/robots.py`.
- **Model Checkpoint**: Ensure the checkpoint path is absolute or relative to where you run the command.

## 5. Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
If running from `c:\isaac-lab`, the script might not find the `src` module if `c:\Users\thoma\AI Projects\Axis` is not in the PYTHONPATH.
The `eval_isaac.py` script tries to fix this automatically, but if it fails, you can add it explicitly:

```powershell
$env:PYTHONPATH = "c:\Users\thoma\AI Projects\Axis;" + $env:PYTHONPATH
.\isaaclab.bat -p ...
```
