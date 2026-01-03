# c:\Users\thoma\AI Projects\Axis\isaac_sim\my_robot_ext\tasks\env_entry.py

try:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
except ImportError:
    try:
        from isaaclab.envs import ManagerBasedRLEnv
    except ImportError:
        # Fallback for environments where Isaac Lab is not installed (e.g. dev machine)
        class ManagerBasedRLEnv:
            def __init__(self, cfg, **kwargs):
                pass

class MyRobotEnv(ManagerBasedRLEnv):
    """
    A skeleton environment class for RL tasks using the Axis model.
    Inherits from ManagerBasedRLEnv which provides the standard Isaac Lab
    manager-based workflow (Scene, Observations, Rewards, Actions, Events).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Initialize the parent class
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        
        # Determine device
        self.device = self.sim.device if hasattr(self, "sim") else "cpu"
        
        print(f"Initialized MyRobotEnv on device: {self.device}")

    def step(self, action):
        # Custom step logic (if needed beyond what Managers provide)
        # Usually ManagerBasedRLEnv handles this via the ActionManager
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
