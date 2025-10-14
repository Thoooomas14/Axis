from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import torch

# Path to your built Unity simulation
env = UnityEnvironment(file_name="Build/UR3eSim.x86_64", no_graphics=True)
env.reset()

# Get the first behavior name (e.g., "UR3e?team=0")
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
print(f"Connected to behavior: {behavior_name}")
print(f"Observation space: {[o.shape for o in spec.observation_specs]}")
print(f"Action space: {spec.action_spec.continuous_size} continuous actions")

# Dummy PyTorch model (replace later)
model = torch.nn.Sequential(
    torch.nn.Linear(sum(np.prod(o.shape) for o in spec.observation_specs), 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, spec.action_spec.continuous_size),
    torch.nn.Tanh()
)

# Example interaction loop
for episode in range(3):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    while len(terminal_steps) == 0:
        obs = decision_steps.obs[0]  # numpy array of shape (N, obs_dim)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # Run your model (this will later be your custom controller)
        with torch.no_grad():
            actions = model(obs_tensor).numpy()

        action_tuple = ActionTuple(continuous=actions)
        env.set_actions(behavior_name, action_tuple)
        env.step()  # physics update

        decision_steps, terminal_steps = env.get_steps(behavior_name)

    print(f"Episode {episode} finished")

env.close()
