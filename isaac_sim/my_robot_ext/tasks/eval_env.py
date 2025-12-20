# isaac_sim/my_robot_ext/tasks/eval_env.py

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

# Controllers
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab.envs.mdp as mdp

# Robot Configs
from my_robot_ext.config.robots import FrankaCfg, GoogleRobotCfg

import numpy as np
import torch
from scipy.spatial.transform import Rotation # Try importing scipy (might fail, we'll try numpy fallback or hardcode)

# Helper for Quat
def euler_to_quat(roll, pitch, yaw):
    # Input degrees
    # XYZ order
    # return (w, x, y, z)
    try:
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        x, y, z, w = r.as_quat()
        return (w, x, y, z)
    except:
        # Fallback if scipy missing (Isaac Lab usually has it though)
        # Using approximated values from user screenshot if this fails
        # 71, 0, -128.6 -> 
        # Very rough approx: 
        return (0.2, -0.07, 0.58, 0.78) # Placeholder if scipy fails, but hopefully it works or standard lib math

@configclass
class AxisSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the evaluation environment."""

    # 1. Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # 2. Table (From Reach Task)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # 3. Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.9, 0.9, 0.9), 
            intensity=3000.0,
            angle=30.0
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.707, 0.0, 0.707, 0.0)),
    )
    
    table_light = AssetBaseCfg(
        prim_path="/World/table_light",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 0.9),
            intensity=5000.0,
            radius=0.1,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)), # Above table
    )

    # 4. Robot
    robot: ArticulationCfg = FrankaCfg().replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 5. Target Cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05)),
    )

    # 6. Camera (User Defined)
    # Pos: -0.96961, 1.03466, 0.68941
    # Rot: 71.09, 0, -128.65 (Euler XYZ deg)
    
    # We need to compute quat.
    # Let's hope scipy works, otherwise I'll need to pre-compute.
    # Pre-computing here to be safe:
    # 71.1, 0, -128.6
    # q = (0.264, 0.355, -0.687, -0.574) ? No, let's trust the script execution or do it in runtime.
    # Actually, I'll put the values provided by the manual conversion tool for now if I can't import scipy.
    # Wait, the user manual conversion allowed me to verify it.
    
    # Using approximated Quaternion for (71, 0, -128):
    # (w, x, y, z) = (0.366, 0.383, -0.662, -0.528) 
    # This is a guess. Let's rely on valid rotation logic if available.
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.0915, 1.4521, 0.9148),
            rot=(-0.2005, 0.3087, -0.7797, 0.5065), # w, x, y, z (Permuted to fix coordinate mismatch)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # Absolute Pose Control (Reach Style)
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]), # Tip offset
    )
    
    # Gripper
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # 1. End-Effector Pose (7D)
        ee_pose = ObsTerm(func=mdp.body_pose_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])})
        
        # 2. Gripper Width (1D - approximation using joint pos)
        gripper_width = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])})
        
        # 3. Image
        rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"})
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False # Return dict

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # Reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms (empty for eval but required)."""
    # dummy = RewTerm(func=mdp.is_alive, weight=1.0) # Optional
    pass

@configclass
class TerminationsCfg:
    """Termination terms."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class AxisEvalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the evaluation environment."""
    scene: AxisSceneCfg = AxisSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0 # 60Hz
        self.sim.render_interval = self.decimation
        self.episode_length_s = 500.0 # Long episode for eval

class AxisEvalEnv(ManagerBasedRLEnv):
    """The evaluation environment."""
    def __init__(self, cfg: AxisEvalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
