# isaac_sim/my_robot_ext/tasks/eval_env.py

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Controllers
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

import isaaclab.envs.mdp as mdp

# Robot Configs
from my_robot_ext.config.robots import FrankaCfg


@configclass
class AxisSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the evaluation environment."""

    # 1. Ground Plane (Replaced by Room Floor)
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )

    # 1. Room Environment
    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.05), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )

    # 2. Table
    # 2. Table (Replaced with specific Matte Material)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.05),  # Approximate table size
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1),  # Dark Grey
                metallic=0.0,  # No metal
                roughness=1.0,  # Fully matte
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, -0.025)
        ),  # Top surface at Z=0
    )

    # 3. Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=5000.0),
    )

    # 4. Robot
    robot: ArticulationCfg = FrankaCfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.0)
    robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # Identity
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -1.2,  # Lower shoulder
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,  # Adjust elbow
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,  # 90 deg
        "panda_joint7": 0.785,  # 45 deg
        "panda_finger_joint1": 0.04,
        "panda_finger_joint2": 0.04,
    }

    # 5. Target Cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0.05)),
    )
    # 6. Camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=35.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.7, 0.5, 0.4),  # Moved back to match training distribution?
            rot=(0.37527, -0.46577, 0.62404, -0.50279),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # Absolute Pose Control
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="pinv"
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.107]
        ),  # Tip offset
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

        # 1. Wrist Pose (Stable Rotation)
        ee_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        # 2. Finger Tips (Accurate Position)
        tool_tips = ObsTerm(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["panda_leftfinger", "panda_rightfinger"]
                )
            },
        )

        # 3. Gripper Width
        gripper_width = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_finger_joint.*"]
                )
            },
        )

        # 4. Image
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
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
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 500.0


class AxisEvalEnv(ManagerBasedRLEnv):
    """The evaluation environment."""

    def __init__(self, cfg: AxisEvalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
