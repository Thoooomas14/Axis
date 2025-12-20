try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg
    from omni.isaac.lab.actuators import ImplicitActuatorCfg
    from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
except ImportError:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration for specific robots
##

class FrankaCfg(ArticulationCfg):
    """Configuration for the Franka Emika Panda robot."""
    def __init__(self, **kwargs):
        super().__init__(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            ),
            actuators={
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit=87.0,
                    velocity_limit=2.175,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit=12.0,
                    velocity_limit=2.61,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit=200.0,
                    velocity_limit=0.2,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
        )

class GoogleRobotCfg(ArticulationCfg):
    """Configuration for the Google Robot (Placeholder)."""
    def __init__(self, **kwargs):
        super().__init__(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                # PLACEHOLDER PATH - User needs to update this
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Google/google_robot_instanceable.usd",
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                # PLACEHOLDER JOINTS
                joint_pos={
                    "joint_1": 0.0,
                    "joint_2": 0.0,
                    # ...
                },
            ),
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=400.0,
                    damping=40.0,
                ),
            },
        )
