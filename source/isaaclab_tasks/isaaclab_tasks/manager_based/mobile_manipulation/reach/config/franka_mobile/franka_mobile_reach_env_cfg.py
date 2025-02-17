# Copyright (c) 2022-2025, Elevate Robotics
# All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import NonHolonomicActionCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.mobile_manipulation.reach.mobile_reach_env_cfg import (
    MobileReachEnvCfg,
)

##
# Pre-defined configs
##


@configclass
class FrankaMobileReachEnvCfg(MobileReachEnvCfg):
    """Configuration for Franka arm on mobile base reach environment."""

    # Custom Panda USD structure.
    # Links:
    #  - base_link
    #  - panda_link[0-6]
    #  - arm_mount_link
    # Joints:
    #  - dummy_base_prismatic_y_joint (Should have x_joint as well?)
    #  - dummy_base_revolute_z_joint
    #  - panda_joint[1-7]
    #  - panda_arm_mount_joint
    usd_path = "C:\\Users\\duhad\\elevate\\cratos\\usd\\ridgeback_panda.usd"

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Setup mobile Franka using USD file
        # self.scene.robot = RIDGEBACK_FRANKA_PANDA_CFG
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/ridgeback_franka",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    # base
                    "dummy_base_prismatic_y_joint": 0.0,
                    "dummy_base_prismatic_x_joint": 0.0,
                    "dummy_base_revolute_z_joint": 0.0,
                    # franka arm
                    "panda_joint1": 0.0,
                    "panda_joint2": 0.0,  # -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": 0.0,  # -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 0.0,  # 2.0,
                    "panda_joint7": 0.0,  # 0.741,
                    # tool
                    "panda_finger_joint.*": 0.035,
                },
                joint_vel={".*": 0.0},
            ),
            actuators={
                "base": ImplicitActuatorCfg(
                    joint_names_expr=["dummy_base_.*"],
                    velocity_limit=100.0,
                    effort_limit=1000.0,
                    stiffness=0.0,
                    damping=1e5,
                ),
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit=87.0,
                    velocity_limit=100.0,
                    stiffness=800.0,
                    damping=40.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[5-7]"],
                    effort_limit=12.0,
                    velocity_limit=100.0,
                    stiffness=800.0,
                    damping=40.0,
                ),
                "panda_hand": ImplicitActuatorCfg(
                    joint_names_expr=["panda_finger_joint.*"],
                    effort_limit=200.0,
                    velocity_limit=0.2,
                    stiffness=1e5,
                    damping=1e3,
                ),
            },
        )

        print(
            f"[FrankaMobileReachEnvCfg] Initial joint positions: {self.scene.robot.init_state.joint_pos}"
        )

        # self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],  # Only Franka arm joints
        #     body_name="panda_hand",
        #     controller=DifferentialIKControllerCfg(
        #         command_type="pose",
        #         use_relative_mode=True,
        #         ik_method="dls",
        #     ),
        #     scale=0.5,  # Slower arm movement for coordination
        #     body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
        #         pos=[0.0, 0.0, 0.107],  # Offset to end effector
        #     ),
        # )
        self.actions.base_action = NonHolonomicActionCfg(
            asset_name="robot",
            body_name="base_link",
            x_joint_name="dummy_base_prismatic_x_joint",
            y_joint_name="dummy_base_prismatic_y_joint",
            yaw_joint_name="dummy_base_revolute_z_joint",
        )

        # Set target body for rewards
        # self.rewards.ee_position_tracking.params["asset_cfg"].body_names = [
        #     "panda_hand"
        # ]
        # self.rewards.ee_position_tracking_fine.params["asset_cfg"].body_names = [
        #     "panda_hand"
        # ]

        self.commands.ee_pose.ranges.pos_x = (-0.0, 0.0)
        self.commands.ee_pose.ranges.pos_y = (-0.0, 0.0)
        self.commands.ee_pose.ranges.pos_z = (0.0, 0.0)

        # Adjust environment settings
        self.scene.env_spacing = 4.0  # More space for mobile base
        self.episode_length_s = 8.0  # Slightly longer episodes
        self.viewer.eye = (6.0, 6.0, 4.0)  # Wider view for mobile workspace

        # Set target body for commands
        self.commands.ee_pose.body_name = "panda_hand"


@configclass
class FrankaMobileReachEnvCfg_PLAY(FrankaMobileReachEnvCfg):
    """Play configuration with smaller number of environments."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
