# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.dual_pick.mdp as mdp

# TODO
# [X] Re-create successful dual pick with minimal changes:
#   [X] new reward terms weighted to zero
#   [X] new boxes far from target box
#   [X] observations for new boxes (should be ignored by policy)
# [X] Add boxes back in
# [ ] New behavior:
#   [ ] Stable grip (reward box being in gripper)
#   [ ] Handle boxes in different positions
#   [ ] Handle neighboring boxes
#   [ ] Place box in new location
#   [ ] Multiple picks
# [O] Ablation  / experiments:
#   [X] Optional extra boxes: slightly faster convergence
#   [X] Fix waypoint offsets and calculations: less reward, slower convergence
#   [X] Remove waypoints: failed to pick box
#   [X] > max height penalty: lower picks, but not stable yet, longer training
#   [X] low velocity reward: best performance ever, long training still
#   [X] Network size: layers: [256, 128, 64]: faster convergence, better grasp
#   [ ] Randomized box positions
#   [ ] shortened episode length
#   [ ] Symmetric grip reward
#   [ ] IK vs joint control
#   [ ] Observations: add relative positions of grippers to box
#   [ ] Extra boxes
#   [ ] Add waypoint observations
#   [ ] fewer waypoints / no waypoint progress
# [ ] Hyper parameter tuning
#   [X] larger policy / value network size
#   [ ] learning rate
#   [ ] batch size
#   [ ] cartesion actions using IK
# [ ] Add randomization:
#   [ ] box poses
#   [ ] box sizes
#   [ ] box masses
# [ ] More realistic environment:
#   [ ] Non-rigid boxes
#   [ ] Boxes on pallet on foor
#   [ ] More realistic / complex box arrangements
# [ ] Better robot
#   [ ] Elevate arm
#   [ ] Mobile base
#   [ ] Paddle grippers
#   [ ] Vaccum grippers
#   [ ] Shoulder lift and rotate joints that move both arms
# [ ] Get video working


# TODO: Move this to proper config
INCLUDE_EXTRA_BOXES = False


@configclass
class DualPickSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with two robotic arms and a box."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # Box to pick
    target_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.087], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(3.0, 3.0, 3.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    if INCLUDE_EXTRA_BOXES:
        left_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/LeftBox",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.9, 0.275, 0.087], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(3.0, 3.0, 3.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        right_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/RightBox",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.9, -0.275, 0.087], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(3.0, 3.0, 3.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

    # robots
    robot_left: ArticulationCfg = MISSING
    robot_right: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Left arm observations
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_left")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Right arm observations
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_right")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # Box pose observation
        target_box_pose = ObsTerm(
            func=mdp.object_pose,
            params={"object_name": "target_box"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        if INCLUDE_EXTRA_BOXES:
            left_box_pose = ObsTerm(
                func=mdp.object_pose,
                params={"object_name": "left_box"},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )
            right_box_pose = ObsTerm(
                func=mdp.object_pose,
                params={"object_name": "right_box"},
                noise=Unoise(n_min=-0.01, n_max=0.01),
            )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task rewards using dynamic waypoints
    left_gripper_to_waypoint = RewTerm(
        func=mdp.gripper_to_dynamic_waypoint,
        weight=-0.2,
        params={
            "robot_cfg": SceneEntityCfg("robot_left", body_names=["panda_hand"]),
            "box_name": "target_box",
            "is_left_arm": True,
        },
    )
    right_gripper_to_waypoint = RewTerm(
        func=mdp.gripper_to_dynamic_waypoint,
        weight=-0.2,
        params={
            "robot_cfg": SceneEntityCfg("robot_right", body_names=["panda_hand"]),
            "box_name": "target_box",
            "is_left_arm": False,
        },
    )
    waypoint_progress = RewTerm(func=mdp.waypoint_progress, weight=0.4, params={})

    # Lifting reward
    box_lift = RewTerm(
        func=mdp.box_height,
        weight=200.0,
        params={"box_name": "target_box", "min_height": 0.087, "max_height": 0.5},
    )

    box_low_velocity = RewTerm(
        func=mdp.box_low_velocity,
        weight=400.0,
        params={"box_name": "target_box", "min_height": 0.2, "max_height": 0.5},
    )

    # Lifting success bonus
    box_lifted = RewTerm(
        func=mdp.box_lifted,
        weight=400.0,
        params={"box_name": "target_box", "min_height": 0.2, "max_height": 0.4},
    )

    # Regularization
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0001,
    )
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_left")},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_right")},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_left_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left"),
        },
    )

    reset_robot_right_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_target_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),  # Fixed x position
                "y": (0.0, 0.0),  # Fixed y position
                "z": (0.0, 0.0),  # Fixed z position
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},  # Empty dict means zero velocities
            "asset_cfg": SceneEntityCfg("target_box"),
        },
    )

    if INCLUDE_EXTRA_BOXES:
        reset_left_box = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},  # Empty dict means zero velocities
                "asset_cfg": SceneEntityCfg("left_box"),
            },
        )

        reset_right_box = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},  # Empty dict means zero velocities
                "asset_cfg": SceneEntityCfg("right_box"),
            },
        )

    # Note that this basically commands panda_hand to hit the table, because it's not accounting for the
    # 10cm between panda_hand and the fingertips, but "fixing" this results in worse experience.
    waypoint_progress = EventTerm(
        func=mdp.WaypointProgress,
        mode="interval",
        interval_range_s=(0.01, 0.01),  # Run often
        params={
            "left_waypoints": [
                [0.0, 0.25, 0.0],  # Move to pre-grasp position
                [0.0, 0.15, 0.05],  # Move in and lift slightly
                [0.0, 0.15, 0.2],  # Lift higher
            ],
            "right_waypoints": [
                [0.0, -0.25, 0.0],  # Move to pre-grasp position
                [0.0, -0.15, 0.05],  # Move in and lift slightly
                [0.0, -0.15, 0.2],  # Lift higher
            ],
            "left_robot_cfg": SceneEntityCfg("robot_left", body_names=["panda_hand"]),
            "right_robot_cfg": SceneEntityCfg("robot_right", body_names=["panda_hand"]),
            "box_name": "target_box",
            "completion_threshold": 0.05,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    box_fall = DoneTerm(func=mdp.box_height_threshold, params={"box_name": "target_box", "min_height": -0.1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 14_500},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 14_500},
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 14_500},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class DualPickEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dual-arm box picking environment."""

    scene: DualPickSceneCfg = DualPickSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
