# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CRATOS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/apptronik/workspaces/isaac_ros-dev/src/cratos/usd/customized_panda.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"dummy_base_prismatic_y_joint": 0.0, "dummy_base_revolute_z_joint": 0.0}
    ),
    actuators={
        "dummy_base_prismatic_y_actuator": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_prismatic_y_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "dummy_base_revolute_z_actuator": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_revolute_z_joint"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
        ),
    },
)
"""Configuration for the Cratos mobile manipulator."""
