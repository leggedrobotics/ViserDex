# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Allegro Hand (Biotac variant) used by ViserDex.

The following configurations are available:

* :obj:`ALLEGRO_HAND_CFG`: Allegro Hand with implicit actuator model.

The USD asset is bundled locally under `viserdex/assets/data/usd/AllegroHand-Biotac/` rather than
streamed from a Nucleus server — it's an ETH Zurich (Robotic Systems Lab) original asset, not the stock
Wonik Robotics model.

"""


import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

VISERDEX_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

ALLEGRO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{VISERDEX_ASSETS_DIR}/usd/AllegroHand-Biotac/allegro_hand_right_biotac.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # rot=(0.0, 0.0, 0.0, 1.0),  # vertical
        rot=(0, 0.7071068, 0, 0.7071068),  # horizontal
        joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.4},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.6,
            velocity_limit_sim=10.0,
            stiffness=7.0,
            damping=0.22,
            friction=0.15,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Allegro Hand robot."""
