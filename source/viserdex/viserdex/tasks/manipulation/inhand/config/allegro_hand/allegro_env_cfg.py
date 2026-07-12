# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import viserdex.tasks.manipulation.inhand.inhand_env_cfg as inhand_env_cfg

##
# Pre-defined configs
##
from viserdex.assets.allegro import ALLEGRO_HAND_CFG  # isort: skip


def add_play_cfg(cfg):
    cfg.scene.num_envs = 50
    cfg.observations.policy.enable_corruption = False
    cfg.terminations.time_out = None


@configclass
class ViserDexAllegroEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ViserDexAllegroPoseEstimatorEnvCfg(ViserDexAllegroEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        inhand_env_cfg.pose_estimator_post_init(self)


@configclass
class ViserDexAllegroPoseEstimatorEvalEnvCfg(ViserDexAllegroPoseEstimatorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        inhand_env_cfg.pose_estimator_eval_post_init(self)
        self.scene.robot.spawn.visible = False


@configclass
class ViserDexAllegroPoseEstimatorCameraEnvCfg(ViserDexAllegroPoseEstimatorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        inhand_env_cfg.pose_estimator_camera_post_init(self)
        self.scene.robot.spawn.visible = False


@configclass
class ViserDexAllegroPoseEstimatorUnaugmentedEnvCfg(ViserDexAllegroPoseEstimatorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        inhand_env_cfg.pose_estimator_unaugmented_post_init(self)
        

@configclass
class ViserDexAllegroEnvCfg_PLAY(ViserDexAllegroEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        add_play_cfg(self)
