# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)


@configclass
class SimpleAllegroPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 1000
    experiment_name = "viserdex_ppo"
    logger = "wandb"
    wandb_project = "stable-grasping"
    empirical_normalization = True
    # NOTE: "policy" here is rsl_rl's algorithmic observation *set* name (what the actor/student network
    # consumes), not to be confused with the environment's "policy" observation *group* below — see
    # rsl_rl.env.vec_env.VecEnv.step()'s docstring.
    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
