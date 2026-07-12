# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to roll out a trained policy and log pose-estimator metrics without further training."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from viserdex.utils import cli_args
# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate the pose estimator in the environment without further training.")
parser.add_argument("--save_images", action="store_true", default=False, help="Save evaluation images.")
parser.add_argument("--play_time", type=int, default=None, help="Time to simulate (in sec).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch
from termcolor import colored
import numpy as np
from datetime import datetime
from collections import defaultdict

import viserdex.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.assets import RigidObject
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
import isaaclab.utils.math as math_utils

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from viserdex.utils.wandb_utils import WandbSummaryWriter
from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # env_cfg.terminations.object_out_of_reach.params["threshold"] = 0.05

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    policy_run = object_cfg["rollout_policy_run_name"]
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, policy_run, agent_cfg.load_checkpoint)

    agent_cfg.experiment_name = "feature_extractor"
    agent_cfg.run_name = args_cli.load_run
    log_root_path = os.path.join("logs", "rsl_rl", "feature_extractor", args_cli.load_run)
    env_cfg.log_dir = log_root_path
    env_cfg.agent = agent_cfg

    # Log stdout to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, "feature_extractor", f"eval_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    print(colored(f"[INFO]: Loading experiment from directory: {log_root_path}", "green"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env_play.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent_play.yaml"), agent_cfg)

    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # initialize writer
    if os.path.exists(os.path.join(log_dir, "wandb_run_id.txt")):
        with open(os.path.join(log_dir, "wandb_run_id.txt"), "r") as f:
            agent_cfg.run_id = f.read().strip()
    writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    writer.log_config(env_cfg, agent_cfg, agent_cfg.algorithm, agent_cfg.policy)

    # reset environment
    obs = env.get_observations()
    errors = defaultdict(list)

    obj: RigidObject = env.unwrapped.scene["object"]
    default_state = obj.data.default_root_state.clone()
    object_pos = default_state[:, :3]
    object_quat = default_state[:, 3:7]

    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # -------------------- PLOT KEYPOINTS ON IMAGES IN FEATURE EXTRACTOR --------------------
            # agent stepping
            # actions = -1.0 * torch.ones(env.action_space.shape, device=env.unwrapped.device)
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)
            # write state to the object
            # object_pos = default_state[:, :3] + torch.rand_like(object_pos) * 0.03 + env.unwrapped.scene.env_origins
            # object_quat = math_utils.random_orientation(object_quat.shape[0], env.unwrapped.device)
            # root_state = default_state.clone()
            # root_state[:, :3] = object_pos
            # root_state[:, 3:7] = object_quat
            # obj.write_root_state_to_sim(root_state)
            # store logs
            for key in extras["log"].keys():
                if "pose_estimator" in key:
                    errors[key].append(extras["log"][key])
                    writer.add_scalar(f"eval/{key}", extras["log"][key], step_count)

        if args_cli.play_time is not None:
            if env.unwrapped.common_step_counter * env.unwrapped.step_dt >= args_cli.play_time:
                break

        if step_count % 100 == 0:
            print(f"[INFO] Step: {step_count}")

        step_count += 1

    file_name = os.path.join(log_dir, "logs.txt")
    with open(file_name, "w") as f:
        max_key_length = max(len(key.strip('feature_extractor/')) for key in errors.keys())
        for key, value in errors.items():
            errors_array = np.array(value)
            average = np.average(errors_array)
            variance = np.sqrt(np.average((errors_array - average)**2))
            f.write(f"{key.strip('feature_extractor/').ljust(max_key_length + 2)}: {average:.5f} +- {variance:.5f}\n")
            writer.add_scalar(f"summary/eval/{key}/average", average, step_count)
            writer.add_scalar(f"summary/eval/{key}/variance", variance, step_count)

    writer.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
