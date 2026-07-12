# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train the pose estimator online using rollouts from a trained manipulation policy."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from viserdex.utils import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=6200, help="Number of steps to run the agent.")
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
import torch
from termcolor import colored
import numpy as np
from datetime import datetime
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from viserdex.utils.log_utils import Tee
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from viserdex.utils.wandb_utils import WandbSummaryWriter
import viserdex.tasks  # noqa: F401
from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    # load pre-trained policy for rollouts
    load_run = object_cfg["rollout_policy_run_name"]
    assert args_cli.task in [
        "Isaac-ViserDex-Repose-Allegro-Estimator-v0",
        "Isaac-ViserDex-Repose-Allegro-Estimator-Eval-v0",
        "Isaac-ViserDex-Repose-Allegro-Estimator-Camera-v0",
        "Isaac-ViserDex-Repose-Allegro-Estimator-NoAug-v0",
    ], "Unsupported task for pose estimator training."
    load_run = object_cfg["rollout_policy_run_name"]
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)


    agent_cfg.experiment_name = "pose_estimator"
    old_log_dir = os.path.dirname(resume_path)
    run_name = args_cli.run_name if args_cli.run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "rsl_rl", "pose_estimator", run_name)
    env_cfg.log_dir = log_dir
    env_cfg.agent = agent_cfg

    # Log stdout to file
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "stdout.log")
    log_file = open(log_file_path, "a")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(colored(f"[INFO]: Loading experiment from directory: {log_root_path}", "green"))
    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(colored(f"[INFO]: Loading model checkpoint from: {resume_path}", "green"))
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # initialize writer
    writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    writer.log_config(env_cfg, agent_cfg, agent_cfg.algorithm, agent_cfg.policy)
    # save run ID to file
    with open(os.path.join(log_dir, "wandb_run_id.txt"), "w") as f:
        f.write(writer.get_run_id())

    # reset environment
    obs = env.get_observations()

    step_counter = 0
    with tqdm(total=args_cli.num_steps, desc="Pose estimator rollout") as progress_bar:
        while simulation_app.is_running() and step_counter < args_cli.num_steps:
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, extras = env.step(actions)

            string = f"[INFO]: Step: {step_counter}, "
            for key, value in extras["log"].items():
                if "pose_estimator" in key:
                    string += f"{key}: {value}, "
                    writer.add_scalar(key, value, step_counter)
            string += "\n----------------------------------\n"
            if step_counter % 100 == 0:
                print(colored(string, "green"))

            step_counter += 1
            progress_bar.update(1)

            if step_counter >= args_cli.num_steps:
                print(colored(f"[INFO]: Reached maximum number of steps: {args_cli.num_steps}", "green"))
                break
    writer.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
