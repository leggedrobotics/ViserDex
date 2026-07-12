from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )

        self.log_dir = log_dir

        if cfg.get("run_id") is not None:
            run = wandb.init(project=project, entity=entity, dir=log_dir, id=cfg.get("run_id"), resume="must")
            print(f"[INFO]: Wandb: Resuming run with ID: {cfg.get('run_id')}")
        else:
            try:
                run = wandb.init(project=project, entity=entity, dir=log_dir)
            except wandb.errors.UsageError as e:
                print(f"[ERROR]: Wandb: {e}")
                print("Running with start method 'fork' to avoid wandb error.")
                run = wandb.init(project=project, entity=entity, dir=log_dir, settings=wandb.Settings(start_method='fork'))
        self.run_id = run.id

        # Change generated name to project-number format
        wandb.run.name = cfg["run_name"] if len(cfg["run_name"]) > 0 else wandb.run.name

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

    def get_run_id(self):
        return self.run_id

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": env_cfg})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        super().add_video(tag, vid_tensor, global_step=global_step, fps=fps, walltime=walltime)
        wandb.log({self._map_path(tag): wandb.Video(vid_tensor, fps=fps)}, step=global_step)

    def add_histogram(self, tag, values, global_step=None, bins=64, walltime=None):
        super().add_histogram(
            tag,
            values,
            global_step=global_step,
            bins=bins,
            walltime=walltime,
        )
        wandb.log({self._map_path(tag): wandb.Histogram(values, num_bins=bins)}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        if not isinstance(env_cfg, dict):
            env_cfg = env_cfg.to_dict()
        if not isinstance(runner_cfg, dict):
            runner_cfg = runner_cfg.to_dict()
        if not isinstance(alg_cfg, dict):
            alg_cfg = alg_cfg.to_dict()
        if not isinstance(policy_cfg, dict):
            policy_cfg = policy_cfg.to_dict()
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
