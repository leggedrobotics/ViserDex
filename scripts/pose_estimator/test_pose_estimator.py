# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test a trained pose estimator against a held-out real-world capture dataset."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from viserdex.utils import cli_args
# add argparse arguments
parser = argparse.ArgumentParser(description="Test the pose estimator against a held-out real-world capture dataset.")
parser.add_argument("--save_images", action="store_true", default=False, help="Save evaluation images.")
parser.add_argument("--play_time", type=int, default=None, help="Time to simulate (in sec).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--interpolation", type=str, choices=["linear", "cubic"], default="linear", help="Interpolation method for resizing images.")
parser.add_argument("--dataset_mode", type=str, choices=["easy", "hard"], default="easy", help="Dataset mode to use for evaluation.")
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
import cv2
import sys
import time
import torch
from termcolor import colored
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

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
from viserdex.tasks.manipulation.inhand.mdp.observations import image_to_pose, _get_observation_term_func
from viserdex.tasks.manipulation.inhand.pose_estimator.pose_estimator import PoseEstimator
from viserdex.utils.wandb_utils import WandbSummaryWriter
from scipy.optimize import linear_sum_assignment
from viserdex.assets.objects import TARGET_OBJECT_CFG as target_object_cfg


intrinsics = [615.82031, 0.0, 324.87790, 0.0, 615.93530, 242.42068, 0.0, 0.0, 1.0]
# intrinsics = [615, 0.0, 320, 0.0, 615, 240, 0.0, 0.0, 1.0]
crop_pos = [220, 180]  # Position of the crop in the original image
crop_size = [300, 300]  # Size of the crop in the original image
img_size = [120, 120]  # Size of the resized image
min_x, min_y = crop_pos
max_x, max_y = min_x + crop_size[0], min_y + crop_size[1]
interpolation_method = cv2.INTER_LINEAR if args_cli.interpolation == "linear" else cv2.INTER_CUBIC


def load_and_process_img(rgb_path, depth_path, process_fn, pose_path, device, use_depth=False):
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img[min_y:max_y, min_x:max_x]
    rgb_img = cv2.resize(rgb_img, img_size, interpolation=interpolation_method)
    # rgb_img = cv2.resize(rgb_img, img_size, interpolation=cv2.INTER_CUBIC)
    rgb_img = torch.tensor(rgb_img, dtype=torch.float32, device=device)
    rgb_img = rgb_img.permute(2, 0, 1).clone(memory_format=torch.contiguous_format)  # Convert to (C, H, W)
    if use_depth:
        assert os.path.exists(depth_path), f"Depth path {depth_path} does not exist."
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = torch.tensor(depth_img, dtype=torch.float32, device=device)
        if depth_img.ndim == 2:
            depth_img = depth_img.unsqueeze(0)
        depth_img = depth_img.permute(2, 0, 1).clone(memory_format=torch.contiguous_format)
    else:
        depth_img = None
    # Process images
    rgb_img, depth_img = process_fn(rgb_img, depth_img, apply_mask=False)
    with open(pose_path, "r") as f:
        pose_mat = torch.tensor(np.loadtxt(f, dtype=np.float32), dtype=torch.float32, device=device)
        fix_rot_path = os.path.join(os.path.dirname(pose_path), "fix_rot.txt")
        if os.path.exists(fix_rot_path):
            with open(fix_rot_path, "r") as fr:
                T_fix_rot = torch.tensor(np.loadtxt(fr, comments="#", dtype=np.float32), dtype=torch.float32, device=device)
        else:
            T_fix_rot = torch.eye(4, dtype=torch.float32, device=device)
        pose_mat = torch.matmul(pose_mat, T_fix_rot)
        pos = pose_mat[:3, 3]
        rot = pose_mat[:3, :3]
        # rot = torch.matmul(rot, t)
        quat = math_utils.quat_from_matrix(rot)
    return rgb_img.unsqueeze(0), depth_img.unsqueeze(0) if depth_img is not None else None, pos.unsqueeze(0), quat.unsqueeze(0)


def dataset_generator(dataset_path, process_fn, device="cpu"):
    files = os.listdir(os.path.join(dataset_path, "rgb_masked"))
    img_files = [f for f in files if f.endswith(".png")]
    for img_file in img_files:
        rgb_img_path = os.path.join(dataset_path, "rgb_masked", img_file)
        depth_img_path = os.path.join(dataset_path, "depth", img_file)
        pose_path = os.path.join(dataset_path, "ob_in_cam", img_file.replace(".png", ".txt"))
        yield load_and_process_img(rgb_img_path, depth_img_path, process_fn, pose_path, device), img_file


def unproject_points_to_camera_frame(points_pixels):
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=points_pixels.device).view(3, 3)
    points_pixels = points_pixels.view(-1, 3)  # (P, 3)
    depth = points_pixels[..., -1].clone() + 1e-6  # (P, 1)
    points_pixels[..., -1] = 1.0
    points_pixels[:, 0] = (points_pixels[:, 0] * crop_size[0] / img_size[0]).int()
    points_pixels[:, 1] = (points_pixels[:, 1] * crop_size[1] / img_size[1]).int()
    points_pixels[:, 0] += crop_pos[0]
    points_pixels[:, 1] += crop_pos[1]
    points = torch.matmul(torch.inverse(intrinsics_tensor), points_pixels.unsqueeze(-1)).squeeze(-1)  # (P, 3)
    points = points / points[..., -1].unsqueeze(-1)  # normalize by last coordinate
    points_xyz = points * depth.unsqueeze(-1)  # (P, 3)
    return points_xyz


def project_keypoints(predictions):
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=predictions.device).view(3, 3)
    points_pixels = math_utils.project_points(predictions.view(9, 3), intrinsics_tensor)[..., :2]
    points_pixels[:, 0] -= crop_pos[0]
    points_pixels[:, 1] -= crop_pos[1]
    points_pixels[:, 0] = (points_pixels[:, 0] * img_size[0] / crop_size[0]).int()
    points_pixels[:, 1] = (points_pixels[:, 1] * img_size[1] / crop_size[1]).int()
    return points_pixels.int()


def rigid_procrustes(X, Y, weights=None):
    """
    X, Y: (N,3) numpy arrays of corresponding points (X in object frame, Y in camera frame)
    weights: None or (N,) positive weights
    Returns R (3x3), t (3,), residuals (N,)
    """
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    if weights is None:
        w = np.ones(X.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        assert w.shape[0] == X.shape[0]
    W = w.sum()

    # weighted centroids
    cx = (w[:, None] * X).sum(axis=0) / W
    cy = (w[:, None] * Y).sum(axis=0) / W

    Xc = X - cx
    Yc = Y - cy

    # weighted covariance
    S = ((w[:, None] * Yc).T @ Xc)  # 3x3

    U, _, Vt = np.linalg.svd(S)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    t = cy - R @ cx

    # residuals
    Y_pred = (R @ X.T).T + t
    residuals = np.linalg.norm(Y_pred - Y, axis=1)
    return R, t, residuals


def save_predictions(rgb_img, depth_img, pixels, gt_pixels, output_path):
    fig_rgb, axs_rgb = plt.subplots()
    img = rgb_img.permute(1, 2, 0).cpu().numpy()
    if img.shape[2] == 1:  # grayscale image
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for j in range(pixels.shape[0]):
        img = cv2.drawMarker(img, [pixels[j, 0].item(), pixels[j, 1].item()], (255, 0, 0), 1, 2)
        img = cv2.drawMarker(img, [gt_pixels[j, 0].item(), gt_pixels[j, 1].item()], (0, 255, 0), 0, 2)
        img = cv2.putText(
            img, f"{j}", (pixels[j, 0].item() + 5, pixels[j, 1].item() + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA
        )
        img = cv2.putText(
            img, f"{j}", (gt_pixels[j, 0].item() - 5, gt_pixels[j, 1].item() - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA
        )
    axs_rgb.imshow(img)
    fig_rgb.savefig(output_path)
    plt.close('all')


def get_matching(predictions, gt_input):
    cost_matrix = torch.cdist(predictions.view(-1, 3)[1:, :2], gt_input.view(-1, 3)[1:, :2], p=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
    return np.concatenate([np.array([0]), col_ind + 1])


def get_metrics(pos_in_cam, quat_in_cam, raw_predictions, func: image_to_pose, rgb_img_shape):
    pos_in_cam = pos_in_cam.expand(func._env.num_envs, -1)
    quat_in_cam = quat_in_cam.expand(func._env.num_envs, -1)
    object_pos, object_rot = math_utils.combine_frame_transforms(
        func.camera_pos - func._env.scene.env_origins, func.camera_quat_ros, pos_in_cam, quat_in_cam
    )

    use_camera_frame = func.cfg.params.get("use_camera_frame", True)
    use_pixel_coordinates = func.cfg.params.get("use_pixel_coordinates", True)
    kp_dims = func.kp_dims
    num_kp = func.num_kp

    gt_keypoints = func.compute_keypoints(object_pos, object_rot)

    # Just for testing
    # gt_keypoints_cam = func.compute_keypoints(object_pos, object_rot
    # gt_keypoints_cam_world = func._get_points_in_world_frame(gt_keypoints_cam)
    # assert torch.allclose(gt_keypoints_cam_world.view(-1, 8, 3), gt_keypoints, atol=1e-5), \
    #     "Keypoints in camera frame to world and world frame do not match."

    gt_keypoints = torch.cat([object_pos, gt_keypoints.view(-1, kp_dims - 3)], dim=-1)

    if use_camera_frame:
        gt_keypoints = func._get_points_in_camera_frame(gt_keypoints.view(-1, num_kp, 3)).view(-1, kp_dims)
    else:
        gt_keypoints = gt_keypoints.view(-1, kp_dims)

    if use_pixel_coordinates:
        assert use_camera_frame, "Pixel coordinates can only be used in camera frame."
        gt_pixels = func._project_points_to_image(gt_keypoints.view(-1, num_kp, 3), use_camera_frame)
        gt_pixels[..., 0] = gt_pixels[..., 0] / rgb_img_shape[3]
        gt_pixels[..., 1] = gt_pixels[..., 1] / rgb_img_shape[2]
        gt_pixels = torch.cat([gt_pixels, gt_keypoints.view(-1, num_kp, 3)[..., -1:]], dim=-1).view(-1, kp_dims)

    # Compute the input for the feature extractor
    if use_pixel_coordinates:
        gt_input = gt_pixels.clone()
    else:
        gt_input = gt_keypoints.clone()

    prediction_keypoints = raw_predictions[:, :kp_dims].clone()

    if use_pixel_coordinates:
        pred_pixels = prediction_keypoints.clone()
        prediction_keypoints = prediction_keypoints.view(-1, num_kp, 3)
        prediction_keypoints[..., 0] *= rgb_img_shape[3]
        prediction_keypoints[..., 1] *= rgb_img_shape[2]
        prediction_keypoints = func._unproject_points_to_camera_frame(prediction_keypoints).view(-1, kp_dims)

    log_dict = func.get_metrics(
        raw_pred=raw_predictions[0:1],
        pred_kp=prediction_keypoints[0:1],
        gt_input=gt_input[0:1],
        gt_kp=gt_keypoints[0:1],
        pose_loss=None,
        pred_pixels=pred_pixels[0:1] if use_pixel_coordinates else None,
        gt_pixels=gt_pixels[0:1] if use_pixel_coordinates else None,
    )

    log_prefix = "/".join(list(log_dict.keys())[0].split('/')[:-1])
    t_error_thresh = 0.01  # 1 cm
    r_error_thresh = 10.0  # 10 degrees
    object_frame_pts = torch.cat((torch.zeros(1, 3, device=pred_pixels.device), func.object_frame_pts[0]), dim=0)
    pred_R, pred_t, residuals = rigid_procrustes(object_frame_pts, prediction_keypoints[0:1].view(-1, 3))
    pred_t = torch.tensor(pred_t, dtype=torch.float32, device=prediction_keypoints.device).unsqueeze(0)
    pred_R = torch.tensor(pred_R, dtype=torch.float32, device=prediction_keypoints.device).unsqueeze(0)
    pred_quat_from_kp = math_utils.quat_from_matrix(pred_R)
    translation_error = torch.norm(pred_t - pos_in_cam[0:1], dim=-1).item()
    rotation_error = math_utils.quat_error_magnitude(pred_quat_from_kp, quat_in_cam[0:1]).item() * (180.0 / np.pi)
    accuracy = translation_error < t_error_thresh and rotation_error < r_error_thresh
    log_dict[f"{log_prefix}/mean translation error (m)"] = translation_error
    log_dict[f"{log_prefix}/mean rotation error (deg)"] = rotation_error
    log_dict[f"{log_prefix}/accuracy ({t_error_thresh * 100} cm, {r_error_thresh} deg)"] = accuracy

    return log_dict, locals()


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "feature_extractor", args_cli.load_run)
    log_root_path = os.path.abspath(log_root_path)
    env_cfg.log_dir = log_root_path
    env_cfg.agent = agent_cfg

    # Log stdout to file
    agent_cfg.experiment_name = "feature_extractor"
    agent_cfg.run_name = args_cli.load_run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args_cli.interpolation}"
    suffix += f"_{args_cli.dataset_mode}"
    log_dir = os.path.join(log_root_path, "feature_extractor", f"test{suffix}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "output"), exist_ok=True)

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

    # initialize writer
    if os.path.exists(os.path.join(log_dir, "wandb_run_id.txt")):
        with open(os.path.join(log_dir, "wandb_run_id.txt"), "r") as f:
            agent_cfg.run_id = f.read().strip()
    writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
    writer.log_config(env_cfg, agent_cfg, agent_cfg.algorithm, agent_cfg.policy)

    # reset environment (return value unused; this just forces one observation computation before the
    # dataset-driven test loop below, which doesn't step the policy through the env)
    env.get_observations()
    feature_extractor_term: image_to_pose = _get_observation_term_func(
        env.unwrapped.observation_manager,
        "gaussian_features" if "eval" in args_cli.task.lower() else "features",
        "image_features",
    )
    feature_extractor: PoseEstimator = feature_extractor_term.pose_estimator

    assert feature_extractor_term.cfg.params.get("use_camera_frame", True), \
        "Feature extractor should be in camera frame for evaluation."

    test_dataset_path = target_object_cfg["test_dataset_path"]
    if args_cli.dataset_mode == "hard" and "hard_test_dataset_path" in target_object_cfg:
        test_dataset_path = target_object_cfg["hard_test_dataset_path"]
    errors = defaultdict(list)
    store = defaultdict(list)
    step_count = 0
    # simulate environment
    for (rgb_img, depth_img, pos, quat), file_name in dataset_generator(
        test_dataset_path, process_fn=feature_extractor_term.preprocess_images, device=env.unwrapped.device
    ):
        # run everything in inference mode
        with torch.inference_mode():
            predictions = feature_extractor.test(rgb_img)
            predictions = predictions.to(env.unwrapped.device)
            metrics, values = get_metrics(
                pos_in_cam=pos,
                quat_in_cam=quat,
                raw_predictions=predictions,
                func=feature_extractor_term,
                rgb_img_shape=rgb_img.shape,
            )
            store["file_name"].append(file_name)
            for key in values:
                if key in ["func"]:
                    continue
                store[key].append(values[key])
            for key in metrics:
                metrics[key] = metrics[key]

            for key, value in metrics.items():
                writer.add_scalar(f"test/{key}", value, step_count)
                errors[key].append(value)
            errors["test/feature_extractor/filename"].append(file_name)

            pred_pixels = project_keypoints(values["prediction_keypoints"][0])
            gt_pixels = project_keypoints(values["gt_keypoints"][0])

            # save predictions
            num_plot_pts = pred_pixels.shape[0]
            output_path = os.path.join(log_dir, "output", f"{file_name}_predictions.png")
            save_predictions(rgb_img.squeeze(0), None, pred_pixels[:num_plot_pts], gt_pixels[:num_plot_pts], output_path)

            step_count += 1

    file_name = os.path.join(log_dir, "logs.txt")
    with open(file_name, "w") as f:
        max_key_length = max(len(key.strip('feature_extractor/')) for key in errors.keys())
        for key, value in errors.items():
            if key == "test/feature_extractor/filename":
                continue
            errors_array = np.array(value)
            average = np.average(errors_array)
            variance = np.sqrt(np.average((errors_array - average)**2))
            f.write(f"{key.strip('feature_extractor/').ljust(max_key_length + 2)}: {average:.5f} +- {variance:.5f}\n")
            writer.add_scalar(f"summary/test/{key}/average", average, step_count)
            writer.add_scalar(f"summary/test/{key}/variance", variance, step_count)
    print(colored(f"[INFO]: Saved test logs to {file_name}", "green"))
    store_file = os.path.join(log_dir, "store.pt")
    torch.save(store, store_file)

    writer.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
