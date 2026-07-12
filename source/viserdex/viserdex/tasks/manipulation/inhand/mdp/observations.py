# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg, ManagerTermBase
from isaaclab.sensors import TiledCamera
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab_tasks.manager_based.manipulation.inhand.mdp.commands import InHandReOrientationCommand
from viserdex.renderer import GaussianSplatCamera
from viserdex.tasks.manipulation.inhand.mdp.masking import DiscretizedJointPositionsMasker
from viserdex.tasks.manipulation.inhand.pose_estimator import PoseEstimator, PoseEstimatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.io import dump_yaml
import h5py


class image_to_pose(ManagerTermBase):
    """Pose extractor for the in-hand manipulation task."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.camera: TiledCamera = env.scene[cfg.params["camera_cfg"].name]
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._robot: Articulation = env.scene[SceneEntityCfg('robot').name]

        self.pts_path: str = cfg.params["points_path"]
        self.object_frame_pts = torch.load(self.pts_path).to(env.device).unsqueeze(0).repeat(env.num_envs, 1, 1)
        assert self.object_frame_pts.shape[-1] == 3, "Points should be in 3D space."
        # number of keypoints for the in-hand cube: 8 corners + 1 center
        self.num_kp = self.object_frame_pts.shape[1] + 1

        self.pose_estimator: PoseEstimator = PoseEstimator(
            num_outputs=self.num_predictions,
            cfg=cfg.params["pose_estimator_cfg"],
            device=env.device,
            log_dir=os.path.join(env.cfg.log_dir, "pose_estimator")
        )

        joint_positions_masker = None
        if cfg.params.get("masking_data") is not None:
            joint_positions_masker = DiscretizedJointPositionsMasker(
                data_path=cfg.params["masking_data"], env=env, robot=SceneEntityCfg('robot')
            )
        self._joint_positions_masker: DiscretizedJointPositionsMasker | None = joint_positions_masker

        if cfg.params["debug_vis"]:
            self.keypoint_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/myKeypointMarkers",
                markers={
                    "sphere_pred": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "sphere_gt": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self.keypoint_visualizer = VisualizationMarkers(self.keypoint_marker_cfg)

    @property
    def num_predictions(self):
        return 3 * self.num_kp

    @property
    def kp_dims(self):
        return 3 * self.num_kp

    @property
    def camera_pos(self):
        return self.camera.data.pos_w

    @property
    def camera_quat_ros(self):
        return self.camera.data.quat_w_ros

    def __str__(self):
        return self.cfg.params["camera_cfg"].name

    def reset(self, env_ids: list[int] | None = None):
        self.raw_predictions = torch.zeros((self._env.num_envs, self.num_predictions), device=self.device)

    def get_metrics(
        self, raw_pred, pred_kp, gt_input, gt_kp, pose_loss=None, pred_pixels=None, gt_pixels=None, name=None,
    ):
        kp_dims = self.kp_dims
        num_kp = self.num_kp
        name = name if name is not None else self.__str__()
        pose_estimator = self.pose_estimator

        log_dict = dict()
        log_prefix = f"pose_estimator/{name}"

        mean_kp_pred_error = (raw_pred - gt_input)[:, :kp_dims].view(-1, num_kp, 3).norm(dim=-1).mean()
        mean_kp_metric_error = (pred_kp - gt_kp).view(-1, num_kp, 3).norm(dim=-1).mean()

        if pose_loss is not None:
            log_dict[f"{log_prefix}/loss"] = pose_loss.detach().item()
            log_dict[f"{log_prefix}/lr"] = pose_estimator.optimizer.state_dict()['param_groups'][0]['lr']
        log_dict[f"{log_prefix}/mean_kp_error (raw)"] = mean_kp_pred_error.detach().item()
        log_dict[f"{log_prefix}/mean_kp_error (m)"] = mean_kp_metric_error.detach().item()

        if gt_pixels is not None:
            mean_pixel_error = (pred_pixels - gt_pixels).view(-1, num_kp, 3)[:, :, :2].norm(dim=-1).mean()
            mean_depth_error = (pred_pixels - gt_pixels).view(-1, num_kp, 3)[:, :, 2].abs().mean()
            log_dict[f"{log_prefix}/mean_pixel_error (pixels)"] = mean_pixel_error.detach().item()
            log_dict[f"{log_prefix}/mean_depth_error (m)"] = mean_depth_error.detach().item()

        return log_dict

    def preprocess_images(self, rgb_img, depth_img, apply_mask=True):
        # mask out pixels with depth >= 1.0 and depth is infinity
        if depth_img is not None:
            rgb_img[depth_img.expand(*rgb_img.shape) >= 1.0] = 0.0
        if apply_mask:
            if self._joint_positions_masker is not None:
                rgb_img, depth_img = self._joint_positions_masker.apply_masks(rgb_img, depth_img)
        if depth_img is not None:
            # set pixels with depth >= 1.0 to 0.0 to match the splat image
            depth_img[depth_img >= 1.0] = 0.0
        return rgb_img, depth_img

    def compute_keypoints(self, object_pos, object_rot):
        pose = torch.cat((object_pos, object_rot), dim=1).view(-1, 1, 7).expand(-1, self.num_kp - 1, 7)
        keypoints = pose[..., :3] + math_utils.quat_apply(pose[..., 3:7], self.object_frame_pts)
        return keypoints

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pose_estimator_cfg: PoseEstimatorCfg,
        camera_cfg: SceneEntityCfg,
        points_path: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg('object'),
        masking_data: str | None = None,
        debug_vis: bool = False,
    ):
        # Prepare the images from the camera
        assert (self.camera._is_outdated.all() or env.common_step_counter == 0), \
            "Camera is already updated before computing the images. Check where data is being accessesd."
        rgb_img = self.camera.data.output["rgb"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)
        depth_img = self.camera.data.output["distance_to_image_plane"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)

        # preprocess the images
        rgb_img, depth_img = self.preprocess_images(rgb_img, depth_img)

        # get the object pose from the environment
        object_pos = self._object.data.root_pos_w - self._env.scene.env_origins
        object_rot = self._object.data.root_quat_w

        # generate ground truth keypoints for in-hand cube
        gt_keypoints = self.compute_keypoints(object_pos, object_rot)
        # add the center point of the cube
        gt_keypoints = torch.cat([object_pos, gt_keypoints.view(-1, self.kp_dims - 3)], dim=-1)
        gt_keypoints = self._get_points_in_camera_frame(gt_keypoints.view(-1, self.num_kp, 3)).view(-1, self.kp_dims)
        self.gt_keypoints = gt_keypoints

        gt_pixels = self._project_points_to_image(self.gt_keypoints.view(-1, self.num_kp, 3))
        gt_pixels[..., 0] = gt_pixels[..., 0] / rgb_img.shape[3]
        gt_pixels[..., 1] = gt_pixels[..., 1] / rgb_img.shape[2]
        self.gt_pixels = torch.cat(
            [gt_pixels, self.gt_keypoints.view(-1, self.num_kp, 3)[..., -1:]], dim=-1
        ).view(-1, self.kp_dims)
        self.gt_input = self.gt_pixels.clone()

        # train CNN to regress on keypoint positions
        raw_predictions, pose_loss = self.pose_estimator.step(rgb_img, self.gt_input)
        raw_predictions = raw_predictions.to(self.device)
        pose_loss = pose_loss.to(self.device) if pose_loss is not None else None
        self.raw_predictions = raw_predictions.clone()

        raw_predictions = raw_predictions.clone().detach()
        prediction_keypoints = raw_predictions[:, :self.kp_dims].clone()
        prediction_quat = raw_predictions[:, self.kp_dims:].clone()

        pred_pixels = prediction_keypoints.clone()
        prediction_keypoints = prediction_keypoints.view(-1, self.num_kp, 3)
        prediction_keypoints[..., 0] *= rgb_img.shape[3]
        prediction_keypoints[..., 1] *= rgb_img.shape[2]
        prediction_keypoints = self._unproject_points_to_camera_frame(prediction_keypoints).view(-1, self.kp_dims)

        self.prediction_keypoints = prediction_keypoints

        if debug_vis:
            self._save_images_grid(rgb_img, depth_img)
            prediction_keypoints_world = self._get_points_in_world_frame(prediction_keypoints)
            gt_keypoints_world = self._get_points_in_world_frame(self.gt_keypoints)
            self._show_keypoint_markers(prediction_keypoints_world, gt_keypoints_world)
            self._save_keypoint_images(rgb_img, prediction_keypoints, self.gt_keypoints)

        log_dict = self.get_metrics(
            raw_pred=raw_predictions,
            pred_kp=prediction_keypoints,
            gt_input=self.gt_input,
            gt_kp=self.gt_keypoints,
            pose_loss=pose_loss,
            pred_pixels=pred_pixels,
            gt_pixels=self.gt_pixels,
        )
        # `ManagerBasedRLEnv` has no `log_info()` convenience method in public Isaac Lab; write directly
        # into the same `extras["log"]` dict it populates itself
        self._env.extras.setdefault("log", {}).update(log_dict)

        ret = self._get_points_in_world_frame(prediction_keypoints)
        return ret

    def _save_images_grid(self, rgb_img, depth_img, max_images: int | None = 100):
        """Writes image buffers to file."""
        output_dir = os.path.join(self._env.cfg.log_dir, "pose_estimator", "camera_images")
        os.makedirs(output_dir, exist_ok=True)
        if max_images is not None and rgb_img.shape[0] > max_images:
            rgb_img = rgb_img[max_images:]
        save_image(make_grid(rgb_img / 255.0, nrow=round(rgb_img.shape[0] ** 0.5)), os.path.join(output_dir, f"{self}_rgb_grid.png"))
        if depth_img is not None:
            if max_images is not None and depth_img.shape[0] > max_images:
                depth_img = depth_img[:max_images]
            save_image(make_grid(depth_img, nrow=round(depth_img.shape[0] ** 0.5)), os.path.join(output_dir, f"{self}_depth_grid.png"))

    def _show_keypoint_markers(self, prediction_keypoints, gt_keypoints):
        marker_indices = [0] * 3 * self._env.num_envs + [1] * 3 * self._env.num_envs
        pred_pos = prediction_keypoints.view(-1, self.num_kp, 3)[:, 1:4, :] + self._env.scene.env_origins.unsqueeze(1)
        gt_pos = gt_keypoints.view(-1, self.num_kp, 3)[:, 1:4, :] + self._env.scene.env_origins.unsqueeze(1)
        marker_pos = torch.cat((pred_pos, gt_pos), dim=0).view(-1, 3)
        self.keypoint_visualizer.visualize(marker_pos, marker_indices=marker_indices)

    def _get_points_in_world_frame(self, points_cam_frame):
        camera_pos = self.camera_pos - self._env.scene.env_origins
        camera_quat_ros = self.camera_quat_ros
        points_cam_frame = points_cam_frame.view(points_cam_frame.shape[0], -1, 3).clone()
        points, _ = math_utils.combine_frame_transforms(
            camera_pos.unsqueeze(1).tile(1, points_cam_frame.shape[1], 1).view(-1, 3),
            camera_quat_ros.tile(1, points_cam_frame.shape[1], 1).view(-1, 4),
            points_cam_frame.view(-1, 3),
            None
        )
        return points.view(-1, points_cam_frame.shape[1] * 3)

    def _get_points_in_camera_frame(self, points):
        camera_pos = self.camera_pos - self._env.scene.env_origins
        camera_quat_ros = self.camera_quat_ros
        points_cam_frame, _ = math_utils.subtract_frame_transforms(
            camera_pos.unsqueeze(1).tile(1, points.shape[1], 1).view(-1, 3),
            camera_quat_ros.tile(1, points.shape[1], 1).view(-1, 4),
            points.view(-1, 3),
            None
        )
        points_cam_frame = points_cam_frame.view(*points.shape)
        return points_cam_frame

    def _project_points_to_image(self, points, camera_frame: bool = True):
        """Projects points from world frame to camera frame and then to pixel coordinates.
        Args:
            points (torch.Tensor): Points in world frame of shape (N, P, 3).
        Returns:
            points_pixels (torch.Tensor): Projected points in pixel coordinates of shape (N, P, 2).
        """
        intrinsics = self.camera.data.intrinsic_matrices.clone()
        if not camera_frame:
            points_cam_frame = self._get_points_in_camera_frame(points)
        else:
            points_cam_frame = points
        points_pixels = math_utils.project_points(points_cam_frame, intrinsics)[..., :2]
        return points_pixels

    def _unproject_points_to_camera_frame(self, points_pixels):
        """Unprojects points from pixel coordinates to camera frame.
        Args:
            points_pixels (torch.Tensor): Points in pixel coordinates of shape (N, P, 3).
        Returns:
            points (torch.Tensor): Unprojected points in camera frame of shape (N, P, 3).
        """
        intrinsics = self.camera.data.intrinsic_matrices.clone().unsqueeze(1)  # (N, 1, 3, 3)
        depth = points_pixels[..., -1].clone() + 1e-6  # (N, P, 1)
        points_pixels[..., -1] = 1.0
        points = torch.matmul(torch.inverse(intrinsics), points_pixels.unsqueeze(-1)).squeeze(-1)  # (N, P, 3)
        points = points / points[..., -1].unsqueeze(-1)  # normalize by last coordinate
        points_xyz = points * depth.unsqueeze(-1)  # (N, P, 3)
        return points_xyz

    def _save_keypoint_images(self, rgb_img, prediction_keypoints, gt_keypoints):
        gt_pixels = self._project_points_to_image(gt_keypoints.view(-1, self.num_kp, 3))
        gt_pixels = gt_pixels.round().int()
        pred_pixels = self._project_points_to_image(prediction_keypoints.view(-1, self.num_kp, 3))
        pred_pixels = pred_pixels.round().int()

        grid_size = 2
        num_envs_to_show = grid_size * grid_size
        fig, axs = plt.subplots(grid_size, grid_size)
        envs_to_show = torch.arange(num_envs_to_show)
        envs_to_show = torch.randint(high=self._env.num_envs, size=(num_envs_to_show,))
        for i, e_i in enumerate(envs_to_show):
            img = np.ascontiguousarray(np.moveaxis(rgb_img[e_i].cpu().numpy(), 0, -1).astype(np.uint8))
            for j in range(gt_pixels.shape[1]):
                img = cv2.circle(img, [gt_pixels[e_i, j, 0].item(), gt_pixels[e_i, j, 1].item()], 2, (0, 255, 0), 1)
            for j in range(pred_pixels.shape[1]):
                img = cv2.drawMarker(img, [pred_pixels[e_i, j, 0].item(), pred_pixels[e_i, j, 1].item()], (255, 0, 0), 1, 2)
            axs[i % grid_size, i // grid_size].imshow(img)
            axs[i % grid_size, i // grid_size].set_title(f"Env {e_i.item()}")
        # save the figure in an output directory at the same level as the script
        output_dir = os.path.join(self._env.cfg.log_dir, "pose_estimator", "camera_images")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "keypoint_predictions.png"))
        plt.close('all')


class image_cropped_to_pose(image_to_pose):
    """Pose feature extractor for the in-hand manipulation task."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        img_size = cfg.params["img_size"]
        crop_pos = cfg.params["crop_pos"]
        crop_size = cfg.params["crop_size"]

        self._min_x = crop_pos[0]
        self._max_x = crop_pos[0] + crop_size[0]
        self._min_y = crop_pos[1]
        self._max_y = crop_pos[1] + crop_size[1]

        self._resize = transforms.Resize(size=(img_size[1], img_size[0]))

    def preprocess_images(self, rgb_img, depth_img, apply_mask=True):

        # crop the images
        rgb_img = rgb_img[:, :, self._min_y:self._max_y, self._min_x:self._max_x]
        if depth_img is not None:
            depth_img = depth_img[:, :, self._min_y:self._max_y, self._min_x:self._max_x]

        # resize the images
        rgb_img = self._resize(rgb_img)
        if depth_img is not None:
            depth_img = self._resize(depth_img)

        return super().preprocess_images(rgb_img, depth_img, apply_mask)

    def _project_points_to_image(self, points, camera_frame: bool = True):
        """Projects points from world frame to camera frame and then to pixel coordinates.
        Args:
            points (torch.Tensor): Points in world frame of shape (N, P, 3).
        Returns:
            points_pixels (torch.Tensor): Projected points in pixel coordinates of shape (N, P, 2).
        """
        points_pixels = super()._project_points_to_image(points, camera_frame)
        points_pixels[:, :, 0] -= self._min_x
        points_pixels[:, :, 1] -= self._min_y
        points_pixels[:, :, 0] = (points_pixels[:, :, 0] * self._resize.size[1] / (self._max_x - self._min_x))
        points_pixels[:, :, 1] = (points_pixels[:, :, 1] * self._resize.size[0] / (self._max_y - self._min_y))
        return points_pixels

    def _unproject_points_to_camera_frame(self, points_pixels):
        """Unprojects points from pixel coordinates to camera frame.
        Args:
            points_pixels (torch.Tensor): Points in pixel coordinates of shape (N, P, 3).
        Returns:
            points (torch.Tensor): Unprojected points in camera frame of shape (N, P, 3).
        """
        points_pixels = points_pixels.clone()
        points_pixels[:, :, 0] = points_pixels[:, :, 0] * (self._max_x - self._min_x) / self._resize.size[1]
        points_pixels[:, :, 1] = points_pixels[:, :, 1] * (self._max_y - self._min_y) / self._resize.size[0]
        points_pixels[:, :, 0] += self._min_x
        points_pixels[:, :, 1] += self._min_y
        return super()._unproject_points_to_camera_frame(points_pixels)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pose_estimator_cfg: PoseEstimatorCfg,
        camera_cfg: SceneEntityCfg,
        img_size: tuple[int, int],
        crop_pos: tuple[float, float],
        crop_size: tuple[int, int],
        points_path: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg('object'),
        masking_data: str | None = None,
        debug_vis: bool = False,
    ):
        return super().__call__(
            env,
            pose_estimator_cfg,
            camera_cfg,
            object_cfg=object_cfg,
            points_path=points_path,
            masking_data=masking_data,
            debug_vis=debug_vis,
        )


class gaussian_splat_image_to_pose(image_to_pose):
    """Pose feature extractor for the in-hand manipulation task."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.camera: GaussianSplatCamera = env.scene[cfg.params["camera_cfg"].name]

        # store the camera position and quaternion in world frame
        camera_pos_w, camera_quat_w_world = self.camera.get_world_poses()
        # store the camera position and quaternion in world frame
        self._init_camera_pos_w = camera_pos_w.clone()
        self._init_camera_quat_w_world = camera_quat_w_world.clone()
        self._init_camera_quat_w_ros = math_utils.convert_camera_frame_orientation_convention(
            camera_quat_w_world, "world", "ros"
        )

    @property
    def camera_pos(self):
        return self._init_camera_pos_w

    @property
    def camera_quat_ros(self):
        return self._init_camera_quat_w_ros

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pose_estimator_cfg: PoseEstimatorCfg,
        camera_cfg: SceneEntityCfg,
        points_path: str,
        object_cfg: SceneEntityCfg = SceneEntityCfg('object'),
        masking_data: str | None = None,
        debug_vis: bool = False,
    ):
        # get the object pose from the environment
        object_pos = self._object.data.root_pos_w - self._env.scene.env_origins
        object_rot = self._object.data.root_quat_w

        camera_pos_w, camera_quat_w_world = self.camera.get_world_poses()

        assert (
            torch.isclose(camera_pos_w, self._init_camera_pos_w, atol=1e-6).all()
            and torch.isclose(camera_quat_w_world, self._init_camera_quat_w_world, atol=1e-6).all()
        ), "Camera position or quaternion has changed since initialization."

        # store the camera position and quaternion in world frame
        self._init_camera_pos_w = camera_pos_w.clone()
        self._init_camera_quat_w_world = camera_quat_w_world.clone()
        self._init_camera_quat_w_ros = math_utils.convert_camera_frame_orientation_convention(
            camera_quat_w_world, "world", "ros"
        )

        camera_pos_frame_object, camera_quat_frame_object = math_utils.subtract_frame_transforms(
            object_pos, object_rot, self._init_camera_pos_w - env.scene.env_origins, self._init_camera_quat_w_world,
        )

        self.camera.set_world_poses(camera_pos_frame_object, camera_quat_frame_object, convention="world")

        ret = super().__call__(
            env,
            pose_estimator_cfg,
            camera_cfg,
            object_cfg=object_cfg,
            points_path=points_path,
            masking_data=masking_data,
            debug_vis=debug_vis,
        )

        self.camera.set_world_poses(self._init_camera_pos_w, self._init_camera_quat_w_world, convention="world")

        return ret


def _get_observation_term_func(manager, group_name: str, term_name: str):
    """Look up a registered observation term's `.func` by group/term name.

    Public Isaac Lab's `ObservationManager` has no public accessor for this — only the internal
    `_group_obs_term_names`/`_group_obs_term_cfgs` parallel lists — so this replicates the lookup that a
    private-fork-only `get_term_cfg()` convenience method used to provide.
    """
    term_names = manager._group_obs_term_names[group_name]
    term_cfgs = manager._group_obs_term_cfgs[group_name]
    return term_cfgs[term_names.index(term_name)].func


def check_images_diff(
    env: ManagerBasedRLEnv,
    splat_term: str = "gaussian_images",
    camera_term: str = "images",
):

    if env.common_step_counter % 5 != 0:
        return torch.zeros((env.num_envs, 1), device=env.device)

    def draw_grid(img, grid_size=15):
        """Draw a grid on the image."""
        h, w = img.shape[:2]
        if img.shape[2] == 1:  # grayscale image
            img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for i in range(0, h, grid_size):
            img[i, :, :] = (0, 255, 0)  # draw horizontal lines
        for j in range(0, w, grid_size):
            img[:, j, :] = (0, 255, 0)
        return img

    def draw_keypoints(img, func, idx):
        camera_frame = func.cfg.params.get("use_camera_frame", True)
        gt_pixels = func._project_points_to_image(func.gt_keypoints.view(-1, func.num_kp, 3), camera_frame)
        gt_pixels = gt_pixels.round().int()
        pred_pixels = func._project_points_to_image(func.prediction_keypoints.view(-1, func.num_kp, 3), camera_frame)
        pred_pixels = pred_pixels.round().int()
        if img.shape[2] == 1:  # grayscale image
            img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for j in range(gt_pixels.shape[1]):
            img = cv2.circle(img, [gt_pixels[idx, j, 0].item(), gt_pixels[idx, j, 1].item()], 2, (0, 255, 0), 1)
        for j in range(pred_pixels.shape[1]):
            img = cv2.drawMarker(img, [pred_pixels[idx, j, 0].item(), pred_pixels[idx, j, 1].item()], (255, 0, 0), 1, 2)
        return img

    if not hasattr(env, "observation_manager"):
        return torch.zeros((env.num_envs, 1), device=env.device)

    manager = env.observation_manager
    splat_term_cls = _get_observation_term_func(manager, splat_term, "image_features")
    camera_term_cls = _get_observation_term_func(manager, camera_term, "image_features")

    splat_term_sensor = splat_term_cls.camera
    camera_term_sensor = camera_term_cls.camera

    # get the splat images
    splat_rgb_img = splat_term_sensor.data.output["rgb"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)
    splat_depth_img = splat_term_sensor.data.output["distance_to_image_plane"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)
    splat_rgb_img, splat_depth_img = splat_term_cls.preprocess_images(splat_rgb_img, splat_depth_img)

    # get the camera images
    cam_rgb_img = camera_term_sensor.data.output["rgb"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)
    cam_depth_img = camera_term_sensor.data.output["distance_to_image_plane"].permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)
    cam_rgb_img, cam_depth_img = camera_term_cls.preprocess_images(cam_rgb_img, cam_depth_img)

    # save the figure in an output directory at the same level as the script
    output_dir = os.path.join(env.cfg.log_dir, "images_output")
    os.makedirs(output_dir, exist_ok=True)

    num_images = 2
    random_env = True
    if random_env:
        eidx = torch.randint(0, env.num_envs, (num_images,)).tolist()
    else:
        eidx = list(range(num_images))
    if True:  # grid images
        # create a matplotlib figure to display the images next to each other
        os.makedirs(os.path.join(output_dir, "grid"), exist_ok=True)
        fig_rgb, axs_rgb = plt.subplots(num_images, 2)
        fig_depth, axs_depth = plt.subplots(num_images, 2)
        for i in range(num_images):
            ei = eidx[i]
            axs_rgb[i, 0].imshow(draw_grid(cam_rgb_img[ei].permute(1, 2, 0).cpu().numpy()))
            axs_rgb[i, 1].imshow(draw_grid(splat_rgb_img[ei].permute(1, 2, 0).cpu().numpy()))
            axs_depth[i, 0].imshow(draw_grid(cam_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1)))
            axs_depth[i, 1].imshow(draw_grid(splat_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1)))

            if i == 0:
                axs_rgb[i, 0].set_title("RGB (Camera)")
                axs_rgb[i, 1].set_title("RGB (Gaussian Splat)")
                axs_depth[i, 0].set_title("Depth (Camera)")
                axs_depth[i, 1].set_title("Depth (Gaussian Splat)")

        fig_rgb.savefig(os.path.join(output_dir, "grid", f"rgb_comparison_{env.common_step_counter}.png"))
        fig_depth.savefig(os.path.join(output_dir, "grid", f"depth_comparison_{env.common_step_counter}.png"))
        plt.close('all')

    if True:  # default images
        # create a matplotlib figure to display the images next to each other
        os.makedirs(os.path.join(output_dir, "default"), exist_ok=True)
        fig_rgb, axs_rgb = plt.subplots(num_images, 2)
        fig_depth, axs_depth = plt.subplots(num_images, 2)
        for i in range(num_images):
            ei = eidx[i]
            axs_rgb[i, 0].imshow(cam_rgb_img[ei].permute(1, 2, 0).cpu().numpy())
            axs_rgb[i, 1].imshow(splat_rgb_img[ei].permute(1, 2, 0).cpu().numpy())
            axs_depth[i, 0].imshow(cam_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axs_depth[i, 1].imshow(splat_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1))

            if i == 0:
                axs_rgb[i, 0].set_title("RGB (Camera)")
                axs_rgb[i, 1].set_title("RGB (Gaussian Splat)")
                axs_depth[i, 0].set_title("Depth (Camera)")
                axs_depth[i, 1].set_title("Depth (Gaussian Splat)")

        fig_rgb.savefig(os.path.join(output_dir, "default", f"rgb_comparison_{env.common_step_counter}.png"))
        fig_depth.savefig(os.path.join(output_dir, "default", f"depth_comparison_{env.common_step_counter}.png"))
        plt.close('all')

    if True:  # keypoint images
        # create a matplotlib figure to display the images next to each other
        os.makedirs(os.path.join(output_dir, "keypoints"), exist_ok=True)
        fig_rgb, axs_rgb = plt.subplots(num_images, 2)
        fig_depth, axs_depth = plt.subplots(num_images, 2)
        for i in range(num_images):
            ei = eidx[i]
            axs_rgb[i, 0].imshow(draw_keypoints(cam_rgb_img[ei].permute(1, 2, 0).cpu().numpy(), camera_term_cls, ei))
            axs_rgb[i, 1].imshow(draw_keypoints(splat_rgb_img[ei].permute(1, 2, 0).cpu().numpy(), splat_term_cls, ei))
            axs_depth[i, 0].imshow(draw_keypoints(cam_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1), camera_term_cls, ei))
            axs_depth[i, 1].imshow(draw_keypoints(splat_depth_img[ei].permute(1, 2, 0).cpu().numpy().clip(0, 1), splat_term_cls, ei))

            if i == 0:
                axs_rgb[i, 0].set_title("RGB (Camera)")
                axs_rgb[i, 1].set_title("RGB (Gaussian Splat)")
                axs_depth[i, 0].set_title("Depth (Camera)")
                axs_depth[i, 1].set_title("Depth (Gaussian Splat)")

        fig_rgb.savefig(os.path.join(output_dir, "keypoints", f"rgb_comparison_{env.common_step_counter}.png"))
        fig_depth.savefig(os.path.join(output_dir, "keypoints", f"depth_comparison_{env.common_step_counter}.png"))
        plt.close('all')

    return torch.zeros((env.num_envs, 1), device=env.device)
