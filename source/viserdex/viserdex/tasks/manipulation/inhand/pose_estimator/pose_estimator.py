# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
from filelock import FileLock
from dataclasses import MISSING

from torchvision.utils import make_grid, save_image

from isaaclab.utils import configclass

from .transforms import get_transforms
from .networks import PoseEstimatorNetwork


FULL_AUGS: list[str] = ["jitter", "iso", "brightness", "random_blur", "motion_blur", "gamma", "hue", "saturation"]
GS_ONLY_AUGS: list[str] = ["iso", "motion_blur"]


@configclass
class PoseEstimatorCfg:
    """Configuration for the pose estimator model."""

    train: bool = True
    """If True, the pose estimator model is trained during the rollout process. Default is False."""

    network_cls: torch.nn.Module = PoseEstimatorNetwork
    """Class used for the pose estimator model. Default is PoseEstimatorNetwork."""

    train_backbone: bool = True
    """If True, the backbone of the pose estimator model is trained. Default is True."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    num_batches: int = 16
    """Number of batches to split the input data into. Default is 16. More batches lead to less memory usage but slower training."""

    separate_depth_loss: bool = True
    """If True, the depth loss is computed separately from the pixel loss. Default is True."""

    depth_loss_coef: float = 1.0
    """Factor to scale the depth loss. Default is 1.0."""

    # augmentation_names: list[str] = ["jitter", "iso", "brightness", "random_blur", "motion_blur", "gamma", "hue", "saturation"]
    # augmentation_names: list[str] = ["iso", "motion_blur"]
    augmentation_names: list[str] = MISSING
    """List of augmentations to apply to the input images."""

    transform_names: list[str] = ["binary_opening"]
    """List of transformations to apply to the input images. Default is ["binary_opening"]."""


class PoseEstimator(nn.Module):
    """Class for extracting pose from image data.

    It uses a CNN to regress keypoint positions from normalized RGB and depth
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, num_outputs: int, cfg: PoseEstimatorCfg, device: str, log_dir: str):
        """Initialize the pose estimator model.

        Args:
            cfg (PoseEstimatorCfg): Configuration for the pose estimator model.
            device (str): Device to run the model on.
        """
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.log_dir = log_dir
        self.num_outputs = num_outputs
        with FileLock(os.path.join(os.path.dirname(self.log_dir), "pose_estimator.lock")):
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        # Pose estimator model
        self.pose_estimator = cfg.network_cls(num_outputs)
        self.pose_estimator.to(self.device)

        self.step_count = 0

        self.load_model(self.log_dir)
        if self.cfg.train:
            self.optimizer, self.lr_scheduler = self.pose_estimator.get_optimizer_and_scheduler()
            # self.loss_fn = nn.MSELoss()
            self.loss_fn = nn.HuberLoss(delta=0.05)
            self.pose_estimator.train()
        else:
            self.pose_estimator.eval()

        self.augmentations = get_transforms(self.cfg.augmentation_names, self.device)
        self.transforms = get_transforms(self.cfg.transform_names, self.device)

    def load_model(self, log_dir: str):
        list_of_files = glob.glob(log_dir + "/*.pth")
        print(f"[INFO]: Looking for pose estimator checkpoints in {log_dir}, found {len(list_of_files)} files.")
        if len(list_of_files) > 0:
            # latest_file = max(list_of_files, key=os.path.getctime)
            latest_file = sorted(list_of_files, key=lambda x: int(os.path.basename(x).split("_")[1]))[-1]
            checkpoint = latest_file
            print(f"[INFO]: Loading pose estimator checkpoint from {checkpoint}")
            self.pose_estimator.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=self.device), strict=True)
            self.cfg.train = False
            self.pose_estimator.eval()

    def eval(self):
        self.cfg.train = False
        self.pose_estimator.eval()

    def train(self, val: bool = True):
        self.cfg.train = val
        if val:
            self.pose_estimator.train()
        else:
            self.pose_estimator.eval()

    def preprocess_images(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
        Returns:
            torch.Tensor: Preprocessed RGB and depth
        """
        # process RGB image
        rgb_img = rgb_img / 255.0
        # apply augmentations
        with torch.inference_mode():
            rgb_img = self.transforms(rgb_img)
            if self.cfg.train:
                rgb_img = self.augmentations(rgb_img)
        if self.cfg.write_image_to_file:
            self.save_images(rgb_img)
        return rgb_img.clone()  # cloned to enable inference mode

    def save_images(self, rgb_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, 3, H, W).
        """
        save_image(
            make_grid(rgb_img, nrow=round(rgb_img.shape[0] ** 0.5)), os.path.join(self.log_dir, "rgb.png")
        )

    def step(
        self,
        rgb_img: torch.Tensor,
        gt_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extracts the pose using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, 3, H, W).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).
            kwargs: Additional keyword arguments for proprioceptive information. Can include:
                joint_pos (torch.Tensor): Joint position tensor. Shape: (N, num_joints).
                last_action (torch.Tensor): Last action tensor. Shape: (N, action_dim).
                last_prediction (torch.Tensor): Last prediction tensor. Shape: (N, num_outputs).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """
        # move tensors to device
        rgb_img = rgb_img.to(self.device)
        gt_pose = gt_pose.to(self.device)
        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    # initialize buffers for batched training
                    avg_pose_loss = torch.tensor(0.0, device=self.device)
                    batch_size = rgb_img.shape[0]
                    random_indices = torch.randperm(batch_size, device=self.device)
                    mini_batch_size = batch_size // self.cfg.num_batches
                    mini_batches = torch.split(random_indices, max(mini_batch_size, 1), dim=0)
                    predictions = torch.zeros(batch_size, self.pose_estimator.num_outputs, device=self.device)

                    # iterate through batches
                    for batch in mini_batches:

                        # clear gradients
                        self.optimizer.zero_grad()

                        # preprocessing
                        img_input_batch = self.preprocess_images(rgb_img[batch])
        
                        # forward pass
                        predictions_batch = self.pose_estimator(img_input_batch)

                        # store batch predictions
                        predictions[batch] = predictions_batch.detach()

                        # backward pass
                        if self.cfg.separate_depth_loss:
                            pixel_pose_loss = self.loss_fn(predictions_batch[..., :2], gt_pose[batch][..., :2].clone())
                            depth_pose_loss = self.loss_fn(predictions_batch[..., 2:], gt_pose[batch][..., 2:].clone())
                            pose_loss = (pixel_pose_loss + self.cfg.depth_loss_coef * depth_pose_loss) * 100
                        else:
                            pose_loss = self.loss_fn(predictions_batch, gt_pose[batch].clone()) * 100
                        if not torch.isnan(pose_loss) and self.step_count > 5:
                            pose_loss.backward()
                            self.optimizer.step()
                            avg_pose_loss += pose_loss.detach() / len(mini_batches)

            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.step_count += 1

            if self.step_count % 2000 == 0:
                print(f"[INFO]: Saving feature extractor checkpoint at step {self.step_count} at path {self.log_dir}")
                torch.save(
                    self.pose_estimator.state_dict(),
                    os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                )

            return predictions, avg_pose_loss
        else:
            with torch.no_grad():
                with torch.inference_mode(True):
                    img_input = self.preprocess_images(rgb_img)
                    predictions = self.pose_estimator(img_input)
            return predictions, None

    def test(
        self,
        rgb_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the pose using the images in inference mode.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, 3, H, W) or (3, H, W).
        Returns:
            torch.Tensor: pose predictions.
        """
        rgb_img = rgb_img.to(self.device)
        with torch.no_grad():
            with torch.inference_mode(True):
                if rgb_img.ndim == 3:
                    rgb_img = rgb_img.unsqueeze(0)
                img_input = self.preprocess_images(rgb_img)
                predictions = self.pose_estimator(img_input)
        return predictions
