# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from collections.abc import Sequence
from dataclasses import MISSING
from matplotlib import pyplot as plt


class DiscretizedJointPositionsMasker:

    def __init__(self, data_path: str, env: ManagerBasedRLEnv, robot: SceneEntityCfg):
        # store the configuration and device

        self.env = env
        self.device = env.device

        masking_data = torch.load(data_path, weights_only=False, map_location=self.device)

        self._robot: Articulation = self.env.scene[robot.name]

        self._chain_num_to_joint_ids = self._resolve_joint_ids(masking_data["joint_names"])
        self._bin_data = masking_data["bin_data"]
        self._bin_size_per_chain = masking_data["bin_size"]
        self._min_bin_per_chain = masking_data["min_bin"]
        self._max_bin_per_chain = masking_data["max_bin"]

    def _resolve_joint_ids(self, joint_names_per_chain):
        chain_num_to_joint_ids = []
        for joint_names in joint_names_per_chain:
            chain_joint_ids, _ = self._robot.find_joints(joint_names)
            chain_num_to_joint_ids.append(torch.tensor(chain_joint_ids, dtype=torch.int32, device=self.device))
        return chain_num_to_joint_ids

    def create_mask(self, joint_positions: torch.Tensor, img_shape: torch.tensor) -> torch.Tensor:
        mask = torch.ones(img_shape[0], img_shape[2], img_shape[3], dtype=torch.float32, device=self.device)
        mask = mask * torch.inf  # Initialize mask with infinity
        for chain in range(len(self._chain_num_to_joint_ids)):
            chain_joint_positions = joint_positions[:, self._chain_num_to_joint_ids[chain]]
            chain_bins = torch.floor(chain_joint_positions / self._bin_size_per_chain[chain]).int()
            # chain_ids = chain_bins - self._min_bin_per_chain[chain].unsqueeze(0)
            min_bins = self._min_bin_per_chain[chain].unsqueeze(0)
            max_bins = self._max_bin_per_chain[chain].unsqueeze(0)
            chain_bins = torch.clamp(chain_bins, min=min_bins, max=max_bins)
            chain_ids = chain_bins - min_bins
            chain_bin_data = self._bin_data[chain]
            for i in range(chain_ids.shape[-1]):
                joint_bin_data = chain_bin_data[i]
                joint_chain_ids = chain_ids[:, :i + 1]
                joint_pixels = joint_bin_data[tuple(joint_chain_ids.T)]
                mask = self.update_mask_from_pixels(joint_pixels, mask)
        return mask.unsqueeze(1)

    @staticmethod
    def update_mask_from_pixels(pixels, mask):
        new_mask = torch.ones_like(mask) * torch.inf
        i = pixels[:, :, 0].long()
        j = pixels[:, :, 1].long()
        ns = torch.arange(i.shape[0], dtype=torch.int, device=pixels.device).view(-1, 1).expand_as(i)
        val = pixels[:, :, 2]
        new_mask[ns, i, j] = val
        return torch.minimum(mask, new_mask)

    def debug_vis(self, mask):
        fig, axs = plt.subplots(2, 1)
        for i in range(2):
            axs[i].imshow(mask[i].cpu().numpy()[0], vmin=0, vmax=1.0)
            if i == 0:
                axs[i].set_title("Mask")
        # save the figure in an output directory at the same level as the script
        output_dir = os.path.join(self.env.cfg.log_dir, "pose_estimator", "camera_images")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "discretized_mask.png"))
        plt.close('all')

    def apply_masks(self, rgb_img: torch.Tensor, depth_img: torch.Tensor):
        original_device = rgb_img.device
        rgb_img = rgb_img.to(self.device)
        depth_img = depth_img.to(self.device)
        mask = self.create_mask(self._robot.data.joint_pos.to(self.device), depth_img.shape)
        # self.debug_vis(mask)
        applied_mask = mask <= depth_img
        rgb_img[(applied_mask).expand_as(rgb_img)] = 0.0
        depth_img[(applied_mask).expand_as(depth_img)] = 0.0
        return rgb_img.to(original_device), depth_img.to(original_device)
