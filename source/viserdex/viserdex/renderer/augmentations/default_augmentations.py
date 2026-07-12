# Copyright (c) 2025 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from isaaclab.utils.noise import UniformNoiseCfg, ConstantNoiseCfg, GaussianNoiseCfg
from viserdex.renderer.augmentations import SplatAugmentationCfg, NoiseAugmentationCfg, ClusteredSplatNoiseAugmentationCfg, ShiftAugmentationCfg

splat_cluster_path = "dummy_splat_directory/clusters/indices"

DefaultAugmentations = [
    # Basic Noise Augmentations
    NoiseAugmentationCfg(
        keys=["sh0"], noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1), probability=0.2
    ),
    NoiseAugmentationCfg(
        keys=["shN"], noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1), probability=0.2
    ),
    NoiseAugmentationCfg(
        keys=["sh0"], noise_cfg=UniformNoiseCfg(n_min=0.8, n_max=1.2, operation="scale"), probability=0.2
    ),
    NoiseAugmentationCfg(
        keys=["shN"], noise_cfg=UniformNoiseCfg(n_min=0.8, n_max=1.2, operation="scale"), probability=0.2
    ),
    # Clustered Means3d Noise Augmentations
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=os.path.join(splat_cluster_path, "splat_clustered_means3d_indices.pt"),
        cluster_fraction=0.1,
        noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
        keys=["sh0", "shN"],
        probability=0.8,
        noise_per_cluster=True,
    ),
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=os.path.join(splat_cluster_path, "splat_clustered_means3d_indices.pt"),
        cluster_fraction=0.2,
        noise_cfg=UniformNoiseCfg(n_min=0.9, n_max=1.1, operation="scale"),
        keys=["sh0", "shN"],
        probability=0.8,
        noise_per_cluster=True,
    ),
    # Clustered Sh0 Noise Augmentations
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=os.path.join(splat_cluster_path, "splat_clustered_sh0_indices.pt"),
        cluster_fraction=0.1,
        noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2, operation="add"),
        keys=["sh0"],
        probability=0.8,
        noise_per_cluster=True,
    ),
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=os.path.join(splat_cluster_path, "splat_clustered_sh0_indices.pt"),
        cluster_fraction=0.1,
        noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
        keys=["shN"],
        probability=0.8,
        noise_per_cluster=True,
    ),
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=os.path.join(splat_cluster_path, "splat_clustered_sh0_indices.pt"),
        cluster_fraction=0.1,
        noise_cfg=UniformNoiseCfg(n_min=0.6, n_max=1.4, operation="scale"),
        keys=["sh0", "shN"],
        probability=0.8,
        noise_per_cluster=True,
    ),
    # Shift Augmentations
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=None,
        cluster_fraction=1.0,
        noise_cfg=UniformNoiseCfg(n_min=0.6, n_max=1.4, operation="scale"),
        keys=["sh0", "shN"],
        probability=0.2,
        noise_per_cluster=True,
    ),
    ClusteredSplatNoiseAugmentationCfg(
        cluster_indices_path=None,
        cluster_fraction=1.0,
        noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
        keys=["shN"],
        probability=0.2,
        noise_per_cluster=True,
    ),
    ShiftAugmentationCfg(
        keys=["sh0", "shN"], noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2), probability=0.8, apply_per_key=False
    ),
    ShiftAugmentationCfg(
        keys=["sh0"], noise_cfg=UniformNoiseCfg(n_min=0.9, n_max=1.4, operation="scale"), probability=0.8, apply_per_key=False
    ),
]
