# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Maximum Wilder-Smith
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from abc import ABC
from viserdex.utils.load_utils import load_splat_file
from typing import Literal, Callable, List, Sequence, Union
from dataclasses import dataclass, field


# ==========================================
# Part 1: Utils and Config (Mocking IsaacLab)
# ==========================================

def configclass(cls):
    return dataclass(cls)

# --- Noise Configs ---



def constant_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)
    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")

def uniform_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)
    
    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")

def gaussian_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    if isinstance(cfg.mean, torch.Tensor):
        cfg.mean = cfg.mean.to(data.device)
    if isinstance(cfg.std, torch.Tensor):
        cfg.std = cfg.std.to(data.device)

    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


@configclass
class NoiseCfg:
    """Base configuration for a noise term."""
    func: Callable = None
    operation: Literal["add", "scale", "abs"] = "add"

@configclass
class ConstantNoiseCfg(NoiseCfg):
    bias: Union[torch.Tensor, float] = 0.0
    func: Callable = staticmethod(constant_noise)

@configclass
class UniformNoiseCfg(NoiseCfg):
    n_min: Union[torch.Tensor, float] = -1.0
    n_max: Union[torch.Tensor, float] = 1.0
    func: Callable = staticmethod(uniform_noise)

@configclass
class GaussianNoiseCfg(NoiseCfg):
    mean: Union[torch.Tensor, float] = 0.0
    std: Union[torch.Tensor, float] = 1.0
    func: Callable = staticmethod(gaussian_noise)

# --- Noise Functions ---
# Link functions to configs
ConstantNoiseCfg.func = staticmethod(constant_noise)
UniformNoiseCfg.func = staticmethod(uniform_noise)
GaussianNoiseCfg.func = staticmethod(gaussian_noise)

# ==========================================
# Part 2: Augmentation Classes & Configs
# ==========================================



class SplatAugmentation(ABC):
    def __init__(self, cfg, splats: dict[str, torch.Tensor], device: str):
        self._cfg = cfg
        self._device = device
        # Handle cases where splats might not have all keys initially (e.g. if loaded partially)
        self._key_dims = {}
        if hasattr(self._cfg, 'keys'):
             for key in self._cfg.keys:
                 if key in splats:
                     self._key_dims[key] = splats[key].shape

    def __call__(self, splats: dict[str, torch.Tensor], num_samples: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses should implement this method.")

class NoiseAugmentation(SplatAugmentation):
    def __init__(self, cfg, splats: dict[str, torch.Tensor], device: str):
        super().__init__(cfg, splats, device)
        self._noise_cfg = cfg.noise_cfg
        self._keys = cfg.keys
        if isinstance(cfg.probability, list):
            assert len(cfg.probability) == len(self._keys), "Probability list must match the number of keys."
            self._probabilities = cfg.probability
        else:
            self._probabilities = [cfg.probability] * len(self._keys)

    def __call__(self, splats: dict[str, torch.Tensor], num_samples: int) -> dict[str, torch.Tensor]:
        """Apply noise augmentation to the splats."""
        samples = torch.rand(len(self._keys), device=self._device)
        for key, prob, sample in zip(self._keys, self._probabilities, samples):
            if sample > prob:
                continue
            data = splats[key]
            data = data.expand(num_samples, *self._key_dims[key])
            splats[key] = self._noise_cfg.func(data, self._noise_cfg)
        return splats


class ClusteredSplatNoiseAugmentation(NoiseAugmentation):
    def __init__(self, cfg, splats: dict[str, torch.Tensor], device: str):
        super().__init__(cfg, splats, device)
        
        # assign clustering parameters
        if cfg.cluster_indices_path is not None and os.path.exists(cfg.cluster_indices_path):
            self._cluster_indices = torch.load(cfg.cluster_indices_path).to(device)
            self._num_clusters = torch.unique(self._cluster_indices).numel()
        else:
            print(f"Warning: Cluster indices path {cfg.cluster_indices_path} not found. Using single cluster.")
            self._cluster_indices = torch.zeros(len(splats["means3d"]), dtype=torch.int64, device=device)
            self._num_clusters = 1

        self._mask = torch.zeros(self._num_clusters, len(self._cluster_indices), dtype=torch.bool, device=device)
        # print("mask shape:", self._mask.shape)
        # print("cluster indices shape:", self._cluster_indices.shape)
        # print("num clusters:", self._num_clusters)
        for cluster in range(self._num_clusters):
            cluster_mask = self._cluster_indices == cluster
            self._mask[cluster] = cluster_mask

    def _apply_noise_per_cluster(self, data: torch.Tensor) -> torch.Tensor:
        noise_per_cluster_shape = (data.shape[0], self._num_clusters, *data.shape[2:])
        if self._noise_cfg.operation == "add":
            noise_per_cluster = self._noise_cfg.func(torch.zeros(noise_per_cluster_shape, device=data.device), self._noise_cfg)
            noise = noise_per_cluster[:, self._cluster_indices]
            return noise + data
        elif self._noise_cfg.operation == "scale":
            noise_per_cluster = self._noise_cfg.func(torch.ones(noise_per_cluster_shape, device=data.device), self._noise_cfg)
            noise = noise_per_cluster[:, self._cluster_indices]
            return noise * data
        elif self._noise_cfg.operation == "abs":
            noise_per_cluster = self._noise_cfg.func(torch.zeros(noise_per_cluster_shape, device=data.device), self._noise_cfg)
            noise = noise_per_cluster[:, self._cluster_indices]
            return noise

    def __call__(self, splats: dict[str, torch.Tensor], num_samples: int) -> dict[str, torch.Tensor]:
        cluster_samples = torch.rand(num_samples, self._num_clusters, device=self._device)
        cluster_masks = cluster_samples[:, self._cluster_indices] < self._cfg.cluster_fraction
        key_samples = torch.rand(len(self._keys), device=self._device)
        for key, prob, key_sample in zip(self._keys, self._probabilities, key_samples):
            if key_sample > prob:
                continue
            data = splats[key]
            # print("Key:", key)
            # print("Before expand:", data.shape)
            data = data.expand(num_samples, *self._key_dims[key])
            # print("After expand:", data.shape)
            # print("Cluster masks shape:", cluster_masks.shape)
            # print("cluster samples shape:", cluster_samples.shape)
            cluster_masks_key = cluster_masks.view(*cluster_masks.shape, *[1] * len(data.shape[2:])).expand_as(data)
            # print("Cluster masks key shape:", cluster_masks_key.shape)
            if self._cfg.noise_per_cluster:
                noisy_data = self._apply_noise_per_cluster(data)
            else:
                noisy_data = self._noise_cfg.func(data, self._noise_cfg)
            data[cluster_masks_key] = noisy_data[cluster_masks_key]
            splats[key] = data
        return splats


class ShiftAugmentation(SplatAugmentation):
    def __init__(self, cfg, splats: dict[str, torch.Tensor], device: str):
        super().__init__(cfg, splats, device)
        self._noise_cfg = cfg.noise_cfg
        self._apply_per_key = cfg.apply_per_key
        if isinstance(cfg.probability, list):
            assert len(cfg.probability) == len(self._key_dims), "Probability list must match the number of keys."
            self._probabilities = cfg.probability
        else:
            self._probabilities = [cfg.probability] * len(self._key_dims)

    def _apply_noise(self, data: torch.Tensor) -> torch.Tensor:
        if self._noise_cfg.operation == "add":
            noise = self._noise_cfg.func(torch.zeros(data.shape[0], device=self._device), self._noise_cfg)
            noise = noise.view(-1, *([1] * (data.ndim - 1)))
            return noise + data
        elif self._noise_cfg.operation == "scale":
            noise = self._noise_cfg.func(torch.ones(data.shape[0], device=self._device), self._noise_cfg)
            noise = noise.view(-1, *([1] * (data.ndim - 1)))
            return noise.expand_as(data) * data
        elif self._noise_cfg.operation == "abs":
            noise = self._noise_cfg.func(torch.zeros(data.shape[0], device=self._device), self._noise_cfg)
            noise = noise.view(-1, *([1] * (data.ndim - 1)))
            return noise.expand_as(data)

    def __call__(self, splats: dict[str, torch.Tensor], num_samples: int) -> dict[str, torch.Tensor]:
        """Apply noise augmentation to the splats."""
        if self._apply_per_key:
            samples = torch.rand(len(self._key_dims), device=self._device)
            for key, prob, sample in zip(self._key_dims, self._probabilities, samples):
                if sample > prob:
                    continue
                data = splats[key]
                data = data.expand(num_samples, *self._key_dims[key])
                splats[key] = self._apply_noise(data)
            return splats
        else:
            sample = torch.rand(1, device=self._device).item()
            if sample > self._probabilities[0]:
                return splats
            else:
                data = torch.cat(
                    [splats[key].expand(num_samples, *dim).flatten(start_dim=2) for key, dim in self._key_dims.items()],
                    dim=2
                )
                data = self._apply_noise(data)
                curr_dim = 0
                for key, dim in self._key_dims.items():
                    numel = torch.prod(torch.tensor(dim[1:], device=self._device))
                    splats[key] = data[:, :, curr_dim:curr_dim + numel].view(num_samples, -1, *dim[1:])
                    curr_dim += numel
            return splats
@configclass
class SplatAugmentationCfg:
    func: Callable = None

@configclass
class NoiseAugmentationCfg(SplatAugmentationCfg):
    func: Callable = NoiseAugmentation
    keys: List[str] = field(default_factory=list)
    noise_cfg: NoiseCfg = None
    probability: Union[float, List[float]] = 1.0

@configclass
class ClusteredSplatNoiseAugmentationCfg(NoiseAugmentationCfg):
    func: Callable = ClusteredSplatNoiseAugmentation
    cluster_indices_path: str = None
    cluster_fraction: float = 0.1
    noise_per_cluster: bool = False

@configclass
class ShiftAugmentationCfg(SplatAugmentationCfg):
    func: Callable = ShiftAugmentation
    keys: List[str] = field(default_factory=list)
    noise_cfg: NoiseCfg = None
    probability: Union[float, List[float]] = 1.0
    apply_per_key: bool = False



# ==========================================
# Part 3: Default Augmentations
# ==========================================

# ==========================================
# Part 4: IO & Main
# ==========================================

def generate_default_augmentations(path):
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
            cluster_indices_path=os.path.join(path, "splat_clustered_means3d_indices.pt"),
            cluster_fraction=0.1,
            noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
            keys=["sh0", "shN"],
            probability=0.8,
            noise_per_cluster=True,
        ),
        ClusteredSplatNoiseAugmentationCfg(
            cluster_indices_path=os.path.join(path, "splat_clustered_means3d_indices.pt"),
            cluster_fraction=0.2,
            noise_cfg=UniformNoiseCfg(n_min=0.9, n_max=1.1, operation="scale"),
            keys=["sh0", "shN"],
            probability=0.8,
            noise_per_cluster=True,
        ),
        # Clustered Sh0 Noise Augmentations
        ClusteredSplatNoiseAugmentationCfg(
            cluster_indices_path=os.path.join(path, "splat_clustered_sh0_indices.pt"),
            cluster_fraction=0.5,
            noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2, operation="add"),
            keys=["sh0"],
            probability=0.8,
            noise_per_cluster=True,
        ),
        ClusteredSplatNoiseAugmentationCfg(
            cluster_indices_path=os.path.join(path, "splat_clustered_sh0_indices.pt"),
            cluster_fraction=0.5,
            noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1, operation="add"),
            keys=["shN"],
            probability=0.8,
            noise_per_cluster=True,
        ),
        ClusteredSplatNoiseAugmentationCfg(
            cluster_indices_path=os.path.join(path, "splat_clustered_sh0_indices.pt"),
            cluster_fraction=0.5,
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
    return DefaultAugmentations

def save_splat_file(path, splats, original_vertex_data=None):
    """Saves the splats to a .ply file."""
    # Convert back to numpy
    xyz = splats["means3d"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy().squeeze(1)
    shN = splats["shN"].detach().cpu().numpy().transpose(0, 2, 1).reshape(-1, 45)
    opacities = splats["opacities"].detach().cpu().numpy().reshape(-1)
    scales = splats["scales"].detach().cpu().numpy()
    rotations = splats["quats"].detach().cpu().numpy()

    # Construct structured array for PlyElement
    # We need to match the standard PLY format for Gaussian Splats
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), # Normals usually 0
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'f4'))
    dtype_list.append(('opacity', 'f4'))
    dtype_list.append(('scale_0', 'f4'))
    dtype_list.append(('scale_1', 'f4'))
    dtype_list.append(('scale_2', 'f4'))
    dtype_list.append(('rot_0', 'f4'))
    dtype_list.append(('rot_1', 'f4'))
    dtype_list.append(('rot_2', 'f4'))
    dtype_list.append(('rot_3', 'f4'))

    num_points = xyz.shape[0]
    vertex_data = np.zeros(num_points, dtype=dtype_list)

    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    # Normals (dummy)
    vertex_data['nx'] = 0
    vertex_data['ny'] = 0
    vertex_data['nz'] = 0
    
    vertex_data['f_dc_0'] = sh0[:, 0]
    vertex_data['f_dc_1'] = sh0[:, 1]
    vertex_data['f_dc_2'] = sh0[:, 2]
    
    for i in range(45):
        vertex_data[f'f_rest_{i}'] = shN[:, i]
        
    vertex_data['opacity'] = opacities
    vertex_data['scale_0'] = scales[:, 0]
    vertex_data['scale_1'] = scales[:, 1]
    vertex_data['scale_2'] = scales[:, 2]
    
    vertex_data['rot_0'] = rotations[:, 0]
    vertex_data['rot_1'] = rotations[:, 1]
    vertex_data['rot_2'] = rotations[:, 2]
    vertex_data['rot_3'] = rotations[:, 3]

    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write(path)
    print(f"Saved augmented splats to {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment Gaussian Splats")
    parser.add_argument("--input", type=str, required=True, help="Input .ply file")
    parser.add_argument("--output", type=str, required=True, help="Output .ply file")
    parser.add_argument("--cluster_dir", type=str, default=".", help="Directory containing cluster indices if needed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load
    print(f"Loading {args.input}...")
    try:
        splats = load_splat_file(args.input, device)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Initialize Augmentations
    augmentations = []
    
    DefaultAugmentations = generate_default_augmentations(args.cluster_dir)
    
    # Update cluster path in configs relative to provided dir
    # for cfg in DefaultAugmentations:
        # if isinstance(cfg, ClusteredSplatNoiseAugmentationCfg) and cfg.cluster_indices_path:
        #      # Just replace the dummy prefix
        #      basename = os.path.basename(cfg.cluster_indices_path)
        #      cfg.cluster_indices_path = os.path.join(args.cluster_dir, basename)

    print("Initializing augmentations...")
    for cfg in DefaultAugmentations:
        augmentations.append(cfg.func(cfg, splats, device))

    # Apply
    print("Applying augmentations...")
    # Since we are augmenting a single file, num_samples=1
    # Note: Our modified Augmentation classes will return [1, N, D] or [N, D] depending on logic
    # But since we save back, we assume they modify 'splats' in place or return modified dict.
    # The classes implemented above modify dict in place but might expand tensors to batch.
    # We need to squeeze batch dim before saving if it exists.
    
    for aug in augmentations:
        aug(splats, num_samples=1)

    # Post-process: Remove batch dimension if added
    for k, v in splats.items():
        if v.ndim > 1 and v.shape[0] == 1 and k != "opacities": # Opacities is [N] or [1, N]
             # Check if first dim is batch 1. 
             # Original dims: means3d [N, 3]. If becomes [1, N, 3], squeeze.
             if v.shape[0] == 1:
                 splats[k] = v.squeeze(0)
        elif k == "opacities" and v.ndim == 2 and v.shape[0] == 1:
             splats[k] = v.squeeze(0)

    # Save
    save_splat_file(args.output, splats)

if __name__ == "__main__":
    main()