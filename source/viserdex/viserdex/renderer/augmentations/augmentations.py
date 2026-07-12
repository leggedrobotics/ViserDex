# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import ABC


class SplatAugmentation(ABC):
    """Base class for splat augmentations."""

    def __init__(self, cfg, splats: dict[str, torch.tensor], device: str):
        """Initialize the augmentation with a configuration."""
        self._cfg = cfg
        self._device = device
        self._key_dims = {key: splats[key].shape for key in self._cfg.keys}

    def __call__(self, splats: dict[str, torch.Tensor], num_samples: int) -> dict[str, torch.Tensor]:
        """Apply the augmentation to the given data."""
        raise NotImplementedError("Subclasses should implement this method.")


class NoiseAugmentation(SplatAugmentation):

    def __init__(self, cfg, splats: dict[str, torch.tensor], device: str):
        # initialize parent class
        super().__init__(cfg, splats, device)

        # assign noise parameters
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
    """Base class for clustered splat augmentations."""

    def __init__(self, cfg, splats: dict[str, torch.tensor], device: str):
        super().__init__(cfg, splats, device)

        # assign clustering parameters
        if cfg.cluster_indices_path is not None:
            self._cluster_indices = torch.load(cfg.cluster_indices_path).to(device)
            self._num_clusters = torch.unique(self._cluster_indices).numel()
        else:
            self._cluster_indices = torch.zeros(len(splats["means3d"]), dtype=torch.int64, device=device)
            self._num_clusters = 1

        # create a mask per cluster
        self._mask = torch.zeros(self._num_clusters, len(self._cluster_indices), dtype=torch.bool, device=device)
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
            data = data.expand(num_samples, *self._key_dims[key])
            cluster_masks_key = cluster_masks.view(*cluster_masks.shape, *[1] * len(data.shape[2:])).expand_as(data)
            if self._cfg.noise_per_cluster:
                noisy_data = self._apply_noise_per_cluster(data)
            else:
                noisy_data = self._noise_cfg.func(data, self._noise_cfg)
            data[cluster_masks_key] = noisy_data[cluster_masks_key]
            splats[key] = data
        return splats


class ShiftAugmentation(SplatAugmentation):

    def __init__(self, cfg, splats: dict[str, torch.tensor], device: str):
        # initialize parent class
        super().__init__(cfg, splats, device)

        # assign noise parameters
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
