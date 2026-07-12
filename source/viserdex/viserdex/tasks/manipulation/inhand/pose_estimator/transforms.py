# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.morphology as M
import kornia.filters as F


class ISOlikeNoise(nn.Module):
    def __init__(self, p=0.25, color_shift=0.05, intensity=0.5):
        super().__init__()
        self.p = p
        self.color_shift = color_shift
        self.intensity = intensity

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        # Simulate Poisson noiseF
        shot_noise = torch.poisson(x * 255.0) / 255.0
        shot_noise = torch.clamp(shot_noise, 0.0, 1.0)

        # Gaussian read noise
        gaussian_noise = torch.randn_like(x) * self.intensity
        noisy = (shot_noise + gaussian_noise) * self.color_shift

        # Channel-wise color shift
        shift = (torch.randn(x.size(0), x.size(1), 1, 1, device=x.device)
                 * self.color_shift)
        noisy = noisy + shift
        return x + torch.clamp(noisy, 0.0, 1.0)


class BinaryOpening(nn.Module):
    def __init__(self, kernel_size=(3, 3), p=1.0, iterations=1):
        super().__init__()
        self.p = p
        self.kernel = torch.ones(kernel_size, dtype=torch.float32)
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        for it in range(self.iterations):
            opened = M.opening((x.sum(dim=1, keepdim=True) > 0).float(), self.kernel.to(x.device))
            x = x * opened.expand_as(x)
        return x


class RandomKernelFilter(nn.Module):
    def __init__(
        self,
        filter_fn,
        blur_limit=(3, 9),
        p=1.0,
        sigma_range=None,
        filter_kwargs=None,
    ):
        """
        Generic random-kernel filter module using Kornia filter functions.

        Parameters:
        - filter_fn: Kornia functional filter (e.g., kornia.filters.gaussian_blur2d)
        - blur_limit: tuple (min, max) odd kernel size range
        - p: probability of applying filter
        - sigma_range: tuple (min, max) for filters that accept sigma
        - filter_kwargs: other static kwargs to pass to the filter_fn
        """
        super().__init__()
        self.filter_fn = filter_fn
        self.blur_limit = blur_limit
        self.p = p
        self.sigma_range = sigma_range
        self.filter_kwargs = filter_kwargs or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.shape
        device = x.device

        # Sample odd kernel sizes
        ksizes = torch.randint(
            self.blur_limit[0] // 2, self.blur_limit[1] // 2 + 1, (B,), device=device
        ) * 2 + 1

        if self.sigma_range is not None:
            sigmas = torch.empty(B, device=device).uniform_(*self.sigma_range)
        else:
            sigmas = None

        unique_ksizes, inverse_indices = torch.unique(ksizes, sorted=True, return_inverse=True)
        k_groups = {
            int(k.item()): (inverse_indices == i).nonzero(as_tuple=False).squeeze(1)
            for i, k in enumerate(unique_ksizes)
        }

        # Apply filter per group of same kernel size
        for k, idxs in k_groups.items():
            x_grp = x[idxs]
            kwargs = self.filter_kwargs.copy()

            if sigmas is not None:
                sigma_grp = sigmas[idxs]
                kwargs['sigma'] = torch.stack([sigma_grp, sigma_grp], dim=1)

            if self.filter_fn.__name__ == 'motion_blur':
                # For motion blur, we need to pass the kernel size and angle
                kwargs['angle'] = torch.rand(len(idxs), device=device) * 2.0 - 1.0  # Random angle in [-1, 1]
                kwargs['direction'] = torch.rand(len(idxs), device=device)  # Random direction in [0, 1]

            x[idxs] = self.filter_fn(x_grp, kernel_size=k, **kwargs).to(x.dtype)

        return x


def get_transforms(transform_names, device) -> nn.Sequential:
    if isinstance(transform_names, str):
        transform_names = [transform_names]
    transforms = []
    for transform in transform_names:
        if transform == "jitter":
            transforms.append(K.ColorJiggle(
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.8, 1.2],
                hue=[-0.2, 0.2],
                p=0.2)
            )
        elif transform == "iso":
            transforms.append(ISOlikeNoise(p=0.25))
        elif transform == "brightness":
            transforms.append(K.RandomBrightness(brightness=(0.5, 1.5), p=0.5))
        elif transform == "contrast":
            transforms.append(K.RandomContrast(contrast=(0.5, 1.5), p=0.5))
        elif transform == "random_blur":
            transforms.append(RandomKernelFilter(F.box_blur, blur_limit=(3, 5), p=0.5))
        elif transform == "blur":
            transforms.append(RandomKernelFilter(F.box_blur, blur_limit=(3, 3), p=1.0))
        elif transform == "motion_blur":
            transforms.append(RandomKernelFilter(F.motion_blur, blur_limit=(3, 17), p=0.5))
        elif transform == "binary_opening":
            transforms.append(BinaryOpening(kernel_size=(3, 3), p=1.0, iterations=1))
        elif transform == "gamma":
            transforms.append(K.RandomGamma(gamma=(0.5, 1.5), p=0.5))
        elif transform == "hue":
            transforms.append(K.RandomHue(hue=(-0.5, 0.5), p=0.5))
        elif transform == "saturation":
            transforms.append(K.RandomSaturation(saturation=(0.5, 1.5), p=0.5))
        else:
            raise ValueError(f"Unknown transform: {transform}")
    transforms = nn.Sequential(K.AugmentationSequential(*transforms, data_keys=["input"])).to(device)
    return transforms
