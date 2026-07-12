# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any, TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from viserdex.renderer.augmentations.augmentations import (
    SplatAugmentation, NoiseAugmentation, ClusteredSplatNoiseAugmentation, ShiftAugmentation
)


@configclass
class SplatAugmentationCfg:
    """Configuration for the Splat augmentation."""

    func: type[SplatAugmentation] = SplatAugmentation

    params: dict[str, Any] = dict()
    """Parameters for the augmentation function."""

    keys: list[str] = MISSING
    """Keys in the splats dictionary to which noise will be applied."""


@configclass
class NoiseAugmentationCfg(SplatAugmentationCfg):
    """Configuration for the Noise augmentation."""

    func: type[NoiseAugmentation] = NoiseAugmentation
    """The function to apply noise augmentation."""

    noise_cfg: NoiseCfg = MISSING
    """Configuration for the noise to be applied."""

    probability: float | list[float] = 1.0
    """Probability of applying the noise augmentation. Can be a single value or a list of probabilities for each key."""


@configclass
class ClusteredSplatNoiseAugmentationCfg(NoiseAugmentationCfg):
    """Configuration for the clustered splat noise augmentation."""

    func: type[ClusteredSplatNoiseAugmentation] = ClusteredSplatNoiseAugmentation
    """The function to apply clustered splat noise augmentation."""

    cluster_indices_path: str | None = MISSING
    """Path to the file containing cluster indices for the splats."""

    cluster_fraction: float = MISSING
    """Fraction of clusters to apply the noise augmentation to. If set to 1.0, all clusters will be affected."""

    noise_per_cluster: bool = True
    """Whether to use the same noise for all splats in a cluster. If False, each splat will receive its own noise sample."""


@configclass
class ShiftAugmentationCfg(SplatAugmentationCfg):
    """Configuration for the Noise augmentation."""

    func: type[ShiftAugmentation] = ShiftAugmentation
    """The function to apply noise augmentation."""

    noise_cfg: NoiseCfg = MISSING
    """Configuration for the noise to be applied."""

    probability: float | list[float] = 1.0
    """Probability of applying the noise augmentation. Can be a single value or a list of probabilities for each key."""

    apply_per_key: bool = True
    """Whether to apply the shift augmentation per key or to all keys at once."""
