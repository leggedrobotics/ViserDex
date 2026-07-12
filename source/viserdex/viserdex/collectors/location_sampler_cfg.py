# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .location_sampler import sample_icosphere_locations


@configclass
class IcoSphereLocationSamplerCfg:
    """Configuration for sampling poses on an icosphere of a given radius around the origin."""

    func: callable = sample_icosphere_locations
    """The sampling function. Called as `cfg.func(cfg)`, returning an `(N, 7)` pose tensor."""

    radius: float = MISSING
    """Radius of the icosphere to sample poses on."""

    subdivisions: int = 0
    """Number of recursive subdivisions applied to the base 12-vertex icosahedron. 0 keeps all 12 base
    vertices; each additional level roughly quadruples the number of sampled points."""
