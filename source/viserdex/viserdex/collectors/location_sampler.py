# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal viewpoint sampler: poses on a sphere around the origin, each looking at the origin.

This is a lightweight, from-scratch replacement for the viewpoint-sampling tooling that shipped with the
splat-*creation* pipeline (out of scope for this release, see `scripts/preprocessing_instructions.md`).
It only exists to give `scripts/preprocessing/check_gs_camera.py` a source of "camera around object"
poses for its own sanity checks -- it makes no attempt to replicate that original tooling.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.utils.math as math_utils


def _icosphere_vertices(subdivisions: int) -> np.ndarray:
    """Unit-sphere vertices from a recursively-subdivided icosahedron (a standard geodesic sphere)."""
    phi = (1.0 + 5.0**0.5) / 2.0
    verts = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ]  # fmt: skip
    verts = [list(np.array(v) / np.linalg.norm(v)) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in midpoint_cache:
                m = (np.array(verts[i]) + np.array(verts[j])) / 2.0
                verts.append(list(m / np.linalg.norm(m)))
                midpoint_cache[key] = len(verts) - 1
            return midpoint_cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    return np.array(verts)


def sample_icosphere_locations(cfg) -> torch.Tensor:
    """Sample poses on an icosphere of `cfg.radius`, each oriented with local +X pointing at the origin.

    Returns:
        A `(N, 7)` tensor of `[x, y, z, qw, qx, qy, qz]` poses (Isaac Lab's usual position + wxyz-quaternion
        layout).
    """
    positions = torch.tensor(_icosphere_vertices(cfg.subdivisions), dtype=torch.float32) * cfg.radius

    forward = -torch.nn.functional.normalize(positions, dim=-1)
    up_ref = torch.tensor([0.0, 0.0, 1.0]).expand_as(forward)
    # fall back to a different "up" reference wherever it's (nearly) parallel to `forward`
    degenerate = torch.cross(forward, up_ref, dim=-1).norm(dim=-1, keepdim=True) < 1e-4
    up_ref = torch.where(degenerate, torch.tensor([0.0, 1.0, 0.0]), up_ref)

    right = torch.nn.functional.normalize(torch.cross(up_ref, forward, dim=-1), dim=-1)
    up = torch.cross(forward, right, dim=-1)

    # columns are the local axes (forward, right, up) expressed in the world frame
    rot = torch.stack([forward, right, up], dim=-1)
    quat = math_utils.quat_from_matrix(rot)

    return torch.cat([positions, quat], dim=-1)
