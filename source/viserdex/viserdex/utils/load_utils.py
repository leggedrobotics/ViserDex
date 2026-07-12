# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from plyfile import PlyData


def load_splat_file(splat_model_path, device="cpu"):
    """Loads a .ply gaussian splat file."""
    plydata = PlyData.read(splat_model_path)
    vertex = plydata["vertex"]

    xyz = torch.tensor(
        np.vstack((vertex["x"], vertex["y"], vertex["z"])).T, dtype=torch.float32, device=device
    )  # [N, 3]
    sh0 = torch.tensor(
        np.vstack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])).T,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)  # [N, 1, 3]

    shN = torch.tensor(
        np.vstack([vertex[f"f_rest_{i}"] for i in range(45)]).T,
        dtype=torch.float32,
        device=device,
    )
    shN = shN.reshape(-1, 3, 15).permute(0, 2, 1)  # [N, 15, 3]

    opacities = torch.tensor(vertex["opacity"], dtype=torch.float32, device=device).view(-1, 1)
    scales = torch.tensor(
        np.vstack((vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])).T,
        dtype=torch.float32,
        device=device,
    )
    rotations = torch.tensor(
        np.vstack((vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])).T,
        dtype=torch.float32,
        device=device,
    )

    splat = {
        "means3d": xyz,
        "sh0": sh0,
        "shN": shN,
        "opacities": opacities.squeeze(1),
        "scales": scales,
        "quats": rotations,
    }

    return splat
