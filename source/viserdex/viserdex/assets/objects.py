# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os


VISERDEX_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


_OBJECT_CFGS = {
    "RSLCube":
    {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/RSLCube/RSLCube_0p9.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/RSLCube/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/RSLCube/cube.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/RSLCube/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "test_release_ppo_docker_4",
    },
    "3DPrintedToy":
    {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/3DPrintedToy/3DPrintedToy.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/3DPrintedToy/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/3DPrintedToy/bulbasaur_resized_1p25x_pm_triangulated_centered.obj",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/3DPrintedToy/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "biotac_3d_printed_toy_1p25",
    },
    "Duck":
    {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/Duck/Duck.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/Duck/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/Duck/Duck.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/Duck/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "priveleged_duck",
    },
    "TabletBottle":
    {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/TabletBottle/TabletBottle.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/TabletBottle/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/TabletBottle/Mentos.glb",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/TabletBottle/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "priveleged_tablet_bottle",
    },
    "Globe":
    {
        "usd_path": f"{VISERDEX_ASSETS_DIR}/usd/Globe/Globe.usd",
        "points_path": f"{VISERDEX_ASSETS_DIR}/splats/Globe/icosphere/keypoints/points_8.pt",
        "mesh_path": f"{VISERDEX_ASSETS_DIR}/meshes/Globe/Globe_o3d.obj",
        "splat_path": f"{VISERDEX_ASSETS_DIR}/splats/Globe/icosphere/exports/splat.ply",
        "rollout_policy_run_name": "privileged_globe",
    },
}

# "3DPrintedToy"  # "RSLCube"  # "Toytruck"  # "Duck"  # "TabletBottle"  # "Globe"
object_name = "RSLCube"
TARGET_OBJECT_CFG = _OBJECT_CFGS[object_name]
TARGET_OBJECT_CFG["name"] = object_name
