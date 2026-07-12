# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sample mesh points")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import open3d as o3d
import torch
import os
import numpy as np
from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg

# Load mesh
mesh_path = object_cfg["mesh_path"]
mesh = o3d.io.read_triangle_mesh(mesh_path)

# Shift mesh so that the center of its bounding box is at origin
bbox = mesh.get_axis_aligned_bounding_box()
print("Bounding box min:", bbox.min_bound)
print("Bounding box max:", bbox.max_bound)
print("Bounding box size:", bbox.get_extent())
bbox_center = bbox.get_center()
mesh.translate(-bbox_center)

# Rotate mesh 90° around X axis
R = mesh.get_rotation_matrix_from_xyz((np.radians(90), 0, 0))
mesh.rotate(R, center=(0, 0, 0))

mesh.compute_vertex_normals()


pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
pts = np.asarray(pcd.points)

picked = o3d.visualization.VisualizerWithEditing()
picked.create_window(window_name="Pick Points", width=1024, height=768)
picked.add_geometry(pcd)
picked.run()       # blocks until window is closed
picked_indices = picked.get_picked_points()
picked.destroy_window()

pcd_points = np.asarray(pcd.points)
mesh_vertices = np.asarray(mesh.vertices)

def closest_vertex(p):
    diffs = mesh_vertices - p
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    return idx, mesh_vertices[idx]

results = []
for idx in picked_indices:
    p = pcd_points[idx]
    vidx, v = closest_vertex(p)
    results.append((idx, p, vidx, v))
selected_points = np.asarray([v for _, _, _, v in results])

if len(selected_points) == 0:
    print("No points were selected. Exiting.")
    exit(0)

print("locations:", selected_points.shape)

print("\n================= RESULTS =================")
for pi, p, vidx, v in results:
    # print(f"Picked point cloud idx: {pi}")
    # print(f"  coordinate: {p}")
    # print(f"Closest mesh vertex idx: {vidx}")
    print(f"  vertex coord: {v}")
    print("-------------------------------------------")


output_path = os.path.join(os.path.dirname(object_cfg["points_path"]), f"points_{len(results)}.pt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

results_points_tensor = torch.tensor(selected_points, dtype=torch.float32)
torch.save(results_points_tensor, output_path)
print(f"Saved selected points of shape {results_points_tensor.shape} to {output_path}")
