# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cluster a splat.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import torch
import numpy as np
from plyfile import PlyData
from viserdex.utils.load_utils import load_splat_file
from sklearn.cluster import KMeans

from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array of shape (..., 3) in [0, 1] to CIELAB."""
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    xyz = np.dot(rgb_lin, matrix.T)
    xyz_ref_white = np.array([0.95047, 1.00000, 1.08883])
    xyz_normalized = xyz / xyz_ref_white
    epsilon = 216 / 24389
    kappa = 24389 / 27
    mask2 = xyz_normalized > epsilon
    f_xyz = np.where(mask2, np.power(np.maximum(xyz_normalized, 1e-8), 1/3), (kappa * xyz_normalized + 16) / 116)
    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def load_gaussian_splat_model(splat_model_path: str, device: str = "cpu") -> tuple[dict[str, torch.Tensor], PlyData]:
    """Load your Gaussian splatting model."""
    # check if model_path exists
    if not os.path.exists(splat_model_path):
        raise FileNotFoundError(f"Model file not found at path: {splat_model_path}")

    if splat_model_path.endswith(".ply"):
        splats = load_splat_file(splat_model_path, device)
        plydata = PlyData.read(splat_model_path)
        return splats, plydata

    else:
        raise ValueError(f"Unsupported model file extension: {splat_model_path}")


def cluster_(data: torch.Tensor, num_clusters: int, auto_clusters: bool = False, max_clusters: int = 10) -> tuple[np.ndarray, int]:

    data_np = data.view(data.shape[0], -1).cpu().numpy()
    
    if auto_clusters:
        from sklearn.metrics import silhouette_score
        best_k = 2
        best_score = -1
        # Subsample for silhouette score to speed it up
        sample_size = min(10000, data_np.shape[0])
        
        # Test a range of k
        k_max = max(2, max_clusters)
        if k_max > 20:
            k_range = np.unique(np.linspace(2, k_max, 10, dtype=int))
        else:
            k_range = range(2, k_max + 1)
            
        print(f"Finding optimal clusters from {list(k_range)} using Silhouette Score...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, max_iter=100, n_init=1, random_state=42).fit(data_np)
            score = silhouette_score(data_np, kmeans.labels_, sample_size=sample_size, random_state=42)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = kmeans.labels_
        print(f"Optimal number of clusters: {best_k} (Silhouette Score: {best_score:.4f})")
        
        kmeans = KMeans(n_clusters=best_k, max_iter=100, n_init=1, random_state=42).fit(data_np)
        labels = kmeans.labels_
        actual_num_clusters = best_k
    else:
        kmeans = KMeans(
            n_clusters=num_clusters,
            max_iter=100,
            n_init=1,
            random_state=42,
        ).fit(data_np)
        labels = kmeans.labels_
        actual_num_clusters = num_clusters

    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")

    return labels, actual_num_clusters


if __name__ == "__main__":

    splat_model_path = object_cfg["splat_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats, plydata = load_gaussian_splat_model(splat_model_path, device)
    cluster_dir = os.path.join(os.path.dirname(splat_model_path), "clusters_auto")

    os.makedirs(os.path.join(cluster_dir, "splats"), exist_ok=True)
    os.makedirs(os.path.join(cluster_dir, "indices"), exist_ok=True)

    clusters = {
        "scales": (64, False),
        "quats": (128, False),
        "means3d": (128, False),
        "sh0": (32, True),
    }

    for key, (num_clusters, use_auto) in clusters.items():
        print(f"Clustering {key} with shape {splats[key].shape[1:]} in {num_clusters} clusters (auto={use_auto})...")
        data_to_cluster = splats[key]
        
        if key == "sh0":
            SH_C0 = 0.28209479177387814
            data_to_cluster = data_to_cluster * SH_C0 + 0.5
            data_to_cluster = data_to_cluster.clamp(min=0.0, max=1.0)  # Ensure RGB values are in [0, 1]
            sh0_np = data_to_cluster.cpu().numpy()
            sh0_lab = rgb_to_lab(sh0_np)
            data_to_cluster = torch.tensor(sh0_lab, device=data_to_cluster.device, dtype=data_to_cluster.dtype)

        cluster_ids, actual_num_clusters = cluster_(data_to_cluster, num_clusters, use_auto, num_clusters)
        # print("Cluster IDs:", cluster_ids)

        colors = np.random.rand(actual_num_clusters, 3)
        plydata["vertex"]["f_dc_0"] = colors[cluster_ids][:, 0]
        plydata["vertex"]["f_dc_1"] = colors[cluster_ids][:, 1]
        plydata["vertex"]["f_dc_2"] = colors[cluster_ids][:, 2]
        plydata.write(os.path.join(cluster_dir, "splats", f"splat_clustered_{key}.ply"))

        torch.save(
            torch.tensor(cluster_ids, device=device, dtype=torch.int64),
            os.path.join(cluster_dir, "indices", f"splat_clustered_{key}_indices.pt"),
        )
