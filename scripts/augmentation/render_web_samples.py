# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-render a bank of static preview images for the project website's augmentation viewer.

For each object, renders one "base" (unaugmented) image plus `--num_samples` independent draws for
each of the four augmentation categories `augment_viewer.py` exposes (random / spatial / color /
global), each using that tool's default slider values. Unlike `augment_viewer.py`, this doesn't open an
interactive viser session -- it rasterizes directly with `gsplat` and writes plain JPGs, so it can run
headless and batch through many samples quickly.

Usage:
    python scripts/augmentation/render_web_samples.py --output-dir /path/to/output --num_samples 20
"""

import argparse
import copy
import os

import numpy as np
import torch
from gsplat.rendering import rasterization
from PIL import Image

from augment_splats import generate_default_augmentations
from viserdex.assets.objects import _OBJECT_CFGS
from viserdex.utils.load_utils import load_splat_file

IMAGE_SIZE = 500
# matches check_gs_camera.py's PinholeCameraCfg ratio (focal_length=24, horizontal_aperture=20.955)
FOCAL_TO_APERTURE_RATIO = 24.0 / 20.955
VIEW_DIRECTION = np.array([1.0, -1.0, 0.6])  # camera offset direction, in a 3/4-isometric-ish view
DISTANCE_MARGIN = 1.6  # multiplies the fitted distance so the object doesn't fill the whole frame


def look_at_camtoworld(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Builds a camera-to-world matrix for a camera at `eye` looking at `target`, in the ROS camera
    convention (`+Z` forward, `+X` right, `+Y` down) that `gsplat`'s rasterizer expects, given a
    `Z`-up world (matching Isaac Sim/USD's convention)."""
    world_up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    camtoworld = np.eye(4, dtype=np.float32)
    camtoworld[:3, 0] = right
    camtoworld[:3, 1] = down
    camtoworld[:3, 2] = forward
    camtoworld[:3, 3] = eye
    return camtoworld


def fit_camera_to_splat(means3d: torch.Tensor, fov_rad: float) -> np.ndarray:
    """Positions the camera so the splat's bounding sphere comfortably fills the frame, regardless of
    the object's own physical scale."""
    points = means3d.detach().cpu().numpy()
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())

    direction = VIEW_DIRECTION / np.linalg.norm(VIEW_DIRECTION)
    distance = (radius / np.sin(fov_rad / 2.0)) * DISTANCE_MARGIN
    eye = center + direction * distance
    return look_at_camtoworld(eye, center)


def render_view(splats: dict, camtoworld: np.ndarray, device: str) -> np.ndarray:
    """Rasterizes `splats` from `camtoworld`, composited onto a white background, as an (H, W, 3)
    uint8 RGB array."""
    means = splats["means3d"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, 16, 3]

    f = IMAGE_SIZE * FOCAL_TO_APERTURE_RATIO
    c = IMAGE_SIZE / 2.0
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, dtype=torch.float32, device=device))

    render_images, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        sh_degree=3,
        near_plane=0.01,
        far_plane=1e10,
        packed=True,
        rasterize_mode="antialiased",
        render_mode="RGB",
    )
    rgb = render_images[0, ..., :3].clamp(0.0, 1.0)
    alpha = render_alphas[0, ..., :1].clamp(0.0, 1.0)
    composited = rgb * alpha + (1.0 - alpha)  # white background
    return (composited.cpu().numpy() * 255.0).round().astype(np.uint8)


# (category name, slice into generate_default_augmentations()'s list, default slider values from
# augment_viewer.py) -- "probability"/"cluster_fraction" is what augment_viewer.py's per-category
# slider actually controls; everything else is left at generate_default_augmentations()'s defaults.
CATEGORIES = {
    "random": (slice(0, 4), {"probability": 1.0, "range": 0.2}),
    "spatial": (slice(4, 6), {"cluster_fraction": 0.8, "range": 0.9}),
    "color": (slice(6, 9), {"cluster_fraction": 0.2, "range": 0.5}),
    "global": (slice(9, None), {"cluster_fraction": 0.2, "shift_probability": 0.8, "range": 0.6}),
}


def apply_category(category: str, base_augmentations, splats: dict, device: str) -> dict:
    """Applies one augmentation category (matching `augment_viewer.py`'s `setup_and_run_*` functions)
    to a fresh copy of `splats`, using a fresh random draw each call."""
    augmented = copy.deepcopy(splats)
    sel, defaults = CATEGORIES[category]
    cfgs = copy.deepcopy(base_augmentations[sel])

    augs = []
    for cfg in cfgs:
        if category == "random":
            cfg.probability = defaults["probability"]
        elif category in ("spatial", "color"):
            cfg.probability = 1.0
            cfg.cluster_fraction = defaults["cluster_fraction"]
        elif category == "global":
            if "Shift" in cfg.__class__.__name__:
                cfg.probability = defaults["shift_probability"]
            else:
                cfg.cluster_fraction = defaults["cluster_fraction"]

        if cfg.noise_cfg.operation == "scale":
            cfg.noise_cfg.n_max = 1 + defaults["range"]
            cfg.noise_cfg.n_min = 1 - defaults["range"]
        else:
            cfg.noise_cfg.n_max = defaults["range"]
            cfg.noise_cfg.n_min = -defaults["range"]

        augs.append(cfg.func(cfg, augmented, device))

    for aug in augs:
        aug(augmented, num_samples=1)

    # augmentations add a leading `num_samples` batch dim only to the keys they actually touch --
    # squeeze it back off (num_samples=1 here) so every key has a consistent, un-batched shape again.
    for key, value in augmented.items():
        if value.dim() == splats[key].dim() + 1:
            augmented[key] = value.squeeze(0)

    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write rendered JPGs to.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of random draws per augmentation category.")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-95).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fov_rad = 2 * np.arctan(IMAGE_SIZE / (2 * IMAGE_SIZE * FOCAL_TO_APERTURE_RATIO))

    for obj_name, cfg in _OBJECT_CFGS.items():
        print(f"[INFO] === {obj_name} ===")
        splat_path = cfg["splat_path"]
        cluster_path = os.path.join(os.path.dirname(splat_path), "clusters", "indices")
        out_dir = os.path.join(args.output_dir, obj_name)
        os.makedirs(out_dir, exist_ok=True)

        splats = load_splat_file(splat_path, device)
        camtoworld = fit_camera_to_splat(splats["means3d"], fov_rad)
        default_augmentations = generate_default_augmentations(cluster_path)

        base_img = render_view(splats, camtoworld, device)
        Image.fromarray(base_img).save(os.path.join(out_dir, "base.jpg"), quality=args.quality)
        print(f"[INFO] wrote base.jpg")

        for category in CATEGORIES:
            for i in range(args.num_samples):
                augmented = apply_category(category, default_augmentations, splats, device)
                img = render_view(augmented, camtoworld, device)
                Image.fromarray(img).save(os.path.join(out_dir, f"{category}_{i:02d}.jpg"), quality=args.quality)
            print(f"[INFO] wrote {args.num_samples} samples for '{category}'")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
