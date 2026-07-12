# Preprocessing & Augmentation Instructions

This document covers the scripts in `scripts/preprocessing/` and `scripts/augmentation/` — the tooling
that prepares per-object assets for the pose estimator and renders/augments Gaussian Splats, starting
from an existing mesh, USD, and Gaussian Splat for the object (see Prerequisites below).

## Prerequisites

Before using these scripts for a new object, you need three assets that all share the **same coordinate
frame**, so that renderings from the USD and the splat stay spatially aligned:

- a **mesh** of the object,
- a **USD** file generated from that mesh (for use as the object's physical/collision representation in
  simulation), and
- a **Gaussian Splat** (`.ply`) of the same object (for photorealistic rendering).

Meshes are typically obtained from a 3D scan or CAD model, USDs via Isaac Lab's standard mesh-to-USD
asset conversion tooling, and splats by training a Gaussian Splatting model (e.g. with
[gsplat](https://github.com/nerfstudio-project/gsplat) or [nerfstudio](https://github.com/nerfstudio-project/nerfstudio))
on captured images of the object.

Once you have them:

1. Copy the mesh to `source/viserdex/viserdex/assets/data/meshes/<object_name>/`.
2. Copy the USD into the Isaac Lab assets directory referenced by `ISAACLAB_ASSETS_DATA_DIR` (see the
   existing entries in `viserdex/assets/objects.py` for the expected layout).
3. Copy the splat `.ply` to `source/viserdex/viserdex/assets/data/splats/<object_name>/`.
4. Add a `TARGET_OBJECT_CFG` entry for the object to `_OBJECT_CFGS` in `viserdex/assets/objects.py`,
   pointing at the paths above.
5. Set `object_name = "<object_name>"` at the bottom of `objects.py`. This one variable controls which
   object every script in this repository targets.

## `scripts/preprocessing/`

These scripts prepare the per-object assets (masks, splat clusters, keypoints) that the pose estimator
and manipulation environment consume, and sanity-check the assets from the Prerequisites step.

1. **`check_gs_camera.py`** — Sanity-check the Gaussian Splat camera renderer against Isaac Sim's native
   camera on the active object. Run this first after adding a new object, to confirm the splat and USD
   are correctly aligned. Writes two comparison images to `scripts/preprocessing/output/`.

   ```bash
   python scripts/preprocessing/check_gs_camera.py --num_envs 2
   ```

2. **`cluster_splat.py`** — Cluster the Gaussians in the active object's splat by color and spatial
   locations, producing the cluster-index files consumed by `scripts/augmentation/augment_splats.py`.
   No CLI arguments; reads the active object from `objects.py`.

   ```bash
   python scripts/preprocessing/cluster_splat.py
   ```

3. **`sample_mesh_keypoints_selection.py`** — Interactively select/sample keypoints on the active
   object's mesh. Set the mesh path inside the script to match the mesh used to create the USD, and make
   sure `points_path` in the object's `TARGET_OBJECT_CFG` entry matches the number of points sampled here.

   ```bash
   python scripts/preprocessing/sample_mesh_keypoints_selection.py
   ```

4. **`visualize_keypoints.py`** — Visualize the keypoints sampled in the previous step on the active
   object, as a sanity check before moving on to training.

   ```bash
   python scripts/preprocessing/visualize_keypoints.py --num_envs 1
   ```

## `scripts/augmentation/`

1. **`augment_splats.py`** — Apply the pre-rasterization augmentation pipeline (noise, clustered
   perturbations, color/spatial shifts) to a splat `.ply` file, using the cluster-index files produced by
   `cluster_splat.py`.

   ```bash
   python scripts/augmentation/augment_splats.py \
       --input path/to/splat.ply \
       --output path/to/augmented_splat.ply \
       --cluster_dir path/to/cluster/indices/
   ```

2. **`augment_viewer.py`** — Interactive [viser](https://github.com/nerfstudio-project/viser)-based
   viewer for inspecting a splat file, its clusters, and augmentations.

   ```bash
   python scripts/augmentation/augment_viewer.py
   ```

## Training and evaluating

Once the assets above exist for an object, see the main [README](../README.md)'s Usage section for the
end-to-end policy-training → pose-estimator-training → evaluation workflow using `scripts/rsl_rl/` and
`scripts/pose_estimator/`.
