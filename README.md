# ViserDex

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-yellow.svg)](LICENCE)

## Overview

This repository contains the Isaac Lab extension for **ViserDex**, developed at the [Robotic Systems
Lab, ETH Zurich](https://rsl.ethz.ch/). ViserDex bridges the visual sim-to-real gap for monocular RGB
in-hand reorientation by performing domain randomization directly in 3D Gaussian Splatting (3DGS)
representation space, producing photorealistic visual data for training.

More details, videos, and results are available on the [project page](https://rffr.leggedrobotics.com/works/viserdex/).

**This release covers the code of ViserDex:**

- The Gaussian Splat rendering and pre-rasterization augmentation pipeline (`viserdex.renderer`)
- The pose estimator model and its training/evaluation loop (`viserdex.tasks.manipulation.inhand.pose_estimator`)
- An in-hand manipulation policy training setup on the default Isaac Lab in-hand reorientation task,
  used to generate the rollout data to train the pose estimator

## Installation

### Option 1: Docker (recommended)

The provided Dockerfile builds directly on top of the public
[`nvcr.io/nvidia/isaac-lab`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab) image.

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build viserdex
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

To run commands inside the running container:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} viserdex /bin/bash
```

For the Isaac Sim GUI window to actually appear on non-headless runs, grant the container access to your
host's X server once per login session, **before** `docker compose up`:

```bash
xhost +local:docker
```

#### Weights & Biases (wandb) login

Training scripts log to [wandb](https://wandb.ai/) by default (`logger = "wandb"` in
`rsl_rl_ppo_cfg.py`). Inside the container, log in once before your first training run:

```bash
wandb login
```

- **`scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py`**: pass `--logger tensorboard` to skip wandb
  entirely, or set `WANDB_MODE=offline` to log to wandb's local format without a login.
- **`scripts/pose_estimator/{train,eval,test}_pose_estimator.py`**: these always use wandb via
  `viserdex`'s own `WandbSummaryWriter`, which additionally requires `WANDB_USERNAME` to be set
  regardless of login/offline status, e.g.:

  ```bash
  export WANDB_USERNAME=your-wandb-username
  export WANDB_MODE=offline   # or `wandb login` instead, if you want runs uploaded live
  ```

### Option 2: Bare metal

- Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Clone this repository outside the `IsaacLab` directory, then install the extension into the same
  Python environment as Isaac Lab:

```bash
cd viserdex
python -m pip install -e .
```

### Data

The meshes, optimized Gaussian Splats, and preprocessed datasets for the ViserDex object set are hosted
separately from the code as a HuggingFace dataset. See [ViserDex-Dataset](#viserdex-dataset) below for
download instructions.

## Usage

All scripts are run from the repository root with the Isaac Lab Python interpreter (`./isaaclab.sh -p`
if installed via the official installer, or `python` inside the Docker container). Simulation scripts
(marked \*) launch an Isaac Sim app instance and accept `AppLauncher`'s standard flags (e.g. `--headless`).

### End-to-end workflow

The manipulation policy and the pose estimator are trained separately, with the policy trained first so
the pose estimator has something to roll out against:

1. **Train the manipulation ("rollout") policy** with RSL-RL PPO on the default in-hand task:

   ```bash
   python scripts/rsl_rl/train.py --task Isaac-ViserDex-Repose-Allegro-v0
   ```

   This writes a checkpoint under `logs/rsl_rl/<experiment_name>/<run_name>/`.

2. **Point the active object at the new checkpoint.** Open `viserdex/assets/objects.py` and set
   `rollout_policy_run_name` for the active object's entry in `_OBJECT_CFGS` to the `<run_name>` from
   step 1.

3. **Train the pose estimator**, which rolls out the policy from step 1/2 through the Gaussian Splat
   camera to generate its training images.

   ```bash
   python scripts/pose_estimator/train_pose_estimator.py --task Isaac-ViserDex-Repose-Allegro-Estimator-v0 --enable_cameras
   ```

4. **Evaluate** with `scripts/pose_estimator/eval_pose_estimator.py`.

**Switching objects:** every script in this repository reads the active object from the single
`object_name` variable at the bottom of `viserdex/assets/objects.py` — change it once there and every
script (paths, policy checkpoints, keypoints, splats) picks up the new object automatically. Adding a new
object entirely is covered in [`scripts/preprocessing_instructions.md`](scripts/preprocessing_instructions.md).

### Script reference

| Script | Description |
| --- | --- |
| `scripts/rsl_rl/train.py`\* | Train the in-hand manipulation policy with RSL-RL PPO on the default Isaac Lab in-hand task. |
| `scripts/rsl_rl/play.py`\* | Play back / evaluate a trained RSL-RL policy checkpoint. |
| `scripts/pose_estimator/train_pose_estimator.py`\* | Train the pose estimator online, using rollouts from a trained manipulation policy rendered through the Gaussian Splat camera. |
| `scripts/pose_estimator/eval_pose_estimator.py`\* | Roll out a trained policy and log pose-estimator metrics without further training. |
| `scripts/pose_estimator/test_pose_estimator.py`\* | Evaluate a trained pose estimator against a held-out real-world capture dataset, reporting translation/rotation error via keypoints + Rigid Procrustes. |
| `scripts/preprocessing/check_gs_camera.py`\* | Sanity-check the Gaussian Splat camera renderer against Isaac Sim's native camera on the active object. |
| `scripts/preprocessing/visualize_keypoints.py`\* | Visualize sampled object keypoints on the active object. |
| `scripts/preprocessing/sample_mesh_keypoints_selection.py`\* | Interactively select/sample keypoints on an object mesh. |
| `scripts/preprocessing/cluster_splat.py`\* | Cluster the Gaussians in an existing splat `.ply` file by color, producing the cluster-index files used by the augmentation pipeline. |
| `scripts/augmentation/augment_splats.py` | Apply the pre-rasterization augmentation pipeline (noise, clustered perturbations, color/spatial shifts) to a splat `.ply` file. |
| `scripts/augmentation/augment_viewer.py` | Interactive [viser](https://github.com/nerfstudio-project/viser)-based viewer for inspecting a splat file and its clusters. |
| `scripts/list_envs.py`\* | Print all Isaac Lab environments registered by this extension. |

See [`scripts/preprocessing_instructions.md`](scripts/preprocessing_instructions.md) for detailed, per-argument
instructions on the `preprocessing`/`augmentation` scripts and how to onboard a new object.

## ViserDex-Dataset

Meshes, splats, and evaluation datasets are hosted on HuggingFace at `leggedrobotics/ViserDexSplats`.
Fetch them with:

```bash
pip install huggingface_hub
python scripts/download_data.py --repo-id leggedrobotics/ViserDexSplats
```

This places meshes/splats under `source/viserdex/viserdex/assets/data/` and the real-world evaluation
keypoint captures under `data/keypoints/`, matching the paths referenced in `viserdex.assets.objects`.
The data is licensed separately from the code, under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(attribution required, non-commercial use only).

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

## Third-party licenses

This project depends on several third-party packages; their license texts are collected under
[`docs/licenses/`](docs/licenses/).

## License

This project is licensed under the [BSD-3-Clause License](LICENCE), Copyright (c) 2026 ETH Zurich
(Robotic Systems Lab). See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the list of code contributors.

## Citation

ViserDex has been accepted to RSS 2026; the official proceedings citation will be added here once
available. Until then, please cite the arXiv preprint:

```bibtex
@article{bhardwaj2026viserdex,
  title={ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation},
  author={Bhardwaj, Arjun and Wilder-Smith, Maximum and Mittal, Mayank and Patil, Vaishakh and Hutter, Marco},
  journal={arXiv preprint arXiv:2604.11138},
  year={2026}
}
```
