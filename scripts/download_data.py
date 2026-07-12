# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download and extract the meshes/splats/keypoint-dataset bundle used by this extension.

The data (~18 GB) is hosted separately from the code on HuggingFace and licensed separately (see the
README's "Data hosting" section). This script downloads the dataset snapshot and places it at the paths
expected by ``viserdex.assets.objects`` (``source/viserdex/viserdex/assets/data/``) and by the
pose-estimator test scripts (``data/keypoints/``).

Usage:
    python scripts/download_data.py [--repo-id leggedrobotics/ViserDexSplats] [--revision main]
"""

import argparse
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DATA_DIR = os.path.join(REPO_ROOT, "source", "viserdex", "viserdex", "assets", "data")
KEYPOINTS_DATA_DIR = os.path.join(REPO_ROOT, "data", "keypoints")

# Note: viewpoint-capture data used to *create* splats (as opposed to render with them) is intentionally
# not part of this download -- that pipeline is not included in this release.
DEFAULT_REPO_ID = "leggedrobotics/ViserDexSplats"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id to pull from.")
    parser.add_argument("--revision", type=str, default="main", help="Dataset revision/tag to download.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("This script requires huggingface_hub: pip install huggingface_hub") from e

    print(f"[INFO] Downloading '{args.repo_id}' (revision={args.revision})...")
    snapshot_path = snapshot_download(repo_id=args.repo_id, revision=args.revision, repo_type="dataset")

    for sub_dir, target_dir in [("assets_data", ASSETS_DATA_DIR), ("keypoints", KEYPOINTS_DATA_DIR)]:
        src = os.path.join(snapshot_path, sub_dir)
        if not os.path.isdir(src):
            print(f"[WARN] '{sub_dir}' not found in dataset snapshot, skipping.")
            continue
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        print(f"[INFO] Copying {src} -> {target_dir}")
        shutil.copytree(src, target_dir, dirs_exist_ok=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
