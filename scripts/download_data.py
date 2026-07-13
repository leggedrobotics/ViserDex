# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download and extract the meshes/splats bundle used by this extension.

The data is hosted separately from the code on HuggingFace and licensed separately (see the README's
"ViserDex-Dataset" section). This script downloads the dataset snapshot and places it at the path
expected by ``viserdex.assets.objects`` (``source/viserdex/viserdex/assets/data/``).

Usage:
    python scripts/download_data.py [--repo-id leggedrobotics/ViserDexSplats] [--revision main]
"""

import argparse
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DATA_DIR = os.path.join(REPO_ROOT, "source", "viserdex", "viserdex", "assets", "data")

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

    src = os.path.join(snapshot_path, "assets_data")
    if not os.path.isdir(src):
        print("[WARN] 'assets_data' not found in dataset snapshot, skipping.")
    else:
        os.makedirs(os.path.dirname(ASSETS_DATA_DIR), exist_ok=True)
        print(f"[INFO] Copying {src} -> {ASSETS_DATA_DIR}")
        shutil.copytree(src, ASSETS_DATA_DIR, dirs_exist_ok=True)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
