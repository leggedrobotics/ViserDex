import os

# NOTE: intentionally not importing `.allegro` here (like `viserdex/__init__.py` does for `.tasks`) — it
# pulls in `isaaclab.sim`, which requires `pxr` (only importable after `AppLauncher` has launched Isaac
# Sim). Scripts that only need `VISERDEX_ASSETS_DIR` (e.g. `scripts/augmentation/augment_viewer.py`, which
# never launches Isaac Sim at all) would otherwise crash with `ModuleNotFoundError: No module named 'pxr'`.
# Import `ALLEGRO_HAND_CFG` directly from `viserdex.assets.allegro` instead.

VISERDEX_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
