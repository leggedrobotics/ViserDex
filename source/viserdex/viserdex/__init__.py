"""
Python module serving as a project/extension template.
"""

# Note: intentionally not importing `.tasks` here (which registers the Gym environments). That import
# chain pulls in the full Isaac Lab stack (including `pxr`, only importable after `AppLauncher` has
# launched Isaac Sim), so scripts that need `viserdex`-package utilities (e.g. `viserdex.utils.cli_args`)
# *before* constructing `AppLauncher` would otherwise crash with `ModuleNotFoundError: No module named
# 'pxr'`. Every script that calls `gym.make` already does `import viserdex.tasks` explicitly, after
# `AppLauncher` is constructed.
