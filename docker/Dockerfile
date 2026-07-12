# we use the public Isaac Lab image as the base
FROM nvcr.io/nvidia/isaac-lab:2.3.2 AS base

USER root

ARG DOCKER_VISERDEX_EXTENSION_PATH_ARG
ENV DOCKER_VISERDEX_EXTENSION_PATH=${DOCKER_VISERDEX_EXTENSION_PATH_ARG}

# Copy the ViserDex extension directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_VISERDEX_EXTENSION_PATH}

# NOTE: we deliberately do NOT upgrade rsl-rl-lib here — we target whatever version the base image ships
# (currently rsl-rl-lib==3.1.2), so that isaaclab_rl.rsl_rl's own bundled wrapper/config classes (which are
# built and tested against that exact pairing) are used as-is, with no patching.

# Upgrade wandb: the base image ships wandb==0.24.0, which has a known bug that can silently fail to
# upload run data. Unlike rsl-rl-lib, wandb has no schema-compatibility coupling with Isaac Lab/rsl_rl to
# worry about, so it's safe to upgrade independently.
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install --upgrade wandb==0.28.0

# Install the CUDA 12.8 nvcc compiler (matching the base image's torch==2.7.0+cu128 build). The base image
# only ships the CUDA *runtime* (driver + libraries, enough to run prebuilt CUDA binaries like Isaac Sim's
# own), not the *toolkit* (nvcc), and doesn't have NVIDIA's CUDA apt repo registered either. gsplat has no
# prebuilt wheel for Python 3.11 on any version, so it always JIT-compiles its CUDA kernels on first use via
# torch's cpp_extension mechanism, which needs a real nvcc on PATH.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-12.8
# Also add Isaac Sim's own bundled Python's bin/ to PATH: several pip packages (ninja, wandb, ...) install
# console scripts there, but it isn't on PATH by default, only reachable via `/isaac-sim/python.sh -m <pkg>`.
# torch's JIT build step for gsplat shells out to the bare `ninja`/`nvcc` commands directly, so both need to
# be resolvable without the `-m` indirection.
ENV PATH=${CUDA_HOME}/bin:/isaac-sim/kit/python/bin:${PATH}
# A console script's shebang launches the bare interpreter directly, bypassing /isaac-sim/python.sh's own
# wrapper (setup_python_env.sh) that normally injects several extra site-packages-like directories via
# PYTHONPATH — notably `extscache/.../pip_prebundle`, where packages like typing_extensions actually live.
# Without this, invoking `wandb` (or any other console script) as a bare command crashes with
# ModuleNotFoundError even though `pip show` finds the package fine (it's on PYTHONPATH, just not applied
# for non-python.sh invocations). Set the same PYTHONPATH here so bare console scripts work too.
ENV PYTHONPATH=/isaac-sim/kit/python/lib/python3.11:/isaac-sim/kit/python/lib/python3.11/site-packages:/isaac-sim/python_packages:/isaac-sim/exts/isaacsim.simulation_app:/isaac-sim/extsDeprecated/omni.isaac.kit:/isaac-sim/kit/kernel/py:/isaac-sim/kit/plugins/bindings-python:/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle:/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle:/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle:/isaac-sim/exts/omni.pip.compute/pip_prebundle:/isaac-sim/exts/omni.pip.cloud/pip_prebundle

# Install the viserdex extension package (setup.py lives at the repo root, not source/viserdex/ —
# that subdirectory only has a build-system-only pyproject.toml for the actual viserdex python package)
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install -e ${DOCKER_VISERDEX_EXTENSION_PATH}

# make working directory as the extension directory
# this is the default directory when the container is run
WORKDIR ${DOCKER_VISERDEX_EXTENSION_PATH}
