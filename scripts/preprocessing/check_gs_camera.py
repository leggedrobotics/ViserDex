# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth, Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to extract data to generate a gaussian splat model.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

import isaaclab.sim as sim_utils
import omni.log
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer
from viserdex.renderer import GaussianSplatCameraCfg
from viserdex.collectors.location_sampler_cfg import IcoSphereLocationSamplerCfg

from viserdex.assets.objects import TARGET_OBJECT_CFG as object_cfg


# Configuration for Gaussian Splat Camera
@configclass
class GSCameraCubeSceneCfg(InteractiveSceneCfg):

    # object
    object_0 = RigidObjectCfg(
        prim_path="/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_cfg["usd_path"],
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # rigid object to attach the cameras to
    base: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/base",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.0, 0.0)),
    )

    camera_0 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/base/rgb_cam",
        update_period=0,
        data_types=["rgb", "distance_to_image_plane"],
        height=480,
        width=848,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1e6),
        ),
        debug_vis=True,
    )

    gsplat_camera = GaussianSplatCameraCfg(
        prim_path="{ENV_REGEX_NS}/base",
        update_period=0,
        data_types=["rgb", "distance_to_image_plane"],
        camera_model=GaussianSplatCameraCfg.PinholeCfg(
            horizontal_aperture=20.955,
            focal_length=24.0,
            height=480,
            width=848,
        ),
        splat_model_path=object_cfg["splat_path"],
        clipping_range=(0.1, 1e10),
        augmentations=[],
        debug_vis=True,
    )

    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((-1.3, 0.0, 0.4), (1.1, 0.0, -0.4))

    # Initialize scene
    scene_cfg = GSCameraCubeSceneCfg(args_cli.num_envs, env_spacing=1.0)
    with Timer("[INFO]: Scene creation time", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    omni.log.info(f"Scene manager: {scene}")
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    move_object = True

    sampling_cfg = IcoSphereLocationSamplerCfg(radius=0.25, subdivisions=0)
    samples = sampling_cfg.func(sampling_cfg)
    camera_poses = samples[torch.tensor(np.random.choice(len(samples), args_cli.num_envs, replace=False))]

    camera_poses[:, 3:] = math_utils.quat_mul(
        camera_poses[:, 3:],
        math_utils.quat_from_angle_axis(
            torch.tensor(np.pi / 2.0), torch.tensor([0.0, 1.0, 0.0]).to(device=camera_poses.device)
        ).repeat(camera_poses.shape[0], 1),
    )

    scene.rigid_objects["base"].write_root_pose_to_sim(camera_poses.to(device=sim.device))

    if move_object:

        # execute some render steps to make sure the camera buffers are filled
        for _ in range(10):
            sim.render()

        # a genuinely random orientation for the object -- NOT `IcoSphereLocationSamplerCfg`, whose
        # "look at the origin" convention is meaningless (and produces degenerate relative transforms
        # below) when applied to the object's own rotation instead of a viewpoint's.
        object_pos = torch.zeros(1, 3, device=sim.device)
        object_quat = math_utils.random_orientation(1, device=sim.device)
        object_poses = torch.cat([object_pos, object_quat], dim=-1)
        scene.rigid_objects["object_0"].write_root_pose_to_sim(object_poses)

        # `camera_0.data.pos_w`/`quat_w_world` are stale here: `camera_0` is a fixed (no-offset) child
        # prim of `base`, a dynamic rigid body driven by the physx GPU tensor pipeline, and the sensor's
        # `XformPrimView` reads a USD/Fabric transform that doesn't get refreshed by `write_root_pose_to_sim`
        # + `sim.render()`/`scene.update()` alone. `base.data.root_pos_w`/`root_quat_w` (read via the
        # physx tensor API directly) already reflect the write immediately -- reuse them instead of
        # `camera_0`'s stale sensor data.
        #
        # `gsplat_camera`'s *actual* rendered world orientation (when left at its default, un-set pose)
        # is `quat_mul(base_quat_w, offset_ros_to_world)`, NOT `base_quat_w` alone: `GaussianSplatCameraCfg`'s
        # default `offset` has `convention="ros"`, so `_offset_quat` is initialized to
        # `convert(identity, "ros" -> "world")`, a fixed non-identity rotation baked in at sensor init and
        # folded into every pose computation. Feeding `base_quat_w` straight into `set_world_poses(...,
        # convention="world")` below discards that offset entirely (that call sets the *absolute* final
        # pose), which is what produced an all-black render even though the position was numerically
        # sane -- the camera pose was correct in position but facing the wrong way. Reconstruct the same
        # effective orientation here before computing the object-relative transform.
        offset_ros_to_world = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device), origin="ros", target="world"
        )
        camera_pos_w = scene.rigid_objects["base"].data.root_pos_w
        camera_quat_w = math_utils.quat_mul(
            scene.rigid_objects["base"].data.root_quat_w, offset_ros_to_world.repeat(args_cli.num_envs, 1)
        )

        object_pos_w = scene.rigid_objects["object_0"].data.root_pos_w
        object_quat_w = scene.rigid_objects["object_0"].data.root_quat_w

        camera_pos_frame_object, camera_quat_frame_object = math_utils.subtract_frame_transforms(
            object_pos_w.repeat(args_cli.num_envs, 1),
            object_quat_w.repeat(args_cli.num_envs, 1),
            camera_pos_w,
            camera_quat_w,
        )
        scene.sensors["gsplat_camera"].set_world_poses(
            camera_pos_frame_object.to(device=sim.device),
            camera_quat_frame_object.to(device=sim.device),
            convention="world",
        )

    # execute some render steps to make sure the buffers are filled
    for _ in range(10):
        sim.render()

    # update the scene
    scene.update(sim.get_physics_dt())

    # get the rendered images of normal rgb and gsplat rgb
    rgb_cam_output = scene.sensors["camera_0"].data.output["rgb"]  # [B, H, W, 3]
    gsplat_cam_output = scene.sensors["gsplat_camera"].data.output["rgb"]  # [B, H, W, 3]

    # create a matplotlib figure to display the images next to each other
    fig, axs = plt.subplots(args_cli.num_envs, 2)
    for i in range(args_cli.num_envs):
        axs[i, 0].imshow(rgb_cam_output[i].cpu().numpy())
        axs[i, 1].imshow(gsplat_cam_output[i].cpu().numpy())

        if i == 0:
            axs[i, 0].set_title("RGB Camera")
            axs[i, 1].set_title("Gaussian Splat RGB Camera")
    # save the figure in an output directory at the same level as the script
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "gsplat_rgb_camera_comparison.png"))
    plt.close('all')

    # get the rendered images of distance to image plane
    depth_cam_output = scene.sensors["camera_0"].data.output["distance_to_image_plane"]  # [B, H, W, 1]
    depth_gsplat_cam_output = scene.sensors["gsplat_camera"].data.output["distance_to_image_plane"]  # [B, H, W, 1]
    depth_gsplat_cam_output[depth_gsplat_cam_output == 0] = float("inf")

    vmin = min(depth_cam_output.min(), depth_gsplat_cam_output.min())
    vmax = max(depth_cam_output.max(), depth_gsplat_cam_output.max())

    # create a matplotlib figure to display the images next to each other
    fig, axs = plt.subplots(args_cli.num_envs, 2)
    for i in range(args_cli.num_envs):
        axs[i, 0].imshow(depth_cam_output[i].cpu().numpy(), vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(depth_gsplat_cam_output[i].cpu().numpy(), vmin=vmin, vmax=vmax)

        if i == 0:
            axs[i, 0].set_title("Depth (RayCaster Cam)")
            axs[i, 1].set_title("Depth (Gaussian Splat Cam)")
    # save the figure in output directory
    print(f"[INFO] Saving output images to {output_dir}")
    fig.savefig(os.path.join(output_dir, "gsplat_distance_to_image_plane_comparison.png"))
    plt.close('all')

    print("[INFO] Done: both comparison images written. If the process doesn't exit right away, that's "
          "Isaac Sim's own shutdown (extension teardown), not this script hanging -- the outputs above are "
          "already complete and safe to inspect.")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
