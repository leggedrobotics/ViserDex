# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth, Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import re
import time
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.physics.tensors.impl.api as physx
from gsplat.rendering import rasterization
from isaaclab import sim as sim_utils
from isaaclab.sensors import SensorBase
from isaaclab.sensors.camera.camera_data import CameraData
from isaacsim.core.prims import XFormPrim
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from plyfile import PlyData
from pxr import UsdPhysics

from viserdex.renderer.augmentations import SplatAugmentationCfg

if TYPE_CHECKING:
    from .gaussian_splat_camera_cfg import GaussianSplatCameraCfg


class GaussianSplatCamera(SensorBase):
    """A gaussian splatting camera sensor.

    The gaussian splatting camera uses a gaussian splatting model to render the scene. The sensor has the same interface as the
    :class:`isaaclab.sensors.Camera` and :class:`isaaclab.sensors.RayCasterCamera` that implements the camera class through USD camera prims or using ray-casting.
    However, this class provides a faster RGB image generation. The sensor requires a gaussian splat of the scene which can be
    obtained using the `nav_collectors` extension.

    Currently, only the following annotators are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.

    .. note::
        Currently, only static meshes are supported. Extending the splatting model to support dynamic meshes
        is a work in progress.
    """

    cfg: GaussianSplatCameraCfg
    """Configuration for the sensor."""

    UNSUPPORTED_TYPES: set[str] = {
        "distance_to_camera",
        "semantic_segmentation",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: GaussianSplatCameraCfg):
        """
        Initialize the Gaussian Splat Camera.

        Args:
            cfg (GaussianSplatCameraCfg): Configuration for the sensor.
        """
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the camera sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        super().__init__(cfg)

        # initialize data
        self._data = CameraData()

        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> CameraData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.camera_model.height, self.cfg.camera_model.width)

    """
    Operations
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set the intrinsic matrix of the camera.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values (in cm). Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = slice(None)
        # save new intrinsic matrices and focal length
        self._data.intrinsic_matrices[env_ids] = matrices.to(self._device)
        self._focal_length = focal_length

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the sensor.

        Args:
            env_ids (Sequence[int]): The indices of the sensors that are ready to capture.
        """
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        # Reset the frame count
        self._frame[env_ids] = 0

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        """Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.utils.maths.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES

        # get current positions
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        if positions is not None:
            # transform to camera frame
            pos_offset_world_frame = positions - pos_w
            self._offset_pos[env_ids] = math_utils.quat_apply(math_utils.quat_inv(quat_w), pos_offset_world_frame)
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                orientations, origin=convention, target="world"
            )
            self._offset_quat[env_ids] = math_utils.quat_mul(math_utils.quat_inv(quat_w), quat_w_set)

        # update the data
        pos_w, quat_w = self._compute_camera_world_poses(env_ids)
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # get up axis of current stage
        up_axis = stage_utils.get_stage_up_axis()
        # camera position and rotation in opengl convention
        orientations = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis=up_axis, device=self._device)
        )
        self.set_world_poses(eyes, orientations, env_ids, convention="opengl")

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(
                f"The prim at path {prim.GetPath().pathString} is not a physics prim. Defaulting to XFormPrim. \n"
                " The pose of this prim will most likely not be updated correctly when running in headless mode."
            )
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # Initialize the Gaussian splatting renderer
        self._load_gaussian_splat_model()
        # Initialze the augmentations
        self._initialize_augmentations()
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # set offsets
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

    def _load_gaussian_splat_model(self):
        """Load your Gaussian splatting model."""
        # check if model_path exists
        if not os.path.exists(self.cfg.splat_model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.cfg.splat_model_path}")
        else:
            self.splat_model_path = self.cfg.splat_model_path

        # Load the splats from the checkpoint
        if self.cfg.splat_model_path.endswith(".ply"):
            plydata = PlyData.read(self.cfg.splat_model_path)
            vertex = plydata["vertex"]

            xyz = torch.tensor(
                np.vstack((vertex["x"], vertex["y"], vertex["z"])).T, dtype=torch.float32, device=self._device
            )  # [N, 3]
            sh0 = torch.tensor(
                np.vstack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])).T,
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(1)  # [N, 1, 3]

            shN = torch.tensor(
                np.vstack([vertex[f"f_rest_{i}"] for i in range(45)]).T,
                dtype=torch.float32,
                device=self._device,
            )
            shN = shN.reshape(-1, 3, 15).permute(0, 2, 1)  # [N, 15, 3]

            opacities = torch.tensor(vertex["opacity"], dtype=torch.float32, device=self._device).view(-1, 1)
            scales = torch.tensor(
                np.vstack((vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])).T,
                dtype=torch.float32,
                device=self._device,
            )
            rotations = torch.tensor(
                np.vstack((vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])).T,
                dtype=torch.float32,
                device=self._device,
            )

            self._splats = {
                "means3d": xyz,
                "sh0": sh0,
                "shN": shN,
                "opacities": opacities.squeeze(1),
                "scales": scales,
                "quats": rotations,
            }

        else:
            raise ValueError(f"Unsupported model file extension: {self.cfg.splat_model_path}")

    def _initialize_augmentations(self):
        """Initialize the augmentations for the camera."""
        # initialize augmentations
        self._augmentations = list()
        splat_dir = os.path.dirname(self.splat_model_path)
        for augmentation_cfg in self.cfg.augmentations:
            assert isinstance(augmentation_cfg, SplatAugmentationCfg), (
                "Augmentation configuration must be of type SplatAugmentationCfg."
            )
            if hasattr(augmentation_cfg, "cluster_indices_path") and augmentation_cfg.cluster_indices_path is not None:
                cluster_indices_path = augmentation_cfg.cluster_indices_path.replace("dummy_splat_directory", splat_dir)
                augmentation_cfg.cluster_indices_path = cluster_indices_path
            self._augmentations.append(augmentation_cfg.func(augmentation_cfg, self._splats, self._device))

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # increment frame count
        self._frame[env_ids] += 1
        # compute poses from current view
        pos_w, quat_w = self._compute_camera_world_poses(self._ALL_INDICES)
        # update the data
        self._data.pos_w[env_ids] = pos_w[env_ids]
        self._data.quat_w_world[env_ids] = quat_w[env_ids]

        quat_w = self._data.quat_w_ros

        # Construct camtoworld matrices
        camtoworld = torch.eye(4, device=self._device).unsqueeze(0).repeat(pos_w.shape[0], 1, 1)  # Shape: (N, 4, 4)
        camtoworld[:, :3, :3] = math_utils.matrix_from_quat(quat_w)  # Assign rotation matrices
        camtoworld[:, :3, 3] = self._data.pos_w  # Assign positions

        depth_rendered = False
        for data_type in self.cfg.data_types:
            if data_type == "rgb":
                sh_degree = 3
            else:
                continue

            if "distance_to_image_plane" in self.cfg.data_types and not depth_rendered:
                depth_rendered = True
                render_mode = "RGB+D"
            else:
                render_mode = "RGB"

            if env_ids is None:
                env_ids = torch.arange(self._num_envs, device=self._device)
            elif not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self._device)
            batched_env_ids = torch.split(env_ids, self.cfg.batch_size)

            # perform batched augmentations
            batches_splat = {key: value.clone() for key, value in self._splats.items()}
            # Apply augmentations
            for augmentation in self._augmentations:
                batches_splat = augmentation(batches_splat, num_samples=len(batched_env_ids))

            # Ensure all the keys are batched
            for key in batches_splat:
                batches_splat[key] = batches_splat[key].expand(len(batched_env_ids), *self._splats[key].shape)

            for batch_num, batch in enumerate(batched_env_ids):
                with torch.no_grad():
                    # Prepare the colors for rendering
                    batch_splat = {key: value[batch_num] for key, value in batches_splat.items()}
                    colors = torch.cat([batch_splat["sh0"], batch_splat["shN"]], 1)  # [N, K, 3]

                    # detailed documentation of the rasterization function can be found here:
                    # https://docs.gsplat.studio/main/apis/rasterization.html
                    render_images, render_alphas, _ = rasterization(
                        means=batch_splat["means3d"],  # [N, 3]
                        quats=batch_splat["quats"],  # [N, 4]
                        scales=torch.exp(batch_splat["scales"]),  # [N, 3]
                        opacities=torch.sigmoid(batch_splat["opacities"]),  # [N,]
                        colors=colors,
                        viewmats=torch.linalg.inv(camtoworld[batch]),
                        Ks=self._data.intrinsic_matrices[batch],
                        width=self.cfg.camera_model.width,
                        height=self.cfg.camera_model.height,
                        sh_degree=sh_degree,
                        near_plane=self.cfg.clipping_range[0],
                        far_plane=self.cfg.clipping_range[1],
                        packed=True,
                        absgrad=False,
                        sparse_grad=True,
                        rasterize_mode="antialiased",  # "classic", "antialiased"
                        render_mode=render_mode,
                        distributed=False,
                    )  # [B, H, W, 3 or 4]

                assert torch.isfinite(render_images).all(), "Rendered images contain NaN or Inf values."

                # Clamp the rendered images to [0, 1] range
                render_images[..., :3] = torch.clamp(render_images[..., :3], 0.0, 1.0)
                # Store the rendered images
                self._data.output[data_type][batch] = (render_images[..., :3] * 255.0).round().to(torch.uint8)

                if render_mode == "RGB+D":
                    self._data.output["distance_to_image_plane"][batch] = render_images[..., 3].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization objects.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        pass

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: GaussianSplatCameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & GaussianSplatCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"GaussianSplatCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained from the splatting model."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.intrinsic_matrices[:, 2, 2] = 1.0
        # -- image shape
        self._data.image_shape = self.image_shape
        # -- output data
        self._data.output = {}
        self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._view.count)]
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.camera_model.height, self.cfg.camera_model.width, 1)
                dtype = torch.float32
            elif name in ["rgb"]:
                shape = (self.cfg.camera_model.height, self.cfg.camera_model.width, 3)
                dtype = torch.uint8
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._data.output[name] = torch.zeros((self._view.count, *shape), device=self._device, dtype=dtype)

    def _compute_intrinsic_matrices(self):
        """Compute intrinsic matrices for the cameras."""
        # check if vertical aperture is provided
        # if not then it is auto-computed based on the aspect ratio to preserve squared pixels
        if self.cfg.camera_model.vertical_aperture is None:
            self.cfg.camera_model.vertical_aperture = (
                self.cfg.camera_model.horizontal_aperture * self.cfg.camera_model.height / self.cfg.camera_model.width
            )

        # Compute the intrinsic matrix components
        f_x = (
            self.cfg.camera_model.width * self.cfg.camera_model.focal_length / self.cfg.camera_model.horizontal_aperture
        )
        f_y = (
            self.cfg.camera_model.height * self.cfg.camera_model.focal_length / self.cfg.camera_model.vertical_aperture
        )
        c_x = self.cfg.camera_model.horizontal_aperture_offset * f_x + self.cfg.camera_model.width / 2
        c_y = self.cfg.camera_model.vertical_aperture_offset * f_y + self.cfg.camera_model.height / 2

        # allocate the intrinsic matrices
        self._data.intrinsic_matrices[:, 0, 0] = f_x
        self._data.intrinsic_matrices[:, 0, 2] = c_x
        self._data.intrinsic_matrices[:, 1, 1] = f_y
        self._data.intrinsic_matrices[:, 1, 2] = c_y

        # save focal length
        self._focal_length = self.cfg.camera_model.focal_length

    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """
        # obtain the poses of the sensors
        # note: clone arg doesn't exist for xform prim view so we need to do this manually
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # return the pose
        return pos_w.clone(), quat_w.clone()

    def _compute_camera_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z) in "world" convention.
        """
        # get the pose of the view the camera is attached to
        pos_w, quat_w = self._compute_view_world_poses(env_ids)
        # apply offsets
        # need to apply quat because offset relative to parent frame
        pos_w += math_utils.quat_apply(quat_w, self._offset_pos[env_ids])
        quat_w = math_utils.quat_mul(quat_w, self._offset_quat[env_ids])

        return pos_w, quat_w

    def get_world_poses(self, env_ids: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the world poses of the camera.

        Args:
            env_ids (Sequence[int]): The indices of the sensors to get the poses for. Defaults to None, which means all sensor indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the positions and orientations of the cameras in world frame.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # compute camera world poses
        return self._compute_camera_world_poses(env_ids)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._view = None
