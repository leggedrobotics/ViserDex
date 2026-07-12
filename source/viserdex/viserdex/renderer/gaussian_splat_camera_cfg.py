# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .gaussian_splat_camera import GaussianSplatCamera
from .augmentations import SplatAugmentationCfg
from .augmentations.default_augmentations import DefaultAugmentations


@configclass
class GaussianSplatCameraCfg(SensorBaseCfg):
    """Configuration for the Gaussian Splat Camera."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.
        """

    @configclass
    class PinholeCfg:
        """Configuration for a pinhole camera depth image pattern for ray-casting.

        .. caution::
            Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
            world unit is meters, so all of these values are in cm. For more information, please check:
            https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html
        """

        camera_model: str = "pinhole"

        focal_length: float = 24.0
        """Perspective focal length (in cm). Defaults to 24.0cm.

        Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
        """

        horizontal_aperture: float = 20.955
        """Horizontal aperture (in cm). Defaults to 20.955 cm.

        Emulates sensor/film width on a camera.

        Note:
            The default value is the horizontal aperture of a 35 mm spherical projector.
        """
        vertical_aperture: float | None = None
        r"""Vertical aperture (in cm). Defaults to None.

        Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
        horizontal aperture and the aspect ratio of the image to maintain squared pixels. In this case, the vertical
        aperture is calculated as:

        .. math::
            \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
        """

        horizontal_aperture_offset: float = 0.0
        """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""

        vertical_aperture_offset: float = 0.0
        """Offsets Resolution/Film gate vertically. Defaults to 0.0."""

        width: int = MISSING
        """Width of the image (in pixels)."""

        height: int = MISSING
        """Height of the image (in pixels)."""

        @classmethod
        def from_intrinsic_matrix(
            cls,
            intrinsic_matrix: list[float],
            width: int,
            height: int,
            focal_length: float = 24.0,
        ) -> GaussianSplatCameraCfg.PinholeCfg:
            r"""Create a :class:`PinholeCameraPatternCfg` class instance from an intrinsic matrix.

            The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
            the 2D image. The matrix is defined as:

            .. math::
                I_{cam} = \begin{bmatrix}
                f_x & 0 & c_x \\
                0 & f_y & c_y \\
                0 & 0 & 1
                \end{bmatrix},

            where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while :math:`c_x` and :math:`c_y` are the
            principle point offsets along x and y direction respectively.

            Args:
                intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                    The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
                width: Width of the image (in pixels).
                height: Height of the image (in pixels).
                focal_length: Focal length of the camera (in cm). Defaults to 24.0 cm.

            Returns:
                An instance of the :class:`PinholeCameraPatternCfg` class.
            """
            # extract parameters from matrix
            f_x = intrinsic_matrix[0]
            c_x = intrinsic_matrix[2]
            f_y = intrinsic_matrix[4]
            c_y = intrinsic_matrix[5]
            # resolve parameters for usd camera
            horizontal_aperture = width * focal_length / f_x
            vertical_aperture = height * focal_length / f_y
            horizontal_aperture_offset = (c_x - width / 2) / f_x
            vertical_aperture_offset = (c_y - height / 2) / f_y

            return cls(
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
                horizontal_aperture_offset=horizontal_aperture_offset,
                vertical_aperture_offset=vertical_aperture_offset,
                width=width,
                height=height,
            )

    @configclass
    class OrthoCameraCfg:
        """Configuration for an orthographic camera.

        NOT YET IMPLEMENTED."""

        camera_model: str = "ortho"

        width: int = MISSING
        """Width of the image (in pixels)."""

        height: int = MISSING
        """Height of the image (in pixels)."""

    @configclass
    class FisheyeCameraCfg:
        """Configuration for a fisheye camera.

        NOT YET IMPLEMENTED."""

        camera_model: str = "fisheye"

        width: int = MISSING
        """Width of the image (in pixels)."""

        height: int = MISSING
        """Height of the image (in pixels)."""

    class_type: type = GaussianSplatCamera

    splat_model_path: str = MISSING
    """Path to the Gaussian splat model file."""

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"].

    Please refer to the :class:`GaussianSplatCamera` class for a list of available data types.
    """

    batch_size: int = 128
    """Batch size for the camera."""

    clipping_range: tuple[float, float] = (0.01, 1e6)
    """Near and far clipping distances (in m). Defaults to (0.01, 1e6).

    The minimum clipping range will shift the camera forward by the specified distance. Don't set it too high to
    avoid issues for distance related data types (e.g., ``distance_to_image_plane``).
    """

    augmentations: list[SplatAugmentationCfg] = DefaultAugmentations
    """List of augmentations to apply to the camera data. Defaults to []."""

    camera_model: PinholeCfg | OrthoCameraCfg | FisheyeCameraCfg = PinholeCfg()
    """Configuration for the camera model."""

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""
