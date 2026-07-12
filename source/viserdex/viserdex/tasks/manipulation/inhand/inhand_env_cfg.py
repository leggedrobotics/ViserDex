# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import viserdex.tasks.manipulation.inhand.pose_estimator.networks as networks
from viserdex.tasks.manipulation.inhand.pose_estimator import PoseEstimatorCfg, FULL_AUGS, GS_ONLY_AUGS
from viserdex.renderer import GaussianSplatCameraCfg
from viserdex.renderer.augmentations.default_augmentations import DefaultAugmentations
import viserdex.tasks.manipulation.inhand.mdp as mdp
from viserdex.assets import VISERDEX_ASSETS_DIR
from viserdex.assets.objects import TARGET_OBJECT_CFG

##
# Scene definition
##


@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = MISSING

    # objects
    object: RigidObjectCfg = MISSING

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.32, 0.32, 0.32), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        marker_pos_offset=(0.06, -0.2, 0.08),
        debug_vis=True,
        goal_pose_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_marker",
            markers={
                "goal": sim_utils.UsdFileCfg(
                    usd_path=TARGET_OBJECT_CFG["usd_path"],
                    scale=(1.0, 1.0, 1.0),
                ),
            },
        )
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class Proprioceptive(ObsGroup):
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class Privileged(ObsGroup):
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )

    @configclass
    class Pose(ObsGroup):
        image_features = ObsTerm(
            func=mdp.gaussian_splat_image_to_pose,
            params={
                "pose_estimator_cfg": PoseEstimatorCfg(augmentation_names=GS_ONLY_AUGS),
                "camera_cfg": SceneEntityCfg('gsplat_camera'),
                "object_cfg": SceneEntityCfg('object'),
                "points_path": TARGET_OBJECT_CFG["points_path"],
                "masking_data": f"{VISERDEX_ASSETS_DIR}/masking_data/masking_data.pt",
                "debug_vis": False,
            }
        )

    # observation groups
    policy: Proprioceptive = Proprioceptive()
    privileged: Privileged = Privileged()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.5),
            "dynamic_friction_range": (0.7, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.6, 1.5),  # default: 3.0
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.5),
            "dynamic_friction_range": (0.7, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [-0.01, 0.01],
                "yaw": [-3.14, 3.14], "pitch": [-3.14, 3.14], "roll": [-3.14, 3.14]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            "position_range": {".*": [0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_pos_l2 = RewTerm(
    #     func=mdp.track_pos_l2,
    #     weight=-10.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )
    track_orientation_inv_l2 = RewTerm(
        func=mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"},
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- optional penalties (these are disabled by default)
    # object_away_penalty = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-0.0,
    #     params={"term_keys": "object_out_of_reach"},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"}
    )

    object_out_of_reach = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})


##
# Environment configuration
##


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    # Scene settings
    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(num_envs=8192, env_spacing=2.0, replicate_physics=True)
    # Viewer settings
    viewer = ViewerCfg(
        eye=(-0.5, 0.0, 1.0),
        lookat=(0.9, 0.0, -0.3),
        origin_type="env",
        env_index=0,
    )
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**22,  # env default: 2**20
            gpu_max_rigid_patch_count=2**23,  # env default: 2**23
            gpu_collision_stack_size=2**26,  # env default: 2**26
            enable_ccd=False,  # env default: False
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rerender_on_reset: bool = True

    def __post_init__(self):
        """Post initialization."""
        # object configuration=
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=TARGET_OBJECT_CFG["usd_path"],
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.02, 0.0, 0.565), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


def pose_estimator_post_init(env_cfg: InHandObjectEnvCfg):

    env_cfg.scene.base: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/base",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    env_cfg.scene.gsplat_camera = GaussianSplatCameraCfg(
        prim_path="{ENV_REGEX_NS}/base",
        update_period=0,
        offset=GaussianSplatCameraCfg.OffsetCfg(
            # Obtained from camera-robot extrinsic calibration
            pos=(-0.14602209, 0.03251047, 0.5 + (0.87840962 - 0.5)),
            rot=(-0.17201962, 0.68861653, -0.67576702, 0.19888554), convention="ros"
        ),
        camera_model=GaussianSplatCameraCfg.PinholeCfg(
            # Simulating off-center cropping with aperture settings
            horizontal_aperture=300 * 24.0 / 615.0,
            horizontal_aperture_offset=-(220.0 + 150.0 - 320.0) / 615.0,
            vertical_aperture_offset=-(180.0 + 150.0 - 240.0) / 615.0,
            focal_length=24.0,
            width=120,
            height=120,
        ),
        data_types=["rgb", "distance_to_image_plane"],
        splat_model_path=TARGET_OBJECT_CFG["splat_path"],
        clipping_range=(0.1, 20.0),
        augmentations=DefaultAugmentations,  # DefaultAugmentations
    )

    env_cfg.observations.pose: ObservationsCfg.Pose = ObservationsCfg.Pose()


def pose_estimator_unaugmented_post_init(env_cfg: InHandObjectEnvCfg):
    env_cfg.scene.gsplat_camera.augmentations = []
    env_cfg.observations.pose.image_features.params["pose_estimator_cfg"].augmentation_names = FULL_AUGS


def pose_estimator_eval_post_init(env_cfg: InHandObjectEnvCfg):
    """Post initialization for evaluation configuration."""
    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.14602209, 0.03251047, 0.5 + (0.87840962 - 0.5)),
            rot=(-0.17201962, 0.68861653, -0.67576702, 0.19888554), convention="ros"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            horizontal_aperture=640.0 * 24.0 / 615.0,
            focal_length=24.0,
            focus_distance=400.0,
            clipping_range=(0.1, 20.0)
        ),
        data_types=["rgb", "distance_to_image_plane"],
        width=640,
        height=480,
    )
    env_cfg.observations.gaussian_features = copy.deepcopy(env_cfg.observations.pose)
    env_cfg.observations.pose.image_features.func = mdp.image_cropped_to_pose
    env_cfg.observations.pose.image_features.params["camera_cfg"] = SceneEntityCfg('tiled_camera')
    env_cfg.observations.pose.image_features.params["img_size"] = (
        env_cfg.scene.gsplat_camera.camera_model.width, env_cfg.scene.gsplat_camera.camera_model.height
    )  # (width, height)
    env_cfg.observations.pose.image_features.params["crop_pos"] = (220, 180)  # (x, y)
    env_cfg.observations.pose.image_features.params["crop_size"] = (300, 300)  # (width, height)
    # env_cfg.scene.base = None
    # env_cfg.scene.gsplat_camera = None
    # env_cfg.observations.gaussian_features = None
    env_cfg.observations.gaussian_features.image_checker = ObsTerm(
        func=mdp.check_images_diff,
        params={
            "splat_term": "gaussian_features",
            "camera_term": "pose",
        }
    )


def pose_estimator_camera_post_init(env_cfg: InHandObjectEnvCfg):
    """Post initialization for camera configuration."""
    env_cfg.scene.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.14602209, 0.03251047, 0.5 + (0.87840962 - 0.5)),
            rot=(-0.17201962, 0.68861653, -0.67576702, 0.19888554), convention="ros"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            horizontal_aperture=640.0 * 24.0 / 615.0,
            focal_length=24.0,
            focus_distance=400.0,
            clipping_range=(0.1, 20.0)
        ),
        data_types=["rgb", "distance_to_image_plane"],
        width=int(640 * 120.0 / 300.0),
        height=int(480 * 120.0 / 300.0),
    )
    env_cfg.observations.pose.image_features.func = mdp.image_cropped_to_pose
    env_cfg.observations.pose.image_features.params["camera_cfg"] = SceneEntityCfg('tiled_camera')
    env_cfg.observations.pose.image_features.params["img_size"] = (
        env_cfg.scene.gsplat_camera.camera_model.width, env_cfg.scene.gsplat_camera.camera_model.height
    )  # (width, height)
    env_cfg.observations.pose.image_features.params["crop_pos"] = (
        int(220 * 120.0 / 300.0), int(180 * 120.0 / 300.0)
    )  # (x, y)
    env_cfg.observations.pose.image_features.params["crop_size"] = (120, 120)  # (width, height)
    env_cfg.scene.base = None
    env_cfg.scene.gsplat_camera = None
    env_cfg.observations.pose.image_features.params["pose_estimator_cfg"].augmentation_names = FULL_AUGS
