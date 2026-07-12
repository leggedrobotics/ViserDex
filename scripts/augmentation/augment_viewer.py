# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import viser
from viser import transforms as tf

import numpy as np
from plyfile import PlyData, PlyElement
import os
import torch
import time
import augment_splats
from augment_splats import generate_default_augmentations
from viserdex.utils.load_utils import load_splat_file
from pathlib import Path
import PIL.Image as Image
import copy
from viserdex.assets import VISERDEX_ASSETS_DIR
# load splat from ply


def convert_splat_for_viz(splat):
    centers = splat["means3d"].contiguous().cpu().numpy()
    SH_C0 = 0.28209479177387814
    rgbs = (0.5 + SH_C0 * np.squeeze(splat["sh0"])).cpu().numpy()
    scales = np.exp(splat["scales"].cpu().numpy())
    opacities = 1.0/ (1.0 + np.exp(-splat["opacities"].unsqueeze(1).cpu().numpy()))
    Rs = tf.SO3(splat["quats"].cpu().numpy()).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    
    print(f"Converted splat for viz with {centers.shape[0]} gaussians.")
    
    return {
        "centers": centers,
        "rgbs": rgbs,
        "opacities": opacities,
        "covariances": covariances,
    }
    




if __name__ == "__main__":
    obj = 'Duck'
    ply_path = f'{VISERDEX_ASSETS_DIR}/splats/{obj}/icosphere/exports/splat.ply'
    ply_path = Path(ply_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    splats = load_splat_file(ply_path, device)
    
    server = viser.ViserServer()
    
    current = 'base'
    
    
    reset_button = server.gui.add_button("Reset")
    render_button = server.gui.add_button("Render All Augmentations")
    
    with server.gui.add_folder("Augmentation Settings"):
        base_button = server.gui.add_button("Base Splat")
        
        with server.gui.add_folder("Random Noise"):
            random_button = server.gui.add_button("Random Noise")
            random_probability = server.gui.add_slider("Random Noise Probability", min=0.0, max=1.0, step=0.01, initial_value=1.0)
            random_max_range = server.gui.add_slider("Random Noise Max Range", min=0.1, max=1.0, step=0.01, initial_value=0.2)
            
        with server.gui.add_folder("Spatial Noise"):
            spatial_button = server.gui.add_button("Spatial Noise")
            spatial_probability = server.gui.add_slider("Spatial Noise Probability", min=0.0, max=1.0, step=0.01, initial_value=0.8)
            spatial_max_range = server.gui.add_slider("Spatial Noise Max Range", min=0.1, max=1.0, step=0.01, initial_value=0.9)
            
        with server.gui.add_folder("Color Noise"):
            color_button = server.gui.add_button("Color Noise")
            color_probability = server.gui.add_slider("Color Noise Probability", min=0.0, max=1.0, step=0.01, initial_value=0.2)
            color_max_range = server.gui.add_slider("Color Noise Max Range", min=0.1, max=10.0, step=0.01, initial_value=0.5)
        
        with server.gui.add_folder("Global Noise"):
            global_button = server.gui.add_button("Global Noise")
            global_probability = server.gui.add_slider("Global Noise Probability", min=0.0, max=1.0, step=0.01, initial_value=0.2) 
            global_shift_probability = server.gui.add_slider("Global Shift Probability", min=0.0, max=1.0, step=0.01, initial_value=0.8)
            global_max_range = server.gui.add_slider("Global Noise Max Range", min=0.1, max=1.0, step=0.01, initial_value=0.6)
            
        
    
    gs_handle = server.scene.add_gaussian_splats(
        f"/base/gaussian_splats",
        **convert_splat_for_viz(splats),
    )
    
    splats_handles = {'base': gs_handle}
    
    cluster_path = ply_path.parent / 'clusters_auto' / 'indices'
    
    
    defaultAugmentations = generate_default_augmentations(cluster_path)
    
    random_augs_cfg = defaultAugmentations[:4]
    spatial_augs_cfg = defaultAugmentations[4:6]
    color_augs_cfg = defaultAugmentations[6:9]
    global_augs_cfg = defaultAugmentations[9:]    

    
    @base_button.on_click
    def on_base_button_click(_):
        global current
        current = 'base'
        splats_handles['base'].visible = True
        for key in splats_handles:
            splats_handles[key].visible = 'base' in key
            
    @random_button.on_click
    def on_random_button_click(_):
        global current
        current = 'random'
        splats_handles['random'].visible = True
        for key in splats_handles:
            splats_handles[key].visible = 'random' in key
        setup_and_run_random()
            
    @spatial_button.on_click
    def on_spatial_button_click(_):
        global current
        current = 'spatial'
        splats_handles['spatial'].visible = True
        for key in splats_handles:
            splats_handles[key].visible = 'spatial' in key
        setup_and_run_spatial()
            
    @color_button.on_click
    def on_color_button_click(_):
        global current
        current = 'color'
        splats_handles['color'].visible = True
        for key in splats_handles:
            splats_handles[key].visible = 'color' in key
        setup_and_run_color()
            
    @global_button.on_click
    def on_global_button_click(_):
        global current
        current = 'global'
        splats_handles['global'].visible = True
        for key in splats_handles:
            splats_handles[key].visible = 'global' in key
        setup_and_run_global()
            
    @render_button.on_click
    def on_render_button_click(event):
        global current
        client = event.client
        assert client is not None
        aspect = client.camera.aspect
        height = 1000
        width = int(height * aspect)
        print(f"Rendering all augmentations at {aspect}...")
        images = {}
        for selected in splats_handles:
            for key in splats_handles:         
                splats_handles[key].visible = selected in key
            time.sleep(0.1)  # wait for the scene to update
            images[f"{obj}_{selected}"] = client.get_render(width=width, height=height)

        for name, img in images.items():
            img_path = ply_path.parent / 'renders' / f"{name}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_pil = Image.fromarray(img)
            img_pil.save(img_path)
        on_base_button_click(None)
            
    
    def setup_and_run_random():
        augmented_splats = copy.deepcopy(splats)
        # for key in augmented_splats:
        #     print(f"Key: {key}, Shape: {augmented_splats[key].shape}")
        print("Running random augmentations...")
        random_augs = []
        for cfg in random_augs_cfg:
            cfg.probability = random_probability.value
            if cfg.noise_cfg.operation == 'scale':
                cfg.noise_cfg.n_max = 1 + random_max_range.value
                cfg.noise_cfg.n_min = 1 - random_max_range.value
            else:
                cfg.noise_cfg.n_max = random_max_range.value
                cfg.noise_cfg.n_min = -random_max_range.value
            print(f"Random Augmentation: {cfg.__class__.__name__}, Probability: {cfg.probability}, Noise Range: [{cfg.noise_cfg.n_min}, {cfg.noise_cfg.n_max}]")
            
            for key in augmented_splats:
                augmented_splats[key] = augmented_splats[key].squeeze(0)
            random_augs.append(cfg.func(cfg, augmented_splats, device))
            
        for aug in random_augs:
            aug(augmented_splats, num_samples=1)
            
        if 'random' in splats_handles:
            splats_handles['random'].remove()
        splats_handles['random'] = server.scene.add_gaussian_splats(
            f"/random_gaussian_splats",
            visible=current=='random',
            **convert_splat_for_viz(augmented_splats),
        )
                    
        
        return augmented_splats.copy()
    
    def setup_and_run_spatial():
        augmented_splats = copy.deepcopy(splats)
        # for key in augmented_splats:
        #     print(f"Key: {key}, Shape: {augmented_splats[key].shape}")
        print("Running spatial augmentations...")
        
        spatial_augs = []
        for cfg in spatial_augs_cfg:
            for key in augmented_splats:
                augmented_splats[key] = augmented_splats[key].squeeze(0)
            cfg.probability = 1.0
            cfg.cluster_fraction = spatial_probability.value
            
            if cfg.noise_cfg.operation == 'scale':
                cfg.noise_cfg.n_max = 1 + spatial_max_range.value
                cfg.noise_cfg.n_min = 1 - spatial_max_range.value
            else:
                cfg.noise_cfg.n_max = spatial_max_range.value
                cfg.noise_cfg.n_min = -spatial_max_range.value
            print(f"Spatial Augmentation: {cfg.__class__.__name__}, Probability: {cfg.probability}, Noise Range: [{cfg.noise_cfg.n_min}, {cfg.noise_cfg.n_max}]")
            
            spatial_augs.append(cfg.func(cfg, augmented_splats, device))
        
        for aug in spatial_augs:
            aug(augmented_splats, num_samples=1)
            
        
        if 'spatial' in splats_handles:
            splats_handles['spatial'].remove()
        splats_handles['spatial'] = server.scene.add_gaussian_splats(
            f"/spatial_gaussian_splats",
            visible=current=='spatial',
            
            **convert_splat_for_viz(augmented_splats),
        )
            
        
        return augmented_splats.copy()
        
    def setup_and_run_color():
        augmented_splats = copy.deepcopy(splats)
        # for key in augmented_splats:
        #     print(f"Key: {key}, Shape: {augmented_splats[key].shape}")
        print("Running color augmentations...")
        color_augs = []
        for cfg in color_augs_cfg:
            for key in augmented_splats:
                augmented_splats[key] = augmented_splats[key].squeeze(0)
            cfg.probability = 1.0
            cfg.cluster_fraction = color_probability.value
            
            if cfg.noise_cfg.operation == 'scale':
                cfg.noise_cfg.n_max = 1 + color_max_range.value
                cfg.noise_cfg.n_min = 1 - color_max_range.value
            else:
                cfg.noise_cfg.n_max = color_max_range.value
                cfg.noise_cfg.n_min = -color_max_range.value
            print(f"Color Augmentation: {cfg.__class__.__name__}, Probability: {cfg.probability}, Noise Range: [{cfg.noise_cfg.n_min}, {cfg.noise_cfg.n_max}]")
            
            color_augs.append(cfg.func(cfg, augmented_splats, device))
        
        for aug in color_augs:
            aug(augmented_splats, num_samples=1)
            
        if 'color' in splats_handles:
            splats_handles['color'].remove()
        splats_handles['color'] = server.scene.add_gaussian_splats(
            f"/color_gaussian_splats",
            visible=current=='color',
            
            **convert_splat_for_viz(augmented_splats),
        )
            
        
        return augmented_splats.copy()
        
        
    def setup_and_run_global():
        augmented_splats = copy.deepcopy(splats)
        
        global_augs = []
        for cfg in global_augs_cfg:
            for key in augmented_splats:
                augmented_splats[key] = augmented_splats[key].squeeze(0)
            if 'Shift' in cfg.__class__.__name__:
                cfg.probability = global_shift_probability.value
            else:
                cfg.cluster_fraction = global_probability.value
            print(f"Global Augmentation: {cfg.__class__.__name__}, Probability: {cfg.probability}, Noise Range: [{cfg.noise_cfg.n_min}, {cfg.noise_cfg.n_max}]")
                
            if cfg.noise_cfg.operation == 'scale':
                cfg.noise_cfg.n_max = 1 + global_max_range.value
                cfg.noise_cfg.n_min = 1 - global_max_range.value
            else:
                cfg.noise_cfg.n_max = global_max_range.value
                cfg.noise_cfg.n_min = -global_max_range.value
                
            global_augs.append(cfg.func(cfg, augmented_splats, device))
        
        for aug in global_augs:
            aug(augmented_splats, num_samples=1)
            
            
        if 'global' in splats_handles:
            splats_handles['global'].remove()
        splats_handles['global'] = server.scene.add_gaussian_splats(
            f"/global_gaussian_splats",
            visible=current=='global',
            
            **convert_splat_for_viz(augmented_splats),
        )
            
        
        return augmented_splats
    
    
    print("Initializing augmentations...")
    setup_and_run_random()
    setup_and_run_spatial()
    setup_and_run_color()
    setup_and_run_global()
    print("Augmentations applied.")
    
    random_probability.on_update(lambda _: setup_and_run_random())
    random_max_range.on_update(lambda _: setup_and_run_random())
    spatial_probability.on_update(lambda _: setup_and_run_spatial())
    spatial_max_range.on_update(lambda _: setup_and_run_spatial())
    color_probability.on_update(lambda _: setup_and_run_color())
    color_max_range.on_update(lambda _: setup_and_run_color())
    global_probability.on_update(lambda _: setup_and_run_global())
    global_shift_probability.on_update(lambda _: setup_and_run_global())    
    global_max_range.on_update(lambda _: setup_and_run_global())
        
    
    
    while True:
        time.sleep(1)