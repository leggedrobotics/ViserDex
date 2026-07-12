# Copyright (c) 2026 ETH Zurich (Robotic Systems Lab)
# Author: Arjun Bhardwaj
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class PoseEstimatorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self, num_outputs: int):
        super().__init__()

        # backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self._data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return_nodes = {"layer4.1.relu": "features"}
        self._feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.train_backbone = True
        if not self.train_backbone:
            self._feature_extractor.eval()
            self._feature_extractor.requires_grad_(False)
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))


        self._linear = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self._head = nn.Sequential(
            nn.Linear(256, num_outputs),
        )
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            x = self._data_transforms(x[:, 0:3])
        if not self.train_backbone:
            with torch.no_grad():
                x = self._feature_extractor(x)["features"]
                x = self._avg_pool(x)
        else:
            x = self._feature_extractor(x)["features"]
            x = self._avg_pool(x)
        features = self._linear(x.view(x.shape[0], -1))
        out = self._head(features)
        return out

    def get_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=500, max_epochs=200, warmup_start_lr=1e-6, eta_min=2e-5
        )
        return optimizer, scheduler
