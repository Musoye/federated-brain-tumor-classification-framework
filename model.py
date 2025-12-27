import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from torchvision import models
import torch.nn as nn

CONFIG = {
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 8,
    "NUM_CLASSES": 2,
    "LR": 1e-3,
    "EPOCHS_LOCAL": 3,
    "ROUNDS": 3,
    "CLIENTS_PATH": "data/",
    "MODEL_NAME": "custom",   # custom, resnet18, mobilenet_v2
    "saved_path": "global_model.pth"
}


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.depth = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.point = nn.Conv2d(inp, oup, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.point(self.depth(x))))

class CustomCNN(nn.Module):
    def __init__(self, num_classes=CONFIG["NUM_CLASSES"]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32,64),
            DepthwiseSeparableConv(64,128,stride=2),
            DepthwiseSeparableConv(128,256,stride=2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

def get_model(name):
    name = name.lower()

    if name == "custom":
        return CustomCNN()

    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, CONFIG["NUM_CLASSES"])
        return m
    
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, CONFIG["NUM_CLASSES"])
        return m

    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, CONFIG["NUM_CLASSES"])
        return m

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, CONFIG["NUM_CLASSES"])
        return m

    else:
        raise ValueError(f"Unknown model name: {name}")

