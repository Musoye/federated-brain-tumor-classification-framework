import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

CONFIG = {
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 8,
    "NUM_CLASSES": 2,
    "LR": 1e-3,
    "EPOCHS_LOCAL": 5,
    "ROUNDS": 5,
    "CLIENTS_PATH": "data/",
    "MODEL_NAME": "custom",   # custom, resnet18, mobilenet_v2
}

def get_transforms(train=True):
    t = [
        transforms.Resize((CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])),
    ]
    if train:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    t += [
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    return transforms.Compose(t)

from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def make_dataloaders(path, batch=CONFIG["BATCH_SIZE"]):
    # Load dataset normally
    ds = datasets.ImageFolder(path, transform=get_transforms(train=True))

    # --- CLASS NORMALIZATION ---
    new_class_to_idx = {}
    for cls in ds.classes:
        if cls.lower() in ["notumor", "no", "negative", "no_tumor"]:
            new_class_to_idx[cls] = 0        # no tumor
        else:
            new_class_to_idx[cls] = 1        # yes tumor

    # Print the classes
    print("Original classes:", ds.classes)
    print("New remapped classes:", new_class_to_idx)
    ds.class_to_idx = new_class_to_idx

    # --- DATASET SPLIT ---
    val_size = int(0.2 * len(ds))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    
    return train_loader, val_loader

