import os
import torch
import flwr as fl
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 8,
    "NUM_CLASSES": 2,
    "LR": 1e-3,
    "EPOCHS_LOCAL": 2,
    "ROUNDS": 3,
    "CLIENTS_PATH": "data/",
    "MODEL_NAME": "custom",   # custom, resnet18, mobilenet_v2
}
