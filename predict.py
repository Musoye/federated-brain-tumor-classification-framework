from PIL import Image
import torch
import torch.nn.functional as F
import os
import torch
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
    "EPOCHS_LOCAL": 3,
    "ROUNDS": 3,
    "CLIENTS_PATH": "data/",
    "MODEL_NAME": "custom",   # custom, resnet18, mobilenet_v2
    "saved_path": "global_model.pth"
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

def load_federated_model(model, checkpoint_path, device=None):
    """
    Loads your federated-trained model from a saved .pth checkpoint.
    Handles both full-state and dict-wrapped checkpoints.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Case 1: Standard PyTorch state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        model.class_to_idx = ckpt.get("class_to_idx", None)

    # Case 2: Direct state_dict saved with torch.save(model.state_dict())
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)

    else:
        raise ValueError("Checkpoint format not recognized")

    return model.to(device)

def predict_image(model, img_path, checkpoint_path, device=None):
    """
    Loads the model checkpoint + predicts on an image.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    model = load_federated_model(model, checkpoint_path, device)

    # Use your true non-augmented transform
    transform = get_transforms(train=False)

    # Load + preprocess image
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    # --- FIX: HARDCODE THE MAPPING HERE ---
    # If the checkpoint didn't have the mapping, we force it:
    if not hasattr(model, "class_to_idx") or model.class_to_idx is None:
        model.class_to_idx = {'no': 0, 'yes': 1}
    # --------------------------------------

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    result = {
        idx_to_class[0]: float(probs[0]),
        idx_to_class[1]: float(probs[1]),
    }

    return result