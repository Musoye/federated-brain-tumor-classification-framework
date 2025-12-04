import os
import torch
import flwr as fl
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np


def client_fn(cid: str):
    cid = int(cid)
    model, trainl, vall = GLOBAL_CLIENTS[cid]
    return FLClient(model, trainl, vall)

DATA_PATH = CONFIG.get("CLIENTS_PATH", "/content/data")

if not os.path.exists(DATA_PATH):
    print(f"Path {DATA_PATH} not found, falling back to default '/content/data'")
    DATA_PATH = "/content/data"

# Load client datasets
GLOBAL_CLIENTS = load_all_clients(DATA_PATH)

def client_fn(cid):
    cid = int(cid)
    model, train_loader, val_loader = GLOBAL_CLIENTS[cid]
    return FLClient(model, train_loader, val_loader)

model_global = get_model(CONFIG["MODEL_NAME"])

strategy = SaveModelStrategy(
    model=model_global,
    min_available_clients=len(GLOBAL_CLIENTS),
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(GLOBAL_CLIENTS),
    config=fl.server.ServerConfig(num_rounds=CONFIG["ROUNDS"]),
    strategy=strategy,
)
