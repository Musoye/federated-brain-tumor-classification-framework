import os
import torch
import flwr as fl
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np



def get_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model, params):
    sd = OrderedDict({k: torch.tensor(v) for k,v in zip(model.state_dict().keys(), params)})
    model.load_state_dict(sd)


class FLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        train_local(self.model, self.trainloader, self.valloader)
        return get_params(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        acc = evaluate(self.model, self.valloader)
        return 0.0, len(self.valloader.dataset), {"accuracy": acc}


def load_all_clients(data_root="/content/data"):
    client_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root)])
    clients = []

    for d in client_dirs:
        train_loader, val_loader = make_dataloaders(d)
        model = get_model(CONFIG["MODEL_NAME"])
        clients.append((model, train_loader, val_loader))

    return clients

