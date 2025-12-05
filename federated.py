import os
import copy
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 8,
    "NUM_CLASSES": 2,
    "LR": 1e-3,
    "EPOCHS_LOCAL": 2,
    "ROUNDS": 3,
    "CLIENTS_PATH": "data/",
    "MODEL_NAME": "custom",   # custom, resnet50, vgg16, mobilenet_v2, efficientnet_b0
}

def get_params(model: nn.Module) -> List[np.ndarray]:
    """Return model parameters as list of numpy arrays (matching state_dict order)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    """Set model parameters from list of numpy arrays (matching state_dict order)."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(params):
        raise ValueError("Parameter length mismatch.")
    sd = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, params)})
    model.load_state_dict(sd)

class FLClient:
    def __init__(self, model_fn, train_loader: DataLoader, val_loader: DataLoader, device: str = DEVICE):
        """
        model_fn: callable returning a fresh model instance (e.g. lambda: get_model(CONFIG['MODEL_NAME']))
        """
        self.device = device
        self._model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # keep a local model instance (copied when needed)
        self.model = self._model_fn().to(self.device)

    def num_samples(self) -> int:
        return len(self.train_loader.dataset)

    def fit(self, global_params: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
        """
        Loads global params into local model, runs train_local (your function),
        and returns updated params + number of samples.
        """
        # ensure we have a fresh model instance then load params
        self.model = self._model_fn().to(self.device)
        set_params(self.model, global_params)

        # train_local is your function that uses CONFIG and returns the trained model
        trained_model = train_local(self.model, self.train_loader, self.val_loader)

        # ensure on cpu for serialization
        trained_model = trained_model.to("cpu")
        updated_params = get_params(trained_model)
        return updated_params, self.num_samples()

    def evaluate(self, global_params: List[np.ndarray]) -> dict:
        """
        Evaluate the provided params on this client's validation loader.
        Returns the metrics dict from your evaluate() function.
        """
        self.model = self._model_fn().to(self.device)
        set_params(self.model, global_params)
        metrics = evaluate(self.model, self.val_loader, self.device)
        return metrics

def fed_avg(client_params: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
    """Weighted average of client parameter lists (FedAvg)."""
    if not client_params:
        raise ValueError("No client parameters provided to fed_avg.")
    n_tensors = len(client_params[0])
    total = float(sum(client_sizes))
    averaged = []
    for t in range(n_tensors):
        acc = np.zeros_like(client_params[0][t], dtype=np.float64)
        for params, size in zip(client_params, client_sizes):
            acc += params[t].astype(np.float64) * (size / total)
        # cast back to original dtype
        averaged.append(acc.astype(client_params[0][t].dtype))
    return averaged

def load_all_clients(data_root: str = None) -> List[FLClient]:
    """
    Expects each client to be a subdirectory of data_root containing an ImageFolder dataset.
    Returns list of FLClient objects.
    """
    data_root = data_root or CONFIG.get("CLIENTS_PATH", "data/")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Clients path not found: {data_root}")

    client_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d))])
    clients = []
    for d in client_dirs:
        train_loader, val_loader = make_dataloaders(d, batch=CONFIG["BATCH_SIZE"])
        # model factory (fresh model each time)
        model_fn = lambda: get_model(CONFIG["MODEL_NAME"])
        clients.append(FLClient(model_fn, train_loader, val_loader))
    if not clients:
        raise ValueError(f"No client subfolders found in {data_root}. Each client should be a subfolder.")
    return clients

def run_fedavg(clients: List[FLClient],
               global_model: nn.Module,
               rounds: int = None,
               clients_per_round: int = None,
               eval_every: int = 1,
               eval_clients: List[int] = None,
               verbose: bool = True) -> nn.Module:
    """
    Runs synchronous FedAvg training and returns the final global model.
    - clients: list of FLClient
    - global_model: initial global model instance
    - rounds: number of federated rounds (defaults to CONFIG["ROUNDS"])
    - clients_per_round: how many clients sampled per round (default = all)
    - eval_every: evaluate every N rounds
    - eval_clients: list of client indices to evaluate the global model on (default = all)
    """
    rounds = rounds or CONFIG.get("ROUNDS", 3)
    n_clients = len(clients)
    if clients_per_round is None or clients_per_round > n_clients:
        clients_per_round = n_clients

    # Prepare global model and params
    global_model = copy.deepcopy(global_model)
    global_model.to("cpu")  # keep global model on cpu for parameter manipulation
    global_params = get_params(global_model)

    for r in range(1, rounds + 1):
        if verbose:
            print(f"\n===== Federated Round {r}/{rounds} =====")

        # sample clients (random without replacement)
        selected = np.random.choice(n_clients, clients_per_round, replace=False).tolist()
        if verbose:
            print(f"Selected clients: {selected}")

        collected_params = []
        collected_sizes = []

        # sequential local training (calls your train_local)
        for cid in selected:
            client = clients[cid]
            if verbose:
                print(f"-> Training client {cid} (samples: {client.num_samples()})")
            updated_params, n_samples = client.fit(global_params)
            collected_params.append(updated_params)
            collected_sizes.append(n_samples)

        # aggregate
        new_global_params = fed_avg(collected_params, collected_sizes)

        # update global model
        set_params(global_model, new_global_params)
        global_params = new_global_params  # use for next round

        # evaluation
        if eval_every and (r % eval_every == 0):
            eval_idx = eval_clients if eval_clients is not None else list(range(n_clients))
            metrics_list = []
            for cid in eval_idx:
                m = clients[cid].evaluate(global_params)
                metrics_list.append(m)
            # Build averaged metrics (we'll show accuracy mean; other metrics can be aggregated similarly)
            accuracies = [m.get("accuracy", 0.0) for m in metrics_list]
            mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
            if verbose:
                print(f"Round {r} evaluation - mean accuracy across eval clients: {mean_acc:.4f}")

    # return final global model (on CPU)
    return global_model

# ------------------------
# Single entrypoint to start everything
# ------------------------
def start_federated(data_root: str = None,
                    rounds: int = None,
                    clients_per_round: int = None,
                    eval_every: int = 1,
                    save_path: str = None,
                    verbose: bool = True) -> str:
    """
    High-level function that:
    - loads clients from data_root (or CONFIG["CLIENTS_PATH"])
    - instantiates global model via get_model(CONFIG['MODEL_NAME'])
    - runs FedAvg for `rounds` rounds (or CONFIG["ROUNDS"])
    - optionally saves the final global model to save_path
    Returns path to saved model if save_path provided, else returns None.
    """
    data_root = data_root or CONFIG.get("CLIENTS_PATH", "data/")
    rounds = rounds or CONFIG.get("ROUNDS", 3)

    if verbose:
        print(f"Loading clients from: {data_root}")
    clients = load_all_clients(data_root)

    if verbose:
        print(f"Instantiating global model: {CONFIG['MODEL_NAME']}")
    global_model = get_model(CONFIG["MODEL_NAME"])

    trained_global = run_fedavg(clients,
                                global_model,
                                rounds=rounds,
                                clients_per_round=clients_per_round,
                                eval_every=eval_every,
                                verbose=verbose)

    if save_path:
        torch.save(trained_global.state_dict(), save_path)
        if verbose:
            print(f"Saved final global model to: {save_path}")
        return save_path

    return None

# ------------------------
# Example usage (uncomment to run as script)
# ------------------------

