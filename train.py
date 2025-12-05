import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from preparation import CONFIG

def train_local(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])

    model.to(DEVICE)

    for ep in range(CONFIG["EPOCHS_LOCAL"]):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{CONFIG['EPOCHS_LOCAL']}")

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / len(train_loader)

        # ---- Evaluation ----
        metrics = evaluate(model, val_loader, DEVICE)

        print("\n========== EVALUATION ==========")
        print(f"Model: {CONFIG['MODEL_NAME']}")
        print(f"Epoch: {ep + 1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision per class: {metrics['precision_per_class']}")
        print(f"Recall per class: {metrics['recall_per_class']}")
        print(f"F1 per class: {metrics['f1_per_class']}")
        print("Confusion Matrix:")
        for row in metrics['confusion_matrix']:
            print(row)
        print("================================\n")

    return model

