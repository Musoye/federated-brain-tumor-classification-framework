import torch

def evaluate(model, loader, device):
    model.eval()

    # Track all predictions & true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # Merge all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # -------- METRICS --------
    # Accuracy
    accuracy = (all_preds == all_labels).float().mean().item()

    # Confusion matrix (for binary classification)
    num_classes = len(torch.unique(all_labels))
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t, p in zip(all_labels, all_preds):
        conf_matrix[t.long(), p.long()] += 1

    # Extract TP, FP, FN (works for both binary and multi-class)
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(0) - tp
    fn = conf_matrix.sum(1) - tp

    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": conf_matrix.tolist()
    }

    return metrics

