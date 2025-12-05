from PIL import Image
import torch
import torch.nn.functional as F

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

    Args:
        model (torch.nn.Module): Model architecture instance.
        img_path (str): Path to image for prediction.
        checkpoint_path (str): Saved federated model state.
        device (str): CPU/GPU.
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

    # Map model outputs to labels
    if not hasattr(model, "class_to_idx") or model.class_to_idx is None:
        raise ValueError("model.class_to_idx not found in checkpoint. Save it when training.")

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    result = {
        idx_to_class[0]: float(probs[0]),
        idx_to_class[1]: float(probs[1]),
    }

    return result

