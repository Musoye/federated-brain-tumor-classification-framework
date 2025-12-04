from PIL import Image
import torch.nn.functional as F
import torch

def predict_image(model, img_path, device=DEVICE):
    # Use your REAL transform pipeline (no augmentation)
    transform = get_transforms(train=False)

    # Load image
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # Evaluate
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    # Map probabilities to class names (no/yes)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    result = {
        idx_to_class[0]: float(probs[0]),
        idx_to_class[1]: float(probs[1]),
    }
    return result
