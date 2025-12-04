import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    print("✔ Pruning applied")
    return model

def apply_quantization(model):
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    print("✔ Quantization applied")
    return model
