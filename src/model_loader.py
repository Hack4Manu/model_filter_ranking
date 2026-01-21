import torch
import sys

def load_deeplab_model(path, device=None):
    """
    === Loads a DeepLabV3 model from a .pth file and sets it to eval mode. ===
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Loading Model from {path} ---")
    
    # 1. Direct Load (Matches your snippet)
    try:
        model = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        sys.exit(1)

    # 2. Force Eval Mode (CRITICAL FIX)
    # This was the missing key. Without this, IoU drops to 0.22.
    model.to(device)
    model.eval()
    print("[SUCCESS] Model loaded and set to EVAL mode.")

    return model

def register_hooks(model, target_substrings=None):
    # (This part remains the same as it is just helper logic)
    hooks = []
    sys.modules[__name__].activations = {} 

    def get_activation(name):
        def hook(model, input, output):
            sys.modules[__name__].activations[name] = output.detach()
        return hook

    count = 0
    for name, module in model.named_modules():
        if target_substrings:
            if not any(sub in name for sub in target_substrings):
                continue
        h = module.register_forward_hook(get_activation(name))
        hooks.append(h)
        count += 1
        
    print(f"[SUCCESS] Attached hooks to {count} layers.")
    return hooks

def clear_activations():
    sys.modules[__name__].activations = {}