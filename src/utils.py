import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

# (Keep get_paired_paths as it works well now)
def get_paired_paths(image_dir, mask_dir):
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Missing: {image_dir} or {mask_dir}")
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    img_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
    mask_names = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts) and not f.startswith('.')])
    
    valid_pairs = []
    mask_set = set(mask_names)
    
    for img_file in img_names:
        name_stem = os.path.splitext(img_file)[0]
        if img_file in mask_set:
            valid_pairs.append((os.path.join(image_dir, img_file), os.path.join(mask_dir, img_file)))
            continue
        expected_mask_png = name_stem + ".png"
        if expected_mask_png in mask_set:
             valid_pairs.append((os.path.join(image_dir, img_file), os.path.join(mask_dir, expected_mask_png)))

    if not valid_pairs:
        print(f"[WARN] No pairs found.")
        return [], []
    img_paths, mask_paths = zip(*valid_pairs)
    return list(img_paths), list(mask_paths)

def load_image_batch(image_paths, size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensors = [transform(Image.open(p).convert('RGB')) for p in image_paths]
    return torch.stack(tensors) if tensors else torch.empty(0)

def load_mask_batch(mask_paths, size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    tensors = []
    for p in mask_paths:
        t = transform(Image.open(p).convert('L'))
        t = (t > 0).float() # Binary threshold
        tensors.append(t)
    return torch.stack(tensors) if tensors else torch.empty(0)

def compute_batch_iou(preds, masks):
    """
    Matches the logic from your provided snippet exactly.
    """
    # Force Sigmoid + Threshold (Binary Mode)
    # This is what your old code did, so we do it here.
    pred_classes = (torch.sigmoid(preds) > 0.5).float()
    
    p_flat = pred_classes.view(pred_classes.size(0), -1)
    m_flat = masks.view(masks.size(0), -1)
    
    intersection = (p_flat * m_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + m_flat.sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.cpu().numpy()