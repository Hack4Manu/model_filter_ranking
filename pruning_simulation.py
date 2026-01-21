import torch
import numpy as np
import h5py
import os
import argparse
import pandas as pd
from tqdm import tqdm
import src.utils as utils
import src.model_loader as loader

# CONFIG
BATCH_SIZE = 8
IMG_SIZE = (512, 512)

def get_ranked_indices(h5_path, metric_name):
    """
    Returns a list of (LayerName, ChannelIndex) sorted from LEAST important to MOST important.
    We prune from the start of the list (Low scores).
    """
    data = []
    with h5py.File(h5_path, 'r') as f:
        for layer in f.keys():
            if layer in ['iou_scores', 'image_names']: continue
            
            # Handle key variations if needed
            key = f"ranking_{metric_name.lower().replace(' ', '_')}"
            if key not in f[layer]: continue
                
            scores = f[layer][key][:]
            
            # Normalize scores per layer (Global Ranking Strategy)
            # This ensures we don't just prune entire layers that happen to have low scale
            if scores.std() > 0:
                scores = (scores - scores.mean()) / scores.std()
                
            for ch, val in enumerate(scores):
                data.append({'layer': layer, 'ch': ch, 'score': val})
                
    df = pd.DataFrame(data)
    # Sort ascending: Lowest score (least important) first
    df = df.sort_values('score', ascending=True)
    return list(zip(df['layer'], df['ch']))

def evaluate_model(model, img_paths, mask_paths, device):
    """Runs validatoin on a subset of data"""
    model.eval()
    ious = []
    
    # We create a simple batch loop
    for i in range(0, len(img_paths), BATCH_SIZE):
        b_imgs = img_paths[i : i+BATCH_SIZE]
        b_masks = mask_paths[i : i+BATCH_SIZE]
        
        imgs = utils.load_image_batch(b_imgs, size=IMG_SIZE).to(device)
        masks = utils.load_mask_batch(b_masks, size=IMG_SIZE).to(device)
        
        if len(imgs) == 0: continue
            
        with torch.no_grad():
            out = model(imgs)['out']
            batch_ious = utils.compute_batch_iou(out, masks)
            ious.extend(batch_ious)
            
    return np.mean(ious)

def apply_masks(model, prune_list, verbose=False):
    """
    Zeros out the channels in prune_list.
    NOTE: We use forward hooks to mask activations dynamically!
    This is cleaner than modifying weights permanently.
    """
    # Clear existing hooks first
    if hasattr(model, 'pruning_hooks'):
        for h in model.pruning_hooks: h.remove()
    model.pruning_hooks = []
    
    # Group by layer for efficiency
    masks_by_layer = {}
    for layer, ch in prune_list:
        if layer not in masks_by_layer: masks_by_layer[layer] = []
        masks_by_layer[layer].append(ch)
        
    # Define Hook
    def get_mask_hook(indices_to_zero):
        # Create a boolean mask tensor (1=Keep, 0=Prune)
        # We construct it lazily inside the hook to match device/shape
        def hook(module, input, output):
            # output shape: (B, C, H, W)
            if not hasattr(module, 'active_mask'):
                # Initialize mask on first run
                C = output.shape[1]
                m = torch.ones(C, device=output.device)
                m[indices_to_zero] = 0.0
                module.active_mask = m.view(1, C, 1, 1)
            
            return output * module.active_mask
        return hook

    # Register Hooks
    count = 0
    for name, module in model.named_modules():
        if name in masks_by_layer:
            indices = masks_by_layer[name]
            h = module.register_forward_hook(get_mask_hook(indices))
            model.pruning_hooks.append(h)
            count += len(indices)
            
    if verbose: print(f"  [Masking] Zeroed out {count} channels.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', type=str, required=True, help="Path to rankings.h5")
    parser.add_argument('--model', type=str, required=True, help="Path to model.pth")
    parser.add_argument('--limit', type=int, default=200, help="Number of test images to use (speed)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (Test Set)
    # Use the Test Set path you confirmed earlier
    TEST_IMG = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM/test/images"
    TEST_MASK = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM/test/masks"
    
    img_paths, mask_paths = utils.get_paired_paths(TEST_IMG, TEST_MASK)
    # Limit for speed
    img_paths = img_paths[:args.limit]
    mask_paths = mask_paths[:args.limit]
    
    print(f"--- Pruning Simulation (Evaluating on {len(img_paths)} Test Images) ---")

    # 2. Load Model
    model = loader.load_deeplab_model(args.model, device)
    
    # 3. Metrics to Compare
    metrics = ['pls', 'dynamic_variance', 'pca', 'centrality', 'static_variance'] 
    pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {'ratio': pruning_ratios}
    
    for metric in metrics:
        print(f"\nTesting Metric: {metric.upper()}")
        
        # Reload model to clear previous hooks/state
        # (Faster to just clear hooks, but safer to reload if unsure)
        if hasattr(model, 'pruning_hooks'):
            for h in model.pruning_hooks: h.remove()
        model.pruning_hooks = []
        # Clear cached masks
        for m in model.modules():
            if hasattr(m, 'active_mask'): del m.active_mask

        # Get Ranked List (Lowest to Highest)
        # We want to prune the LOWEST scores first
        ranked_list = get_ranked_indices(args.rankings, metric)
        total_kernels = len(ranked_list)
        
        metric_ious = []
        
        for ratio in pruning_ratios:
            n_prune = int(total_kernels * ratio)
            to_prune = ranked_list[:n_prune] # Bottom K%
            
            # Apply Mask
            apply_masks(model, to_prune)
            
            # Evaluate
            iou = evaluate_model(model, img_paths, mask_paths, device)
            print(f"  Pruned {int(ratio*100)}% -> IoU: {iou:.4f}")
            metric_ious.append(iou)
            
        results[metric] = metric_ious

    # 4. Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/pruning_results.csv", index=False)
    print("\n[DONE] Results saved to results/pruning_results.csv")

if __name__ == "__main__":
    main()