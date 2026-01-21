import torch
import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm
import src.utils as utils
import src.model_loader as loader
import src.rankers as rankers

# CONFIG
# You can change these defaults via command line args
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Ranking Pipeline")
    parser.add_argument('--data_dir', type=str, default="data/train/images", help="Path to images")
    parser.add_argument('--mask_dir', type=str, default="data/train/masks", help="Path to masks")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument('--save_path', type=str, default="results/rankings.h5", help="Output H5 file path")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on {device} ---")

    # 1. Setup Data
    print(f"Matching images in {args.data_dir} with masks in {args.mask_dir}...")
    img_paths, mask_paths = utils.get_paired_paths(args.data_dir, args.mask_dir)
    
    if len(img_paths) == 0:
        print("[ERROR] No data found. Exiting.")
        return

    print(f"Found {len(img_paths)} paired samples.")
    
    # 2. Load Model
    # We assume the path is fixed for your project, or you can add it as an arg
    MODEL_PATH = "/ediss_data/ediss2/xai-texture/saved_models/Deeplab/CBIS_DDSM/Feature_10/deeplab_v3_segmentation.pth"
    model = loader.load_deeplab_model(MODEL_PATH, device)

    # 3. Hook Layers
    # We focus on the ResNet Backbone layers
    target_layers = [name for name, _ in model.named_modules() if "backbone" in name and "conv2" in name]
    # Add initial conv1 just in case
    if hasattr(model.backbone, 'conv1'): target_layers.insert(0, 'backbone.conv1')
    
    print(f"--- Registering Hooks (Filter: {target_layers[:3]} ... {target_layers[-1]}) ---")
    loader.register_hooks(model, target_substrings=target_layers)

    # 4. Run Inference & Collect Data
    # We process in batches to save memory
    activations_buffer = {layer: [] for layer in loader.sys.modules['src.model_loader'].activations.keys()}
    # Note: If activations.keys() is empty now, it will fill up after first batch. 
    # Better strategy: run one batch to init, or just append dynamically.
    
    # Re-init buffer dynamically
    activations_buffer = {} 
    all_ious = []
    all_names = []

    # Manual Batching Loop (Simpler than DataLoader for this custom logic)
    for i in tqdm(range(0, len(img_paths), args.batch_size)):
        batch_img_paths = img_paths[i : i + args.batch_size]
        batch_mask_paths = mask_paths[i : i + args.batch_size]
        
        # Load Batch
        imgs = utils.load_image_batch(batch_img_paths, size=(512, 512)).to(device)
        masks = utils.load_mask_batch(batch_mask_paths, size=(512, 512)).to(device)
        
        if len(imgs) == 0: continue

        # Forward Pass
        with torch.no_grad():
            out = model(imgs)['out']
            
            # --- CRITICAL: Calculate IoU ---
            batch_ious = utils.compute_batch_iou(out, masks)
            all_ious.extend(batch_ious)
            all_names.extend([os.path.basename(p) for p in batch_img_paths])

            # Collect Activations (GAP)
            # Access the global dictionary in model_loader
            current_acts = loader.sys.modules['src.model_loader'].activations
            
            for layer_name, tensor in current_acts.items():
                if layer_name not in activations_buffer:
                    activations_buffer[layer_name] = []
                
                # Global Average Pooling (B, C, H, W) -> (B, C)
                gap = torch.mean(tensor, dim=(2, 3)).cpu().numpy()
                activations_buffer[layer_name].append(gap)
                
            loader.clear_activations()

    # 5. Aggregate Results
    print("Aggregating results...")
    final_activations = {}
    for layer, list_of_arrays in activations_buffer.items():
        final_activations[layer] = np.concatenate(list_of_arrays, axis=0) # (N_images, Channels)

    all_ious = np.array(all_ious)
    print(f"Mean IoU of Dataset: {np.mean(all_ious):.4f}")

    # 6. Compute Rankings
    results = {}
    
    print("--- Computing Static Variance (Weights) ---")
    static_vars = rankers.compute_static_variance(model, final_activations.keys())
    
    print("--- Computing Dynamic Variance (Activations) ---")
    # Convert back to tensor for rankers (if needed) or adapt rankers to numpy
    # rankers.py expects Dictionary of Tensors usually
    tensor_acts = {k: torch.tensor(v) for k, v in final_activations.items()}
    
    dyn_vars = rankers.compute_dynamic_variance(tensor_acts)
    
    print("--- Computing PCA Loadings ---")
    pca_loads = rankers.compute_pca_loading(tensor_acts)
    
    print("--- Computing Eigenvector Centrality ---")
    centralities = rankers.compute_eigen_centrality(tensor_acts)
    
    print("--- Computing PLS Importance (Supervised) ---")
    pls_scores = rankers.compute_pls_importance(tensor_acts, all_ious)

    # 7. Save to H5
    print(f"--- Saving results to {args.save_path} ---")
    
    # Helper: Handles both Tensors (needs .numpy()) and Arrays (already numpy)
    def safe_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    with h5py.File(args.save_path, 'w') as f:
        # --- CRITICAL: Save Metadata (IoU & Names) ---
        f.create_dataset('iou_scores', data=all_ious)
        f.create_dataset('image_names', data=np.array(all_names, dtype='S'))

        # Save Per-Layer Metrics
        for layer in final_activations.keys():
            grp = f.create_group(layer)
            
            # Save Metric Scores using safe_numpy
            if layer in static_vars:
                grp.create_dataset('ranking_static_variance', data=safe_numpy(static_vars[layer]))
            
            if layer in dyn_vars:
                grp.create_dataset('ranking_dynamic_variance', data=safe_numpy(dyn_vars[layer]))
                
            if layer in pca_loads:
                grp.create_dataset('ranking_pca', data=safe_numpy(pca_loads[layer]))
                
            if layer in centralities:
                grp.create_dataset('ranking_centrality', data=safe_numpy(centralities[layer]))
                
            if layer in pls_scores:
                grp.create_dataset('ranking_pls', data=safe_numpy(pls_scores[layer]))

            # --- CRITICAL: Save Raw Vectors for Stability Notebook ---
            grp.create_dataset('raw_gap', data=final_activations[layer])

    print("[SUCCESS] Saved.")

if __name__ == "__main__":
    main()
