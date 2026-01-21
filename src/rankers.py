import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import scipy.stats

def get_layer_module(model, layer_name):
    """Helper to retrieve a specific layer module from string name."""
    if layer_name == "backbone.conv1": return model.backbone.conv1
    
    module = model
    try:
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    except AttributeError:
        print(f"[WARN] Could not find layer: {layer_name}")
        return None

# --- 1. STATIC METRICS (Weights Only) ---
def compute_static_variance(model, target_layers):
    """
    Calculates the variance of the weights for each kernel.
    High Variance = Strong Edges/Structure.
    """
    print("--- Computing Static Variance (Weights) ---")
    results = {}
    
    for layer in target_layers:
        module = get_layer_module(model, layer)
        if module is None: continue
        
        # Weights shape: (Out, In, H, W) -> Average over In -> (Out, H, W)
        w = module.weight.data.mean(dim=1).cpu().numpy()
        
        # Calculate Variance per kernel (flatten H,W)
        # shape: (Out_Channels,)
        variances = np.var(w.reshape(w.shape[0], -1), axis=1)
        results[layer] = variances
        
    return results

# --- 2. DYNAMIC METRICS (Activations) ---
def compute_dynamic_variance(activations_dict):
    """
    Calculates variance of GAP activations across the batch.
    High Variance = Highly Selective (discriminative) feature.
    """
    print("--- Computing Dynamic Variance (Activations) ---")
    results = {}
    
    for layer, act_tensor in activations_dict.items():
        # act_tensor shape: (Batch, Channels)
        # We want variance across the Batch dimension
        # Result shape: (Channels,)
        var_per_channel = torch.var(act_tensor, dim=0).cpu().numpy()
        results[layer] = var_per_channel
        
    return results

def compute_pca_loading(activations_dict):
    """
    Calculates how much each kernel contributes to the layer's 'Main Theme'.
    Uses Pearson correlation with PC1.
    """
    print("--- Computing PCA Loadings ---")
    results = {}
    
    for layer, act_tensor in activations_dict.items():
        X = act_tensor.cpu().numpy() # (Batch, Channels)
        
        # 1. Run PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).flatten() # (Batch,) The "Main Theme"
        
        # 2. Correlate each channel with PC1
        n_channels = X.shape[1]
        loadings = []
        for c in range(n_channels):
            channel_data = X[:, c]
            # Handle constant channels (variance=0) to avoid NaN
            if np.std(channel_data) < 1e-9:
                corr = 0.0
            else:
                corr, _ = scipy.stats.pearsonr(channel_data, pc1)
            loadings.append(abs(corr))
            
        results[layer] = np.array(loadings)
        
    return results

def compute_eigen_centrality(activations_dict):
    """
    Constructs a graph where nodes=kernels and edges=correlation.
    Calculates Eigenvector Centrality (Hubness).
    """
    print("--- Computing Eigenvector Centrality ---")
    results = {}
    
    for layer, act_tensor in activations_dict.items():
        X = act_tensor.cpu().numpy()
        
        # 1. Correlation Matrix (Similarity)
        # Add small noise to avoid division by zero
        X = X + np.random.normal(0, 1e-9, X.shape)
        corr_matrix = np.corrcoef(X, rowvar=False) # (Channels, Channels)
        
        # 2. Absolute value (we care about connection strength, neg or pos)
        adj_matrix = np.abs(np.nan_to_num(corr_matrix))
        
        # 3. Power Iteration for Eigenvector Centrality
        # (Faster/More stable than networkx for dense matrices)
        n_nodes = adj_matrix.shape[0]
        v = np.ones(n_nodes) / np.sqrt(n_nodes)
        
        for _ in range(20): # 20 iterations is usually enough for convergence
            v_next = adj_matrix @ v
            norm = np.linalg.norm(v_next)
            if norm == 0: break
            v = v_next / norm
            
        results[layer] = v
        
    return results

# --- 3. SUPERVISED METRICS (Needs IoU/Labels) ---
def compute_pls_importance(activations_dict, iou_scores):
    """
    Uses Partial Least Squares to find kernels that drive Accuracy (IoU).
    """
    print("--- Computing PLS Importance (Supervised) ---")
    results = {}
    
    # Normalize targets
    Y = np.array(iou_scores)
    Y = (Y - Y.mean()) / (Y.std() + 1e-9)
    
    for layer, act_tensor in activations_dict.items():
        X = act_tensor.cpu().numpy()
        
        # PLS Regression
        pls = PLSRegression(n_components=1)
        pls.fit(X, Y)
        
        # The weights tell us importance
        # shape: (Channels, 1) -> flatten
        importance = np.abs(pls.x_weights_.flatten())
        results[layer] = importance
        
    return results