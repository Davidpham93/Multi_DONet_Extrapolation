import torch
import torch.nn as nn
from build_model_2 import MultiBranchDeepONet
import numpy as np
# ============================================================

# ============================================================
# Loader utility
# ============================================================

def load_model(checkpoint_path: str, device: torch.device | None = None):
    """
    Load a trained MultiBranchDeepONet from a .pt checkpoint file.

    The checkpoint must be the one saved by the training script, i.e. a dict with:
        - "model_state_dict"
        - "model_config"

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file (e.g., "mb_deeponet_checkpoints/multibranch_deeponet_best.pt").
    device : torch.device or None
        Device to map the model to. If None, uses "cuda" if available, else "cpu".

    Returns
    -------
    model : MultiBranchDeepONet
        Loaded model in eval() mode, moved to the chosen device.
    model_config : dict
        Configuration dict used to build the model (b1_dim, b2_dim, b3_dim, trunk_in_dim, ...).
    device : torch.device
        The device where the model is located.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt["model_config"]
    state_dict   = ckpt["model_state_dict"]

    model = MultiBranchDeepONet(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Model config: {model_config}")
    print(f"Device: {device}")

    return model, model_config, device


# ============================================================
# Optional: example usage
# ============================================================

if __name__ == "__main__":
    # Example: load model
    checkpoint = "mb_deeponet_checkpoints/multibranch_deeponet_best.pt"
    model, cfg, dev = load_model(checkpoint)


    val_npz   = "data/train_sin_moving_window_data.npz"
    val = np.load(val_npz)
    y_lf_br1_val   = val["y_lf_br1"]             # [1, Nb1]
    y_hf_br2_val   = val["y_hf_br2"]             # [1, Nb2]
    Y_lf_extra_wd_br3_val = val["Y_lf_extra_wd_br3"]  # [Bv, Nw1]
    Y_hf_extra_wd_br4_val = val["Y_hf_extra_wd_br4"]  # [Bv, Nw1]
    X_prd_coord_t_val     = val["X_prd_extra_coord_t"]      # [Bv, Nt,1]
    Y_HF_extra_out_val    = val["Y_HF_extra_out"]           # [Bv, Nt]

    print(f"  y_lf_br1            : {y_lf_br1_val.shape}")
    print(f"  y_hf_br2            : {y_hf_br2_val.shape}")
    print(f"  Y_lf_extra_wd_br3   : {Y_lf_extra_wd_br3_val.shape}")
    print(f"  Y_hf_extra_wd_br4   : {Y_hf_extra_wd_br4_val.shape}")
    print(f"  X_prd_extra_coord_t : {X_prd_coord_t_val.shape}")
    print(f"  Y_HF_extra_out      : {Y_HF_extra_out_val.shape}")

    y_lf_br1_val  = torch.as_tensor(y_lf_br1_val,  dtype=torch.float32, device=dev)  # [1, Nb1]
    y_hf_br2_val  = torch.as_tensor(y_hf_br2_val,  dtype=torch.float32, device=dev)  # [1, Nb2]
    Y_lf_extra_wd_br3_val = torch.as_tensor(Y_lf_extra_wd_br3_val, dtype=torch.float32, device=dev)  # [1, Nb1]
    Y_hf_extra_wd_br4_val = torch.as_tensor(Y_hf_extra_wd_br4_val, dtype=torch.float32, device=dev)  # [1, Nb2]
    X_prd_coord_t_val = torch.as_tensor(X_prd_coord_t_val, dtype=torch.float32, device=dev)  # [B, Nt, 1 ]
    Y_HF_extra_out_val = torch.as_tensor(Y_HF_extra_out_val, dtype=torch.float32, device=dev)  # [B, Nt]

    with torch.no_grad():
        y_pred = model(y_lf_br1_val, y_hf_br2_val, Y_lf_extra_wd_br3_val, Y_hf_extra_wd_br4_val, X_prd_coord_t_val)       # [B, Nt]

    print("Output shape:", y_pred.shape)   # should be [4, 50]
     
    # 5. Error metrics
    # -------------------------------------------------
    criterion = nn.MSELoss()
    mse = criterion(y_pred, Y_HF_extra_out_val)
    rmse = torch.sqrt(mse)
    rel_l2 = torch.norm(y_pred - Y_HF_extra_out_val) / torch.norm(Y_HF_extra_out_val)

    print("MSE Error :", mse.item())
    print("RMSE      :", rmse.item())
    print("Rel L2    :", rel_l2.item())

    