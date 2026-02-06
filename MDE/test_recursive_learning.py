import torch
import numpy as np
from eval_model import load_model
import matplotlib.pyplot as plt


# -------------------------------------------------
# 1. Load trained model
# -------------------------------------------------
checkpoint = "mb_deeponet_checkpoints/multibranch_deeponet_best.pt"
model, cfg, dev = load_model(checkpoint)

kwd = 5          # length of window 1 in unit of dx
n_extra = 5      # window 2 is longer by n_extra * dx (currently unused)
moving_step = 1  # move windows by moving_step * dx each time

# -------------------------------------------------
# 2. Load data
# -------------------------------------------------
val_npz = "data/train_sin_moving_window_data.npz"
val = np.load(val_npz)

y_lf_br1            = val["y_lf_br1"]             # [1, Nb1]
y_hf_br2            = val["y_hf_br2"]             # [1, Nb2]
Y_lf_extra_wd_br3   = val["Y_lf_extra_wd_br3"]    # [B, Nw1]
Y_hf_extra_wd_br4   = val["Y_hf_extra_wd_br4"]    # [B, Nw1]
X_prd_extra_coord_t = val["X_prd_extra_coord_t"]  # [B, Nt, 1]
X_prd_extra_coord_t_global = val["X_prd_extra_coord_t_global"]  # [B, Nt, 1]
Y_HF_extra_out      = val["Y_HF_extra_out"]       # [B, Nt]

print(f"  y_lf_br1            : {y_lf_br1.shape}")
print(f"  y_hf_br2            : {y_hf_br2.shape}")
print(f"  Y_lf_extra_wd_br3   : {Y_lf_extra_wd_br3.shape}")
print(f"  Y_hf_extra_wd_br4   : {Y_hf_extra_wd_br4.shape}")
print(f"  X_prd_extra_coord_t : {X_prd_extra_coord_t.shape}")
print(f"  Y_HF_extra_out      : {Y_HF_extra_out.shape}")

# -------------------------------------------------
# 3. Convert to tensors (fixed inputs)
# -------------------------------------------------
y_lf_br1_tensor = torch.as_tensor(y_lf_br1, dtype=torch.float32, device=dev)  # [1, Nb1]

y_hf_br2_tensor = torch.as_tensor(y_hf_br2, dtype=torch.float32, device=dev)  # [1, Nb2]

Y_lf_extra_wd_br3_tensor = torch.as_tensor(Y_lf_extra_wd_br3, dtype=torch.float32, device=dev)  # [B, Nw1]

# HF window kept as NumPy array for easy concatenation
Y_hf_extra_wd_br4_np = Y_hf_extra_wd_br4.copy()    # [B, Nw1]

X_prd_extra_coord_t_tensor = torch.as_tensor(X_prd_extra_coord_t, dtype=torch.float32, device=dev)  # [B, Nt, 1]

Y_HF_extra_out_tensor = torch.as_tensor(Y_HF_extra_out, dtype=torch.float32, device=dev)  # [B, Nt]

# Container for predictions (NumPy)
Y_HF_extra_pred = np.zeros_like(Y_HF_extra_out, dtype=np.float32)

B, Nt, _ = X_prd_extra_coord_t_tensor.shape
print("batch size:", B)

# -------------------------------------------------
# 4. Autoregressive loop over batch
# -------------------------------------------------
Y_hf_window_np = None  # current HF window [1, Nw1]

for i in range(B):
    # Slice current sample
    x_i = X_prd_extra_coord_t_tensor[i:i+1, :, :]      # [1, Nt, 1]
    u3_i = Y_lf_extra_wd_br3_tensor[i:i+1, :]          # [1, Nw1]

    if i == 0:
        # Start from true HF window for sample 0
        Y_hf_window_np = Y_hf_extra_wd_br4_np[i:i+1, :]  # [1, Nw1]
    # else: use updated Y_hf_window_np from previous iteration

    u4_i = torch.as_tensor(Y_hf_window_np, dtype=torch.float32, device=dev)  # [1, Nw1]

    print("Y_hf_extra_wd_br4 window shape:", u4_i.shape)
    print("X_prd_extra_coord_t_tensor shape:", x_i.shape)

    # Forward pass
    with torch.no_grad():
        y_pred = model(
            y_lf_br1_tensor,  # [1, Nb1]
            y_hf_br2_tensor,  # [1, Nb2]
            u3_i,             # [1, Nw1]
            u4_i,             # [1, Nw1]
            x_i,              # [1, Nt, 1]
        )                     # -> [1, Nt]

    y_pred_np = y_pred.detach().cpu().numpy()  # [1, Nt]

    # Save full predicted field for this sample
    Y_HF_extra_pred[i, :] = y_pred_np[0, :]

    # ---- Update HF window for next iteration ----
    # predicted segment at [kwd : kwd + moving_step]
    y_add = y_pred_np[:, kwd:kwd + moving_step]               # [1, moving_step]

    # base window: original HF window of sample i, shifted by moving_step
    base = Y_hf_extra_wd_br4_np[i:i+1, moving_step:]          # [1, Nw1 - moving_step]

    # new current HF window for next sample
    Y_hf_window_np = np.concatenate([base, y_add], axis=1)    # [1, Nw1]

# -------------------------------------------------
# 5. Optional: convert predictions to tensor & compute error
# -------------------------------------------------
Y_HF_extra_pred_tensor = torch.as_tensor(
    Y_HF_extra_pred, dtype=torch.float32, device=dev
)  # [B, Nt]

mse = torch.mean((Y_HF_extra_pred_tensor - Y_HF_extra_out_tensor) ** 2)
rmse = torch.sqrt(mse)
rel_l2 = torch.norm(
    Y_HF_extra_pred_tensor - Y_HF_extra_out_tensor
) / torch.norm(Y_HF_extra_out_tensor)

print("Prediction shape:", Y_HF_extra_pred.shape)
print("MSE :", mse.item())
print("RMSE:", rmse.item())
print("Rel L2:", rel_l2.item())

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
fig3.suptitle("Set 3: Window 2")

import matplotlib.pyplot as plt
Xw_exptra_global = X_prd_extra_coord_t_global #set_HF_extra_prd_wd["X_global"] # [B, Nw2, 1]

B = Xw_exptra_global.shape[0]
# Global
for b in range(B):
    xg = Xw_exptra_global[b, :, 0]
    y_pred = Y_HF_extra_pred_tensor[b, :].detach().cpu().numpy()
    y_val = Y_HF_extra_out_tensor[b, :].detach().cpu().numpy()

    label = f"batch {b}" 
    ax3a.plot(xg, y_pred, marker="o", linestyle="-", alpha=0.7, label=label)
    ax3a.plot(xg, y_val, marker="*", linestyle="-", alpha=0.7, label=label)

ax3a.set_xlabel("x (global)")
ax3a.set_ylabel("y = sin(x)")
ax3a.set_title("Global coords")
ax3a.grid(True)
if b < 10:
    ax3a.legend()
