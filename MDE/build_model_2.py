import os
import csv
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Model definition
# ============================================================

class MLP(nn.Module):
    """Simple fully-connected MLP."""
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.GELU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [..., in_dim]
        return self.net(x)


class MultiBranchDeepONet(nn.Module):
    """
    DeepONet with 4 branch nets and 1 trunk net.

    Inputs to forward():
      u1: [B, b1]    # branch 1 (y_lf_br1, fixed per epoch, expanded over B)
      u2: [B, b2]    # branch 2 (y_hf_br2, fixed per epoch, expanded over B)
      u3: [B, b3]    # branch 3 (Y_*_wd_br3, batch-dependent)
      u4: [B, b3]    # branch 4 (Y_*_wd_br4, batch-dependent)
      x : [B, t1, d_x]   # trunk input coordinates

    Output:
      y:  [B, t1]
    """

    def __init__(
        self,
        b1_dim: int,
        b2_dim: int,
        b3_dim: int,       # used for both branch 3 and 4
        trunk_in_dim: int,
        width: int = 64,
        branch_hidden=(128, 128),
        trunk_hidden=(128, 128),
    ):
        super().__init__()
        # Four branch networks, each mapping its own sensor values -> latent width
        self.branch1 = MLP(b1_dim, branch_hidden, width)
        self.branch2 = MLP(b2_dim, branch_hidden, width)
        self.branch3 = MLP(b3_dim, branch_hidden, width)
        self.branch4 = MLP(b3_dim, branch_hidden, width)

        # Trunk net acts pointwise on coordinates
        self.trunk = MLP(trunk_in_dim, trunk_hidden, width)

    def forward(self, u1, u2, u3, u4, x):
        """
        u1: [B, b1]
        u2: [B, b2]
        u3: [B, b3]
        u4: [B, b3]
        x:  [B, t1, d_x]
        """
        B, t1, d_x = x.shape

        # ----- Branch side -----
        b1 = self.branch1(u1)  # [B, width]
        b2 = self.branch2(u2)  # [B, width]
        b3 = self.branch3(u3)  # [B, width]
        b4 = self.branch4(u4)  # [B, width]

        # Elementwise product across all 4 branches -> one combined branch vector
        # shape: [B, width]
        branch = b1 * b2 * b3 * b4

        # ----- Trunk side -----
        # Apply the trunk net pointwise to each coordinate x_i
        x_flat = x.reshape(B * t1, d_x)      # [B*t1, d_x]
        t_flat = self.trunk(x_flat)          # [B*t1, width]
        trunk_feats = t_flat.view(B, t1, -1) # [B, t1, width]

        # ----- DeepONet combination -----
        # vanilla DeepONet: elementwise product in latent dim and sum over latent dim
        # branch:      [B, width]      -> [B, 1, width]
        # trunk_feats: [B, t1, width]
        y = (branch.unsqueeze(1) * trunk_feats).sum(dim=-1)  # [B, t1]

        return y


# ============================================================
# Dataset for moving-window data (branch 3 & 4 + trunk + output)
# ============================================================

class WindowDataset(Dataset):
    """
    Dataset for dynamic inputs only:

      Y_br3  : [B, Nw1]  (branch 3 input)
      Y_br4  : [B, Nw1]  (branch 4 input)
      X_t    : [B, Nt]   (trunk coordinates)
      Y_out  : [B, Nt]   (target HF output)

    Branch 1 and 2 inputs (y_lf_br1, y_hf_br2) are fixed and
    handled outside the dataset.
    """
    def __init__(self, Y_br3, Y_br4, X_t, Y_out):
        super().__init__()
        # Store as torch tensors (float32)
        self.Y_br3 = torch.as_tensor(Y_br3, dtype=torch.float32)
        self.Y_br4 = torch.as_tensor(Y_br4, dtype=torch.float32)
        self.X_t   = torch.as_tensor(X_t,   dtype=torch.float32)
        self.Y_out = torch.as_tensor(Y_out, dtype=torch.float32)

        assert self.Y_br3.shape[0] == self.Y_br4.shape[0] == self.X_t.shape[0] == self.Y_out.shape[0]
        self.n_samples = self.Y_br3.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.Y_br3[idx],   # [Nw1]
            self.Y_br4[idx],   # [Nw1]
            self.X_t[idx],     # [Nt]
            self.Y_out[idx],   # [Nt]
        )


# ============================================================
# Training function (train + validation)
# ============================================================

def train_multibranch_deeponet(
    train_npz_path: str,
    val_npz_path: str,
    out_dir: str = "mb_deeponet_checkpoints",
    batch_size: int = 16,
    max_epochs: int = 500,
    lr: float = 3e-3,
    early_stop_patience: int = 30,
    lr_factor: float = 0.5,
    lr_patience: int = 10,
    min_lr: float = 1e-6,
):

    """
    Train MultiBranchDeepONet from train + validation NPZ datasets.

    Training NPZ file must contain:
      y_lf_br1            : [1, Nb1]    (branch 1 fixed input)
      y_hf_br2            : [1, Nb2]    (branch 2 fixed input)
      Y_lf_learn_wd_br3   : [Btr, Nw1]  (branch 3 train)
      Y_hf_learn_wd_br4   : [Btr, Nw1]  (branch 4 train)
      X_prd_train_coord_t : [Btr, Nt]   (trunk coords train)
      Y_HF_train_out      : [Btr, Nt]   (HF train output)

    Validation NPZ file must contain:
      y_lf_br1            : [1, Nb1]    (branch 1 fixed, val)
      y_hf_br2            : [1, Nb2]    (branch 2 fixed, val)
      Y_lf_extra_wd_br3   : [Bv, Nw1]   (branch 3 val)
      Y_hf_extra_wd_br4   : [Bv, Nw1]   (branch 4 val)
      X_prd_extra_coord_t : [Bv, Nt]    (trunk coords val)
      Y_HF_extra_out      : [Bv, Nt]    (HF val output)
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------- Load training data --------------------
    train = np.load(train_npz_path)
    y_lf_br1_tr   = train["y_lf_br1"]            # [1, Nb1]
    y_hf_br2_tr   = train["y_hf_br2"]            # [1, Nb2]
    Y_lf_learn_wd_br3_tr = train["Y_lf_learn_wd_br3"]  # [Btr, Nw1]
    Y_hf_learn_wd_br4_tr = train["Y_hf_learn_wd_br4"]  # [Btr, Nw1]
    X_prd_coord_t_tr     = train["X_prd_train_coord_t"]      # [Btr, Nt]
    Y_HF_train_out_tr    = train["Y_HF_train_out"]           # [Btr, Nt]

    # Shapes
    Nb1 = y_lf_br1_tr.shape[1]
    Nb2 = y_hf_br2_tr.shape[1]
    Nw1 = Y_lf_learn_wd_br3_tr.shape[1]
    Nt  = X_prd_coord_t_tr.shape[1]

    print("Loaded TRAIN data from:", train_npz_path)
    print(f"  y_lf_br1            : {y_lf_br1_tr.shape}")
    print(f"  y_hf_br2            : {y_hf_br2_tr.shape}")
    print(f"  Y_lf_learn_wd_br3   : {Y_lf_learn_wd_br3_tr.shape}")
    print(f"  Y_hf_learn_wd_br4   : {Y_hf_learn_wd_br4_tr.shape}")
    print(f"  X_prd_train_coord_t : {X_prd_coord_t_tr.shape}")
    print(f"  Y_HF_train_out      : {Y_HF_train_out_tr.shape}")

    # -------------------- Load validation data --------------------
    val = np.load(val_npz_path)
    y_lf_br1_val   = val["y_lf_br1"]             # [1, Nb1]
    y_hf_br2_val   = val["y_hf_br2"]             # [1, Nb2]
    Y_lf_extra_wd_br3_val = val["Y_lf_extra_wd_br3"]  # [Bv, Nw1]
    Y_hf_extra_wd_br4_val = val["Y_hf_extra_wd_br4"]  # [Bv, Nw1]
    X_prd_coord_t_val     = val["X_prd_extra_coord_t"]      # [Bv, Nt]
    Y_HF_extra_out_val    = val["Y_HF_extra_out"]           # [Bv, Nt]

    print("Loaded VAL data from:", val_npz_path)
    print(f"  y_lf_br1            : {y_lf_br1_val.shape}")
    print(f"  y_hf_br2            : {y_hf_br2_val.shape}")
    print(f"  Y_lf_extra_wd_br3   : {Y_lf_extra_wd_br3_val.shape}")
    print(f"  Y_hf_extra_wd_br4   : {Y_hf_extra_wd_br4_val.shape}")
    print(f"  X_prd_extra_coord_t : {X_prd_coord_t_val.shape}")
    print(f"  Y_HF_extra_out      : {Y_HF_extra_out_val.shape}")

    # Sanity checks on shapes (Nb1, Nb2, Nw1, Nt must match)
    assert y_lf_br1_val.shape[1] == Nb1, "Nb1 mismatch between train and val"
    assert y_hf_br2_val.shape[1] == Nb2, "Nb2 mismatch between train and val"
    assert Y_lf_extra_wd_br3_val.shape[1] == Nw1, "Nw1 mismatch between train and val"
    assert X_prd_coord_t_val.shape[1] == Nt, "Nt mismatch between train and val"

    # -------------------- Datasets & DataLoaders --------------------
    train_dataset = WindowDataset(
        Y_lf_learn_wd_br3_tr,
        Y_hf_learn_wd_br4_tr,
        X_prd_coord_t_tr,
        Y_HF_train_out_tr,
    )
    val_dataset = WindowDataset(
        Y_lf_extra_wd_br3_val,
        Y_hf_extra_wd_br4_val,
        X_prd_coord_t_val,
        Y_HF_extra_out_val,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # -------------------- Device & model --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model configuration dict (will be saved inside checkpoint)
    model_config = {
        "b1_dim": Nb1,
        "b2_dim": Nb2,
        "b3_dim": Nw1,
        "trunk_in_dim": 1,
        "width": 64,
        "branch_hidden": (128, 128),
        "trunk_hidden": (128, 128),
    }

    model = MultiBranchDeepONet(**model_config).to(device)

    # Fixed branch inputs (train & val) to device
    y_lf_tr_tensor  = torch.as_tensor(y_lf_br1_tr,  dtype=torch.float32, device=device)  # [1, Nb1]
    y_hf_tr_tensor  = torch.as_tensor(y_hf_br2_tr,  dtype=torch.float32, device=device)  # [1, Nb2]
    y_lf_val_tensor = torch.as_tensor(y_lf_br1_val, dtype=torch.float32, device=device)  # [1, Nb1]
    y_hf_val_tensor = torch.as_tensor(y_hf_br2_val, dtype=torch.float32, device=device)  # [1, Nb2]

    # -------------------- Optimizer, scheduler, loss --------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
        verbose=False,
    )
    criterion = nn.MSELoss()

    # -------------------- Training loop with val + early stopping --------------------
    best_val_loss = float("inf")
    best_state = None
    no_improve_epochs = 0

    history = []  # list of dicts for CSV rows

    for epoch in range(1, max_epochs + 1):
        # ======== Train ========
        model.train()
        train_running_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            Y_br3_b, Y_br4_b, X_t_b, Y_out_b = batch
            # Move to device
            Y_br3_b = Y_br3_b.to(device)    # [B_batch, Nw1]
            Y_br4_b = Y_br4_b.to(device)    # [B_batch, Nw1]
            X_t_b   = X_t_b.to(device)      # [B_batch, Nt,1]
            Y_out_b = Y_out_b.to(device)    # [B_batch, Nt]

            B_batch = X_t_b.shape[0]

            # Branch 1 & 2 inputs: fixed, expanded along batch dim
            u1_tr = y_lf_tr_tensor.expand(B_batch, -1)  # [B_batch, Nb1]
            u2_tr = y_hf_tr_tensor.expand(B_batch, -1)  # [B_batch, Nb2]

            # Branch 3 & 4 inputs: batch-dependent
            u3_tr = Y_br3_b                          # [B_batch, Nw1]
            u4_tr = Y_br4_b                          # [B_batch, Nw1]

            # Trunk input: X_t_b [B_batch, Nt] -> [B_batch, Nt, 1]
            x_trunk_tr = X_t_b

            optimizer.zero_grad()
            y_pred_tr = model(u1_tr, u2_tr, u3_tr, u4_tr, x_trunk_tr)  # [B_batch, Nt]

            loss_tr = criterion(y_pred_tr, Y_out_b)
            loss_tr.backward()
            optimizer.step()

            train_running_loss += loss_tr.item()
            train_batches += 1

        train_loss = train_running_loss / max(train_batches, 1)

        # ======== Validation ========
        model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                Y_br3_b, Y_br4_b, X_t_b, Y_out_b = batch
                Y_br3_b = Y_br3_b.to(device)    # [B_batch, Nw1]
                Y_br4_b = Y_br4_b.to(device)    # [B_batch, Nw1]
                X_t_b   = X_t_b.to(device)      # [B_batch, Nt]
                Y_out_b = Y_out_b.to(device)    # [B_batch, Nt]

                B_batch = X_t_b.shape[0]

                # Branch 1 & 2 (val)
                u1_val = y_lf_val_tensor.expand(B_batch, -1)  # [B_batch, Nb1]
                u2_val = y_hf_val_tensor.expand(B_batch, -1)  # [B_batch, Nb2]

                u3_val = Y_br3_b
                u4_val = Y_br4_b
                x_trunk_val = X_t_b

                y_pred_val = model(u1_val, u2_val, u3_val, u4_val, x_trunk_val)  # [B_batch, Nt]

                loss_val = criterion(y_pred_val, Y_out_b)

                val_running_loss += loss_val.item()
                val_batches += 1

        val_loss = val_running_loss / max(val_batches, 1)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Early stopping logic based on val_loss
        status = ""
        if val_loss < best_val_loss - 1e-8:  # small tolerance
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve_epochs = 0
            status = "improved"
        else:
            no_improve_epochs += 1
            status = f"no_improve ({no_improve_epochs}/{early_stop_patience})"

        print(
            f"Epoch {epoch:4d} | "
            f"train_loss = {train_loss:.4e} | "
            f"val_loss = {val_loss:.4e} | "
            f"lr = {current_lr:.2e} | {status}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "no_improve_epochs": no_improve_epochs,
            }
        )

        if no_improve_epochs >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # -------------------- Save best model + config to a .pt file --------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "best_val_loss": best_val_loss,
        "history": history,
        "train_npz_path": train_npz_path,
        "val_npz_path": val_npz_path,
    }

    model_path = os.path.join(out_dir, "multibranch_deeponet_best.pt")
    torch.save(checkpoint, model_path)
    print("Saved model + config checkpoint to:", model_path)

    # -------------------- Save training history to CSV --------------------
    history_path = os.path.join(out_dir, "training_history.csv")
    with open(history_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "no_improve_epochs"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    print("Saved training history to:", history_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Change these to your actual file paths
    train_npz = "data/train_sin_moving_window_data.npz"
    val_npz   = "data/train_sin_moving_window_data.npz"
    

    train_multibranch_deeponet(
        train_npz_path=train_npz,
        val_npz_path=val_npz,
        out_dir="mb_deeponet_checkpoints",
        batch_size=2,
        max_epochs=1000,
        lr=3e-3,
        early_stop_patience=30,
        lr_factor=0.5,
        lr_patience=10,
        min_lr=1e-6,
    )
