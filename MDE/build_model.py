
from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from MDONE import MultiBranchDeepONet

# =========================
# DDP helpers
# =========================
def ddp_is_available() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_setup() -> None:
    if "RANK" in os.environ and not ddp_is_available():
        dist.init_process_group(backend="nccl")

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_available() else 0

def ddp_world() -> int:
    return dist.get_world_size() if ddp_is_available() else 1

def ddp_primary() -> bool:
    return ddp_rank() == 0

# ---------- safe checkpoint helpers ----------
def unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m

def to_cpu_state_dict(m: nn.Module) -> Dict[str, torch.Tensor]:
    raw = unwrap_ddp(m)
    sd = raw.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
    return out
#---------------------------------------------
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)
@torch.no_grad()
def eval_groups(
    model: nn.Module,
    group_loaders: List[Tuple[DataLoader, Optional[DistributedSampler]]],
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    model.eval()
    total, n = 0.0, 0
    per_ch_sum = np.zeros(3, dtype=np.float64)

    for loader, _ in group_loaders:
        for Xb, Xt, Xph_grad, Xph_U, Xph_Delta, Y in loader:
            Xb  = Xb.to(device, non_blocking=True)
            Xt  = Xt.to(device, non_blocking=True)
            Xph_grad = Xph_grad.to(device, non_blocking=True)
            Xph_U     = Xph_U.to(device, non_blocking=True)
            Xph_Delta = Xph_Delta.to(device, non_blocking=True)
            Y   = Y.to(device, non_blocking=True)
            out = model(Xb, Xt, Xph_grad, Xph_U, Xph_Delta)           # (B,Nt,3)
            loss = mse_loss(out, Y)
            total += float(loss.item()) * Xb.shape[0]
            per_ch_sum += ((out - Y) ** 2).mean(dim=(0, 1)).detach().cpu().numpy()
            n += Xb.shape[0]

    if ddp_is_available():
        t = torch.tensor([total, n], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, n = float(t[0].item()), int(t[1].item())
        ch = torch.from_numpy(per_ch_sum).to(device)
        dist.all_reduce(ch, op=dist.ReduceOp.SUM)
        per_ch_sum = ch.cpu().numpy()

    world = ddp_world()
    return total / max(n, 1), per_ch_sum / max(world, 1)

def train_over_groups(
    head: str,
    train_paths: List[str],
    val_paths: List[str],
    device: torch.device,
    cfg: HybridCfg,
    epochs: int,
    batch_size: int,
    micro_batches: int,
    num_workers: int,
    weight_decay: float,
    out_dir: str,
    lr: float = 3e-4,
    *,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.0,
    early_stop_min_lr: float = 1e-7,
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Build per-group loaders (train + val)
    train_group_loaders: List[Tuple[DataLoader, Optional[DistributedSampler]]] = []
    val_group_loaders:   List[Tuple[DataLoader, Optional[DistributedSampler]]] = []

    if ddp_primary():
        print("[groups] building loaders...")

    for gid, p in enumerate(train_paths):
        ldr, sam, info = make_group_loader(
            p, head=head, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        if ddp_primary():
            print(f"  [train group {gid}] {p}  shape={info}")
        train_group_loaders.append((ldr, sam))

    for gid, p in enumerate(val_paths):
        ldr, sam, info = make_group_loader(
            p, head=head, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        if ddp_primary():
            print(f"  [val   group {gid}] {p}  shape={info}")
        val_group_loaders.append((ldr, sam))

    # Model
    model = MultiBranchDeepONet(cfg=cfg, head=head).to(device)
    if ddp_is_available():
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=ddp_primary())

    tag = f"{head}_grouped"
    hist_csv = (out / f"history_{tag}.csv") if ddp_primary() else None
    if ddp_primary() and hist_csv is not None:
        with hist_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train", "val", "val_ch0", "val_ch1", "val_ch2", "lr"])

    best_val = float("inf")
    best_state = to_cpu_state_dict(model)
    no_improve = 0

    for ep in range(1, epochs + 1):
        # Ensure unique shuffles per group in DDP
        for gid, (_, sam) in enumerate(train_group_loaders):
            if isinstance(sam, DistributedSampler):
                sam.set_epoch(ep * 100 + gid)

        # ================= Train (sequential over groups) =================
        model.train()
        loss_sum, n_sum = 0.0, 0

        for ldr, _ in train_group_loaders:  # group by group
            for Xb, Xt, Xph_grad, Xph_U, Xph_Delta, Y in ldr:
                Xb  = Xb.to(device, non_blocking=True)
                Xt  = Xt.to(device, non_blocking=True)
                Xph_grad = Xph_grad.to(device, non_blocking=True)
                Xph_U     = Xph_U.to(device, non_blocking=True)
                Xph_Delta = Xph_Delta.to(device, non_blocking=True)
                Y   = Y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                B = Xb.shape[0]
                parts = max(1, int(micro_batches))
                size = math.ceil(B / parts)
                batch_loss_total = 0.0

                for i in range(parts):
                    s, e = i * size, min(B, (i + 1) * size)
                    if s >= e: break
                    out = model(Xb[s:e], Xt[s:e], Xph_grad[s:e], Xph_U[s:e], Xph_Delta[s:e])  # (b,Nt,3)
                    loss = mse_loss(out, Y[s:e]) / parts
                    loss.backward()
                    batch_loss_total += float(loss.detach().item()) * (e - s)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                loss_sum += batch_loss_total
                n_sum += B

        # DDP reduction for train
        if ddp_is_available():
            t = torch.tensor([loss_sum, n_sum], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum, n_sum = float(t[0].item()), int(t[1].item())

        train_loss = loss_sum / max(n_sum, 1)

        # ================= Validation (sequential over groups) =================
        val_loss, val_per_ch = eval_groups(unwrap_ddp(model), val_group_loaders, device)

        # Scheduler + early stop
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        if val_loss < (best_val - early_stop_min_delta):
            best_val = val_loss
            best_state = to_cpu_state_dict(model)
            no_improve = 0
        else:
            no_improve += 1

        if ddp_primary():
            print(
                f"[epoch {ep:03d}] train={train_loss:.6e}  val={val_loss:.6e}  "
                f"val_ch=({val_per_ch[0]:.6e},{val_per_ch[1]:.6e},{val_per_ch[2]:.6e})  "
                f"lr={cur_lr:.2e}  no_improve={no_improve}/{early_stop_patience}",
                flush=True,
            )
            if hist_csv is not None:
                with hist_csv.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        ep, f"{train_loss:.8e}", f"{val_loss:.8e}",
                        f"{val_per_ch[0]:.8e}", f"{val_per_ch[1]:.8e}", f"{val_per_ch[2]:.8e}",
                        f"{cur_lr:.4e}"
                    ])

        if (cur_lr <= early_stop_min_lr) or (no_improve >= early_stop_patience):
            if ddp_primary():
                reason = "min_lr reached" if cur_lr <= early_stop_min_lr else "patience exhausted"
                print(f"[early-stop] Stopping at epoch {ep} ({reason}). best_val={best_val:.6e}", flush=True)
            break

    if ddp_primary():
        ckpt = {
            "model_state_dict": best_state,
            "model_cfg": asdict(cfg),
            "meta": {
                "head": head,
                "spatial_shape": "grouped",
                "train_files": train_paths,
                "val_files": val_paths,
                "best_val": best_val,
            },
        }
        ckpt_path = Path(out_dir) / f"best_{tag}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[saved] {ckpt_path}  best_val={best_val:.6e}", flush=True)