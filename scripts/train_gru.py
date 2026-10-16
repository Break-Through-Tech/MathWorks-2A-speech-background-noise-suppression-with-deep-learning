"""
Train the GRU denoiser baseline.

Run from the repo root:

    # Quick smoke test (2 epochs, small subset)
    python scripts/train_gru.py --max-epochs 2 --subset 64

    # Full training (default, same as GRU v1)
    python scripts/train_gru.py

    # GRU v2 — IRM target loss
    python scripts/train_gru.py --loss irm --checkpoint-name gru_v2.pt --exp-name gru_v2

    # GRU v3 — log-magnitude loss
    python scripts/train_gru.py --loss log_magnitude --checkpoint-name gru_v3.pt --exp-name gru_v3

Checkpoints are saved to models/<checkpoint-name> (git-ignored).
Experiment records are appended to outputs/experiments/log.csv.
"""

import argparse
import csv
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import features as F
from models.gru import GRUDenoiser


# ============================================================
# DATASET
# ============================================================

class SpeechDataset(Dataset):
    """
    Loads pre-computed spectrograms from a .npz file and applies
    log-magnitude transformation + per-frequency normalization.

    __getitem__ returns four tensors needed for training:
        x          [T, 257]  normalized log-magnitude (GRU input)
        noisy_mag  [257, T]  linear magnitude (for loss)
        clean_mag  [257, T]  linear target magnitude
        frame_mask [T]       True for real frames, False for padding
    """

    def __init__(self, npz_path: Path, mean: np.ndarray, std: np.ndarray):
        data = np.load(npz_path, allow_pickle=True)

        noisy_mag = data["noisy_mag"]   # [N, 257, 126]
        clean_mag = data["clean_mag"]   # [N, 257, 126]
        frame_mask = data["frame_mask"] # [N, 126]

        # Log-magnitude + normalize → [N, 257, 126]
        log_noisy = F.log_magnitude(noisy_mag)
        norm_input = F.normalize(log_noisy, mean, std)

        # Transpose to time-major [N, 126, 257] for the GRU
        x = norm_input.transpose(0, 2, 1).astype(np.float32)

        self.x = torch.from_numpy(x)
        self.noisy_mag = torch.from_numpy(noisy_mag)
        self.clean_mag = torch.from_numpy(clean_mag)
        self.frame_mask = torch.from_numpy(frame_mask)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            self.x[idx],          # [126, 257]
            self.noisy_mag[idx],  # [257, 126]
            self.clean_mag[idx],  # [257, 126]
            self.frame_mask[idx], # [126]
        )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def magnitude_masked_loss(
    mask: torch.Tensor,
    noisy_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked MSE on the enhanced magnitude (GRU v1 / default loss).

        loss = Σ frame_mask · (mask × noisy_mag − clean_mag)²
               ─────────────────────────────────────────────
                          Σ frame_mask · 257

    Args:
        mask:       [B, 126, 257] model output (time-major)
        noisy_mag:  [B, 257, 126] linear magnitude
        clean_mag:  [B, 257, 126] linear magnitude target
        frame_mask: [B, 126]      True = real audio frame
    """
    mask_fm = mask.transpose(1, 2)                        # [B, 257, 126]
    enhanced = mask_fm * noisy_mag                        # [B, 257, 126]
    sq_err = (enhanced - clean_mag) ** 2                  # [B, 257, 126]
    fm = frame_mask.unsqueeze(1).float()                  # [B, 1, 126]
    numerator = (sq_err * fm).sum()
    denominator = fm.sum() * 257.0
    if denominator < 1.0:
        return numerator
    return numerator / denominator


def irm_masked_loss(
    mask: torch.Tensor,
    noisy_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Direct IRM target loss (GRU v2).

    Trains the model to predict the Ideal Ratio Mask directly:

        IRM = clip(clean_mag / (noisy_mag + eps), 0, 1)
        loss = mean over real frames of (mask - IRM)²

    Advantages over magnitude MSE:
    - At high SNR: IRM ≈ 1, model explicitly learns not to suppress
    - At low SNR:  IRM < 1, model learns to suppress noise
    - No magnitude scaling — all T-F bins contribute equally
    - Target is in [0,1], same range as sigmoid output
    """
    mask_fm = mask.transpose(1, 2)                        # [B, 257, 126]
    irm = torch.clamp(clean_mag / (noisy_mag + 1e-8), 0.0, 1.0)  # [B, 257, 126]
    sq_err = (mask_fm - irm) ** 2                         # [B, 257, 126]
    fm = frame_mask.unsqueeze(1).float()                  # [B, 1, 126]
    numerator = (sq_err * fm).sum()
    denominator = fm.sum() * 257.0
    if denominator < 1.0:
        return numerator
    return numerator / denominator


def log_magnitude_masked_loss(
    mask: torch.Tensor,
    noisy_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    frame_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Log-magnitude reconstruction loss (GRU v3).

        loss = mean over real frames of (log(mask × noisy + eps) − log(clean + eps))²

    Advantages:
    - Reduces dynamic-range differences across SNR levels
    - Perceptually motivated (human hearing is roughly log-scale)
    - Penalises both under- and over-suppression more uniformly
    """
    mask_fm = mask.transpose(1, 2)                        # [B, 257, 126]
    enhanced = mask_fm * noisy_mag                        # [B, 257, 126]
    log_enhanced = torch.log(enhanced + eps)
    log_clean = torch.log(clean_mag + eps)
    sq_err = (log_enhanced - log_clean) ** 2              # [B, 257, 126]
    fm = frame_mask.unsqueeze(1).float()                  # [B, 1, 126]
    numerator = (sq_err * fm).sum()
    denominator = fm.sum() * 257.0
    if denominator < 1.0:
        return numerator
    return numerator / denominator


LOSS_FNS = {
    "magnitude":     magnitude_masked_loss,
    "irm":           irm_masked_loss,
    "log_magnitude": log_magnitude_masked_loss,
}


# ============================================================
# TRAINING / VALIDATION LOOPS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    loss_fn,
    epoch: int,
    max_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0

    with tqdm(loader, desc=f"Epoch {epoch:3d}/{max_epochs}", unit="batch", leave=True) as pbar:
        for x, noisy_mag, clean_mag, frame_mask in pbar:
            x = x.to(device)
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)
            frame_mask = frame_mask.to(device)

            optimizer.zero_grad()
            mask = model(x)
            loss = loss_fn(mask, noisy_mag, clean_mag, frame_mask)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
) -> float:
    model.eval()
    total_loss = 0.0

    for x, noisy_mag, clean_mag, frame_mask in loader:
        x = x.to(device)
        noisy_mag = noisy_mag.to(device)
        clean_mag = clean_mag.to(device)
        frame_mask = frame_mask.to(device)

        mask = model(x)
        loss = loss_fn(mask, noisy_mag, clean_mag, frame_mask)
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# POST-TRAINING VAL STOI/PESQ
# ============================================================

def compute_val_metrics(
    checkpoint_path: Path,
    data_dir: Path,
    device: torch.device,
    sample_size: int = 0,
) -> dict:
    """
    Load the best checkpoint and compute STOI + PESQ on the validation set.

    sample_size=0 means use the full val set.
    Returns a dict with val_noisy_stoi, val_enhanced_stoi, val_stoi_delta,
    val_noisy_pesq, val_enhanced_pesq, val_pesq_delta.
    """
    try:
        from pesq import pesq as pesq_fn
        from pystoi import stoi as stoi_fn
    except ImportError:
        print("  [val metrics] pesq/pystoi not installed — skipping STOI/PESQ")
        return {}

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = GRUDenoiser(
        input_size=257,
        hidden_size=ckpt_args.get("hidden_size", 256),
        num_layers=2,
        dropout=ckpt_args.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    norm = np.load(data_dir / "norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    val_data = np.load(data_dir / "val.npz", allow_pickle=True)
    noisy_mag    = val_data["noisy_mag"]    # [N, 257, 126]
    noisy_wav    = val_data["noisy_wav"]    # [N, 32000]
    clean_wav    = val_data["clean_wav"]    # [N, 32000]
    valid_length = val_data["valid_length"] # [N]

    n = len(noisy_mag)
    if sample_size > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(sample_size, n), replace=False)
        noisy_mag    = noisy_mag[idx]
        noisy_wav    = noisy_wav[idx]
        clean_wav    = clean_wav[idx]
        valid_length = valid_length[idx]

    # Run batched inference
    log_noisy = F.log_magnitude(noisy_mag)
    norm_input = F.normalize(log_noisy, mean, std)
    x_all = torch.from_numpy(norm_input.transpose(0, 2, 1).astype(np.float32))
    masks = np.zeros_like(noisy_mag)

    with torch.no_grad():
        for start in range(0, len(noisy_mag), 64):
            end = min(start + 64, len(noisy_mag))
            x = x_all[start:end].to(device)
            m = model(x)
            masks[start:end] = m.cpu().numpy().transpose(0, 2, 1)

    noisy_stoi_vals, enh_stoi_vals = [], []
    noisy_pesq_vals, enh_pesq_vals = [], []

    for i in range(len(noisy_mag)):
        vlen = int(valid_length[i])
        _, phase = F.magnitude_phase(noisy_wav[i])
        enh_mag = masks[i] * noisy_mag[i]
        enh_wav = F.reconstruct(enh_mag, phase, vlen)

        clean_ref  = clean_wav[i][:vlen].astype(np.float32)
        noisy_in   = noisy_wav[i][:vlen].astype(np.float32)
        enh_t      = enh_wav.astype(np.float32)

        try:
            noisy_pesq_vals.append(float(pesq_fn(16000, clean_ref, noisy_in, "wb")))
            enh_pesq_vals.append(float(pesq_fn(16000, clean_ref, enh_t, "wb")))
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                noisy_stoi_vals.append(float(stoi_fn(clean_ref, noisy_in, 16000, extended=False)))
                enh_stoi_vals.append(float(stoi_fn(clean_ref, enh_t, 16000, extended=False)))
        except Exception:
            pass

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    ns  = safe_mean(noisy_stoi_vals)
    es  = safe_mean(enh_stoi_vals)
    np_ = safe_mean(noisy_pesq_vals)
    ep  = safe_mean(enh_pesq_vals)

    return {
        "val_noisy_stoi":    round(ns,  4),
        "val_enhanced_stoi": round(es,  4),
        "val_stoi_delta":    round(es - ns, 4),
        "val_noisy_pesq":    round(np_, 4),
        "val_enhanced_pesq": round(ep,  4),
        "val_pesq_delta":    round(ep - np_, 4),
        "val_n_segments":    len(noisy_stoi_vals),
    }


# ============================================================
# EXPERIMENT LOG
# ============================================================

EXP_LOG_FIELDS = [
    "timestamp", "exp_name", "loss_type", "hidden_size", "dropout",
    "lr", "batch_size", "max_epochs", "best_epoch", "best_val_loss",
    "val_noisy_stoi", "val_enhanced_stoi", "val_stoi_delta",
    "val_noisy_pesq", "val_enhanced_pesq", "val_pesq_delta",
    "val_n_segments", "checkpoint",
]


def save_experiment(record: dict, output_dir: Path):
    log_path = output_dir / "log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXP_LOG_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    print(f"  Experiment record: {log_path}")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GRU speech denoiser baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",        default="data/processed/features",
                        help="Directory containing train/val/test .npz files")
    parser.add_argument("--model-dir",       default="models",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-name", default="gru_best.pt",
                        help="Filename for the saved checkpoint (inside --model-dir)")
    parser.add_argument("--exp-name",        default="",
                        help="Short label for the experiment record (e.g. gru_v2)")
    parser.add_argument("--loss",            default="magnitude",
                        choices=["magnitude", "irm", "log_magnitude"],
                        help="Training loss function")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Initial Adam learning rate")
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--max-epochs",      type=int,   default=50)
    parser.add_argument("--lr-patience",     type=int,   default=3,
                        help="ReduceLROnPlateau patience (epochs)")
    parser.add_argument("--lr-factor",       type=float, default=0.5,
                        help="ReduceLROnPlateau multiplicative factor")
    parser.add_argument("--es-patience",     type=int,   default=8,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--grad-clip",       type=float, default=5.0,
                        help="Global gradient norm clip (0 to disable)")
    parser.add_argument("--hidden-size",     type=int,   default=256)
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--num-workers",     type=int,   default=0,
                        help="DataLoader worker processes")
    parser.add_argument("--subset",          type=int,   default=0,
                        help="Use only first N training samples (0 = all). "
                             "Useful for quick smoke tests.")
    parser.add_argument("--no-val-metrics",  action="store_true",
                        help="Skip post-training val STOI/PESQ computation")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loss:   {args.loss}")

    data_dir  = REPO_ROOT / args.data_dir
    model_dir = REPO_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization statistics (training set only)
    norm = np.load(data_dir / "norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    train_ds = SpeechDataset(data_dir / "train.npz", mean, std)
    val_ds   = SpeechDataset(data_dir / "val.npz",   mean, std)

    if args.subset > 0:
        indices = list(range(min(args.subset, len(train_ds))))
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"Using subset of {len(train_ds)} training samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GRUDenoiser(
        input_size=257,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Training:   {len(train_ds)} samples, {len(train_loader)} batches/epoch")
    print(f"Validation: {len(val_ds)} samples")

    loss_fn = LOSS_FNS[args.loss]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    checkpoint_path = model_dir / args.checkpoint_name
    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0

    print(f"\nCheckpoint: {checkpoint_path}\n")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            args.grad_clip, loss_fn, epoch, args.max_epochs,
        )
        val_loss = validate(model, val_loader, device, loss_fn)
        current_lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            marker = "  [best]"
        else:
            patience_count += 1

        print(
            f"  train={train_loss:.4f}  val={val_loss:.4f}"
            f"  lr={current_lr:.2e}{marker}"
        )

        scheduler.step(val_loss)

        if patience_count >= args.es_patience:
            print(f"\nEarly stopping (no improvement for {args.es_patience} epochs).")
            break

    print(f"\nDone. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    # ----------------------------------------------------------
    # Post-training: compute val STOI / PESQ on best checkpoint
    # ----------------------------------------------------------
    val_metrics = {}
    if not args.no_val_metrics:
        print("\nComputing val STOI/PESQ on best checkpoint...")
        val_metrics = compute_val_metrics(checkpoint_path, data_dir, device)
        if val_metrics:
            print(f"  Val STOI: noisy={val_metrics['val_noisy_stoi']:.4f}  "
                  f"enhanced={val_metrics['val_enhanced_stoi']:.4f}  "
                  f"Δ={val_metrics['val_stoi_delta']:+.4f}")
            print(f"  Val PESQ: noisy={val_metrics['val_noisy_pesq']:.4f}  "
                  f"enhanced={val_metrics['val_enhanced_pesq']:.4f}  "
                  f"Δ={val_metrics['val_pesq_delta']:+.4f}")

    # ----------------------------------------------------------
    # Save experiment record
    # ----------------------------------------------------------
    exp_dir = REPO_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":       args.exp_name or args.checkpoint_name.replace(".pt", ""),
        "loss_type":      args.loss,
        "hidden_size":    args.hidden_size,
        "dropout":        args.dropout,
        "lr":             args.lr,
        "batch_size":     args.batch_size,
        "max_epochs":     args.max_epochs,
        "best_epoch":     best_epoch,
        "best_val_loss":  round(best_val_loss, 6),
        "checkpoint":     str(checkpoint_path.relative_to(REPO_ROOT)),
        **val_metrics,
    }
    save_experiment(record, exp_dir)


if __name__ == "__main__":
    main()
