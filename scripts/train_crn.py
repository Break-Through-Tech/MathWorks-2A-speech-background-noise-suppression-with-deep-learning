"""
Train the CRN-Lite denoiser (advanced model).

Spec: docs/advanced_model.md. Reuses the dataset, loss functions,
train/val loops and experiment log from train_gru.py, so the CRN
and the GRU are trained on exactly the same pipeline.

Run from the repo root:

    # Dry run (2 epochs, small subset, no PESQ/STOI)
    python scripts/train_crn.py --max-epochs 2 --subset 64 --no-val-metrics --checkpoint-name crn_dryrun.pt --exp-name crn_dryrun

    # Full training (doc defaults: IRM loss, batch 16, 60 epochs)
    python scripts/train_crn.py

    # Smaller model if it overfits
    python scripts/train_crn.py --hidden-size 128 --checkpoint-name crn_h128.pt --exp-name crn_h128

Checkpoints are saved to models/<checkpoint-name> (git-ignored).
Experiment records are appended to outputs/experiments/log.csv.
The test set is never loaded here.
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import features as F
from models.crn import CRNDenoiser
from train_gru import (
    LOSS_FNS,
    SpeechDataset,
    save_experiment,
    train_one_epoch,
    validate,
)


# ============================================================
# POST-TRAINING VAL STOI/PESQ  (CRN version of the GRU helper)
# ============================================================

def compute_val_metrics(checkpoint_path, data_dir, device):
    """STOI + PESQ on the validation set using the best checkpoint."""
    try:
        from pesq import pesq as pesq_fn
        from pystoi import stoi as stoi_fn
    except ImportError:
        print("  [val metrics] pesq/pystoi not installed - skipping")
        return {}

    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden = ckpt.get("args", {}).get("hidden_size", 256)
    model = CRNDenoiser(hidden_size=hidden)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    norm = np.load(data_dir / "norm_stats.npz")
    val = np.load(data_dir / "val.npz", allow_pickle=True)
    noisy_mag = val["noisy_mag"]
    noisy_wav = val["noisy_wav"]
    clean_wav = val["clean_wav"]
    valid_length = val["valid_length"]

    x_all = F.normalize(
        F.log_magnitude(noisy_mag), norm["mean"], norm["std"]
    ).transpose(0, 2, 1).astype(np.float32)
    x_all = torch.from_numpy(x_all)

    masks = np.zeros_like(noisy_mag)
    with torch.no_grad():
        for s in range(0, len(noisy_mag), 64):
            m = model(x_all[s:s + 64].to(device))
            masks[s:s + 64] = m.cpu().numpy().transpose(0, 2, 1)

    ns, es, np_, ep = [], [], [], []
    for i in range(len(noisy_mag)):
        n = int(valid_length[i])
        _, phase = F.magnitude_phase(noisy_wav[i])
        enh = F.reconstruct(masks[i] * noisy_mag[i], phase, n)
        ref = clean_wav[i][:n].astype(np.float32)
        noi = noisy_wav[i][:n].astype(np.float32)
        enh = enh.astype(np.float32)
        try:
            np_.append(float(pesq_fn(16000, ref, noi, "wb")))
            ep.append(float(pesq_fn(16000, ref, enh, "wb")))
        except Exception:
            pass
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ns.append(float(stoi_fn(ref, noi, 16000)))
                es.append(float(stoi_fn(ref, enh, 16000)))
        except Exception:
            pass

    def mean(v):
        return float(np.mean(v)) if v else float("nan")

    return {
        "val_noisy_stoi":    round(mean(ns), 4),
        "val_enhanced_stoi": round(mean(es), 4),
        "val_stoi_delta":    round(mean(es) - mean(ns), 4),
        "val_noisy_pesq":    round(mean(np_), 4),
        "val_enhanced_pesq": round(mean(ep), 4),
        "val_pesq_delta":    round(mean(ep) - mean(np_), 4),
        "val_n_segments":    len(ns),
    }


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CRN-Lite speech denoiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/processed/features")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--checkpoint-name", default="crn_best.pt")
    p.add_argument("--exp-name", default="")
    p.add_argument("--loss", default="irm",
                   choices=list(LOSS_FNS))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--es-patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--subset", type=int, default=0,
                   help="Use only first N training samples (0 = all)")
    p.add_argument("--no-val-metrics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loss:   {args.loss}")

    data_dir = REPO_ROOT / args.data_dir
    model_dir = REPO_ROOT / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    norm = np.load(data_dir / "norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    train_ds = SpeechDataset(data_dir / "train.npz", mean, std)
    val_ds = SpeechDataset(data_dir / "val.npz", mean, std)

    if args.subset > 0:
        idx = list(range(min(args.subset, len(train_ds))))
        train_ds = torch.utils.data.Subset(train_ds, idx)
        print(f"Using subset of {len(train_ds)} training samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = CRNDenoiser(hidden_size=args.hidden_size).to(device)

    # Integration check on one real batch before training
    x0 = next(iter(train_loader))[0][:2].to(device)
    with torch.no_grad():
        y0 = model.eval()(x0)
    assert y0.shape == x0.shape, (x0.shape, y0.shape)
    print(f"Shape check: {tuple(x0.shape)} -> {tuple(y0.shape)} OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Training:   {len(train_ds)} samples, "
          f"{len(train_loader)} batches/epoch")
    print(f"Validation: {len(val_ds)} samples")

    loss_fn = LOSS_FNS[args.loss]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=args.lr_factor, patience=args.lr_patience,
    )

    ckpt_path = model_dir / args.checkpoint_name
    best_val, best_epoch, patience = float("inf"), 0, 0
    print(f"\nCheckpoint: {ckpt_path}\n")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            args.grad_clip, loss_fn, epoch, args.max_epochs,
        )
        val_loss = validate(model, val_loader, device, loss_fn)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_loss < best_val:
            best_val, best_epoch, patience = val_loss, epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "model": "crn",
            }, ckpt_path)
            marker = "  [best]"
        else:
            patience += 1

        print(f"  train={train_loss:.4f}  val={val_loss:.4f}"
              f"  lr={lr:.2e}{marker}")
        scheduler.step(val_loss)

        if patience >= args.es_patience:
            print(f"\nEarly stopping ({args.es_patience} epochs "
                  f"without improvement).")
            break

    print(f"\nDone. Best epoch: {best_epoch}, "
          f"best val_loss: {best_val:.4f}")
    print("GRU v2 best val IRM loss to beat: 0.1124")

    val_metrics = {}
    if not args.no_val_metrics:
        print("\nComputing val STOI/PESQ on best checkpoint...")
        val_metrics = compute_val_metrics(ckpt_path, data_dir, device)
        if val_metrics:
            print(f"  Val STOI: noisy={val_metrics['val_noisy_stoi']:.4f}"
                  f"  enhanced={val_metrics['val_enhanced_stoi']:.4f}"
                  f"  d={val_metrics['val_stoi_delta']:+.4f}")
            print(f"  Val PESQ: noisy={val_metrics['val_noisy_pesq']:.4f}"
                  f"  enhanced={val_metrics['val_enhanced_pesq']:.4f}"
                  f"  d={val_metrics['val_pesq_delta']:+.4f}")

    exp_dir = REPO_ROOT / "outputs" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_experiment({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name": args.exp_name or args.checkpoint_name[:-3],
        "loss_type": args.loss,
        "hidden_size": args.hidden_size,
        "dropout": 0.0,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6),
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        **val_metrics,
    }, exp_dir)


if __name__ == "__main__":
    main()
