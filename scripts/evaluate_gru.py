"""
Evaluate the trained GRU denoiser on the test set.

Run from the repo root:

    python scripts/evaluate_gru.py

Outputs (saved to outputs/evaluation/):
    results.csv            per-segment scores
    summary.txt            overall + per-noise-type + per-SNR averages
    examples/              a few enhanced WAV files per noise type

Requirements: pesq, pystoi  (pip install pesq pystoi)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import features as F
from models.gru import GRUDenoiser


# ============================================================
# HELPERS
# ============================================================

def load_splits_metadata(splits_csv: Path) -> dict:
    """
    Returns {mixed_file: {noise_category, target_snr_db}} for non-excluded clips.
    """
    meta = {}
    with open(splits_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["excluded"]:
                continue
            meta[row["mixed_file"]] = {
                "noise_category": row["noise_category"],
                "target_snr_db": row["target_snr_db"],
            }
    return meta


def safe_pesq(ref: np.ndarray, deg: np.ndarray, fs: int = 16000) -> float:
    try:
        from pesq import pesq
        return float(pesq(fs, ref.astype(np.float32), deg.astype(np.float32), "wb"))
    except Exception:
        return float("nan")


def safe_stoi(ref: np.ndarray, deg: np.ndarray, fs: int = 16000) -> float:
    try:
        from pystoi import stoi
        return float(stoi(ref, deg, fs, extended=False))
    except Exception:
        return float("nan")


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    noisy_mag: np.ndarray,  # [N, 257, 126]
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Returns predicted masks [N, 257, 126] as numpy float32.
    """
    model.eval()
    n = len(noisy_mag)
    masks = np.zeros_like(noisy_mag)

    log_noisy = F.log_magnitude(noisy_mag)           # [N, 257, 126]
    norm_input = F.normalize(log_noisy, mean, std)   # [N, 257, 126]
    x_all = torch.from_numpy(norm_input.transpose(0, 2, 1).astype(np.float32))  # [N, 126, 257]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = x_all[start:end].to(device)              # [B, 126, 257]
        mask = model(x)                               # [B, 126, 257]
        # Transpose back to freq-major [B, 257, 126]
        masks[start:end] = mask.cpu().numpy().transpose(0, 2, 1)

    return masks


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GRU denoiser on the test or validation set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  default="models/gru_best.pt",
                        help="Path to the saved model checkpoint")
    parser.add_argument("--data-dir",    default="data/processed/features")
    parser.add_argument("--splits-csv",  default="data/splits.csv")
    parser.add_argument("--output-dir",  default="",
                        help="Output directory (default: outputs/evaluation for test, "
                             "outputs/val_eval/<checkpoint-stem> for val)")
    parser.add_argument("--split",       default="test", choices=["test", "val"],
                        help="Which split to evaluate on (test = held-out; val = development)")
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of enhanced WAV examples to save per noise type")
    parser.add_argument("--batch-size",  type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = REPO_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the model first:  python scripts/train_gru.py")
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    hidden_size = ckpt_args.get("hidden_size", 256)
    dropout     = ckpt_args.get("dropout", 0.2)

    model = GRUDenoiser(input_size=257, hidden_size=hidden_size, num_layers=2, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    data_dir = REPO_ROOT / args.data_dir
    norm = np.load(data_dir / "norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    split_data = np.load(data_dir / f"{args.split}.npz", allow_pickle=True)
    noisy_mag    = split_data["noisy_mag"]    # [N, 257, 126]
    clean_mag    = split_data["clean_mag"]    # [N, 257, 126]
    noisy_wav    = split_data["noisy_wav"]    # [N, 32000]
    clean_wav    = split_data["clean_wav"]    # [N, 32000]
    valid_length = split_data["valid_length"] # [N]
    mixed_files  = split_data["mixed_file"]   # [N] str

    splits_meta = load_splits_metadata(REPO_ROOT / args.splits_csv)

    # Default output directory depends on split + checkpoint
    if args.output_dir:
        output_dir = REPO_ROOT / args.output_dir
    elif args.split == "val":
        ckpt_stem = Path(args.checkpoint).stem
        output_dir = REPO_ROOT / "outputs" / "val_eval" / ckpt_stem
    else:
        output_dir = REPO_ROOT / "outputs" / "evaluation"

    example_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on {len(noisy_mag)} {args.split} segments...")
    masks = run_inference(model, noisy_mag, mean, std, device, args.batch_size)

    # --------------------------------------------------------
    # Per-segment scoring
    # --------------------------------------------------------

    rows = []
    saved_examples: dict[str, int] = {}  # noise_category → count

    for i in range(len(noisy_mag)):
        vlen = int(valid_length[i])
        mfile = str(mixed_files[i])
        meta = splits_meta.get(mfile, {"noise_category": "unknown", "target_snr_db": "?"})
        noise_cat = meta["noise_category"]
        snr_db    = meta["target_snr_db"]

        # Reconstruct enhanced audio
        _, noisy_phase = F.magnitude_phase(noisy_wav[i])          # [257, 126]
        enhanced_mag   = masks[i] * noisy_mag[i]                  # [257, 126]
        enhanced_wav   = F.reconstruct(enhanced_mag, noisy_phase, vlen)  # (vlen,)

        clean_ref  = clean_wav[i][:vlen].astype(np.float32)
        noisy_in   = noisy_wav[i][:vlen].astype(np.float32)
        enhanced_t = enhanced_wav.astype(np.float32)

        noisy_pesq   = safe_pesq(clean_ref, noisy_in)
        enh_pesq     = safe_pesq(clean_ref, enhanced_t)
        noisy_stoi   = safe_stoi(clean_ref, noisy_in)
        enh_stoi     = safe_stoi(clean_ref, enhanced_t)

        rows.append({
            "segment":       i,
            "mixed_file":    mfile,
            "noise_category": noise_cat,
            "snr_db":        snr_db,
            "valid_length":  vlen,
            "noisy_pesq":    noisy_pesq,
            "enhanced_pesq": enh_pesq,
            "noisy_stoi":    noisy_stoi,
            "enhanced_stoi": enh_stoi,
        })

        # Save example WAV files (a few per noise type)
        count = saved_examples.get(noise_cat, 0)
        if count < args.num_examples:
            prefix = f"{noise_cat}_snr{snr_db}dB_seg{i}"
            sf.write(example_dir / f"{prefix}_noisy.wav",    noisy_in,   F.TARGET_SR, subtype="FLOAT")
            sf.write(example_dir / f"{prefix}_enhanced.wav", enhanced_t, F.TARGET_SR, subtype="FLOAT")
            sf.write(example_dir / f"{prefix}_clean.wav",    clean_ref,  F.TARGET_SR, subtype="FLOAT")
            saved_examples[noise_cat] = count + 1

        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(noisy_mag)} segments...")

    # --------------------------------------------------------
    # Save per-segment CSV
    # --------------------------------------------------------

    results_csv = output_dir / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    def nanmean(vals):
        v = [x for x in vals if not (x != x)]  # exclude NaN
        return sum(v) / len(v) if v else float("nan")

    def fmt(v):
        return f"{v:.3f}" if v == v else "n/a"

    lines = []

    def section(header, subset):
        n_noisy_pesq   = nanmean([r["noisy_pesq"]   for r in subset])
        n_enh_pesq     = nanmean([r["enhanced_pesq"] for r in subset])
        n_noisy_stoi   = nanmean([r["noisy_stoi"]   for r in subset])
        n_enh_stoi     = nanmean([r["enhanced_stoi"] for r in subset])
        lines.append(f"{header}")
        lines.append(f"  N segments:     {len(subset)}")
        lines.append(f"  PESQ  noisy:    {fmt(n_noisy_pesq)}  |  enhanced: {fmt(n_enh_pesq)}  "
                     f"(delta {'+' if n_enh_pesq >= n_noisy_pesq else ''}{fmt(n_enh_pesq - n_noisy_pesq)})")
        lines.append(f"  STOI  noisy:    {fmt(n_noisy_stoi)}  |  enhanced: {fmt(n_enh_stoi)}  "
                     f"(delta {'+' if n_enh_stoi >= n_noisy_stoi else ''}{fmt(n_enh_stoi - n_noisy_stoi)})")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"GRU DENOISER — {args.split.upper()} SET EVALUATION")
    lines.append("=" * 60)
    lines.append("")
    section("Overall", rows)

    noise_cats = sorted({r["noise_category"] for r in rows})
    lines.append("--- By noise type ---")
    lines.append("")
    for cat in noise_cats:
        subset = [r for r in rows if r["noise_category"] == cat]
        section(cat, subset)

    snr_levels = sorted({r["snr_db"] for r in rows}, key=lambda x: float(x) if x != "?" else 99)
    lines.append("--- By SNR ---")
    lines.append("")
    for snr in snr_levels:
        subset = [r for r in rows if r["snr_db"] == snr]
        section(f"SNR {snr} dB", subset)

    lines.append("Note: copy-machine category has very few mixtures (~15).")
    lines.append("      Its averages have high variance — interpret with care.")
    lines.append("")
    lines.append(f"Results CSV:   {results_csv}")
    lines.append(f"Example WAVs:  {example_dir}")

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
