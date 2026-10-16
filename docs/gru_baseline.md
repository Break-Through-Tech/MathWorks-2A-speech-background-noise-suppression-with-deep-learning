# GRU Baseline Model

**Status:** Training and evaluation complete. GRU v2 (IRM loss) beats the noisy baseline.  
**Milestone:** October — Model Design & Training  
**Spec:** [`docs/baseline_model.md`](baseline_model.md) | **Results:** [`docs/model_comparison.md`](model_comparison.md)

---

## What this is

The GRU baseline is a 2-layer recurrent neural network that suppresses background noise from speech recordings. It operates entirely in the frequency domain: it takes the magnitude spectrogram of a noisy recording, predicts a mask that separates speech from noise, and uses that mask to reconstruct cleaner audio.

It exists to set a reproducible performance floor before trying more complex architectures.

---

## Data flow

This diagram shows the complete speech-enhancement pipeline — what happens to an audio recording from the time the noisy mixture is created until an enhanced WAV is produced.

```
clean speech ─────────┐
                       ├─── create_mixtures.py ──► noisy/mixed WAV
background noise ──────┘                               │
                                                      STFT
                                                       │
                               ┌───────────────────────┴──────────────────────┐
                          magnitude [257×126]                           phase [257×126]
                      (strength of each frequency)                (timing / alignment)
                               │                                              │
                     GRU modifies this side                         saved; reused later
                               │                                              │
                    log + per-frequency z-score                               │
                               │                                              │
                      GRU [2 layers, 256 hidden]                              │
                               │                                              │
                     predicted mask [257×126]                                 │
                       values in (0, 1)                                       │
                               │                                              │
                   mask × noisy magnitude                                     │
                   = enhanced magnitude ──────────────────────────────────────┘
                                                       │
                                          enhanced magnitude + noisy phase
                                                       │
                                                     iSTFT
                                                       │
                                              enhanced speech WAV
```

**What each stage does:**

- **Mixture creation:** clean speech and a standalone background-noise recording are summed at a target SNR. The result — the noisy/mixed WAV — is what the model processes. The raw noise recordings are not used anywhere else in the pipeline.
- **STFT** (Short-Time Fourier Transform): converts the 1-D waveform into a 2-D time-frequency representation. It produces two arrays of the same shape — *magnitude* (how strong each frequency is at each moment) and *phase* (timing and alignment information). Both have shape `[257 freq bins × 126 time frames]` for a 2-second segment.
- **Phase is saved, not modified.** The GRU operates only on magnitude. Phase is extracted from the noisy input right after the STFT and held aside; it is reused unchanged at the final reconstruction step. Reusing noisy phase is a deliberate simplifying choice for this baseline: predicting accurate phase is substantially harder, and reusing noisy phase is sufficient to produce a playable waveform. It is also a limitation — at low SNR the noisy phase carries phase noise that even a perfect magnitude mask cannot remove.
- **Log + normalize:** log-magnitude compresses the dynamic range; per-frequency z-score normalization (statistics computed from the training set only) ensures no single frequency bin dominates the GRU input.
- **GRU:** processes the normalized log-magnitude spectrogram one 16 ms frame at a time, left to right, and outputs a predicted enhancement mask — values in (0, 1) for each time-frequency bin. A value near 1 means "keep this bin"; near 0 means "suppress it."
- **Mask application:** `mask × noisy magnitude` scales each bin. Bins judged as mostly speech are kept; bins judged as mostly noise are attenuated.
- **iSTFT + noisy phase reuse:** the enhanced magnitude is combined with the *original noisy phase* (held aside since the STFT step) and passed through the inverse STFT to reconstruct a time-domain waveform.

**Terminology note:** "noisy" and "mixed" mean the same thing throughout this project — a recording of clean speech with one background-noise source added to it. The model always receives the mixed recording as input. The standalone noise recordings (door, typing, etc.) appear only during dataset construction; they are never fed to the model.

The clean speech is available during **training** (as a target for computing the loss) but not during **inference** — the model must predict the mask from the noisy input alone.

---

## Files

### `scripts/create_mixtures.py`

Mixes clean speech and background-noise recordings at various signal-to-noise ratios. Outputs `data/mixed/*.wav` and `data/mixed/metadata.csv` (one row per noisy clip with SNR, noise type, file paths). This step runs once before anything else.

### `scripts/features.py`

Shared audio processing utilities: STFT/iSTFT, log magnitude, normalization, segmentation, phase reconstruction, the IRM formula. Everything is plain NumPy so it works independently of the training framework.

Called by `build_features.py`, `train_gru.py`, and `evaluate_gru.py`.

### `scripts/build_features.py`

Reads the mixed WAVs and clean WAVs, applies the speaker-based train/val/test split, segments into 2-second clips, computes spectrograms, and saves everything to `.npz` files:

```
data/processed/features/
    train.npz          1,292 segments (readers: 00986, 02277, 05983, 07176, 07326, 08465, 08797)
    val.npz            184 segments   (reader: 08897)
    test.npz           172 segments   (reader: 04408)
    norm_stats.npz     per-frequency mean and std from train.npz only
```

Each `.npz` contains:
- `noisy_mag`    — mixed spectrogram magnitude `[N, 257, 126]`
- `clean_mag`    — clean spectrogram magnitude `[N, 257, 126]`
- `noisy_wav`    — mixed waveform `[N, 32000]`
- `clean_wav`    — clean waveform `[N, 32000]`
- `frame_mask`   — boolean `[N, 126]`, False for zero-padded frames at the end of a clip
- `valid_length` — number of real samples in each segment `[N]`
- `mixed_file`   — source filename for each segment `[N]`

The normalization statistics in `norm_stats.npz` are computed from the **training set only**. The same stats are applied to val and test at inference time.

### `models/gru.py`

Defines the neural-network architecture only. It does not load data or call any optimizer. Given an input tensor of shape `[B, T, 257]` (batch × time frames × frequency bins), it outputs a predicted mask `[B, T, 257]` with values in (0, 1).

```python
class GRUDenoiser(nn.Module):
    def __init__(self, input_size=257, hidden_size=256, num_layers=2, dropout=0.2):
        ...
    def forward(self, x):   # x: [B, T, 257] → mask: [B, T, 257]
        out, _ = self.gru(x)
        out = self.fc(out)
        return torch.sigmoid(out)
```

### `scripts/train_gru.py`

Loads the `.npz` files, creates `SpeechDataset` and `DataLoader` objects, instantiates `GRUDenoiser`, and runs the training loop.

The data-flow diagram above describes what the enhancement system does at any point in time. The diagram below zooms into the **GRU training process**: how the GRU learns to predict the mask from noisy input.

**Key distinction:** during training, the code loads both the noisy spectrogram and the matching clean spectrogram from the `.npz` file. The clean spectrogram acts like an *answer key* — it is used to compute how wrong the GRU's prediction was and to calculate the gradients that update the weights. **The clean spectrogram is NOT fed into the GRU as an input.** The GRU's input is always and only the noisy/mixed spectrogram.

The diagram below shows the GRU v1 (magnitude MSE) training path. GRU v2 uses a different loss — see the Training section below for details.

```
    noisy_mag ──► log + normalize ──► GRU input x  [B, 126, 257]
         │                                  │
    (the GRU sees                       model(x)
     only the noisy                         │
     spectrogram)                  predicted mask [B, 126, 257]
                                            │
                        mask × noisy_mag = enhanced_mag  [B, 257, 126]
                                            │
    clean_mag ──────────────────────────────┘
    (answer key:                            │
     NEVER fed                 (enhanced_mag - clean_mag)²
     into the GRU)                          │
                                       masked MSE loss  ← GRU v1
                                            │
                                     backpropagation
                           (computes how each weight contributed
                            to the error, producing a gradient)
                                            │
                               Adam optimizer updates GRU weights
                               (uses the gradient to nudge each
                                weight in the loss-reducing direction)
```

**During inference (test time) there is no answer key:**
- No clean reference is available or needed.
- No loss is calculated.
- No backpropagation is run.
- The GRU weights are frozen.
- The GRU receives only the noisy/mixed spectrogram and predicts a mask.

**Checkpoint naming:** the output filename is configurable via `--checkpoint-name`. By convention in this project:
- `models/gru_best.pt` — GRU v1 (magnitude MSE loss)
- `models/gru_v2.pt` — GRU v2 (IRM loss) — the currently selected GRU baseline

A checkpoint contains the model weights, optimizer state, epoch number, val loss, and training arguments. It is the only artifact needed to run inference.

### `scripts/evaluate_gru.py`

Loads the model architecture from `models/gru.py`, weights from the checkpoint, and test data from `test.npz`. Reconstructs enhanced audio and computes STOI/PESQ against the clean reference. Outputs:

- `outputs/evaluation/results.csv` — per-segment scores
- `outputs/evaluation/summary.txt` — overall + breakdown by noise type and SNR
- `outputs/evaluation/examples/` — a few noisy/enhanced/clean WAV triples per noise type

Use `--split val` to run on the validation set instead (e.g., for comparing checkpoints without touching the test set).

---

## Architecture

| Property | Value |
|---|---|
| Architecture | 2-layer unidirectional GRU + Linear + Sigmoid |
| Parameters | 856,321 (~0.86 M, ~3.4 MB float32) |
| Input | Normalized log-magnitude `[B, 126, 257]` (batch × time × freq) |
| Output | Predicted enhancement mask `[B, 126, 257]`, values in (0, 1) |
| GRU hidden size | 256 |
| Inter-layer dropout | 0.2 |

The GRU processes the spectrogram one 16 ms frame at a time, left to right. Each frame can attend to all previous frames through the hidden state. The unidirectional design means the same weights could run in real time without buffering future context.

The sigmoid output constrains the mask to (0, 1). When the mask is 1 at a frequency bin, that bin passes through unchanged. When it is 0, that bin is zeroed out.

**GRU v1 and GRU v2 use this same architecture.** The only difference is the training objective:

- **GRU v1 (magnitude MSE):** trains the mask *indirectly*. The loss measures how well `mask × noisy magnitude` approximates `clean magnitude`. The individual mask values are not supervised; the model discovers which mask produces an accurate enhanced magnitude.
- **GRU v2 (IRM loss):** trains the mask *directly* against a per-bin target. For each training example the code computes an Ideal Ratio Mask (IRM) from the clean/noisy pair and trains the GRU to match it. The IRM serves as a per-bin answer key for what value the mask should have. The GRU input remains only the noisy spectrogram; the IRM exists only inside the loss computation and is not available at inference time.

---

## Training

**Run from the repo root:**

```bash
# Full training (saves to models/gru_best.pt)
python scripts/train_gru.py

# Quick smoke test (2 epochs, 64 samples)
python scripts/train_gru.py --max-epochs 2 --subset 64 --no-val-metrics
```

**Key settings (defaults):**

| Setting | Value | Note |
|---|---|---|
| Loss | Magnitude MSE | `(mask × noisy − clean)²`, averaged over real frames |
| Optimizer | Adam | lr 1e-3 |
| LR schedule | ReduceLROnPlateau | ×0.5 when val loss stalls for 3 epochs |
| Early stopping | patience 8 | stops when val loss has not improved for 8 epochs |
| Gradient clip | global norm 5.0 | prevents training instability |
| Batch size | 32 | |
| Max epochs | 50 | |

**What validation is checking:** validation loss uses the same formula as training loss but on the validation set (speaker 08897). The model's weights are not updated on validation data. The checkpoint is saved whenever the validation loss improves, so the checkpoint always corresponds to the best-seen validation performance, not the final epoch.

**After training completes,** the script computes STOI and PESQ on the full validation set using the best checkpoint, and appends an experiment record to `outputs/experiments/log.csv`.

**Loss functions** are selected via `--loss`:

*GRU v1 — magnitude MSE (default):*
```
loss = mean over real frames of (mask × noisy_mag − clean_mag)²
```
The model learns indirectly: it discovers which mask produces an enhanced magnitude close to the clean magnitude.

*GRU v2 — IRM target loss (`--loss irm`):*

The training code first computes the Ideal Ratio Mask (IRM) for each T-F bin using the clean/noisy training pair. From `scripts/features.py`:
```python
IRM = clip(clean_mag / (noisy_mag + 1e-8), 0.0, 1.0)
```
The `1e-8` floor prevents division by zero. The clip to `[0, 1]` is essential: if `clean_mag > noisy_mag` at some bin (which can happen due to mixture clipping or phase interactions), the raw ratio would exceed 1 — but the GRU's sigmoid output is bounded to (0, 1), so the target must also be in (0, 1). Clipping makes the target reachable.

Then:
```
loss = mean over real frames of (predicted_mask − IRM)²
```
The GRU is trained to predict the ratio directly. At high SNR, where speech dominates over noise, the IRM is close to 1 — explicitly telling the model "leave this bin alone." This is why v2 does not over-suppress at high SNR.

```bash
# GRU v1 (default, magnitude MSE)
python scripts/train_gru.py --checkpoint-name gru_best.pt --exp-name gru_v1

# GRU v2: IRM target loss (selected model)
python scripts/train_gru.py --loss irm --checkpoint-name gru_v2.pt --exp-name gru_v2

# GRU v3: log-magnitude loss (not run; v2 sufficient)
python scripts/train_gru.py --loss log_magnitude --checkpoint-name gru_v3.pt --exp-name gru_v3
```

---

## Evaluation

```bash
# Evaluate selected model (GRU v2) on the held-out test set
python scripts/evaluate_gru.py --checkpoint models/gru_v2.pt

# Evaluate GRU v1 (for comparison)
python scripts/evaluate_gru.py --checkpoint models/gru_best.pt --output-dir outputs/evaluation_v1

# Evaluate on validation set (for comparing checkpoints without touching test)
python scripts/evaluate_gru.py --split val --checkpoint models/gru_v2.pt
```

Requirements: `pip install pesq pystoi`

---

## What STOI and PESQ mean

Both metrics compare a processed signal against a clean reference.

**STOI** (Short-Time Objective Intelligibility) estimates how intelligible the speech is. It is bounded between 0 and 1, with 1 being most intelligible. It should not be read as "X% of words will be understood" — the absolute value depends on the conditions. What matters is the **delta**: a model that increases STOI is preserving or improving intelligibility; one that decreases it is degrading it.

**PESQ** (Perceptual Evaluation of Speech Quality) estimates perceived quality against the clean reference. In wideband mode (which this project uses), scores span roughly 1 to 4.6. Higher is better. Again, the delta is what matters for comparing a model to the noisy baseline.

Both metrics are computed by comparing:
- The **original mixed/noisy** audio against the clean reference → this is the score to beat
- The **enhanced** audio against the same clean reference → this is what the model achieves

A model is useful if its enhanced scores are higher than the noisy baseline. A lower score means the model made things worse.

---

## Results

### GRU v1 (magnitude MSE loss) — confirmed

| Split | N segments | Noisy PESQ | Enhanced PESQ | PESQ Δ | Noisy STOI | Enhanced STOI | STOI Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | 184 | 1.580 | 1.525 | −0.055 | 0.839 | 0.793 | −0.046 |
| **Test** | 172 | 2.077 | 1.915 | −0.162 | 0.890 | 0.862 | −0.027 |

**Training:** 42 epochs, best val MSE at epoch 34 (0.1437), early stopped.

**Breakdown by SNR (test set):**

| SNR | Noisy PESQ | Enhanced PESQ | Δ PESQ | Noisy STOI | Enhanced STOI | Δ STOI |
|---|---:|---:|---:|---:|---:|---:|
| −5 dB | 1.207 | 1.281 | **+0.074** | 0.710 | 0.703 | −0.006 |
| 0 dB | 1.320 | 1.440 | **+0.120** | 0.834 | 0.824 | −0.010 |
| 5 dB | 1.551 | 1.598 | **+0.047** | 0.871 | 0.871 | +0.001 |
| 10 dB | 2.260 | 2.191 | −0.070 | 0.962 | 0.926 | −0.036 |
| 15 dB | 2.567 | 2.226 | −0.342 | 0.968 | 0.922 | −0.046 |
| 20 dB | 2.876 | 2.411 | −0.465 | 0.954 | 0.914 | −0.040 |

**What this means:** GRU v1 learned to suppress noise and improved quality at low SNR (−5 to 5 dB), which is where suppression is genuinely needed. At high SNR (10–20 dB), the noisy input was already quite clean, and the model degraded it — a classic over-suppression pattern. This drove the overall average negative because the test set contains many high-SNR segments (84/172 are ≥15 dB).

**Why it happens:** The magnitude-domain MSE loss (`(mask × noisy − clean)²`) does not distinguish high-SNR from low-SNR T-F bins. High-amplitude speech bins dominate the loss regardless of SNR. The model learned a general suppression strategy that works at average SNR, but cannot tell when to leave the signal alone. The IRM loss (v2) trains the model to predict the ratio `clean/noisy` directly, which is close to 1 at high SNR — fixing this.

---

### GRU v2 (IRM target loss) — selected model

| Split | N segments | Noisy PESQ | Enhanced PESQ | PESQ Δ | Noisy STOI | Enhanced STOI | STOI Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Validation** | 184 | 1.580 | 1.734 | **+0.154** | 0.839 | 0.840 | **+0.001** |
| **Test** | 172 | 2.077 | 2.293 | **+0.216** | 0.890 | 0.892 | **+0.002** |

**Training:** 26 epochs, best val IRM MSE at epoch 18 (0.1124), early stopped.

**Selection:** v2 was selected over v1 based on validation metrics alone — val PESQ Δ +0.154 vs v1's −0.055.

**Breakdown by SNR (test set):**

| SNR | Noisy PESQ | Enhanced PESQ | Δ PESQ | Noisy STOI | Enhanced STOI | Δ STOI |
|---|---:|---:|---:|---:|---:|---:|
| −5 dB | 1.207 | 1.340 | **+0.133** | 0.710 | 0.748 | **+0.038** |
| 0 dB | 1.320 | 1.606 | **+0.285** | 0.834 | 0.860 | **+0.026** |
| 5 dB | 1.551 | 1.712 | **+0.161** | 0.871 | 0.884 | **+0.013** |
| 10 dB | 2.260 | 2.640 | **+0.380** | 0.962 | 0.952 | −0.010 |
| 15 dB | 2.567 | 2.828 | **+0.261** | 0.968 | 0.951 | −0.016 |
| 20 dB | 2.876 | 2.991 | **+0.115** | 0.954 | 0.933 | −0.021 |

**What this means:** GRU v2 substantially improves PESQ at every tested SNR level, including the high-SNR range where v1 failed badly. For STOI, the picture is more nuanced: STOI improves at −5, 0, and +5 dB (where the model is actively suppressing real noise), but decreases slightly at 10, 15, and 20 dB (−0.010 to −0.021), where the noisy input was already highly intelligible. The overall STOI shift is 0.890 → 0.892 — a very small positive difference. No statistical significance analysis was performed, so this should be interpreted conservatively as approximately preserving intelligibility, not as a meaningful improvement. The primary result is that GRU v2 substantially improves perceptual quality (PESQ +0.216 vs the noisy baseline) while approximately preserving overall intelligibility (STOI +0.002). The selected GRU checkpoint is `models/gru_v2.pt`.

See [`docs/model_comparison.md`](model_comparison.md) and `outputs/experiments/log.csv` for the full comparison table.

---

## Limitations

- **Noisy phase reuse:** The reconstructed audio uses the noisy recording's phase. At low SNR, phase noise is substantial and limits quality gains — even a perfect mask cannot fully recover the original speech. This is a fundamental limitation of magnitude-only approaches.
- **Small dataset:** 1,292 training segments from 7 readers. The model generalises across speakers to some degree (val and test speakers are held out), but the dataset is small enough that further speakers and noise conditions would improve robustness.
- **Single-channel only:** No spatial information.
- **Magnitude MSE loss (v1 only):** Does not correlate well with perceived quality metrics at all SNR levels, as the v1 results demonstrate.

---

## Regenerating the processed data

The files under `data/processed/` are gitignored and must be regenerated after cloning:

```bash
# 1. Generate noisy/mixed WAVs (if not already done)
python scripts/create_mixtures.py

# 2. Segment and compute spectrograms
python scripts/build_features.py
```

The model checkpoints (`models/*.pt`) and evaluation outputs (`outputs/`) are also gitignored.

---

## Building on this baseline

The data pipeline (mixture generation, segmentation, STFT, normalization, reconstruction) and the evaluation methodology (speaker-based split, STOI/PESQ against the same clean references) are reusable. A teammate implementing the next architecture should:

1. Use the same `data/processed/features/*.npz` files as inputs — do not rebuild the split
2. Apply the same normalization from `norm_stats.npz`
3. Use `features.py` for STFT/iSTFT/reconstruction unless the architecture needs a different transform
4. Evaluate with `evaluate_gru.py` (or a version adapted for the new model's output format) so results are directly comparable to GRU
5. Add a row to `docs/model_comparison.md`

The architecture-specific pieces are:
- `models/gru.py` — replace or extend with the new model
- The loss function in `train_gru.py` — adapt to the new objective
- The checkpoint format — ensure `evaluate_gru.py` can load it, or write a parallel eval script

A follow-on model does not need to preserve the mask-based approach. The evaluation chain (feature files → inference → reconstruct with noisy phase → STOI/PESQ) remains the same regardless of whether the model predicts a mask, an enhanced spectrogram directly, or something else.
