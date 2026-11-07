# Advanced Model: Architecture Decision and Design

**Status:** Design — awaiting team review before implementation
**Ticket:** [#15 Research and select the advanced model architecture](https://github.com/Break-Through-Tech/MathWorks-2A-speech-background-noise-suppression-with-deep-learning/issues/15)
**Milestone:** November — Evaluation & Delivery
**Baseline to beat:** GRU v2 — PESQ 2.293 (+0.216 vs noisy), STOI 0.892 (+0.002 vs noisy)

---

## 1. Decision Summary

**Selected architecture: CRN-Lite** (Convolutional Recurrent Network, lightweight variant)

Complex U-Net was considered and rejected. The reasoning is in section 3.

| Decision | Value |
|---|---|
| Architecture | CRN-Lite: CNN encoder → GRU bottleneck → CNN decoder with skip connections |
| Prediction target | Ideal ratio mask (IRM), same as GRU v2 |
| Loss function | IRM target loss (MSE between predicted mask and IRM), same as GRU v2 |
| Input | Normalized log-magnitude `[B, T, 257]`, same format as GRU |
| Output | Mask `[B, T, 257]`, values in (0, 1), same format as GRU |
| Estimated parameters | ~1.5–1.8 M (to be confirmed at implementation; tune down if val loss diverges) |
| Reconstruction | Same as GRU: `mask × noisy_mag` + noisy phase → iSTFT |
| Data pipeline | No changes — uses the same `.npz` files and `norm_stats.npz` |
| Evaluation | Same scripts as GRU (or a lightly adapted version); results go in `docs/model_comparison.md` |

---

## 2. What CRN-Lite is and why it improves on the GRU

The GRU baseline treats each frequency frame as an independent 257-dimensional vector and passes them through a recurrent layer one frame at a time. It has no way to look at a local neighborhood of time-frequency bins — if 10 adjacent frequency bins are all noise, the GRU treats each one in isolation.

CRN-Lite adds a convolutional encoder before the GRU and a convolutional decoder after it. The encoder compresses the spectrogram into a compact representation that captures local patterns across both time and frequency. The GRU then processes this representation sequentially, keeping the temporal modeling that worked in our baseline. The decoder uncompresses the representation back to full resolution, with skip connections that pass fine-grained details from the encoder directly to the decoder at each scale.

The result: the model can recognize noise textures (like the spread of a door slam across a frequency band, or the comb pattern of office typing), not just frame-by-frame loudness, while still tracking how those patterns change over time.

**What CRN-Lite does not change:**
- The magnitude-only approach — it still uses noisy phase for reconstruction, so the phase limitation from the GRU carries over
- The IRM target — same well-understood prediction objective
- The existing preprocessing and evaluation pipeline

---

## 3. Why not Complex U-Net

The ticket asks to select Complex U-Net if practical. After weighing the factors below, it is not practical for this project.

### What Complex U-Net (DCUNet) is

DCUNet operates on the full complex spectrogram (both real and imaginary parts) rather than just the magnitude. It predicts a Complex Ratio Mask (CRM) that modifies both magnitude and phase, which eliminates the noisy-phase limitation of the GRU baseline. It uses an encoder-decoder with skip connections (U-Net structure) with complex-valued convolutions at each layer.

### Why it is not selected

**Dataset size is the primary blocker.** DCUNet and its variants were developed and validated on the full Microsoft DNS Challenge dataset, which contains 60,000–500,000+ training mixtures depending on the edition. Our training set has 1,292 two-second segments from 7 speakers — roughly two to three orders of magnitude smaller. The original DCUNet-10 paper reports results with 10,000+ training files. At our scale, a ~780K–2M parameter model with no prior on magnitude vs. phase has very high overfitting risk, especially without the data augmentation infrastructure (on-the-fly SNR randomization, speed perturbation, noise mixing) that larger training sets implicitly provide.

The GRU v2 — 856K parameters, same dataset — already showed marginal STOI improvement (+0.002) and modest PESQ improvement (+0.216). There is not enough data to expect a substantially larger or more complex model to generalize better without additional data augmentation work that is out of scope for this ticket.

**Implementation complexity is high relative to expected benefit.** PyTorch does not have native complex convolution layers. Implementing DCUNet requires either:
- A custom complex conv module (wrapping two real-valued convolutions and tracking the cross terms), or
- An external library that is not already in `requirements.txt`.

This is non-trivial to implement correctly, debug, and validate, and adds a week or more of implementation overhead in November — the evaluation and delivery month. CRN-Lite, by contrast, uses standard `nn.Conv2d` and `nn.ConvTranspose2d`.

**The phase benefit is real but secondary.** The main weakness the GRU baseline leaves on the table is not noisy phase but local time-frequency feature extraction. The SNR-by-SNR results show the GRU already improves PESQ at every SNR level. The STOI dip at high SNR (−0.010 to −0.021) is a mild over-processing artifact that better local T-F features can address without complex math.

A revisit of Complex U-Net would be warranted if: (a) the team adds substantial data augmentation, (b) the GRU or CRN-Lite plateau without phase improvements, or (c) the project timeline extends.

---

## 4. Architecture detail

### 4.1 Input / output

| Property | Value |
|---|---|
| Input format | `[B, T, 257]` normalized log-magnitude (same as GRU) |
| Output format | `[B, T, 257]` predicted IRM mask, sigmoid activated, values in (0, 1) |
| Reconstruction | identical to GRU: `mask × noisy_mag` + noisy phase → iSTFT |

Internally the model reshapes the input to `[B, 1, 257, T]` for 2D convolution (batch, channel, freq, time). The GRU processes the bottleneck representation along the time axis.

### 4.2 Architecture diagram

```
input [B, T, 257]
  │
  └─ reshape → [B, 1, 257, T]
                    │
         ┌──────────▼──────────┐
         │      ENCODER        │
         │                     │
         │  E1: Conv2d         │ [B, 1, 257, T]  → [B, 16, 129, T]  skip s1
         │  E2: Conv2d         │ [B, 16, 129, T] → [B, 32,  65, T]  skip s2
         │  E3: Conv2d         │ [B, 32,  65, T] → [B, 64,  33, T]  skip s3
         │  E4: Conv2d         │ [B, 64,  33, T] → [B, 64,  17, T]  skip s4
         └──────────┬──────────┘
                    │ flatten freq×ch: [B, T, 1088]
         ┌──────────▼──────────┐
         │   GRU BOTTLENECK    │
         │  GRU(1088, 256,     │
         │        1 layer)     │  [B, T, 256]
         └──────────┬──────────┘
                    │ project back: Linear(256, 1088) → [B, T, 1088]
                    │ reshape: [B, 64, 17, T]
         ┌──────────▼──────────┐
         │      DECODER        │
         │                     │
         │  D4: ConvT2d + s4   │ [B, 128, 17, T] → [B, 64,  33, T]
         │  D3: ConvT2d + s3   │ [B, 128, 33, T] → [B, 32,  65, T]
         │  D2: ConvT2d + s2   │ [B,  64, 65, T] → [B, 16, 129, T]
         │  D1: ConvT2d + s1   │ [B,  32,129, T] → [B,  1, 257, T]
         └──────────┬──────────┘
                    │ Sigmoid
                    │ squeeze channel → [B, 257, T]
                    │ permute → [B, T, 257]
                  mask

skip connections: s_k = encoder layer k output, concatenated channel-wise in decoder
```

### 4.3 Layer-by-layer spec

**Encoder** — stride on frequency only; time resolution is preserved throughout.

| Layer | Op | In channels | Out channels | Freq in | Freq out | Kernel | Stride | Padding | After |
|---|---|---:|---:|---:|---:|---|---|---|---|
| E1 | Conv2d | 1 | 16 | 257 | 129 | (3,3) | (2,1) | (1,1) | BatchNorm + ELU |
| E2 | Conv2d | 16 | 32 | 129 | 65 | (3,3) | (2,1) | (1,1) | BatchNorm + ELU |
| E3 | Conv2d | 32 | 64 | 65 | 33 | (3,3) | (2,1) | (1,1) | BatchNorm + ELU |
| E4 | Conv2d | 64 | 64 | 33 | 17 | (3,3) | (2,1) | (1,1) | BatchNorm + ELU |

Frequency formula: `ceil((F + 2p - k) / s) + 1`

**Bottleneck**

| Step | Op | In | Out |
|---|---|---|---|
| Flatten | reshape | [B, 64, 17, T] | [B, T, 1088] |
| GRU | nn.GRU(1088, 256, 1 layer) | [B, T, 1088] | [B, T, 256] |
| Project | Linear(256, 1088) | [B, T, 256] | [B, T, 1088] |
| Unflatten | reshape | [B, T, 1088] | [B, 64, 17, T] |

**Decoder** — each layer concatenates the matching encoder skip connection before the ConvTranspose.

| Layer | Op | In (after concat) | Out channels | Freq in | Freq out | Kernel | Stride | Padding | Out padding | After |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| D4 | ConvTranspose2d | 128 | 64 | 17 | 33 | (3,3) | (2,1) | (1,1) | (1,0) | BatchNorm + ELU |
| D3 | ConvTranspose2d | 128 | 32 | 33 | 65 | (3,3) | (2,1) | (1,1) | (1,0) | BatchNorm + ELU |
| D2 | ConvTranspose2d | 64 | 16 | 65 | 129 | (3,3) | (2,1) | (1,1) | (1,0) | BatchNorm + ELU |
| D1 | ConvTranspose2d | 32 | 1 | 129 | 257 | (3,3) | (2,1) | (1,1) | (1,0) | Sigmoid |

Frequency transpose formula: `(F - 1) × s - 2p + k + out_padding`
D4: (17−1)×2 − 2×1 + 3 + 1 = 32 − 2 + 3 + 1 = **34** ... need to verify this in code and adjust `output_padding` if the shapes are off. The shapes must be an exact mirror of the encoder; a brief shape-check forward pass before full training is strongly recommended.

### 4.4 Approximate parameter count

| Component | Approximate params |
|---|---:|
| Encoder (E1–E4) | ~63,000 |
| GRU bottleneck (1 layer, h=256) | ~1,030,000 |
| Linear projection (256→1088) | ~280,000 |
| Decoder (D1–D4) | ~125,000 |
| BatchNorm layers | ~5,000 |
| **Total** | **~1.5 M** |

This is roughly 1.75× the GRU baseline (856K). It is still well within what can train on a free-tier Colab GPU in a few hours.

**If validation loss diverges or the model overfits:** reduce GRU hidden size from 256 → 128 (saves ~550K params, total ~960K — smaller than the GRU baseline) and re-run. The encoder/decoder convolutions are a small fraction of the total, so the GRU hidden size is the main knob.

---

## 5. Prediction target and loss function

Both are unchanged from GRU v2.

**Target:** Ideal Ratio Mask (IRM), computed per T-F bin from the training pair:
```python
IRM = clip(clean_mag / (noisy_mag + 1e-8), 0.0, 1.0)
```

**Loss:** masked MSE between the predicted mask and IRM:
```
loss = mean over real frames of (predicted_mask − IRM)²
```
`frame_mask` from the `.npz` file excludes zero-padded frames at the end of short clips, same as GRU v2.

**Why keep IRM loss?** GRU v1 (magnitude MSE) failed at high SNR. GRU v2 (IRM loss) fixed that by giving the model an explicit per-bin target that is near 1 at high SNR. That signal is as useful for CRN-Lite as it was for the GRU.

---

## 6. Training configuration

Initial hyperparameters follow the GRU v2 recipe; adjust based on validation loss curves.

| Hyperparameter | Initial value | Notes |
|---|---|---|
| Optimizer | Adam | β=(0.9, 0.999) |
| Learning rate | 1e-3 | ReduceLROnPlateau ×0.5, patience 3 |
| Batch size | 16 | Halved from GRU (larger model per sample; increase to 32 if Colab VRAM allows) |
| Max epochs | 60 | +10 over GRU to account for more parameters to converge |
| Early stopping | patience 10 | Slightly more patience than GRU (more capacity to explore) |
| Gradient clip | 5.0 (global norm) | Same as GRU |
| Dropout | 0.0 (encoder/decoder) | BatchNorm provides regularization; GRU inter-layer dropout disabled (1 layer) |
| Weight decay | 1e-5 | Added to Adam; can drop to 0 if training is stable |
| Random seed | 42 | Same as GRU for reproducibility |
| Checkpoint | best val IRM loss | Save to `models/crn_best.pt` |

**Suggested first tuning experiments** (one at a time, using validation IRM loss to decide):
1. Learning rate: 3e-4 vs 1e-3
2. GRU hidden size: 128 vs 256 (if overfitting is observed)
3. Batch size: 16 vs 32 (memory permitting)

---

## 7. Integration with the existing pipeline

CRN-Lite slots into the pipeline with minimal changes.

**Files to add:**
- `models/crn.py` — model definition (CRNDenoiser class)
- `scripts/train_crn.py` — training script (adapted from `train_gru.py`)
- `scripts/evaluate_crn.py` — evaluation script (adapted from `evaluate_gru.py`)

**Files that do not change:**
- `scripts/features.py` — same STFT, iSTFT, normalization, IRM
- `data/processed/features/*.npz` — same training/val/test split, same feature format
- `data/processed/features/norm_stats.npz` — same per-frequency normalization stats
- `docs/model_comparison.md` — add a row when test evaluation is complete

**Input format is identical to the GRU:** normalized log-magnitude `[B, T, 257]`. The model internally reshapes to `[B, 1, 257, T]` for convolution, then reshapes the output back to `[B, T, 257]` before returning. The training script sees the same tensor shapes as `train_gru.py`.

**Reconstruction is identical:** `mask × noisy_mag` + noisy phase → iSTFT via `features.reconstruct()`.

**Evaluation is identical:** PESQ (wideband 16 kHz) and STOI on the 172 test segments (speaker 04408), same clean reference files. The only change in `evaluate_crn.py` is loading `CRNDenoiser` instead of `GRUDenoiser`.

---

## 8. Expected performance and what to watch for

There are no published CRN results on our exact dataset, so these are directional estimates based on what has been reported in the literature for similar-scale tasks.

**Hopeful scenario:** PESQ improves by 0.1–0.3 beyond GRU v2 (reaching ~2.4–2.6), and the STOI dip at high SNR either disappears or narrows. The convolutional layers' ability to recognize local noise patterns should help at mid and high SNR where GRU v2 already works, and may help at low SNR where the noise is more complex.

**Concerning signals during training:**
- Validation IRM loss does not improve below GRU v2's 0.1124 by epoch ~20: check the learning rate, reduce model size (GRU hidden to 128), or add weight decay
- Training IRM loss much lower than val IRM loss (overfitting): reduce model size or add dropout (0.1) to BatchNorm layers
- Shape errors in the decoder: ConvTranspose2d output sizes may be off by 1; check with a dry-run forward pass on a dummy batch before full training
- NaN loss early in training: reduce learning rate to 3e-4 and check gradient norms

**Limitation CRN-Lite does not fix:** noisy phase reuse. The model still reconstructs audio by recombining the predicted mask with the noisy input's phase. The phase limitation noted in the GRU baseline applies here as well, most visibly at −5 and 0 dB SNR. This is an intentional scope boundary to keep the implementation manageable in the November timeline.

---

## 9. Open questions for the team

- [ ] Who is implementing `models/crn.py` and `scripts/train_crn.py`?
- [ ] Should we run a shape-check unit test before full training to confirm the encoder/decoder dimensions are symmetric?
- [ ] If CRN-Lite underperforms the GRU at any SNR, do we iterate on hyperparameters or move to evaluation with both models?
- [ ] Do we want to add on-the-fly SNR augmentation (randomly remixing clean + noise at training time) before training the advanced model? This would help with the small dataset but requires changes to the data loader.
