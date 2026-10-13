# Baseline Model Design

**Status:** Draft for team review
**Milestone:** October: Model Design & Training
**Branch:** `feature/document-baseline`

This document defines the first (baseline) noise suppression model:
what goes in, what comes out, what it learns to predict, how it is
trained, and how we judge it. The goal of the baseline is not to be the
best model. It is a simple, reproducible reference point that every
later experiment is compared against.

---

## 1. Summary

| Decision | Baseline choice |
|---|---|
| Framework | PyTorch (Python) |
| Input | Normalized log-magnitude STFT of the noisy segment, `[257 × 126]` |
| Prediction target | Ideal ratio mask (IRM), values in [0, 1] |
| Output | Mask `[257 × 126]`, sigmoid activation |
| Reconstruction | `mask × noisy magnitude` + noisy phase → inverse STFT |
| Architecture | 2-layer unidirectional GRU (256 units) + linear + sigmoid |
| Model size | ~0.86 M parameters (~3.4 MB float32) |
| Loss | Masked MSE on magnitude (padded frames ignored) |
| Optimizer | Adam, lr 1e-3, ReduceLROnPlateau (×0.5, patience 3) |
| Batch / epochs | 32 / up to 50, early stopping (patience 8) on val loss |
| Metrics | PESQ (wideband) and STOI, noisy vs. denoised, on the test set |

### In plain words

1. We cut each noisy recording into 2-second pieces and turn each piece
   into a spectrogram: a picture of which frequencies are loud at each
   moment.
2. The model looks at that picture one 16 ms slice at a time and, for
   every frequency, outputs a number between 0 and 1: "how much of this
   is speech?" That grid of numbers is the **mask**.
3. We multiply the noisy spectrogram by the mask (1 keeps the sound, 0
   removes it) and turn the result back into audio.
4. We check whether it worked with two scores: **PESQ** (how good it
   sounds) and **STOI** (how easy the words are to understand).

### Glossary

| Term | Meaning |
|---|---|
| STFT / spectrogram | Splits audio into short overlapping frames and measures how loud each frequency is in each frame. |
| Frame | One 32 ms slice of audio; a new frame starts every 16 ms. 2 s → 126 frames. |
| Frequency bin | One of 257 frequency bands from 0 to 8 kHz. |
| Log-magnitude | Loudness on a log scale, closer to how we hear and easier for the model. |
| Mask / IRM | A 0–1 value per frame and bin saying how much of the sound to keep. IRM is the "perfect" mask computed from the clean audio. |
| GRU | A type of recurrent neural network layer that remembers what it saw in earlier frames. |
| Phase | The timing part of the STFT. We don't predict it; we reuse it from the noisy audio. |
| PESQ | Perceptual speech quality score, about −0.5 to 4.5. Higher is better. |
| STOI | Speech intelligibility score, 0 to 1. Higher is better. |
| SNR | Signal-to-noise ratio in dB. Lower (e.g. −5 dB) means louder noise. |

---

## 2. Data the model sees

Produced by `scripts/build_features.py` (see `notebooks/01_input_representation.ipynb`).
Rebuild with `python scripts/build_features.py`; outputs go to
`data/processed/` (not tracked in git).

| Split | Mixtures | 2 s segments | Speaker(s) |
|---|---|---|---|
| train | 533 | 1,292 | 7 readers |
| val | 50 | 184 | 08897 |
| test | 60 | 172 | 04408 |

- 646 mixtures total; 3 excluded (silent noise files).
- Noise types: door (283), typing (268), squeakyChair (80), copy-machine (15).
- SNRs: −5, 0, 5, 10, 15, 20 dB (~107 mixtures each).
- Split is by speaker so no voice appears in more than one split.

### Signal settings (`scripts/features.py`)

| Setting | Value |
|---|---|
| Sample rate | 16 kHz |
| STFT window | 512 samples (32 ms), periodic Hann |
| Hop | 256 samples (16 ms) |
| Frequency bins | 257 (0–8 kHz) |
| Segment length | 2 s = 32,000 samples → 126 frames |

Arrays in each `{split}.npz`: `noisy_mag`, `clean_mag` `[N, 257, 126]`,
`noisy_wav`, `clean_wav` `[N, 32000]`, `frame_mask` `[N, 126]` (True =
real audio, False = padding), `valid_length`, `start`, `mixed_file`.

---

## 3. Input / output structure

### Input

```
noisy_mag  [B, 257, 126]
  → log(noisy_mag + 1e-6)              # features.log_magnitude
  → (x − mean) / std                   # per-frequency, norm_stats.npz (train only)
  → transpose to [B, 126, 257]         # time-major for the GRU
```

Each time step is one 16 ms frame described by 257 frequency values.

### Output

```
mask  [B, 126, 257]  → transpose → [B, 257, 126], values in [0, 1]
```

### From mask to audio (inference)

```
enhanced_mag = mask × noisy_mag
enhanced_wav = istft(enhanced_mag, noisy_phase)   # features.reconstruct
```

The model never has to predict phase; it reuses the noisy phase.

---

## 4. Prediction target

**Ideal ratio mask (IRM)** (`features.ideal_ratio_mask`):

```
IRM = clip(clean_mag / (noisy_mag + eps), 0, 1)
```

Why a mask instead of predicting the clean spectrogram directly:

- Bounded output in [0, 1] → stable training with a sigmoid.
- The model only has to decide *how much to keep* at each time-frequency
  point, which is easier to learn from a small dataset.
- Standard, well-documented approach for lightweight speech enhancement
  (e.g., the MathWorks "Denoise Speech Using Deep Learning Networks" example).

Known limitation: reusing the noisy phase caps quality, especially at
low SNR (−5, 0 dB). Complex masks are a possible later improvement.

---

## 5. Loss function

**Masked MSE on the enhanced magnitude** (the "signal approximation" loss):

```
loss = Σ frame_mask · (mask × noisy_mag − clean_mag)²  /  Σ frame_mask · 257
```

- Compared with plain MSE on the mask, this weights errors by how loud
  the bin is, so the model focuses on bins that matter for audio quality.
- `frame_mask` excludes zero-padded frames at the end of short clips.
- Alternative to try if training is unstable: MSE directly between the
  predicted mask and the IRM.
- Later improvement: add a waveform loss (SI-SDR) after inverse STFT.

---

## 6. Architecture

```
input            [B, 126, 257]
GRU layer 1      257 → 256, unidirectional
GRU layer 2      256 → 256, unidirectional, dropout 0.2 between layers
Linear           256 → 257
Sigmoid          → mask [B, 126, 257]
```

| Part | Parameters |
|---|---|
| GRU layer 1 | 395,520 |
| GRU layer 2 | 394,752 |
| Linear | 66,049 |
| **Total** | **~856,000** |

Why this design:

- **Small:** under 1 M parameters, so it trains on a CPU or free-tier Colab
  and is less likely to overfit about 1,300 training segments.
- **Recurrent over time:** noise like typing and door slams is defined by
  how it changes over time; a GRU tracks that context.
- **Unidirectional (causal):** it only uses past frames, so the same
  model can later run in real time (stretch goal 1).

Alternative to compare against once the baseline works: a small
convolutional encoder-decoder (CRN-lite) with a similar parameter budget.

---

## 7. Initial hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | β = (0.9, 0.999) |
| Learning rate | 1e-3 | halve on val-loss plateau (patience 3) |
| Batch size | 32 | ~40 steps per epoch |
| Max epochs | 50 | early stopping, patience 8 on val loss |
| Dropout | 0.2 | between GRU layers |
| Weight decay | 0 | add 1e-5 if overfitting |
| Gradient clipping | 5.0 (global norm) | standard for RNNs |
| Random seed | 42 | fixed for reproducibility |
| Checkpoint | best val loss | saved to `models/` (not tracked) |

First tuning candidates, one at a time: learning rate (3e-4, 1e-3,
3e-3), hidden size (128, 256, 384), dropout (0, 0.2, 0.4).

---

## 8. Evaluation plan

1. **Noisy baseline first:** compute PESQ (wideband, 16 kHz) and STOI of
   the unprocessed noisy test audio vs. clean. This is the score to beat.
2. **Model:** compute the same metrics for the denoised test audio.
3. **Report breakdowns:** overall, **per noise type**, and **per SNR
   level**. Copy-machine has only 15 mixtures, so its averages are noisy;
   say so when reporting.
4. **Listening test:** pick a few test clips per noise type and compare
   noisy vs. denoised by ear.

Metrics are computed on `valid_length` samples only (no padding).

---

## 9. Risks and limitations

- **Small dataset (about 1,300 segments):** overfitting is the main risk.
  Watch the gap between training and validation loss. Remedy: on-the-fly
  re-mixing of clean + noise at random offsets and SNRs.
- **Imbalanced noise types:** copy-machine (15) and squeakyChair (80) are
  under-represented. Consider adding more samples or oversampling.
- **9 speakers only:** val and test are one speaker each, so results may
  reflect those voices more than the model.
- **Noisy phase:** limits achievable PESQ at low SNR.
- **Compute:** the dev codespace has 2 CPUs and no GPU. The baseline is
  sized to train on CPU, but Colab is recommended for tuning.

---

## 10. Dependencies to add

Add to `requirements.txt` before the training task:

```
torch
pesq
pystoi
```

---

## 11. FAQ

**Why do we need a baseline at all?**
Without a simple reference model we can't tell whether a fancier model is
actually better. Every later experiment is compared against this one.

**Why spectrograms instead of raw waveforms?**
Noise and speech are easier to tell apart by frequency, and spectrogram
masking is a well-known approach that works with small models and small
datasets. Waveforms are still saved in the `.npz` files if we want to try
a waveform model later.

**Why predict a mask instead of the clean audio directly?**
A mask is always between 0 and 1, so training is more stable, and the
model only has to learn "keep or remove" instead of generating sound from
scratch.

**Why 2-second segments?**
They give the GRU enough context to recognize noises like door slams and
typing, while keeping batches small enough to train on a CPU. Longer
clips are cut into several segments; shorter ones are zero-padded, and
`frame_mask` tells the loss to ignore the padding.

**Why is the data split by speaker instead of randomly?**
If the same voice appears in training and test, the model can "memorize"
that voice and the test score looks better than it really is. Splitting
by speaker tests whether it works on a voice it has never heard.

**Why are 3 mixtures excluded?**
Their noise files were silent, so they aren't really noisy examples (SNR
above 50 dB). See `scripts/build_features.py`.

**Where are the normalization stats computed?**
On the training split only (`norm_stats.npz`). Using val/test data there
would leak information about those splits into training.

**Why a unidirectional GRU and not a bidirectional one?**
A unidirectional GRU only looks at past frames, so the same model could
run in real time later. A bidirectional one would likely score a bit
higher but needs the whole clip first.

**Can I train this on my laptop?**
Yes. At ~0.86 M parameters it is small enough for a CPU. Google Colab
(free GPU) is faster for trying many hyperparameter settings.

**How do I regenerate the data?**
Run `python scripts/build_features.py`. The outputs in `data/processed/`
are not tracked in git, so everyone builds them locally.

**What PESQ / STOI scores should we expect?**
We don't know yet. Step 1 of the evaluation plan is to measure the noisy
audio itself; the model has to beat that number to be useful.

**What if the results are bad?**
Check the training vs. validation loss first. If validation loss is much
higher, the model is overfitting (see section 9). If both are high, try
the mask-MSE loss or the tuning candidates in section 7.

---

## 12. Open questions for the team

- [ ] Confirm PyTorch (vs. TensorFlow/Keras or MATLAB).
- [ ] Train on Colab or locally?
- [ ] GRU baseline vs. CRN-lite first, or both in parallel?
- [ ] Add more copy-machine / squeakyChair data now or after the baseline?
- [ ] Who owns the training script, and who owns the evaluation script?
