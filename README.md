# Speech Background Noise Suppression with Deep Learning

**MathWorks × Break Through Tech AI Studio — Fall 2026**

---

## Team Members

- Ashley
- Jaya
- Kwabena
- Angela
- Heidy

---

## Project Highlights

- Built a GRU-based deep learning model (856K parameters) to suppress background noise from speech recordings using the Microsoft DNS Challenge dataset.
- **GRU v2 (IRM loss)** achieves **PESQ 2.293 (+0.216 vs. noisy)** and **STOI 0.892 (+0.002 vs. noisy)** on the held-out test set, beating the unprocessed noisy input across all tested SNR levels for perceptual quality.
- Demonstrated that training the mask directly against an Ideal Ratio Mask (IRM) target fixes the over-suppression problem seen in a magnitude-MSE baseline, particularly at high SNR (10–20 dB).
- Fully reproducible pipeline: fixed random seed + speaker-based train/val/test split = identical results across independent runs.
- Designed a next-stage **CRN-Lite** (Convolutional Recurrent Network) architecture to extend the GRU with local time-frequency feature extraction while reusing the same data pipeline and evaluation methodology.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- [Git](https://git-scm.com/)
- GPU recommended for training (free-tier Google Colab works)

### 1. Clone the repository

```bash
git clone https://github.com/Break-Through-Tech/MathWorks-2A-speech-background-noise-suppression-with-deep-learning.git
cd MathWorks-2A-speech-background-noise-suppression-with-deep-learning
```

### 2. Install dependencies

```bash
pip install torch torchaudio numpy scipy librosa pesq pystoi soundfile
```

### 3. Download the dataset

The project uses a subset of the [Microsoft DNS Challenge dataset](https://github.com/microsoft/DNS-Challenge).

1. Clone the DNS Challenge repo and follow its instructions to download clean speech (Librivox) and noise clips (AudioSet / Freesound).
2. Place clean speech WAVs in `data/clean/` and noise WAVs in `data/noise/`.
3. The project targets these noise types: `door`, `typing`, `squeakyChair`, `copy-machine` at SNRs from −5 to +20 dB.

A small sample dataset for testing is provided in `data/sample/` — see [Sample Dataset](#sample-dataset) below.

### 4. Build the data pipeline

```bash
# Mix clean speech + noise at target SNRs → data/mixed/
python scripts/create_mixtures.py

# Segment, compute spectrograms, build train/val/test splits → data/processed/
python scripts/build_features.py
```

This produces:
```
data/processed/features/
    train.npz       1,292 segments (7 speakers)
    val.npz         184 segments   (1 speaker)
    test.npz        172 segments   (1 speaker)
    norm_stats.npz  per-frequency mean and std from train only
```

### 5. Train the model

```bash
# Quick smoke test (2 epochs, 64 samples)
python scripts/train_gru.py --max-epochs 2 --subset 64 --no-val-metrics

# GRU v2 — IRM loss (selected model)
python scripts/train_gru.py \
    --loss irm \
    --checkpoint-name gru_v2.pt \
    --exp-name gru_v2
```

Checkpoints are saved to `models/`. The best validation checkpoint is kept automatically.

### 6. Evaluate model performance

```bash
# Evaluate GRU v2 on the held-out test set
python scripts/evaluate_gru.py --checkpoint models/gru_v2.pt
```

Output files:
- `outputs/evaluation/results.csv` — per-segment STOI/PESQ scores
- `outputs/evaluation/summary.txt` — overall and per-SNR/noise-type breakdown
- `outputs/evaluation/examples/` — noisy / enhanced / clean WAV triples

---

## Project Overview

This project was completed as part of the **Break Through Tech AI Studio Fall 2026** program in partnership with **MathWorks**.

### Challenge

Background noise degrades speech quality in hearing aids, online meetings, and voice communication systems. The goal is to build and validate a deep learning model that suppresses background noise from speech signals, evaluated on objective metrics (PESQ and STOI) and informal listening tests.

### Approach

We trained a recurrent neural network on paired noisy/clean speech segments from the Microsoft DNS Challenge dataset. The model operates in the frequency domain: it predicts a time-frequency mask that separates speech energy from noise energy and applies that mask to reconstruct cleaner audio.

Two training objectives were compared:
- **GRU v1 (Magnitude MSE):** trains the mask indirectly via reconstruction error
- **GRU v2 (IRM loss):** trains the mask directly against an Ideal Ratio Mask per T-F bin

GRU v2 is the selected model. A next-stage **CRN-Lite** architecture (convolutional encoder + GRU bottleneck + convolutional decoder with skip connections) has been designed and is pending implementation.

### Real-world impact

Noise suppression is a core enabling technology for hearing aids and assistive devices. MathWorks develops audio and signal processing tools used by engineers across industries; this project advances speech enhancement capability toward the company's accessibility and communications goals.

---

## Data Exploration

### Dataset

| Property | Value |
|---|---|
| Source | [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge) |
| Clean speech | Librivox (subset) — multiple speakers, 16 kHz WAV |
| Noise types used | door, typing, squeakyChair, copy-machine |
| SNR range | −5, 0, 5, 10, 15, 20 dB |
| Segment length | 2 seconds (32,000 samples at 16 kHz) |
| Train / val / test split | Speaker-based: 7 / 1 / 1 speakers |
| Train segments | 1,292 |
| Val segments | 184 |
| Test segments | 172 |

The dataset is small by deep learning standards (~10 hours of mixed audio), which shaped our model selection decisions throughout the project.

### Preprocessing

Each noisy/clean pair goes through:
1. **STFT** (512-point, 256 hop, Hann window) → magnitude `[257, T]` and phase `[257, T]`
2. **Log magnitude** — compresses dynamic range
3. **Per-frequency z-score normalization** — statistics from training set only
4. The **phase is never modified**; it is saved and reused at reconstruction (a deliberate simplification)

### Key EDA findings

- Noise types are unevenly represented in the test set: copy-machine has only ~15 segments, so its per-type averages carry high variance.
- High-SNR segments (≥15 dB) make up nearly half the test set (84/172). A model that over-suppresses clean audio will show negative overall deltas even if it helps at low SNR.
- The clean speech source is Librivox (read-aloud audiobooks), so the data skews toward careful, articulate speech. Real-world robustness to spontaneous speech is untested.

See [`notebooks/01_input_representation.ipynb`](notebooks/01_input_representation.ipynb) for spectrograms and feature distribution plots.

---

## Model Development

### Architecture: GRU Denoiser

A 2-layer unidirectional GRU followed by a linear + sigmoid head.

| Property | Value |
|---|---|
| Architecture | 2-layer GRU + Linear + Sigmoid |
| Parameters | 856,321 (~3.4 MB float32) |
| Input | Normalized log-magnitude `[B, T, 257]` |
| Output | Enhancement mask `[B, T, 257]`, values in (0, 1) |
| GRU hidden size | 256 |
| Dropout | 0.2 (inter-layer) |

**Pipeline:**
```
noisy WAV → STFT → log + normalize → GRU → mask → mask × noisy magnitude
         → iSTFT (with noisy phase) → enhanced WAV
```

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam, lr = 0.001 |
| Batch size | 32 |
| Max epochs | 50 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early stopping | patience 8 |
| Gradient clipping | max norm 5.0 |

### GRU v1 vs. v2

| | GRU v1 | GRU v2 |
|---|---|---|
| Loss | Magnitude MSE | IRM MSE |
| Training target | Enhanced magnitude ≈ clean magnitude | Predicted mask ≈ IRM |
| Val PESQ Δ | −0.055 | **+0.154** |
| Test PESQ Δ | −0.162 | **+0.216** |
| High-SNR behavior | Over-suppresses | Leaves clean signal intact |

The IRM (Ideal Ratio Mask) is computed per T-F bin as `clip(clean_mag / (noisy_mag + ε), 0, 1)`. Because the IRM approaches 1 at high SNR, GRU v2 learns to suppress less when the input is already clean — fixing v1's core failure mode.

### Advanced model: CRN-Lite (design complete, implementation pending)

A Convolutional Recurrent Network adds a CNN encoder before the GRU and a CNN decoder with skip connections after it, allowing the model to recognize local noise textures across time and frequency instead of treating each frame independently. See [`docs/advanced_model.md`](docs/advanced_model.md) for the full architecture specification.

---

## Results & Key Findings

### Overall test-set performance (172 segments, speaker 04408)

| Model | Test PESQ | Δ vs. Noisy | Test STOI | Δ vs. Noisy |
|---|---:|---:|---:|---:|
| Noisy input (baseline) | 2.077 | — | 0.890 | — |
| GRU v1 (Magnitude MSE) | 1.915 | −0.162 | 0.862 | −0.027 |
| **GRU v2 (IRM loss)** | **2.293** | **+0.216** | **0.892** | **+0.002** |

### GRU v2 breakdown by SNR

| SNR | Noisy PESQ | Enhanced PESQ | Δ PESQ | Noisy STOI | Enhanced STOI | Δ STOI |
|---|---:|---:|---:|---:|---:|---:|
| −5 dB | 1.207 | 1.340 | **+0.133** | 0.710 | 0.748 | **+0.038** |
| 0 dB | 1.320 | 1.606 | **+0.285** | 0.834 | 0.860 | **+0.026** |
| 5 dB | 1.551 | 1.712 | **+0.161** | 0.871 | 0.884 | **+0.013** |
| 10 dB | 2.260 | 2.640 | **+0.380** | 0.962 | 0.952 | −0.010 |
| 15 dB | 2.567 | 2.828 | **+0.261** | 0.968 | 0.951 | −0.016 |
| 20 dB | 2.876 | 2.991 | **+0.115** | 0.954 | 0.933 | −0.021 |

**Key findings:**

- GRU v2 improves **PESQ at every SNR level**, including at high SNR where v1 degraded.
- **STOI improves at low-to-mid SNR** (−5 to +5 dB) where active suppression is needed, and shows a small measured dip at high SNR (−0.010 to −0.021) that was not perceptually obvious in informal listening tests.
- The largest PESQ gain (+0.380) is at 10 dB — mid-range SNR — suggesting the model is best calibrated for moderately noisy conditions.
- Results are **fully reproducible**: independent re-training with the same seed produces bit-identical metrics.

### Informal listening test (GRU v2)

Listened to noisy/enhanced/clean triples across all four noise categories:
- Noise was audibly reduced in all categories.
- No noticeable distortion, thinning, or robotic artifacts in the enhanced audio.
- At high SNR, enhanced audio was perceptually indistinguishable from the noisy input, consistent with the near-zero STOI delta at that range.

---

## Models

| Model | Doc | Status | Test STOI | Test PESQ |
|---|---|---|---:|---:|
| GRU v1 (Magnitude MSE) | [docs/gru_baseline.md](docs/gru_baseline.md) | Complete | 0.862 | 1.915 |
| **GRU v2 (IRM loss)** | [docs/gru_baseline.md](docs/gru_baseline.md) | **Selected** — checkpoint: `models/gru_v2.pt` | 0.892 | 2.293 |
| CRN-Lite | [docs/advanced_model.md](docs/advanced_model.md) | Design complete — implementation in progress | — | — |

See [docs/model_comparison.md](docs/model_comparison.md) for the full comparison table.

---

## Sample Dataset

A small sample dataset is included in `data/sample/` to allow testing without downloading the full DNS Challenge dataset:

```
data/sample/
    clean/          A few clean speech WAV files (16 kHz, mono)
    noise/          One clip per noise type (door, typing, squeakyChair, copy-machine)
    mixed/          Pre-mixed noisy examples at selected SNR levels
```

To run the full evaluation pipeline on the sample data, pass `--data-dir data/sample` to the scripts.

> **Note:** The sample dataset is for testing setup and code only. Model training requires the full DNS Challenge dataset subset.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/01_input_representation.ipynb`](notebooks/01_input_representation.ipynb) | Loads a noisy/clean audio pair, plots waveforms, spectrograms, and magnitude distributions; illustrates what the GRU input looks like before and after normalization |

Additional notebooks demonstrating training and evaluation are in progress.

---

## Project Documentation

Full technical documentation is in the [`docs/`](docs/) folder:

| Document | Contents |
|---|---|
| [`docs/baseline_model.md`](docs/baseline_model.md) | Data flow diagram, feature extraction spec, loss formulas |
| [`docs/gru_baseline.md`](docs/gru_baseline.md) | GRU architecture, training settings, v1 vs. v2 comparison, results, limitations |
| [`docs/advanced_model.md`](docs/advanced_model.md) | CRN-Lite architecture decision, layer-by-layer spec, training config, expected performance |
| [`docs/model_comparison.md`](docs/model_comparison.md) | Authoritative comparison table across all trained models |
| [`docs/task5_evaluation_notes.md`](docs/task5_evaluation_notes.md) | Independent reproduction and manual listening test for GRU v2 |

### Replicating results end-to-end

```bash
# 1. Build data
python scripts/create_mixtures.py
python scripts/build_features.py

# 2. Train GRU v2
python scripts/train_gru.py --loss irm --checkpoint-name gru_v2.pt --exp-name gru_v2

# 3. Evaluate on test set
python scripts/evaluate_gru.py --checkpoint models/gru_v2.pt

# Expected test results:
#   PESQ: noisy 2.077 → enhanced 2.293  (Δ +0.216)
#   STOI: noisy 0.890 → enhanced 0.892  (Δ +0.002)
```

---

## Next Steps

1. **Implement CRN-Lite** (`models/crn.py`, `scripts/train_crn.py`, `scripts/evaluate_crn.py`) and compare against GRU v2 on the same test set. Architecture spec is in [`docs/advanced_model.md`](docs/advanced_model.md).
2. **Address noisy phase reuse** — the current magnitude-only approach reuses the noisy phase for reconstruction, capping achievable quality at low SNR. A phase-aware approach (e.g., Complex U-Net or complex IRM) could unlock further gains once more training data or augmentation is in place.
3. **Expand the dataset** — the training set has 1,292 segments from 7 speakers. On-the-fly SNR augmentation and additional speaker/noise-type coverage would improve generalization.
4. **Real-time streaming** — the GRU architecture is causal (unidirectional), making it a candidate for real-time inference with a small buffer. Benchmarking latency and throughput on a CPU would demonstrate viability for hearing-aid applications.
5. **Broader noise generalization** — evaluate on DNS noise types beyond the four used in training to quantify out-of-distribution robustness.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- Reddy, C. et al. (2020). [The Interspeech 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results](https://arxiv.org/abs/2009.06122). INTERSPEECH 2020.
- Tan, K., & Wang, D. (2018). [A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement](https://www.isca-speech.org/archive/interspeech_2018/tan18b_interspeech.html). INTERSPEECH 2018.
- Librosa documentation: [https://librosa.org/doc/latest/index.html](https://librosa.org/doc/latest/index.html)
- pesq package: [https://pypi.org/project/pesq/](https://pypi.org/project/pesq/)
- pystoi package: [https://github.com/mpariente/pystoi](https://github.com/mpariente/pystoi)
- Microsoft DNS Challenge repo: [https://github.com/microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge)

---

## Acknowledgements

Thank you to our Challenge Advisor **Neha Sardesai** (MathWorks) for project guidance and feedback, our AI Studio Coach **Bhavya Gopal** (Break Through Tech) for support throughout the program, and the Break Through Tech AI Studio program for this opportunity.
