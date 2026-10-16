# Model Comparison

This table is the authoritative comparison across all models trained on the speech noise suppression dataset.
Add a row when a new model's test evaluation is complete. Never update this table from validation numbers.

**Dataset:** 172 test segments, speaker 04408, noise types: door / typing / squeakyChair / copy-machine, SNRs: −5 to 20 dB.
**Metrics:** PESQ = wideband at 16 kHz; STOI = extended=False; both computed on `valid_length` samples only.

| Model | Params | Best Val Loss | Test STOI | Δ STOI vs Noisy | Test PESQ | Δ PESQ vs Noisy | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Noisy input (baseline) | — | — | 0.890 | — | 2.077 | — | From GRU eval run (172 test segments, speaker 04408) |
| **GRU Baseline v1** | 856,321 | 0.1437 (ep. 34) | 0.862 | −0.027 | 1.915 | −0.162 | Magnitude MSE loss; 42 epochs early stopped; helps at low SNR (≤5 dB), degrades at high SNR; see [`docs/gru_baseline.md`](gru_baseline.md) |
| **GRU v2 (IRM loss)** | 856,321 | **0.1124** (ep. 18) | **0.892** | **+0.002** | **2.293** | **+0.216** | IRM target loss; 26 epochs early stopped; improves PESQ at all SNR levels; see [`docs/gru_baseline.md`](gru_baseline.md) |

---

### Notes

- **Params:** trainable parameters only.
- **Best Val Loss:** masked MSE on enhanced magnitude (see loss formula in [`docs/baseline_model.md`](baseline_model.md)). Losses are not directly comparable across models with different reconstruction pipelines.
- **Δ vs Noisy:** positive = improvement. A model that does not beat the noisy baseline is not useful.
- **TBD:** run `python scripts/evaluate_gru.py` to produce these numbers. Fill them in here after each model's evaluation.
- Copy-machine has ~15 test segments — its per-type averages have high variance. Interpret with care.

---

### How to add a row

1. Train your new model and save its best checkpoint.
2. Run its evaluation script on the test set.
3. Copy the `noisy_pesq` and `noisy_stoi` averages from `outputs/evaluation/results.csv` (these come from the same test set, so they should be stable across models — use the GRU run's values once available).
4. Add a row here with model name, parameter count, best val loss, test STOI, STOI delta, test PESQ, PESQ delta, and a short note with a link to the model's own doc.
