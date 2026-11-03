# Task #5: Evaluate Baseline Performance

**Status:** Complete
**Milestone:** October, Model Design & Training (Task #5)
**Model evaluated:** GRU v2 (IRM loss), the selected baseline. See [`gru_baseline.md`](gru_baseline.md) and [`model_comparison.md`](model_comparison.md)

This document records independent verification of the GRU v2 baseline's evaluation results, plus a manual listening test, per Task #5's description (calculate STOI/PESQ, compare enhanced vs. noisy/clean, listen for perceived improvements, document strengths and limitations).

---

## What I did

1. Rebuilt the data pipeline from scratch (`create_mixtures.py`, `build_features.py`) on my own machine.
2. Trained GRU v2 myself (`train_gru.py --loss irm --checkpoint-name gru_v2.pt --exp-name gru_v2`) rather than relying on a shared checkpoint, to get an independent reproduction rather than just re-scoring someone else's run.
3. Ran `evaluate_gru.py --checkpoint models/gru_v2.pt` on the held-out test set.
4. Manually listened to noisy, enhanced, and clean WAV triples across all four noise categories in `outputs/evaluation/examples/`.

## Reproducibility check

My independently trained and evaluated results matched the numbers already published in `docs/model_comparison.md` and `docs/gru_baseline.md` exactly (training uses a fixed seed):

| Metric | Documented | My reproduction |
|---|---|---|
| Best epoch | 18 | 18 |
| Best val loss | 0.1124 | 0.1124 |
| Val PESQ Δ | +0.154 | +0.154 |
| Val STOI Δ | +0.001 | +0.001 |
| Test PESQ (noisy to enhanced) | 2.077 to 2.293 | 2.077 to 2.293 |
| Test STOI (noisy to enhanced) | 0.890 to 0.892 | 0.890 to 0.892 |

This confirms the training pipeline and evaluation script are fully reproducible. It's a useful sanity check that the results in `model_comparison.md` aren't an artifact of one person's environment.

**Per-SNR breakdown (test set)**, also reproduced exactly:

| SNR | Noisy PESQ | Enhanced PESQ | Δ PESQ | Noisy STOI | Enhanced STOI | Δ STOI |
|---|---:|---:|---:|---:|---:|---:|
| −5 dB | 1.207 | 1.340 | +0.133 | 0.710 | 0.748 | +0.038 |
| 0 dB | 1.320 | 1.606 | +0.285 | 0.834 | 0.860 | +0.026 |
| 5 dB | 1.551 | 1.712 | +0.161 | 0.871 | 0.884 | +0.013 |
| 10 dB | 2.260 | 2.640 | +0.380 | 0.962 | 0.952 | −0.010 |
| 15 dB | 2.567 | 2.828 | +0.261 | 0.968 | 0.951 | −0.016 |
| 20 dB | 2.876 | 2.991 | +0.115 | 0.954 | 0.933 | −0.021 |

## Manual listening test

I listened to noisy, enhanced, and clean triples across all four noise categories (door, typing, squeakyChair, copy machine) from `outputs/evaluation/examples/`.

- Noise was audibly reduced in the enhanced audio relative to the noisy input across all categories checked.
- Enhanced speech stayed close to the clean reference, with no noticeable distortion, thinness, or robotic artifacts.
- I did not hear the over suppression behavior reported for GRU v1. This is consistent with GRU v2 being the better performing, selected model.
- At high SNR (15 dB, already relatively clean input), the enhanced audio did not sound perceptibly worse than the noisy original, even though the STOI numbers show a small measured dip (−0.010 to −0.021) at that range. The degradation is real in the numbers but not obviously audible.

## Strengths

- PESQ improves at every tested SNR level (+0.115 to +0.380), with the largest gains at low to mid SNR where suppression is most needed.
- Fully reproducible: same seed, same data pipeline, same results across independent runs.
- No audible artifacts or over suppression in manual listening, consistent with the IRM loss fixing the over suppression problem seen in GRU v1.

## Limitations

- STOI slightly decreases at high SNR (10 to 20 dB, Δ −0.010 to −0.021). The model mildly over processes audio that was already fairly clean. Not perceptually obvious in listening, but measurable.
- Copy machine has only about 15 test segments. Its per category average has high variance and should be interpreted cautiously.
- Reconstruction reuses the noisy signal's phase rather than predicting it, which caps achievable quality, especially at low SNR. This is a known architectural limitation, not a training issue.
- Evaluation is on 2 second segments from a single held out speaker (reader 04408). Broader speaker coverage would strengthen confidence in generalization.

## Conclusion

GRU v2 is a validated, reproducible baseline that beats the noisy signal on both PESQ and STOI overall. Its one real weakness, slight high SNR over processing, is well understood and documented rather than a surprise.
