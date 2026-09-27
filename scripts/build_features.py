"""
Turn the noisy/clean pairs in data/mixed/metadata.csv into model inputs.

Run from the repo root:

    python scripts/build_features.py

Writes:

    data/splits.csv                          which clip is in train/val/test
    data/processed/wav16k/noisy/*.wav        full noisy clips at 16 kHz
    data/processed/wav16k/clean/*.wav        matching clean targets at 16 kHz
    data/processed/features/{split}.npz      2 s segments, waveforms + spectrograms
    data/processed/features/norm_stats.npz   per-frequency mean/std (train only)

Everything under data/processed/ is git-ignored; rerun this script to
rebuild it.
"""

from pathlib import Path
import csv
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F


# ============================================================
# SETTINGS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

MIXED_DIR = DATA_DIR / "mixed"
CLEAN_DIR = DATA_DIR / "clean_read_speech"
METADATA_PATH = MIXED_DIR / "metadata.csv"

SPLITS_PATH = DATA_DIR / "splits.csv"

PROCESSED_DIR = DATA_DIR / "processed"
WAV_DIR = PROCESSED_DIR / "wav16k"
FEATURE_DIR = PROCESSED_DIR / "features"


# ------------------------------------------------------------
# Speaker-based split
#
# The clean speech comes from only 9 readers. Splitting clips at
# random would put the same voices in train and test, so whole
# readers are held out instead. Both held-out readers have all
# four noise types.
# ------------------------------------------------------------

VAL_READERS = {"08897"}
TEST_READERS = {"04408"}

# Mixtures whose noise is this far below the speech contain no
# real noise (the source noise file is silent).
MAX_VALID_SNR_DB = 50


# ============================================================
# HELPERS
# ============================================================

def reader_id(clean_file):
    """
    book_00028_chp_0038_reader_05919_3_seg_1_seg1.wav -> "05919"
    """

    return clean_file.split("reader_")[1].split("_")[0]


def assign_split(reader):

    if reader in TEST_READERS:
        return "test"

    if reader in VAL_READERS:
        return "val"

    return "train"


# ============================================================
# MAIN
# ============================================================

def main():

    with open(METADATA_PATH, newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))

    for folder in ["noisy", "clean"]:
        (WAV_DIR / folder).mkdir(parents=True, exist_ok=True)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    split_rows = []

    segments = {
        split: {
            "noisy_wav": [],
            "clean_wav": [],
            "valid_length": [],
            "start": [],
            "mixed_file": [],
        }
        for split in ["train", "val", "test"]
    }

    # --------------------------------------------------------
    # Pass 1: load, fix target level, resample, segment
    # --------------------------------------------------------

    for row in tqdm(metadata, desc="Processing clips", unit="clip"):

        reader = reader_id(row["clean_file"])
        split = assign_split(reader)

        excluded = ""

        if float(row["actual_snr_db"]) > MAX_VALID_SNR_DB:
            excluded = "silent noise file"

        noisy, noisy_sr = F.load_audio(MIXED_DIR / row["mixed_file"])
        clean, clean_sr = F.load_audio(CLEAN_DIR / row["clean_file"])

        if len(noisy) != len(clean) or noisy_sr != clean_sr:
            excluded = "noisy/clean length or sample rate mismatch"

        gain = 1.0

        if not excluded:
            gain = F.estimate_clean_gain(noisy, clean)

            # Only clipped mixtures were rescaled; for the rest the
            # estimate is ~1 and the small deviation is just noise.
            if np.max(np.abs(noisy)) < 0.985:
                gain = 1.0

        split_rows.append({
            "mixed_file": row["mixed_file"],
            "clean_file": row["clean_file"],
            "reader": reader,
            "noise_category": row["noise_category"],
            "target_snr_db": row["target_snr_db"],
            "split": split,
            "clean_gain": round(gain, 5),
            "excluded": excluded,
        })

        if excluded:
            continue

        noisy = F.resample_audio(noisy, noisy_sr)
        clean = F.resample_audio(clean * gain, clean_sr)

        sf.write(
            WAV_DIR / "noisy" / row["mixed_file"],
            noisy, F.TARGET_SR, subtype="FLOAT"
        )

        sf.write(
            WAV_DIR / "clean" / row["mixed_file"],
            clean, F.TARGET_SR, subtype="FLOAT"
        )

        noisy_segments, starts, valid = F.segment_waveform(noisy)
        clean_segments, _, _ = F.segment_waveform(clean)

        store = segments[split]
        store["noisy_wav"].append(noisy_segments)
        store["clean_wav"].append(clean_segments)
        store["valid_length"].append(valid)
        store["start"].append(starts)
        store["mixed_file"].extend([row["mixed_file"]] * len(starts))

    with open(SPLITS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    # --------------------------------------------------------
    # Pass 2: spectrograms, normalization stats, save
    # --------------------------------------------------------

    stats = None

    for split in ["train", "val", "test"]:

        store = segments[split]

        noisy_wav = np.concatenate(store["noisy_wav"])
        clean_wav = np.concatenate(store["clean_wav"])
        valid_length = np.concatenate(store["valid_length"])

        noisy_mag, _ = F.magnitude_phase(noisy_wav)
        clean_mag, _ = F.magnitude_phase(clean_wav)

        frames = F.frame_mask(valid_length, noisy_mag.shape[-1])

        if split == "train":

            # (segments, freq, frames) -> (freq, real frames)
            log_mag = F.log_magnitude(noisy_mag)
            real = np.transpose(log_mag, (1, 0, 2))[:, frames]

            stats = {
                "mean": real.mean(axis=1).astype(np.float32),
                "std": (real.std(axis=1) + 1e-5).astype(np.float32),
            }

            np.savez(FEATURE_DIR / "norm_stats.npz", **stats)

        np.savez(
            FEATURE_DIR / f"{split}.npz",
            noisy_wav=noisy_wav,
            clean_wav=clean_wav,
            noisy_mag=noisy_mag,
            clean_mag=clean_mag,
            frame_mask=frames,
            valid_length=valid_length,
            start=np.concatenate(store["start"]),
            mixed_file=np.array(store["mixed_file"]),
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    excluded = [r for r in split_rows if r["excluded"]]
    rescaled = [r for r in split_rows if r["clean_gain"] != 1.0]

    print(f"Excluded clips:          {len(excluded)}")

    for r in excluded:
        print(f"  {r['mixed_file']}: {r['excluded']}")

    print(f"Clean targets rescaled:  {len(rescaled)}")
    print()

    for split in ["train", "val", "test"]:

        clips = [
            r for r in split_rows
            if r["split"] == split and not r["excluded"]
        ]

        readers = sorted({r["reader"] for r in clips})

        print(
            f"{split:<6} {len(clips):>4} clips  "
            f"{len(segments[split]['mixed_file']):>5} segments  "
            f"readers: {', '.join(readers)}"
        )

    print()
    print(
        f"Segment shapes:  waveform ({F.SEGMENT_SAMPLES},)  "
        f"spectrogram {noisy_mag.shape[1:]}"
    )
    print(f"Output: {FEATURE_DIR}")


if __name__ == "__main__":
    main()
