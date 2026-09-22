from pathlib import Path
from math import gcd
import csv
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

# Repo structure:
#
# repo/
# ├── data/
# │   ├── clean_read_speech/
# │   ├── copy-machine/
# │   ├── door/
# │   ├── squeakyChair/
# │   └── typing/
# │
# └── scripts/
#     └── create_mixtures.py

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

CLEAN_DIR = DATA_DIR / "clean_read_speech"

NOISE_DIRS = [
    DATA_DIR / "copy-machine",
    DATA_DIR / "door",
    DATA_DIR / "squeakyChair",
    DATA_DIR / "typing",
]

OUTPUT_DIR = DATA_DIR / "mixed"


# ------------------------------------------------------------
# SNR levels
#
# Lower = MORE noise
#
# -5 dB = very noisy
#  0 dB = heavy noise
#  5 dB = substantial noise
# 10 dB = moderate noise
# 15 dB = light noise
# 20 dB = very light noise
# ------------------------------------------------------------

SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


# Fixed seed means everybody on your team gets the same
# clean/noise pairings and SNR assignments.
RANDOM_SEED = 42


# ============================================================
# AUDIO FUNCTIONS
# ============================================================

def load_audio(path):
    """
    Load audio as float32.

    If the file is stereo/multichannel, convert it to mono.
    """

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def resample_audio(audio, original_sr, target_sr):
    """
    Resample noise if its sample rate is different from
    the clean speech sample rate.
    """

    if original_sr == target_sr:
        return audio

    common = gcd(original_sr, target_sr)

    up = target_sr // common
    down = original_sr // common

    return resample_poly(
        audio,
        up,
        down
    ).astype(np.float32)


def match_noise_length(noise, target_length, rng):
    """
    Make noise exactly the same duration as clean speech.

    If noise is LONGER:
        Take a random section.

    If noise is SHORTER:
        Start at a random location and loop/repeat it
        until it is long enough.
    """

    if len(noise) == 0:
        raise ValueError(
            "Noise file contains no audio samples."
        )

    # --------------------------------------------------------
    # Noise is longer than clean speech
    # --------------------------------------------------------

    if len(noise) >= target_length:

        max_start = len(noise) - target_length

        if max_start > 0:
            start = rng.randint(0, max_start)
        else:
            start = 0

        return noise[
            start:start + target_length
        ]

    # --------------------------------------------------------
    # Noise is shorter than clean speech
    #
    # Rotate it first so looping doesn't always start
    # from the exact same point.
    # --------------------------------------------------------

    start = rng.randint(
        0,
        len(noise) - 1
    )

    rotated_noise = np.concatenate([
        noise[start:],
        noise[:start]
    ])

    repetitions = int(
        np.ceil(
            target_length / len(rotated_noise)
        )
    )

    repeated_noise = np.tile(
        rotated_noise,
        repetitions
    )

    return repeated_noise[:target_length]


def calculate_rms(audio):
    """
    Calculate RMS (root mean square) audio energy.
    """

    return np.sqrt(
        np.mean(
            np.square(audio),
            dtype=np.float64
        ) + 1e-12
    )


def mix_at_snr(clean, noise, snr_db):
    """
    Scale the noise relative to clean speech so that the
    resulting mixture has the requested SNR.

    This means the ORIGINAL volume of a noise recording
    doesn't determine how loud it becomes in the mixture.

    Returns:
        mixed audio
        actual SNR
    """

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms < 1e-8:
        raise ValueError(
            "Clean audio appears to be silent."
        )

    if noise_rms < 1e-8:
        raise ValueError(
            "Noise audio appears to be silent."
        )

    # --------------------------------------------------------
    # Determine required noise level
    #
    # SNR = 20 * log10(clean_rms / noise_rms)
    # --------------------------------------------------------

    target_noise_rms = (
        clean_rms
        / (10 ** (snr_db / 20.0))
    )

    noise_scale = (
        target_noise_rms / noise_rms
    )

    scaled_noise = (
        noise * noise_scale
    )

    mixed = (
        clean + scaled_noise
    )

    # --------------------------------------------------------
    # Prevent clipping
    #
    # Scale both components together so the SNR stays
    # unchanged.
    # --------------------------------------------------------

    peak = np.max(
        np.abs(mixed)
    )

    clean_for_measurement = clean.copy()

    if peak > 0.99:

        scale = 0.99 / peak

        mixed *= scale
        scaled_noise *= scale
        clean_for_measurement *= scale

    # --------------------------------------------------------
    # Verify actual SNR
    # --------------------------------------------------------

    actual_snr = (
        20
        * np.log10(
            calculate_rms(clean_for_measurement)
            / calculate_rms(scaled_noise)
        )
    )

    return (
        mixed.astype(np.float32),
        float(actual_snr)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find clean speech
    # --------------------------------------------------------

    clean_files = sorted(
        CLEAN_DIR.rglob("*.wav")
    )

    # --------------------------------------------------------
    # Find noise
    # --------------------------------------------------------

    noise_files = []

    print()
    print("Finding audio files...")
    print()

    for noise_dir in NOISE_DIRS:

        files = sorted(
            noise_dir.rglob("*.wav")
        )

        print(
            f"{noise_dir.name:<15} "
            f"{len(files):>5} files"
        )

        for file in files:

            noise_files.append(
                (
                    file,
                    noise_dir.name
                )
            )

    print()
    print(
        f"Clean speech:     "
        f"{len(clean_files)} files"
    )

    print(
        f"Total noise:      "
        f"{len(noise_files)} files"
    )

    # --------------------------------------------------------
    # Make sure data exists
    # --------------------------------------------------------

    if not clean_files:

        raise RuntimeError(
            f"No WAV files found in:\n"
            f"{CLEAN_DIR}"
        )

    if not noise_files:

        raise RuntimeError(
            "No noise WAV files found."
        )

    # --------------------------------------------------------
    # ONE-TO-ONE PAIRING
    #
    # Shuffle clean and noise independently.
    #
    # No clean file is intentionally reused.
    # No noise file is intentionally reused.
    #
    # Number of mixtures = whichever dataset is smaller.
    # --------------------------------------------------------

    rng.shuffle(clean_files)
    rng.shuffle(noise_files)

    number_of_pairs = min(
        len(clean_files),
        len(noise_files)
    )

    clean_files = (
        clean_files[:number_of_pairs]
    )

    noise_files = (
        noise_files[:number_of_pairs]
    )

    print()
    print(
        f"Will create "
        f"{number_of_pairs} mixtures."
    )

    print(
        "Each mixture uses one unique clean "
        "file and one unique noise file."
    )

    # --------------------------------------------------------
    # BALANCED SNR DISTRIBUTION
    #
    # This makes approximately the same number of examples
    # for each SNR.
    # --------------------------------------------------------

    snrs = []

    while len(snrs) < number_of_pairs:

        snrs.extend(
            SNR_LEVELS
        )

    snrs = (
        snrs[:number_of_pairs]
    )

    rng.shuffle(snrs)

    # --------------------------------------------------------
    # Generate mixtures
    # --------------------------------------------------------

    metadata = []

    skipped = 0

    print()
    print("Creating mixtures...")
    print()

    progress = tqdm(
        range(number_of_pairs),
        total=number_of_pairs,
        desc="Mixing audio",
        unit="file"
    )

    for index in progress:

        clean_path = (
            clean_files[index]
        )

        noise_path, noise_category = (
            noise_files[index]
        )

        snr_db = (
            snrs[index]
        )

        # Show useful information beside progress bar
        progress.set_postfix(
            noise=noise_category,
            snr=f"{snr_db:+}dB"
        )

        try:

            # -----------------------------------------------
            # Load clean speech
            # -----------------------------------------------

            clean, clean_sr = (
                load_audio(clean_path)
            )

            # -----------------------------------------------
            # Load noise
            # -----------------------------------------------

            noise, noise_sr = (
                load_audio(noise_path)
            )

            # -----------------------------------------------
            # Match sample rates
            # -----------------------------------------------

            noise = resample_audio(
                noise,
                noise_sr,
                clean_sr
            )

            # -----------------------------------------------
            # Match duration
            # -----------------------------------------------

            noise = match_noise_length(
                noise,
                len(clean),
                rng
            )

            # -----------------------------------------------
            # Mix
            # -----------------------------------------------

            mixed, actual_snr = (
                mix_at_snr(
                    clean,
                    noise,
                    snr_db
                )
            )

            # -----------------------------------------------
            # Filename
            # -----------------------------------------------

            mixed_name = (
                f"mixed_{index + 1:04d}.wav"
            )

            mixed_path = (
                OUTPUT_DIR / mixed_name
            )

            # -----------------------------------------------
            # Save WAV
            # -----------------------------------------------

            sf.write(
                mixed_path,
                mixed,
                clean_sr,
                subtype="PCM_16"
            )

            # -----------------------------------------------
            # Save metadata in memory
            # -----------------------------------------------

            metadata.append({

                "mixed_file":
                    mixed_name,

                "clean_file":
                    clean_path.name,

                "noise_file":
                    noise_path.name,

                "noise_category":
                    noise_category,

                "target_snr_db":
                    snr_db,

                "actual_snr_db":
                    round(actual_snr, 3),

                "sample_rate":
                    clean_sr,

                "duration_seconds":
                    round(
                        len(mixed)
                        / clean_sr,
                        3
                    ),
            })

        except Exception as error:

            skipped += 1

            tqdm.write(
                "\n"
                "WARNING: File skipped\n"
                f"Clean: {clean_path}\n"
                f"Noise: {noise_path}\n"
                f"Reason: {error}\n"
            )

    # --------------------------------------------------------
    # Save metadata CSV
    # --------------------------------------------------------

    metadata_path = (
        OUTPUT_DIR / "metadata.csv"
    )

    fieldnames = [

        "mixed_file",
        "clean_file",
        "noise_file",
        "noise_category",
        "target_snr_db",
        "actual_snr_db",
        "sample_rate",
        "duration_seconds",

    ]

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Mixtures created: "
        f"{len(metadata)}"
    )

    print(
        f"Files skipped:     "
        f"{skipped}"
    )

    print()
    print(
        f"Mixed audio:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        f"Metadata:\n"
        f"{metadata_path}"
    )

    # --------------------------------------------------------
    # SNR distribution
    # --------------------------------------------------------

    print()
    print("SNR distribution:")

    for snr in SNR_LEVELS:

        count = sum(
            row["target_snr_db"] == snr
            for row in metadata
        )

        print(
            f"  {snr:+3} dB: "
            f"{count} files"
        )

    # --------------------------------------------------------
    # Noise category distribution
    # --------------------------------------------------------

    print()
    print("Noise categories:")

    for noise_dir in NOISE_DIRS:

        category = (
            noise_dir.name
        )

        count = sum(
            row["noise_category"]
            == category
            for row in metadata
        )

        print(
            f"  {category:<15} "
            f"{count} files"
        )

    print()
    print("Done!")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()