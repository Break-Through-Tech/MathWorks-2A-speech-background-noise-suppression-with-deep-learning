"""
Input representations for the noise suppression model.

Everything here is plain numpy/scipy, so the same features work with
PyTorch, TensorFlow, or anything else.

Conventions (match torch.stft / librosa.stft with center=True):

    sample rate   16 kHz
    window        512 samples (32 ms), periodic Hann
    hop           256 samples (16 ms)
    frequency     257 bins (0 - 8 kHz)
    segment       2 s = 32000 samples -> 126 STFT frames
"""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 16000

N_FFT = 512
HOP_LENGTH = 256
N_FREQ = N_FFT // 2 + 1

SEGMENT_SECONDS = 2.0
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * TARGET_SR)

# A trailing piece shorter than this is dropped instead of padded
MIN_LAST_SEGMENT_SECONDS = 0.5

# Added before taking the log so silence doesn't become -inf
LOG_EPS = 1e-6

# Periodic Hann (same as torch.hann_window / scipy "hann", fftbins=True)
WINDOW = (
    0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
).astype(np.float32)


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    """
    Load audio as float32 mono.
    """

    audio, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sample_rate


def resample_audio(audio, original_sr, target_sr=TARGET_SR):
    """
    Resample with a polyphase filter (48 kHz -> 16 kHz is 1/3).
    """

    if original_sr == target_sr:
        return audio

    common = gcd(original_sr, target_sr)

    return resample_poly(
        audio,
        target_sr // common,
        original_sr // common
    ).astype(np.float32)


def estimate_clean_gain(mixed, clean):
    """
    Estimate how much the clean speech was scaled inside the mixture.

    create_mixtures.py scales clean AND noise down together when
    the mixture would clip, but the clean file on disk is not
    scaled. The least-squares gain  <mixed, clean> / <clean, clean>
    recovers that factor, so  gain * clean  is the speech that is
    actually inside the mixture.
    """

    mixed = mixed.astype(np.float64)
    clean = clean.astype(np.float64)

    return float(
        np.dot(mixed, clean)
        / (np.dot(clean, clean) + 1e-12)
    )


# ============================================================
# SEGMENTING
# ============================================================

def segment_waveform(
    audio,
    segment_samples=SEGMENT_SAMPLES,
    hop_samples=None,
    min_last_samples=int(MIN_LAST_SEGMENT_SECONDS * TARGET_SR),
):
    """
    Cut audio into fixed-length segments, zero-padding the last one.

    Returns:
        segments       (num_segments, segment_samples)
        starts         start sample of each segment
        valid_lengths  number of real (non-padded) samples in each
    """

    if hop_samples is None:
        hop_samples = segment_samples

    segments = []
    starts = []
    valid_lengths = []

    start = 0

    while True:

        piece = audio[start:start + segment_samples]

        # Always keep at least one segment per clip
        if len(piece) < min_last_samples and segments:
            break

        valid_lengths.append(len(piece))
        starts.append(start)

        if len(piece) < segment_samples:
            piece = np.pad(
                piece,
                (0, segment_samples - len(piece))
            )

        segments.append(piece)

        if start + segment_samples >= len(audio):
            break

        start += hop_samples

    return (
        np.stack(segments).astype(np.float32),
        np.array(starts),
        np.array(valid_lengths),
    )


# ============================================================
# STFT
# ============================================================

def stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Complex STFT, shape (..., n_fft // 2 + 1, frames).

    Reflect-pads n_fft // 2 on both sides (center=True), so frame t
    is centred on sample t * hop_length.
    """

    audio = np.asarray(audio, dtype=np.float32)

    pad = n_fft // 2

    padded = np.pad(
        audio,
        [(0, 0)] * (audio.ndim - 1) + [(pad, pad)],
        mode="reflect"
    )

    num_frames = 1 + (padded.shape[-1] - n_fft) // hop_length

    index = (
        np.arange(n_fft)[None, :]
        + hop_length * np.arange(num_frames)[:, None]
    )

    frames = padded[..., index] * WINDOW

    spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)

    # (..., frames, freq) -> (..., freq, frames)
    return np.swapaxes(spectrum, -1, -2).astype(np.complex64)


def istft(spectrum, length, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Inverse of stft(): windowed overlap-add.

    length is the number of output samples (the original audio length).
    """

    # (..., freq, frames) -> (..., frames, freq)
    frames = np.fft.irfft(
        np.swapaxes(spectrum, -1, -2),
        n=n_fft,
        axis=-1
    ).astype(np.float32) * WINDOW

    num_frames = frames.shape[-2]
    padded_length = n_fft + hop_length * (num_frames - 1)

    output = np.zeros(
        frames.shape[:-2] + (padded_length,),
        dtype=np.float32
    )

    window_sum = np.zeros(padded_length, dtype=np.float32)

    for t in range(num_frames):

        start = t * hop_length

        output[..., start:start + n_fft] += frames[..., t, :]
        window_sum[start:start + n_fft] += WINDOW ** 2

    output /= np.maximum(window_sum, 1e-8)

    pad = n_fft // 2

    return output[..., pad:pad + length]


def magnitude_phase(audio):
    """
    Magnitude and phase spectrograms of a waveform.
    """

    spectrum = stft(audio)

    return np.abs(spectrum), np.angle(spectrum)


def log_magnitude(magnitude):
    """
    Compress magnitude to a log scale (the usual network input).
    """

    return np.log(magnitude + LOG_EPS)


def reconstruct(magnitude, phase, length):
    """
    Turn a (predicted) magnitude back into audio using a phase,
    normally the phase of the noisy input.
    """

    return istft(
        magnitude * np.exp(1j * phase),
        length
    )


# ============================================================
# TARGETS AND NORMALIZATION
# ============================================================

def ideal_ratio_mask(clean_magnitude, noisy_magnitude):
    """
    Mask in [0, 1] such that  mask * noisy_magnitude ~ clean_magnitude.

    A common training target: the network predicts the mask, and
    the denoised magnitude is  mask * noisy_magnitude.
    """

    return np.clip(
        clean_magnitude / (noisy_magnitude + 1e-8),
        0.0,
        1.0
    ).astype(np.float32)


def frame_mask(valid_lengths, num_frames, hop_length=HOP_LENGTH):
    """
    Boolean (segments, frames) array: True where a frame is centred
    on real audio rather than zero padding.
    """

    centres = np.arange(num_frames) * hop_length

    return centres[None, :] < np.asarray(valid_lengths)[:, None]


def normalize(log_mag, mean, std):
    """
    Standardize log magnitude with per-frequency training statistics.

    mean, std have shape (freq,).
    """

    return (
        (log_mag - mean[:, None])
        / std[:, None]
    ).astype(np.float32)
