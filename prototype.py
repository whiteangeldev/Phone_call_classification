import os
import numpy as np
import soundfile as sf
from tabulate import tabulate
import torch
from scipy import signal

# -------------------------
# 1. Load Silero VAD model
# -------------------------
print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

SR = 16000  # we normalized everything to 16kHz


# -------------------------
# 2. Helper: speech duration using VAD
# -------------------------
def read_audio_soundfile(path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Read audio file using soundfile and resample to target sampling rate.
    Returns a torch tensor compatible with Silero VAD.
    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # take mono if stereo

    # Resample if needed
    if sr != sampling_rate:
        num_samples = int(len(data) * sampling_rate / sr)
        data = signal.resample(data, num_samples)

    # Convert to torch tensor
    return torch.from_numpy(data).float()


def get_speech_duration_seconds(path: str) -> float:
    """
    Uses Silero VAD to estimate total speech duration in seconds.
    """
    wav = read_audio_soundfile(path, sampling_rate=SR)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SR)

    if not speech_timestamps:
        return 0.0

    total_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    return total_samples / SR


# -------------------------
# 3. Helper: detect SIT tones (invalid number)
# -------------------------
SIT_FREQS = [950, 1400, 1800]  # Hz
SIT_TOLERANCE = 40  # +/- Hz around each freq


def has_sit_tones(path: str) -> bool:
    """
    Simple FFT-based detector: checks if there is strong energy
    near the three standard SIT frequencies (950, 1400, 1800 Hz).
    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]  # take mono if stereo

    # If sample rate is not SR, we won't resample here (for prototype it's ok)
    # but you can add resample if needed.

    # Take FFT of a small chunk (e.g. first 2 seconds) to find tones
    max_duration_seconds = 2.0
    max_samples = int(sr * max_duration_seconds)
    data = data[:max_samples]

    # FFT
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1 / sr)
    magnitudes = np.abs(spectrum)

    # Base noise level
    mean_mag = np.mean(magnitudes)
    threshold = mean_mag * 5  # strong peaks

    detected_count = 0

    for target_freq in SIT_FREQS:
        # find index nearest to target frequency
        idx = np.argmin(np.abs(freqs - target_freq))
        if magnitudes[idx] > threshold:
            detected_count += 1

    # if at least 2 of the 3 tones are strong → treat as SIT
    return detected_count >= 2


# -------------------------
# 4. Classification logic
# -------------------------
def classify_call(path: str):
    """
    Returns (label, speech_duration, sit_detected)
    label is one of: ANSWERED, NO_ANSWER, INVALID_NUMBER
    """

    sit_detected = has_sit_tones(path)
    if sit_detected:
        return "INVALID_NUMBER", 0.0, True

    speech_sec = get_speech_duration_seconds(path)

    # Heuristic thresholds (tune if you want):
    #  - 0   sec speech → NO_ANSWER (ringing, silence)
    #  - 0–1.1 sec     → ANSWERED (short human pickup)
    #  - >1.1 sec      → NO_ANSWER (voicemail / long message)
    if speech_sec == 0:
        label = "NO_ANSWER"
    elif speech_sec < 1.5:
        label = "ANSWERED"
    else:
        label = "NO_ANSWER"

    return label, speech_sec, False


# -------------------------
# 5. Run on all audio files
# -------------------------
def main():
    audio_dir = "audio"
    rows = []

    for fname in sorted(os.listdir(audio_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(audio_dir, fname)

        label, speech_sec, sit_flag = classify_call(path)

        rows.append([fname, f"{speech_sec:.2f}", "yes" if sit_flag else "no", label])

    print("\nClassification results:\n")
    print(
        tabulate(
            rows,
            headers=["File", "Speech_sec", "SIT_detected", "Result"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
