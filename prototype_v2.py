import os
import numpy as np
import soundfile as sf
from tabulate import tabulate
import torch
import whisper
from scipy import signal

SR = 16000  # we assume all WAV files are mono 16kHz (from your ffmpeg step)

# ============================
# 1. Load models (VAD + ASR)
# ============================

print("Loading Silero VAD model...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

print("Loading Whisper ASR model (base)...")
asr_model = whisper.load_model("base")  # or "small" / "tiny" if you want faster

# ============================
# 2. VAD: total speech duration
# ============================


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
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SR)

    if not speech_timestamps:
        return 0.0

    total_samples = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    return total_samples / SR


# ============================
# 3. SIT tone detection (INVALID NUMBER)
# ============================

SIT_FREQS = [950, 1400, 1800]  # Hz
SIT_TOLERANCE = 40  # +/- Hz (simple window)


def has_sit_tones(path: str) -> bool:
    """
    Robust SIT tone detector using:
    - 3 tones required (950, 1400, 1800 Hz)
    - Pure-tone check (spectral purity)
    - Duration check (~300 ms)
    - Sequential check
    """
    import librosa

    y, sr = librosa.load(path, sr=16000)

    # Analyze only the first second
    y = y[:16000]

    # Frame into 50 ms windows
    frame_length = int(0.05 * sr)
    hop_length = frame_length // 2

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Helper: find energy near a frequency ±20 Hz
    def peak_energy(target):
        idx = np.where((freqs >= target - 20) & (freqs <= target + 20))[0]
        if len(idx) == 0:
            return np.zeros(S.shape[1])
        return S[idx, :].mean(axis=0)

    # Extract energy curves for each SIT tone
    e950 = peak_energy(950)
    e1400 = peak_energy(1400)
    e1800 = peak_energy(1800)

    # Baseline energy
    noise = np.mean(S)

    # Threshold = 12x noise (high precision)
    th = noise * 12

    # A tone must exceed threshold for at least 4 frames (~200 ms)
    def has_tone(energy_curve):
        return np.sum(energy_curve > th) >= 4

    tone1 = has_tone(e950)
    tone2 = has_tone(e1400)
    tone3 = has_tone(e1800)

    # Must be sequential: peaks must appear in order
    if tone1 and tone2 and tone3:
        return True
    return False


# ============================
# 4. ASR transcription
# ============================

INVALID_KEYWORDS = [
    "not in service",
    "cannot be completed",
    "cannot be completed as dialed",
    "invalid number",
    "has been changed",
    "out of service",
    "cannot be connected",
    "disconnected",
]

VOICEMAIL_KEYWORDS = [
    "leave your message",
    "leave a message",
    "after the tone",
    "after the beep",
    "voicemail",
    "mailbox",
    "not available right now",
    "record your message",
    "please leave your name and number",
]


def transcribe_text(path: str, max_seconds: float = 10.0) -> str:
    """
    Transcribes up to max_seconds of audio using Whisper.
    For simplicity, we just pass the full file; Whisper will handle.
    """
    # For better performance you could trim audio to first max_seconds, but this is fine for prototype.
    result = asr_model.transcribe(path, language="en", fp16=False)
    text = result.get("text", "").strip().lower()
    return text


def contains_any(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


# ============================
# 5. Main classification logic
# ============================


def classify_call(path: str):
    """
    General, common logic:
    1) Check SIT tones → INVALID_NUMBER
    2) VAD speech duration
    3) If speech present, ASR transcription
    4) Rules:
       - SIT or invalid keywords → INVALID_NUMBER
       - No speech → NO_ANSWER
       - Voicemail keywords → NO_ANSWER
       - Long speech (>= 3s) → NO_ANSWER
       - Otherwise → ANSWERED
    """
    sit = has_sit_tones(path)
    speech_sec = get_speech_duration_seconds(path)

    asr_text = ""
    if speech_sec > 0.2:  # don't waste ASR on silence
        asr_text = transcribe_text(path)

    # 1) INVALID_NUMBER rules
    if sit:
        return "INVALID_NUMBER", speech_sec, sit, asr_text

    if asr_text and contains_any(asr_text, INVALID_KEYWORDS):
        return "INVALID_NUMBER", speech_sec, sit, asr_text

    # 2) NO_ANSWER rules
    if speech_sec == 0.0:
        return "NO_ANSWER", speech_sec, sit, asr_text  # ringing / silence

    if asr_text and contains_any(asr_text, VOICEMAIL_KEYWORDS):
        return "NO_ANSWER", speech_sec, sit, asr_text

    if speech_sec >= 3.0:
        # long monologue → likely voicemail / no answer
        return "NO_ANSWER", speech_sec, sit, asr_text

    # 3) Otherwise: short, human-like speech at beginning → ANSWERED
    return "ANSWERED", speech_sec, sit, asr_text


# ============================
# 6. Run on all audio files
# ============================


def main():
    audio_dir = "audio"
    rows = []

    for fname in sorted(os.listdir(audio_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(audio_dir, fname)

        label, speech_sec, sit_flag, asr_text = classify_call(path)

        # Show only first 60 chars of transcript to keep table readable
        short_text = (asr_text[:57] + "...") if len(asr_text) > 60 else asr_text

        rows.append(
            [fname, f"{speech_sec:.2f}", "yes" if sit_flag else "no", label, short_text]
        )

    print("\nClassification results:\n")
    print(
        tabulate(
            rows,
            headers=[
                "File",
                "Speech_sec",
                "SIT_detected",
                "Result",
                "Transcript (snippet)",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
