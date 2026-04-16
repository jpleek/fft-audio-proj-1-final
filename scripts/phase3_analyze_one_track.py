from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly, welch


# ============================================================
# User settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
FIG_ROOT = PROJECT_ROOT / "output" / "figures"
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Pick one source WAV relative to AUDIO_ROOT
SOURCE_RELATIVE_WAV = Path(
    "Deftones/around the fur/01 Deftones - My Own Summer (Shove It).wav"
)

# Analysis settings
TARGET_SR = 44100
MONO = True

# Plot settings
WAVEFORM_SECONDS = 30
MAX_FFT_SECONDS = 60
WELCH_NPERSEG = 8192

# Small floor to avoid log(0)
EPS = 1e-12


@dataclass(frozen=True)
class Preset:
    folder_name: str
    label: str


PRESETS = [
    Preset("mp3_128", "MP3 128 kbps CBR"),
    Preset("mp3_192", "MP3 192 kbps CBR"),
    Preset("mp3_320", "MP3 320 kbps CBR"),
    Preset("mp3_v0", "MP3 VBR q0"),
    Preset("mp3_v2", "MP3 VBR q2"),
    Preset("mp3_v5", "MP3 VBR q5"),
]


# ============================================================
# Path helpers
# ============================================================

def build_output_track_stem(source_rel_wav: Path) -> str:
    """
    Create a clean track stem for output filenames.
    Example:
      Deftones/around the fur/01 ... .wav
      -> Deftones__around_the_fur__01_Deftones_-_My_Own_Summer_(Shove_It)
    """
    parts = list(source_rel_wav.with_suffix("").parts)
    safe_parts = []
    for p in parts:
        p = (
            p.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        safe_parts.append(p)
    return "__".join(safe_parts)


def source_wav_path() -> Path:
    return AUDIO_ROOT / SOURCE_RELATIVE_WAV


def encoded_mp3_path(preset: Preset) -> Path:
    return ENCODED_ROOT / preset.folder_name / SOURCE_RELATIVE_WAV.with_suffix(".mp3")


# ============================================================
# Audio loading with ffmpeg
# ============================================================

def ffprobe_sample_rate_and_channels(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr}")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Could not parse ffprobe sample rate/channels for {path}")

    sample_rate = int(lines[0])
    channels = int(lines[1])
    return sample_rate, channels


def load_audio_ffmpeg(path: Path, target_sr: int = TARGET_SR, mono: bool = MONO) -> tuple[np.ndarray, int]:
    """
    Load audio via ffmpeg -> raw float32 PCM.
    This avoids notebook-style dependency pain and handles WAV/MP3 reliably.
    """
    ac_value = "1" if mono else "2"

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", ac_value,
        "-ar", str(target_sr),
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True, check=False)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed for {path}:\n{stderr}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)

    if not mono:
        audio = audio.reshape(-1, 2)

    return audio, target_sr


# ============================================================
# Metrics
# ============================================================

def dbfs_rms(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x)) + EPS)
    return 20 * np.log10(rms + EPS)


def dbfs_peak(x: np.ndarray) -> float:
    peak = np.max(np.abs(x)) + EPS
    return 20 * np.log10(peak)


def crest_factor_db(x: np.ndarray) -> float:
    peak = np.max(np.abs(x)) + EPS
    rms = np.sqrt(np.mean(np.square(x)) + EPS)
    return 20 * np.log10(peak / rms + EPS)


def align_to_shortest(arrays: list[np.ndarray]) -> list[np.ndarray]:
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]


def fft_magnitude_db(x: np.ndarray, sr: int, max_seconds: int = MAX_FFT_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT on up to max_seconds of audio using a Hann window.
    """
    n_max = min(len(x), sr * max_seconds)
    x_use = x[:n_max]

    window = np.hanning(len(x_use))
    xw = x_use * window

    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(len(xw), d=1 / sr)

    mag = np.abs(X)
    mag_db = 20 * np.log10(mag + EPS)

    return f, mag_db


def welch_psd_db(x: np.ndarray, sr: int, nperseg: int = WELCH_NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    f, pxx = welch(
        x,
        fs=sr,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(nperseg, len(x)) // 2,
        detrend="constant",
        scaling="density",
    )
    pxx_db = 10 * np.log10(pxx + EPS)
    return f, pxx_db


def spectral_difference_metrics(ref_db: np.ndarray, test_db: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    diff = test_db - ref_db

    overall_mae_db = float(np.mean(np.abs(diff)))
    overall_rmse_db = float(np.sqrt(np.mean(diff ** 2)))
    overall_bias_db = float(np.mean(diff))
    spectral_corr = float(np.corrcoef(ref_db, test_db)[0, 1])

    def band_mask(f_lo: float, f_hi: float) -> np.ndarray:
        return (freqs >= f_lo) & (freqs < f_hi)

    bands = {
        "mae_20_200_db": band_mask(20, 200),
        "mae_200_2000_db": band_mask(200, 2000),
        "mae_2000_8000_db": band_mask(2000, 8000),
        "mae_8000_20000_db": band_mask(8000, 20000),
    }

    out = {
        "spectral_mae_db": overall_mae_db,
        "spectral_rmse_db": overall_rmse_db,
        "spectral_bias_db": overall_bias_db,
        "spectral_corr": spectral_corr,
    }

    for key, mask in bands.items():
        if np.any(mask):
            out[key] = float(np.mean(np.abs(diff[mask])))
        else:
            out[key] = math.nan

    return out


# ============================================================
# Plotting
# ============================================================

def save_waveform_plot(audio_map: dict[str, np.ndarray], sr: int, out_path: Path) -> None:
    plt.figure(figsize=(12, 7))

    n = min(sr * WAVEFORM_SECONDS, min(len(x) for x in audio_map.values()))
    t = np.arange(n) / sr

    for label, x in audio_map.items():
        plt.plot(t, x[:n], linewidth=0.8, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform Comparison (first {n / sr:.0f} s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_fft_plot(fft_map: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(12, 7))

    for label, (f, db) in fft_map.items():
        mask = (f >= 20) & (f <= 22050)
        plt.semilogx(f[mask], db[mask], linewidth=1.0, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, relative)")
    plt.title("FFT Magnitude Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_welch_plot(welch_map: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure(figsize=(12, 7))

    for label, (f, db) in welch_map.items():
        mask = (f >= 20) & (f <= 22050)
        plt.semilogx(f[mask], db[mask], linewidth=1.0, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = source_wav_path()
    if not src_path.exists():
        raise FileNotFoundError(f"Source WAV not found: {src_path}")

    missing_mp3s = [encoded_mp3_path(p) for p in PRESETS if not encoded_mp3_path(p).exists()]
    if missing_mp3s:
        raise FileNotFoundError(
            "One or more encoded MP3s are missing:\n" + "\n".join(str(p) for p in missing_mp3s)
        )

    track_stem = build_output_track_stem(SOURCE_RELATIVE_WAV)

    print(f"Loading source WAV: {src_path}")
    src_audio, sr = load_audio_ffmpeg(src_path, target_sr=TARGET_SR, mono=MONO)

    audio_map: dict[str, np.ndarray] = {"WAV source": src_audio}

    for preset in PRESETS:
        mp3_path = encoded_mp3_path(preset)
        print(f"Loading encoded file: {mp3_path}")
        mp3_audio, mp3_sr = load_audio_ffmpeg(mp3_path, target_sr=TARGET_SR, mono=MONO)

        if mp3_sr != sr:
            raise RuntimeError(f"Sample rate mismatch after decoding: {mp3_path}")

        audio_map[preset.label] = mp3_audio

    # Align all arrays to the shortest decoded length
    labels = list(audio_map.keys())
    aligned = align_to_shortest([audio_map[label] for label in labels])
    audio_map = {label: arr for label, arr in zip(labels, aligned)}

    fft_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    welch_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, str | float]] = []

    # Reference spectra from WAV
    ref_fft_f, ref_fft_db = fft_magnitude_db(audio_map["WAV source"], sr=sr)
    ref_welch_f, ref_welch_db = welch_psd_db(audio_map["WAV source"], sr=sr)

    for label, x in audio_map.items():
        fft_f, fft_db = fft_magnitude_db(x, sr=sr)
        welch_f, welch_db = welch_psd_db(x, sr=sr)

        fft_map[label] = (fft_f, fft_db)
        welch_map[label] = (welch_f, welch_db)

        row: dict[str, str | float] = {
            "track": str(SOURCE_RELATIVE_WAV),
            "label": label,
            "duration_sec": len(x) / sr,
            "sample_rate_hz": sr,
            "peak_dbfs": dbfs_peak(x),
            "rms_dbfs": dbfs_rms(x),
            "crest_factor_db": crest_factor_db(x),
        }

        if label == "WAV source":
            row.update({
                "spectral_mae_db": 0.0,
                "spectral_rmse_db": 0.0,
                "spectral_bias_db": 0.0,
                "spectral_corr": 1.0,
                "mae_20_200_db": 0.0,
                "mae_200_2000_db": 0.0,
                "mae_2000_8000_db": 0.0,
                "mae_8000_20000_db": 0.0,
            })
        else:
            diff_metrics = spectral_difference_metrics(ref_welch_db, welch_db, welch_f)
            row.update(diff_metrics)

        metric_rows.append(row)

    waveform_fig = FIG_ROOT / f"{track_stem}__waveform_{RUN_TIMESTAMP}.png"
    fft_fig = FIG_ROOT / f"{track_stem}__fft_{RUN_TIMESTAMP}.png"
    welch_fig = FIG_ROOT / f"{track_stem}__welch_{RUN_TIMESTAMP}.png"
    summary_csv = TABLE_ROOT / f"{track_stem}__analysis_{RUN_TIMESTAMP}.csv"

    save_waveform_plot(audio_map, sr, waveform_fig)
    save_fft_plot(fft_map, fft_fig)
    save_welch_plot(welch_map, welch_fig)

    fieldnames = [
        "track",
        "label",
        "duration_sec",
        "sample_rate_hz",
        "peak_dbfs",
        "rms_dbfs",
        "crest_factor_db",
        "spectral_mae_db",
        "spectral_rmse_db",
        "spectral_bias_db",
        "spectral_corr",
        "mae_20_200_db",
        "mae_200_2000_db",
        "mae_2000_8000_db",
        "mae_8000_20000_db",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    print("\nAnalysis complete.")
    print(f"Waveform figure: {waveform_fig}")
    print(f"FFT figure:      {fft_fig}")
    print(f"Welch figure:    {welch_fig}")
    print(f"Summary CSV:     {summary_csv}")


if __name__ == "__main__":
    main()