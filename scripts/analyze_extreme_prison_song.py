from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
DEMO_ROOT = PROJECT_ROOT / "output" / "extreme_demo" / "Prison_Song"
DEMO_AUDIO_ROOT = DEMO_ROOT / "audio"
FIG_ROOT = DEMO_ROOT / "figures"
TABLE_ROOT = DEMO_ROOT / "tables"

SEARCH_TERM = "Prison Song"

TARGET_SR = 44100
MONO = True

MAX_WAVEFORM_SECONDS = 5
MAX_FFT_SECONDS = 60
WELCH_NPERSEG = 8192
EPS = 1e-12


def find_target_wav(audio_root: Path, search_term: str) -> Path:
    matches = sorted(
        [p for p in audio_root.rglob("*.wav") if search_term.lower() in p.name.lower()]
    )

    if not matches:
        raise FileNotFoundError(
            f'No WAV file found under {audio_root} containing "{search_term}" in the filename.'
        )

    if len(matches) > 1:
        print("Multiple matches found:")
        for m in matches:
            print(f"  {m}")
        raise RuntimeError(
            f'More than one WAV matched "{search_term}". Narrow SEARCH_TERM or hardcode the source path.'
        )

    return matches[0]


def load_audio_ffmpeg(path: Path, target_sr: int = TARGET_SR, mono: bool = MONO) -> tuple[np.ndarray, int]:
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


def fft_magnitude_db(x: np.ndarray, sr: int, max_seconds: int = MAX_FFT_SECONDS) -> tuple[np.ndarray, np.ndarray]:
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

    out = {
        "spectral_mae_db": float(np.mean(np.abs(diff))),
        "spectral_rmse_db": float(np.sqrt(np.mean(diff**2))),
        "spectral_bias_db": float(np.mean(diff)),
        "spectral_corr": float(np.corrcoef(ref_db, test_db)[0, 1]),
    }

    bands = [
        ("mae_20_200_db", 20, 200),
        ("mae_200_2000_db", 200, 2000),
        ("mae_2000_8000_db", 2000, 8000),
        ("mae_8000_20000_db", 8000, 20000),
    ]

    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else np.nan

    return out


def basic_level_metrics(x: np.ndarray) -> dict[str, float]:
    peak = np.max(np.abs(x)) + EPS
    rms = np.sqrt(np.mean(x**2) + EPS)

    peak_dbfs = 20 * np.log10(peak)
    rms_dbfs = 20 * np.log10(rms)
    crest_factor_db = peak_dbfs - rms_dbfs

    return {
        "peak_dbfs": float(peak_dbfs),
        "rms_dbfs": float(rms_dbfs),
        "crest_factor_db": float(crest_factor_db),
    }


def plot_waveform(x: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    n = min(len(x), sr * MAX_WAVEFORM_SECONDS)
    x_use = x[:n]
    t = np.arange(len(x_use)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(t, x_use, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fft_overlay(curves: list[tuple[np.ndarray, np.ndarray, str]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 5))
    for f, db, label in curves:
        plt.plot(f, db, linewidth=1.0, label=label)
    plt.xscale("log")
    plt.xlim(20, 20000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_welch_overlay(curves: list[tuple[np.ndarray, np.ndarray, str]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 5))
    for f, db, label in curves:
        plt.plot(f, db, linewidth=1.0, label=label)
    plt.xscale("log")
    plt.xlim(20, 20000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_curve(freqs: np.ndarray, diff_db: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, diff_db, linewidth=1.0)
    plt.axhline(0, linewidth=1.0)
    plt.xscale("log")
    plt.xlim(20, 20000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Δ PSD vs WAV (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def discover_demo_audio(audio_root: Path) -> list[Path]:
    allowed_ext = {".wav", ".mp3"}
    return sorted([p for p in audio_root.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext])


def main() -> None:
    if not DEMO_AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"Demo audio folder not found: {DEMO_AUDIO_ROOT}\nRun extreme_degrade_prison_song.py first."
        )

    source_wav = find_target_wav(AUDIO_ROOT, SEARCH_TERM)

    waveform_dir = FIG_ROOT / "waveform"
    fft_dir = FIG_ROOT / "fft"
    welch_dir = FIG_ROOT / "welch"
    difference_dir = FIG_ROOT / "difference"

    for d in [waveform_dir, fft_dir, welch_dir, difference_dir, TABLE_ROOT]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading source WAV: {source_wav}")
    wav_audio, sr = load_audio_ffmpeg(source_wav, target_sr=TARGET_SR, mono=MONO)

    demo_files = discover_demo_audio(DEMO_AUDIO_ROOT)
    if not demo_files:
        raise FileNotFoundError(f"No demo audio files found in: {DEMO_AUDIO_ROOT}")

    decoded = {"WAV source": wav_audio}

    for path in demo_files:
        print(f"Loading demo file: {path}")
        audio, sr_demo = load_audio_ffmpeg(path, target_sr=TARGET_SR, mono=MONO)
        if sr_demo != sr:
            raise RuntimeError(f"Sample rate mismatch for {path}: {sr_demo} vs {sr}")
        decoded[path.stem] = audio

    # Align all files to shortest length
    min_len = min(len(x) for x in decoded.values())
    decoded = {label: x[:min_len] for label, x in decoded.items()}

    rows = []

    # Waveform plots for each file
    for label, audio in decoded.items():
        safe_label = label.replace("/", "_")
        plot_waveform(
            audio,
            sr,
            f"Waveform\n{label}",
            waveform_dir / f"{safe_label}__waveform.png",
        )

    # Reference FFT / Welch
    wav_fft_f, wav_fft_db = fft_magnitude_db(decoded["WAV source"], sr)
    wav_welch_f, wav_welch_db = welch_psd_db(decoded["WAV source"], sr)

    fft_curves = [(wav_fft_f, wav_fft_db, "WAV source")]
    welch_curves = [(wav_welch_f, wav_welch_db, "WAV source")]

    # WAV metrics row
    rows.append({
        "label": "WAV source",
        "file_name": source_wav.name,
        "duration_sec": len(decoded["WAV source"]) / sr,
        "sample_rate_hz": sr,
        **basic_level_metrics(decoded["WAV source"]),
        "spectral_mae_db": 0.0,
        "spectral_rmse_db": 0.0,
        "spectral_bias_db": 0.0,
        "spectral_corr": 1.0,
        "mae_20_200_db": 0.0,
        "mae_200_2000_db": 0.0,
        "mae_2000_8000_db": 0.0,
        "mae_8000_20000_db": 0.0,
    })

    for path in demo_files:
        label = path.stem
        audio = decoded[label]

        fft_f, fft_db = fft_magnitude_db(audio, sr)
        welch_f, welch_db = welch_psd_db(audio, sr)

        fft_curves.append((fft_f, fft_db, label))
        welch_curves.append((welch_f, welch_db, label))

        diff_db = welch_db - wav_welch_db
        metrics = spectral_difference_metrics(wav_welch_db, welch_db, wav_welch_f)
        levels = basic_level_metrics(audio)

        plot_difference_curve(
            wav_welch_f,
            diff_db,
            difference_dir / f"{label}__difference_from_wav.png",
            f"Difference from WAV\n{label}",
        )

        rows.append({
            "label": label,
            "file_name": path.name,
            "duration_sec": len(audio) / sr,
            "sample_rate_hz": sr,
            **levels,
            **metrics,
        })

    plot_fft_overlay(
        fft_curves,
        fft_dir / "Prison_Song__fft_overlay.png",
        "FFT Overlay\nPrison Song Extreme Demo",
    )

    plot_welch_overlay(
        welch_curves,
        welch_dir / "Prison_Song__welch_overlay.png",
        "Welch Overlay\nPrison Song Extreme Demo",
    )

    csv_path = TABLE_ROOT / "Prison_Song__extreme_demo_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Waveform plots: {waveform_dir}")
    print(f"FFT overlay:    {fft_dir}")
    print(f"Welch overlay:  {welch_dir}")
    print(f"Differences:    {difference_dir}")
    print(f"Metrics CSV:    {csv_path}")


if __name__ == "__main__":
    main()