from __future__ import annotations

import subprocess
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
FIGURES_ROOT = PROJECT_ROOT / "output" / "figures"
TABLES_ROOT = PROJECT_ROOT / "output" / "tables"
LOGS_ROOT = PROJECT_ROOT / "output" / "logs"
PER_TRACK_TABLES = TABLES_ROOT / "per_track_analysis"

TARGET_SR = 44100
SPECTRAL_MONO = True
LEVEL_MONO = False

MAX_WAVEFORM_SECONDS = 5
MAX_FFT_SECONDS = 60
WELCH_NPERSEG = 8192
EPS = 1e-12

PRESETS = [
    {"folder": "mp3_128", "label": "MP3 128 kbps CBR", "short": "128 CBR"},
    {"folder": "mp3_192", "label": "MP3 192 kbps CBR", "short": "192 CBR"},
    {"folder": "mp3_320", "label": "MP3 320 kbps CBR", "short": "320 CBR"},
    {"folder": "mp3_v0",  "label": "MP3 VBR q0",       "short": "V0"},
    {"folder": "mp3_v2",  "label": "MP3 VBR q2",       "short": "V2"},
    {"folder": "mp3_v5",  "label": "MP3 VBR q5",       "short": "V5"},
]

BANDS = [
    ("band_20_200_hz_mae", 20, 200),
    ("band_200_2000_hz_mae", 200, 2000),
    ("band_2000_8000_hz_mae", 2000, 8000),
    ("band_8000_20000_hz_mae", 8000, 20000),
]


def sanitize_name(text: str) -> str:
    replacements = {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
        "?": "",
        "!": "",
        "\"": "",
        "'": "",
        "*": "",
        "|": "_",
        "<": "_",
        ">": "_",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def build_track_stem(source_rel_wav: Path) -> str:
    parts = list(source_rel_wav.with_suffix("").parts)
    return "__".join(sanitize_name(p) for p in parts)


def discover_wav_files(audio_root: Path) -> list[Path]:
    return sorted([p for p in audio_root.rglob("*.wav") if p.is_file()])


def load_audio_ffmpeg(path: Path, target_sr: int = TARGET_SR, mono: bool = True) -> tuple[np.ndarray, int]:
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

    if mono:
        return audio, target_sr

    audio = audio.reshape(-1, 2)
    return audio, target_sr


def align_mono_signals(*signals: np.ndarray) -> list[np.ndarray]:
    min_len = min(len(x) for x in signals)
    return [x[:min_len] for x in signals]


def align_stereo_signals(*signals: np.ndarray) -> list[np.ndarray]:
    min_len = min(x.shape[0] for x in signals)
    return [x[:min_len] for x in signals]


def basic_level_metrics(x: np.ndarray) -> dict[str, float]:
    x_use = x.reshape(-1) if x.ndim == 2 else x
    peak = np.max(np.abs(x_use)) + EPS
    rms = np.sqrt(np.mean(x_use**2) + EPS)

    peak_dbfs = 20 * np.log10(peak)
    rms_dbfs = 20 * np.log10(rms)
    crest_factor_db = peak_dbfs - rms_dbfs

    return {
        "peak_dbfs": float(peak_dbfs),
        "rms_dbfs": float(rms_dbfs),
        "crest_factor_db": float(crest_factor_db),
    }


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
    }

    corr = np.corrcoef(ref_db, test_db)[0, 1]
    out["spectral_corr"] = float(corr)

    for band_name, f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        out[band_name] = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else np.nan

    return out


def encoded_path_for(wav_rel: Path, preset_folder: str) -> Path:
    return ENCODED_ROOT / preset_folder / wav_rel.with_suffix(".mp3")


def ensure_dirs() -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    PER_TRACK_TABLES.mkdir(parents=True, exist_ok=True)


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
    plt.xlim(20, 20000)
    plt.xscale("log")
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
    plt.xlim(20, 20000)
    plt.xscale("log")
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
    plt.xlim(20, 20000)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Δ PSD vs WAV (dB)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_bars(metric_map: dict[str, float], ylabel: str, title: str, out_path: Path) -> None:
    labels = list(metric_map.keys())
    values = list(metric_map.values())

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, width=0.6)
    plt.ylabel(ylabel)
    plt.xlabel("Preset")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_track(wav_path: Path, timestamp: str) -> list[dict]:
    wav_rel = wav_path.relative_to(AUDIO_ROOT)
    track_stem = build_track_stem(wav_rel)

    print(f"\nProcessing: {wav_rel}")

    # Decode WAV
    wav_spec, sr = load_audio_ffmpeg(wav_path, mono=SPECTRAL_MONO)
    wav_level, sr_level = load_audio_ffmpeg(wav_path, mono=LEVEL_MONO)
    if sr != sr_level:
        raise RuntimeError(f"Sample-rate mismatch for WAV: {wav_path}")

    encoded_spec = {}
    encoded_level = {}

    for preset in PRESETS:
        mp3_path = encoded_path_for(wav_rel, preset["folder"])
        if not mp3_path.exists():
            raise FileNotFoundError(f"Missing encoded file: {mp3_path}")

        enc_spec, enc_sr = load_audio_ffmpeg(mp3_path, mono=SPECTRAL_MONO)
        enc_level, enc_sr_level = load_audio_ffmpeg(mp3_path, mono=LEVEL_MONO)

        if enc_sr != sr or enc_sr_level != sr:
            raise RuntimeError(f"Sample-rate mismatch for encoded file: {mp3_path}")

        encoded_spec[preset["folder"]] = enc_spec
        encoded_level[preset["folder"]] = enc_level

    # Align separately
    mono_aligned = align_mono_signals(wav_spec, *[encoded_spec[p["folder"]] for p in PRESETS])
    wav_spec = mono_aligned[0]
    for i, preset in enumerate(PRESETS, start=1):
        encoded_spec[preset["folder"]] = mono_aligned[i]

    stereo_aligned = align_stereo_signals(wav_level, *[encoded_level[p["folder"]] for p in PRESETS])
    wav_level = stereo_aligned[0]
    for i, preset in enumerate(PRESETS, start=1):
        encoded_level[preset["folder"]] = stereo_aligned[i]

    # Figure tree
    fig_root = FIGURES_ROOT / track_stem
    waveform_dir = fig_root / "waveform"
    fft_dir = fig_root / "fft"
    welch_dir = fig_root / "welch"
    difference_dir = fig_root / "difference"
    bars_dir = fig_root / "bars"

    for d in [waveform_dir, fft_dir, welch_dir, difference_dir, bars_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []

    # WAV metrics
    wav_level_metrics = basic_level_metrics(wav_level)
    wav_fft_f, wav_fft_db = fft_magnitude_db(wav_spec, sr)
    wav_welch_f, wav_welch_db = welch_psd_db(wav_spec, sr)

    rows.append({
        "track": str(wav_rel),
        "track_stem": track_stem,
        "label": "WAV source",
        "preset_folder": "source_wav",
        "preset_short_label": "WAV",
        "duration_sec": len(wav_spec) / sr,
        "sample_rate_hz": sr,
        **wav_level_metrics,
        "spectral_mae_db": 0.0,
        "spectral_rmse_db": 0.0,
        "spectral_bias_db": 0.0,
        "spectral_corr": 1.0,
        "band_20_200_hz_mae": 0.0,
        "band_200_2000_hz_mae": 0.0,
        "band_2000_8000_hz_mae": 0.0,
        "band_8000_20000_hz_mae": 0.0,
    })

    plot_waveform(
        wav_spec, sr,
        f"WAV waveform\n{wav_rel}",
        waveform_dir / f"{track_stem}__wav_waveform__{timestamp}.png"
    )

    fft_curves = [(wav_fft_f, wav_fft_db, "WAV")]
    welch_curves = [(wav_welch_f, wav_welch_db, "WAV")]

    mae_map = {}
    rmse_map = {}
    corr_map = {}
    delta_rms_map = {}
    delta_crest_map = {}

    for preset in PRESETS:
        spec_audio = encoded_spec[preset["folder"]]
        level_audio = encoded_level[preset["folder"]]

        level_metrics = basic_level_metrics(level_audio)
        fft_f, fft_db = fft_magnitude_db(spec_audio, sr)
        welch_f, welch_db = welch_psd_db(spec_audio, sr)

        spec_metrics = spectral_difference_metrics(wav_welch_db, welch_db, wav_welch_f)

        rows.append({
            "track": str(wav_rel),
            "track_stem": track_stem,
            "label": preset["label"],
            "preset_folder": preset["folder"],
            "preset_short_label": preset["short"],
            "duration_sec": len(spec_audio) / sr,
            "sample_rate_hz": sr,
            **level_metrics,
            **spec_metrics,
        })

        fft_curves.append((fft_f, fft_db, preset["short"]))
        welch_curves.append((welch_f, welch_db, preset["short"]))

        diff_db = welch_db - wav_welch_db
        plot_difference_curve(
            wav_welch_f,
            diff_db,
            difference_dir / f"{track_stem}__difference__{preset['short']}__{timestamp}.png",
            f"Welch PSD Difference from WAV\n{wav_rel} | {preset['short']}"
        )

        plot_waveform(
            spec_audio, sr,
            f"{preset['short']} waveform\n{wav_rel}",
            waveform_dir / f"{track_stem}__waveform__{preset['short']}__{timestamp}.png"
        )

        mae_map[preset["short"]] = spec_metrics["spectral_mae_db"]
        rmse_map[preset["short"]] = spec_metrics["spectral_rmse_db"]
        corr_map[preset["short"]] = spec_metrics["spectral_corr"]
        delta_rms_map[preset["short"]] = level_metrics["rms_dbfs"] - wav_level_metrics["rms_dbfs"]
        delta_crest_map[preset["short"]] = level_metrics["crest_factor_db"] - wav_level_metrics["crest_factor_db"]

    plot_fft_overlay(
        fft_curves,
        fft_dir / f"{track_stem}__fft_overlay__{timestamp}.png",
        f"FFT Magnitude Overlay\n{wav_rel}"
    )

    plot_welch_overlay(
        welch_curves,
        welch_dir / f"{track_stem}__welch_overlay__{timestamp}.png",
        f"Welch PSD Overlay\n{wav_rel}"
    )

    plot_metric_bars(
        mae_map,
        "Spectral MAE (dB)",
        f"Spectral MAE by Preset\n{wav_rel}",
        bars_dir / f"{track_stem}__bars_spectral_mae__{timestamp}.png"
    )

    plot_metric_bars(
        rmse_map,
        "Spectral RMSE (dB)",
        f"Spectral RMSE by Preset\n{wav_rel}",
        bars_dir / f"{track_stem}__bars_spectral_rmse__{timestamp}.png"
    )

    plot_metric_bars(
        corr_map,
        "Spectral Correlation",
        f"Spectral Correlation by Preset\n{wav_rel}",
        bars_dir / f"{track_stem}__bars_spectral_corr__{timestamp}.png"
    )

    plot_metric_bars(
        delta_rms_map,
        "Δ RMS (dB)",
        f"RMS Change Relative to WAV\n{wav_rel}",
        bars_dir / f"{track_stem}__bars_delta_rms__{timestamp}.png"
    )

    plot_metric_bars(
        delta_crest_map,
        "Δ Crest Factor (dB)",
        f"Crest Factor Change Relative to WAV\n{wav_rel}",
        bars_dir / f"{track_stem}__bars_delta_crest__{timestamp}.png"
    )

    per_track_df = pd.DataFrame(rows)
    per_track_csv = PER_TRACK_TABLES / f"{track_stem}__analysis__{timestamp}.csv"
    per_track_df.to_csv(per_track_csv, index=False)

    return rows


def main() -> None:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wav_files = discover_wav_files(AUDIO_ROOT)
    all_rows = []

    for idx, wav_path in enumerate(wav_files, start=1):
        print(f"[{idx}/{len(wav_files)}] {wav_path.relative_to(AUDIO_ROOT)}")
        rows = process_track(wav_path, timestamp)
        all_rows.extend(rows)

    batch_df = pd.DataFrame(all_rows)
    batch_csv = TABLES_ROOT / f"phase3_batch_analysis_v2_{timestamp}.csv"
    batch_df.to_csv(batch_csv, index=False)

    log_path = LOGS_ROOT / f"phase3_batch_analysis_v2_{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Batch analysis v2 complete.\n")
        f.write(f"Processed tracks: {len(wav_files)}\n")
        f.write(f"Batch CSV: {batch_csv}\n")
        f.write(f"Per-track tables: {PER_TRACK_TABLES}\n")

    print("\nBatch analysis v2 complete.")
    print(f"Processed tracks: {len(wav_files)}")
    print(f"Batch CSV:        {batch_csv}")
    print(f"Batch log:        {log_path}")
    print(f"Per-track tables: {PER_TRACK_TABLES}")


if __name__ == "__main__":
    main()