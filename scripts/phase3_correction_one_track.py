from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


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

# Decode / analysis settings
TARGET_SR = 44100
MONO = True

# Waveform plotting settings
WAVEFORM_SECONDS = 3
WAVEFORM_START_SECONDS = 0

# FFT / Welch settings
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

# Reduced overlay set for cleaner FFT / Welch comparisons
OVERLAY_PRESET_LABELS = [
    "MP3 128 kbps CBR",
    "MP3 320 kbps CBR",
    "MP3 VBR q5",
]


# ============================================================
# Path helpers
# ============================================================

def sanitize_name(text: str) -> str:
    """
    Make a filename/folder-safe version while keeping it readable.
    """
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
    safe_parts = [sanitize_name(p) for p in parts]
    return "__".join(safe_parts)


def source_wav_path() -> Path:
    return AUDIO_ROOT / SOURCE_RELATIVE_WAV


def encoded_mp3_path(preset: Preset) -> Path:
    return ENCODED_ROOT / preset.folder_name / SOURCE_RELATIVE_WAV.with_suffix(".mp3")


def preset_slug(label: str) -> str:
    return sanitize_name(label)


def make_track_figure_dirs(track_stem: str) -> dict[str, Path]:
    """
    Create a per-track figure tree:
      output/figures/<track_stem>/waveform
      output/figures/<track_stem>/fft
      output/figures/<track_stem>/welch
      output/figures/<track_stem>/difference
      output/figures/<track_stem>/bars
    """
    base = FIG_ROOT / track_stem
    dirs = {
        "base": base,
        "waveform": base / "waveform",
        "fft": base / "fft",
        "welch": base / "welch",
        "difference": base / "difference",
        "bars": base / "bars",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ============================================================
# Audio loading with ffmpeg
# ============================================================

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
        out[key] = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else math.nan

    return out


# ============================================================
# Plotting helpers
# ============================================================

def add_legend_outside() -> None:
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)


def waveform_segment(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    start_idx = int(WAVEFORM_START_SECONDS * sr)
    end_idx = min(len(x), start_idx + int(WAVEFORM_SECONDS * sr))
    x_seg = x[start_idx:end_idx]
    t = np.arange(start_idx, end_idx) / sr
    return t, x_seg


def save_waveform_pair_plot(
    wav_audio: np.ndarray,
    test_audio: np.ndarray,
    test_label: str,
    sr: int,
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))

    t_wav, wav_seg = waveform_segment(wav_audio, sr)
    t_test, test_seg = waveform_segment(test_audio, sr)

    plt.plot(t_wav, wav_seg, linewidth=1.0, label="WAV source")
    plt.plot(t_test, test_seg, linewidth=1.0, label=test_label)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Waveform Comparison: WAV vs {test_label}\n"
        f"Window = {WAVEFORM_START_SECONDS:.2f} to {WAVEFORM_START_SECONDS + WAVEFORM_SECONDS:.2f} s"
    )
    add_legend_outside()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_fft_overlay_plot(
    fft_map: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))

    for label, (f, db) in fft_map.items():
        if label != "WAV source" and label not in OVERLAY_PRESET_LABELS:
            continue
        mask = (f >= 20) & (f <= 22050)
        plt.semilogx(f[mask], db[mask], linewidth=1.0, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, relative)")
    plt.title("FFT Magnitude Comparison")
    add_legend_outside()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_welch_overlay_plot(
    welch_map: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))

    for label, (f, db) in welch_map.items():
        if label != "WAV source" and label not in OVERLAY_PRESET_LABELS:
            continue
        mask = (f >= 20) & (f <= 22050)
        plt.semilogx(f[mask], db[mask], linewidth=1.0, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title("Welch PSD Comparison")
    add_legend_outside()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_welch_difference_plot(
    freqs: np.ndarray,
    ref_db: np.ndarray,
    test_db: np.ndarray,
    test_label: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 5))

    diff_db = test_db - ref_db
    mask = (freqs >= 20) & (freqs <= 22050)

    plt.semilogx(freqs[mask], diff_db[mask], linewidth=1.0, label=f"{test_label} - WAV")
    plt.axhline(0.0, linewidth=1.0, linestyle="--")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Δ PSD (dB)")
    plt.title(f"Welch PSD Difference from WAV: {test_label}")
    add_legend_outside()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_bar_chart(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = source_wav_path()
    if not src_path.exists():
        raise FileNotFoundError(f"Source WAV not found: {src_path}")

    missing_mp3s = [encoded_mp3_path(p) for p in PRESETS if not encoded_mp3_path(p).exists()]
    if missing_mp3s:
        raise FileNotFoundError(
            "One or more encoded MP3s are missing:\n" + "\n".join(str(p) for p in missing_mp3s)
        )

    track_stem = build_track_stem(SOURCE_RELATIVE_WAV)
    fig_dirs = make_track_figure_dirs(track_stem)

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

    labels = list(audio_map.keys())
    aligned = align_to_shortest([audio_map[label] for label in labels])
    audio_map = {label: arr for label, arr in zip(labels, aligned)}

    fft_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    welch_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows: list[dict[str, str | float]] = []

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

    # --------------------------------------------------------
    # Save separate waveform plots for each encode
    # --------------------------------------------------------
    for preset in PRESETS:
        label = preset.label
        out_path = (
            fig_dirs["waveform"]
            / f"{track_stem}__waveform__WAV_vs_{preset_slug(label)}__{RUN_TIMESTAMP}.png"
        )
        save_waveform_pair_plot(
            wav_audio=audio_map["WAV source"],
            test_audio=audio_map[label],
            test_label=label,
            sr=sr,
            out_path=out_path,
        )

    # --------------------------------------------------------
    # Save reduced overlay FFT / Welch plots
    # --------------------------------------------------------
    fft_out = fig_dirs["fft"] / f"{track_stem}__fft_overlay__{RUN_TIMESTAMP}.png"
    welch_out = fig_dirs["welch"] / f"{track_stem}__welch_overlay__{RUN_TIMESTAMP}.png"

    save_fft_overlay_plot(fft_map, fft_out)
    save_welch_overlay_plot(welch_map, welch_out)

    # --------------------------------------------------------
    # Save one Welch difference-from-WAV plot per preset
    # --------------------------------------------------------
    for preset in PRESETS:
        label = preset.label
        f_test, db_test = welch_map[label]
        diff_out = (
            fig_dirs["difference"]
            / f"{track_stem}__welch_difference__{preset_slug(label)}__{RUN_TIMESTAMP}.png"
        )
        save_welch_difference_plot(
            freqs=f_test,
            ref_db=ref_welch_db,
            test_db=db_test,
            test_label=label,
            out_path=diff_out,
        )

    # --------------------------------------------------------
    # Save summary bar chart
    # --------------------------------------------------------
    bar_labels = []
    mae_values = []
    crest_values = []

    for row in metric_rows:
        if row["label"] == "WAV source":
            continue
        bar_labels.append(str(row["label"]))
        mae_values.append(float(row["spectral_mae_db"]))
        crest_values.append(float(row["crest_factor_db"]))

    mae_bar_out = fig_dirs["bars"] / f"{track_stem}__bar__spectral_mae__{RUN_TIMESTAMP}.png"
    crest_bar_out = fig_dirs["bars"] / f"{track_stem}__bar__crest_factor__{RUN_TIMESTAMP}.png"

    save_metric_bar_chart(
        labels=bar_labels,
        values=mae_values,
        ylabel="Spectral MAE (dB)",
        title="Spectral Difference from WAV by Preset",
        out_path=mae_bar_out,
    )

    save_metric_bar_chart(
        labels=bar_labels,
        values=crest_values,
        ylabel="Crest Factor (dB)",
        title="Crest Factor by Preset",
        out_path=crest_bar_out,
    )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------
    summary_csv = TABLE_ROOT / f"{track_stem}__analysis_{RUN_TIMESTAMP}.csv"

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
    print(f"Figure root:      {fig_dirs['base']}")
    print(f"Waveform dir:     {fig_dirs['waveform']}")
    print(f"FFT dir:          {fig_dirs['fft']}")
    print(f"Welch dir:        {fig_dirs['welch']}")
    print(f"Difference dir:   {fig_dirs['difference']}")
    print(f"Bar chart dir:    {fig_dirs['bars']}")
    print(f"Summary CSV:      {summary_csv}")


if __name__ == "__main__":
    main()