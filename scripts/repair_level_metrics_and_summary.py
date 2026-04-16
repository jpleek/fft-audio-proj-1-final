from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
PER_TRACK_DIR = PROJECT_ROOT / "output" / "tables" / "per_track_analysis"
TABLES_ROOT = PROJECT_ROOT / "output" / "tables"
FIGURES_ROOT = PROJECT_ROOT / "output" / "figures"

BATCH_SUMMARY_DIR = TABLES_ROOT / "batch_summary_repaired_levels"
BATCH_SUMMARY_FIG_DIR = FIGURES_ROOT / "batch_summary_repaired_levels"

TARGET_SR = 44100
EPS = 1e-12

PRESETS = [
    {"folder": "mp3_128", "short_label": "128 CBR", "long_label": "MP3 128 kbps CBR"},
    {"folder": "mp3_192", "short_label": "192 CBR", "long_label": "MP3 192 kbps CBR"},
    {"folder": "mp3_320", "short_label": "320 CBR", "long_label": "MP3 320 kbps CBR"},
    {"folder": "mp3_v0",  "short_label": "V0",      "long_label": "MP3 VBR q0"},
    {"folder": "mp3_v2",  "short_label": "V2",      "long_label": "MP3 VBR q2"},
    {"folder": "mp3_v5",  "short_label": "V5",      "long_label": "MP3 VBR q5"},
]

PRESET_ORDER = ["WAV", "128 CBR", "192 CBR", "320 CBR", "V0", "V2", "V5"]
ENCODE_ORDER = ["128 CBR", "192 CBR", "320 CBR", "V0", "V2", "V5"]


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
    safe_parts = [sanitize_name(p) for p in parts]
    return "__".join(safe_parts)


def discover_wav_files(audio_root: Path) -> list[Path]:
    return sorted([p for p in audio_root.rglob("*.wav") if p.is_file()])


def load_audio_ffmpeg(path: Path, target_sr: int = TARGET_SR, mono: bool = False) -> tuple[np.ndarray, int]:
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


def basic_level_metrics(x: np.ndarray) -> dict[str, float]:
    if x.ndim == 2:
        x_use = x.reshape(-1)
    else:
        x_use = x

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


def encoded_path_for(wav_rel: Path, preset_folder: str) -> Path:
    return ENCODED_ROOT / preset_folder / wav_rel.with_suffix(".mp3")


def latest_per_track_csv(track_stem: str) -> Path:
    matches = sorted(PER_TRACK_DIR.glob(f"{track_stem}__analysis_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No per-track CSV found for track_stem={track_stem}")
    return matches[-1]


def save_bar_chart(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(series.index, series.values, width=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Preset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    BATCH_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_SUMMARY_FIG_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = discover_wav_files(AUDIO_ROOT)
    repaired_rows = []

    for idx, wav_path in enumerate(wav_files, start=1):
        wav_rel = wav_path.relative_to(AUDIO_ROOT)
        track_stem = build_track_stem(wav_rel)

        print(f"[{idx}/{len(wav_files)}] Repairing level metrics: {wav_rel}")

        csv_path = latest_per_track_csv(track_stem)
        df = pd.read_csv(csv_path)

        # Repair WAV row
        wav_audio, sr = load_audio_ffmpeg(wav_path, mono=False)
        wav_levels = basic_level_metrics(wav_audio)

        wav_mask = df["preset_short_label"] == "WAV"
        df.loc[wav_mask, "peak_dbfs"] = wav_levels["peak_dbfs"]
        df.loc[wav_mask, "rms_dbfs"] = wav_levels["rms_dbfs"]
        df.loc[wav_mask, "crest_factor_db"] = wav_levels["crest_factor_db"]

        # Repair encoded rows
        for preset in PRESETS:
            mp3_path = encoded_path_for(wav_rel, preset["folder"])
            if not mp3_path.exists():
                raise FileNotFoundError(f"Missing encoded file: {mp3_path}")

            mp3_audio, sr_mp3 = load_audio_ffmpeg(mp3_path, mono=False)
            levels = basic_level_metrics(mp3_audio)

            mask = df["preset_short_label"] == preset["short_label"]
            df.loc[mask, "peak_dbfs"] = levels["peak_dbfs"]
            df.loc[mask, "rms_dbfs"] = levels["rms_dbfs"]
            df.loc[mask, "crest_factor_db"] = levels["crest_factor_db"]

        # Overwrite the per-track CSV in place
        df.to_csv(csv_path, index=False)
        repaired_rows.append(df)

    # Rebuild repaired batch CSV
    batch_df = pd.concat(repaired_rows, ignore_index=True)
    repaired_batch_csv = TABLES_ROOT / "phase3_batch_analysis_REPAIRED_LEVELS.csv"
    batch_df.to_csv(repaired_batch_csv, index=False)

    # Rebuild repaired summary tables
    batch_df["preset_short_label"] = pd.Categorical(
        batch_df["preset_short_label"],
        categories=PRESET_ORDER,
        ordered=True
    )

    print("\nColumns available in repaired batch dataframe:")
    for col in batch_df.columns:
        print(f"  {col}")

    agg_map = {
        "track_count": ("track_stem", "count"),
        "mean_duration_sec": ("duration_sec", "mean"),
        "mean_peak_dbfs": ("peak_dbfs", "mean"),
        "mean_rms_dbfs": ("rms_dbfs", "mean"),
        "mean_crest_factor_db": ("crest_factor_db", "mean"),
    }

    optional_cols = {
        "mean_spectral_mae_db": "spectral_mae_db",
        "mean_spectral_rmse_db": "spectral_rmse_db",
        "mean_spectral_bias_db": "spectral_bias_db",
        "mean_spectral_corr": "spectral_corr",
        "mean_band_20_200_hz_mae": "band_20_200_hz_mae",
        "mean_band_200_2000_hz_mae": "band_200_2000_hz_mae",
        "mean_band_2000_8000_hz_mae": "band_2000_8000_hz_mae",
        "mean_band_8000_20000_hz_mae": "band_8000_20000_hz_mae",
    }

    for out_name, src_col in optional_cols.items():
        if src_col in batch_df.columns:
            agg_map[out_name] = (src_col, "mean")

    preset_summary = (
        batch_df.groupby("preset_short_label", observed=False)
        .agg(**agg_map)
        .reset_index()
    )  
    preset_summary_csv = BATCH_SUMMARY_DIR / "phase35_preset_summary_REPAIRED_LEVELS.csv"
    preset_summary.to_csv(preset_summary_csv, index=False)

    # Compute repaired deltas relative to WAV per track
    wav_ref = (
        batch_df[batch_df["preset_short_label"] == "WAV"][
            ["track_stem", "peak_dbfs", "rms_dbfs", "crest_factor_db"]
        ]
        .rename(columns={
            "peak_dbfs": "wav_peak_dbfs",
            "rms_dbfs": "wav_rms_dbfs",
            "crest_factor_db": "wav_crest_factor_db",
        })
    )

    encoded_df = batch_df[batch_df["preset_short_label"] != "WAV"].copy()
    encoded_df = encoded_df.merge(wav_ref, on="track_stem", how="left")

    encoded_df["delta_peak_db"] = encoded_df["peak_dbfs"] - encoded_df["wav_peak_dbfs"]
    encoded_df["delta_rms_db"] = encoded_df["rms_dbfs"] - encoded_df["wav_rms_dbfs"]
    encoded_df["delta_crest_factor_db"] = encoded_df["crest_factor_db"] - encoded_df["wav_crest_factor_db"]

    delta_summary = (
        encoded_df.groupby("preset_short_label", observed=False)
        .agg(
            track_count=("track_stem", "count"),
            mean_delta_peak_db=("delta_peak_db", "mean"),
            mean_delta_rms_db=("delta_rms_db", "mean"),
            mean_delta_crest_factor_db=("delta_crest_factor_db", "mean"),
        )
        .reset_index()
    )

    delta_summary["preset_short_label"] = pd.Categorical(
        delta_summary["preset_short_label"],
        categories=ENCODE_ORDER,
        ordered=True
    )
    delta_summary = delta_summary.sort_values("preset_short_label")

    delta_summary_csv = BATCH_SUMMARY_DIR / "phase35_delta_summary_REPAIRED_LEVELS.csv"
    delta_summary.to_csv(delta_summary_csv, index=False)

    # Rebuild repaired level-summary charts
    delta_rms = delta_summary.set_index("preset_short_label")["mean_delta_rms_db"]
    delta_crest = delta_summary.set_index("preset_short_label")["mean_delta_crest_factor_db"]
    delta_peak = delta_summary.set_index("preset_short_label")["mean_delta_peak_db"]

    save_bar_chart(
        delta_rms,
        "Mean RMS Change Relative to WAV (Repaired Levels)",
        "Mean Δ RMS (dB)",
        BATCH_SUMMARY_FIG_DIR / "phase35_mean_delta_rms_REPAIRED_LEVELS.png",
    )

    save_bar_chart(
        delta_crest,
        "Mean Crest Factor Change Relative to WAV (Repaired Levels)",
        "Mean Δ Crest Factor (dB)",
        BATCH_SUMMARY_FIG_DIR / "phase35_mean_delta_crest_REPAIRED_LEVELS.png",
    )

    save_bar_chart(
        delta_peak,
        "Mean Peak Change Relative to WAV (Repaired Levels)",
        "Mean Δ Peak (dB)",
        BATCH_SUMMARY_FIG_DIR / "phase35_mean_delta_peak_REPAIRED_LEVELS.png",
    )

    print("\nRepair complete.")
    print(f"Updated per-track CSVs:     {PER_TRACK_DIR}")
    print(f"Repaired batch CSV:        {repaired_batch_csv}")
    print(f"Repaired preset summary:   {preset_summary_csv}")
    print(f"Repaired delta summary:    {delta_summary_csv}")
    print(f"Repaired summary figures:  {BATCH_SUMMARY_FIG_DIR}")


if __name__ == "__main__":
    main()