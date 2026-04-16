from __future__ import annotations

import csv
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"
FIG_ROOT = PROJECT_ROOT / "output" / "figures"

BATCH_SUMMARY_TABLE_ROOT = TABLE_ROOT / "batch_summary"
BATCH_SUMMARY_FIG_ROOT = FIG_ROOT / "batch_summary"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# If None, auto-pick the latest batch CSV from output/tables/
BATCH_CSV_PATH: Path | None = None

# Plot label order
PRESET_ORDER = ["128 CBR", "192 CBR", "320 CBR", "V0", "V2", "V5"]


# ============================================================
# Helpers
# ============================================================

def latest_batch_csv(table_root: Path) -> Path:
    candidates = sorted(table_root.glob("phase3_batch_analysis_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No phase3_batch_analysis_*.csv files found in {table_root}")
    return candidates[-1]


def parse_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def mean_ignore_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(statistics.mean(clean)) if clean else math.nan


def std_ignore_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0 if clean else math.nan
    return float(statistics.stdev(clean))


def min_ignore_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(min(clean)) if clean else math.nan


def max_ignore_nan(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(max(clean)) if clean else math.nan


def ensure_dirs() -> None:
    BATCH_SUMMARY_TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    BATCH_SUMMARY_FIG_ROOT.mkdir(parents=True, exist_ok=True)


def load_batch_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# Data shaping
# ============================================================

def build_track_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        track_stem = row["track_stem"]
        groups.setdefault(track_stem, []).append(row)
    return groups


def preset_rows_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["label"] != "WAV source"]


def summarize_by_preset(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    preset_rows = preset_rows_only(rows)

    metrics = [
        "spectral_mae_db",
        "spectral_rmse_db",
        "spectral_bias_db",
        "spectral_corr",
        "mae_20_200_db",
        "mae_200_2000_db",
        "mae_2000_8000_db",
        "mae_8000_20000_db",
        "peak_dbfs",
        "rms_dbfs",
        "crest_factor_db",
    ]

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in preset_rows:
        grouped.setdefault(row["preset_short_label"], []).append(row)

    summary_rows: list[dict[str, str | float]] = []

    for preset in PRESET_ORDER:
        rows_for_preset = grouped.get(preset, [])
        if not rows_for_preset:
            continue

        out: dict[str, str | float] = {
            "preset_short_label": preset,
            "n_tracks": len(rows_for_preset),
        }

        for metric in metrics:
            vals = [parse_float(r[metric]) for r in rows_for_preset]
            out[f"{metric}_mean"] = mean_ignore_nan(vals)
            out[f"{metric}_std"] = std_ignore_nan(vals)
            out[f"{metric}_min"] = min_ignore_nan(vals)
            out[f"{metric}_max"] = max_ignore_nan(vals)

        summary_rows.append(out)

    return summary_rows


def build_delta_rows(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    """
    Build per-track preset deltas relative to the WAV source row.
    """
    grouped = build_track_groups(rows)
    delta_rows: list[dict[str, str | float]] = []

    for track_stem, track_rows in grouped.items():
        wav_rows = [r for r in track_rows if r["label"] == "WAV source"]
        if len(wav_rows) != 1:
            continue

        wav = wav_rows[0]

        wav_peak = parse_float(wav["peak_dbfs"])
        wav_rms = parse_float(wav["rms_dbfs"])
        wav_crest = parse_float(wav["crest_factor_db"])

        for row in track_rows:
            if row["label"] == "WAV source":
                continue

            delta_row: dict[str, str | float] = {
                "track": row["track"],
                "track_stem": row["track_stem"],
                "preset_short_label": row["preset_short_label"],
                "label": row["label"],
                "spectral_mae_db": parse_float(row["spectral_mae_db"]),
                "spectral_rmse_db": parse_float(row["spectral_rmse_db"]),
                "spectral_corr": parse_float(row["spectral_corr"]),
                "mae_20_200_db": parse_float(row["mae_20_200_db"]),
                "mae_200_2000_db": parse_float(row["mae_200_2000_db"]),
                "mae_2000_8000_db": parse_float(row["mae_2000_8000_db"]),
                "mae_8000_20000_db": parse_float(row["mae_8000_20000_db"]),
                "delta_peak_db": parse_float(row["peak_dbfs"]) - wav_peak,
                "delta_rms_db": parse_float(row["rms_dbfs"]) - wav_rms,
                "delta_crest_db": parse_float(row["crest_factor_db"]) - wav_crest,
            }
            delta_rows.append(delta_row)

    return delta_rows


def summarize_deltas_by_preset(delta_rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped: dict[str, list[dict[str, str | float]]] = {}
    for row in delta_rows:
        grouped.setdefault(str(row["preset_short_label"]), []).append(row)

    metrics = ["delta_peak_db", "delta_rms_db", "delta_crest_db"]

    summary_rows: list[dict[str, str | float]] = []
    for preset in PRESET_ORDER:
        rows_for_preset = grouped.get(preset, [])
        if not rows_for_preset:
            continue

        out: dict[str, str | float] = {
            "preset_short_label": preset,
            "n_tracks": len(rows_for_preset),
        }

        for metric in metrics:
            vals = [float(r[metric]) for r in rows_for_preset]
            out[f"{metric}_mean"] = mean_ignore_nan(vals)
            out[f"{metric}_std"] = std_ignore_nan(vals)
            out[f"{metric}_min"] = min_ignore_nan(vals)
            out[f"{metric}_max"] = max_ignore_nan(vals)

        summary_rows.append(out)

    return summary_rows


def rank_tracks(delta_rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    """
    Rank all preset-track combinations by spectral MAE, descending.
    """
    ranked = sorted(
        delta_rows,
        key=lambda r: float(r["spectral_mae_db"]),
        reverse=True,
    )
    return ranked


# ============================================================
# CSV writing
# ============================================================

def write_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plotting
# ============================================================

def save_bar_chart(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, values, width=0.55)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.margins(x=0.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_grouped_band_chart(
    preset_labels: list[str],
    low_vals: list[float],
    mid_vals: list[float],
    upper_vals: list[float],
    high_vals: list[float],
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    x = np.arange(len(preset_labels))
    width = 0.18

    plt.bar(x - 1.5 * width, low_vals, width=width, label="20–200 Hz")
    plt.bar(x - 0.5 * width, mid_vals, width=width, label="200–2000 Hz")
    plt.bar(x + 0.5 * width, upper_vals, width=width, label="2000–8000 Hz")
    plt.bar(x + 1.5 * width, high_vals, width=width, label="8000–20000 Hz")

    plt.xticks(x, preset_labels)
    plt.ylabel("Mean MAE (dB)")
    plt.title("Mean Band-Limited Spectral Difference from WAV")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dirs()

    batch_csv = BATCH_CSV_PATH if BATCH_CSV_PATH is not None else latest_batch_csv(TABLE_ROOT)
    print(f"Reading batch CSV: {batch_csv}")

    rows = load_batch_rows(batch_csv)
    if not rows:
        raise ValueError(f"No rows found in {batch_csv}")

    preset_summary_rows = summarize_by_preset(rows)
    delta_rows = build_delta_rows(rows)
    delta_summary_rows = summarize_deltas_by_preset(delta_rows)
    ranked_rows = rank_tracks(delta_rows)

    preset_summary_csv = BATCH_SUMMARY_TABLE_ROOT / f"phase35_preset_summary_{RUN_TIMESTAMP}.csv"
    delta_csv = BATCH_SUMMARY_TABLE_ROOT / f"phase35_track_deltas_{RUN_TIMESTAMP}.csv"
    delta_summary_csv = BATCH_SUMMARY_TABLE_ROOT / f"phase35_delta_summary_{RUN_TIMESTAMP}.csv"
    ranked_csv = BATCH_SUMMARY_TABLE_ROOT / f"phase35_ranked_tracks_{RUN_TIMESTAMP}.csv"

    write_csv(preset_summary_rows, preset_summary_csv)
    write_csv(delta_rows, delta_csv)
    write_csv(delta_summary_rows, delta_summary_csv)
    write_csv(ranked_rows, ranked_csv)

    # Build lookup from preset summary
    summary_lookup = {str(r["preset_short_label"]): r for r in preset_summary_rows}
    delta_lookup = {str(r["preset_short_label"]): r for r in delta_summary_rows}

    preset_labels = [p for p in PRESET_ORDER if p in summary_lookup]

    mean_spectral_mae = [float(summary_lookup[p]["spectral_mae_db_mean"]) for p in preset_labels]
    mean_spectral_rmse = [float(summary_lookup[p]["spectral_rmse_db_mean"]) for p in preset_labels]
    mean_spectral_corr = [float(summary_lookup[p]["spectral_corr_mean"]) for p in preset_labels]

    mean_band_low = [float(summary_lookup[p]["mae_20_200_db_mean"]) for p in preset_labels]
    mean_band_mid = [float(summary_lookup[p]["mae_200_2000_db_mean"]) for p in preset_labels]
    mean_band_upper = [float(summary_lookup[p]["mae_2000_8000_db_mean"]) for p in preset_labels]
    mean_band_high = [float(summary_lookup[p]["mae_8000_20000_db_mean"]) for p in preset_labels]

    mean_delta_crest = [float(delta_lookup[p]["delta_crest_db_mean"]) for p in preset_labels]
    mean_delta_rms = [float(delta_lookup[p]["delta_rms_db_mean"]) for p in preset_labels]

    mae_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_spectral_mae_{RUN_TIMESTAMP}.png"
    rmse_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_spectral_rmse_{RUN_TIMESTAMP}.png"
    corr_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_spectral_corr_{RUN_TIMESTAMP}.png"
    band_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_band_mae_{RUN_TIMESTAMP}.png"
    crest_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_delta_crest_{RUN_TIMESTAMP}.png"
    rms_fig = BATCH_SUMMARY_FIG_ROOT / f"phase35_mean_delta_rms_{RUN_TIMESTAMP}.png"

    save_bar_chart(
        labels=preset_labels,
        values=mean_spectral_mae,
        ylabel="Mean Spectral MAE (dB)",
        title="Mean Spectral Difference from WAV by Preset",
        out_path=mae_fig,
    )

    save_bar_chart(
        labels=preset_labels,
        values=mean_spectral_rmse,
        ylabel="Mean Spectral RMSE (dB)",
        title="Mean Spectral RMSE from WAV by Preset",
        out_path=rmse_fig,
    )

    save_bar_chart(
        labels=preset_labels,
        values=mean_spectral_corr,
        ylabel="Mean Spectral Correlation",
        title="Mean Spectral Correlation with WAV by Preset",
        out_path=corr_fig,
    )

    save_grouped_band_chart(
        preset_labels=preset_labels,
        low_vals=mean_band_low,
        mid_vals=mean_band_mid,
        upper_vals=mean_band_upper,
        high_vals=mean_band_high,
        out_path=band_fig,
    )

    save_bar_chart(
        labels=preset_labels,
        values=mean_delta_crest,
        ylabel="Mean Δ Crest Factor (dB)",
        title="Mean Crest Factor Change Relative to WAV",
        out_path=crest_fig,
    )

    save_bar_chart(
        labels=preset_labels,
        values=mean_delta_rms,
        ylabel="Mean Δ RMS (dB)",
        title="Mean RMS Change Relative to WAV",
        out_path=rms_fig,
    )

    print("\nPhase 3.5 summary complete.")
    print(f"Input batch CSV:       {batch_csv}")
    print(f"Preset summary CSV:    {preset_summary_csv}")
    print(f"Track delta CSV:       {delta_csv}")
    print(f"Delta summary CSV:     {delta_summary_csv}")
    print(f"Ranked tracks CSV:     {ranked_csv}")
    print(f"Summary figure folder: {BATCH_SUMMARY_FIG_ROOT}")


if __name__ == "__main__":
    main()