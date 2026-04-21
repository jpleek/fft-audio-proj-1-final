from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "output" / "tables" / "phase4_stft_selected_tracks_summary.csv"

TABLE_ROOT = PROJECT_ROOT / "output" / "tables" / "phase4_stft_summary"
FIG_ROOT = PROJECT_ROOT / "output" / "figures" / "phase4_stft_summary"

MAIN_ENCODE_ORDER = ["128 CBR", "192 CBR", "320 CBR", "V0", "V2", "V5"]
TRACK_ORDER = [
    "my_own_summer_shove_it",
    "how_i_could_just_kill_a_man",
    "sugar",
    "prison_song",
]


def save_bar_chart(
    series: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    plt.figure(figsize=figsize)
    plt.bar(series.index.astype(str), series.values, width=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_grouped_band_chart(
    df: pd.DataFrame,
    x_col: str,
    band_cols: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    x = range(len(df))
    width = 0.2

    plt.figure(figsize=(12, 5))

    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    labels = [
        "20–200 Hz",
        "200–2000 Hz",
        "2000–8000 Hz",
        "8000–20000 Hz",
    ]

    for offset, col, label in zip(offsets, band_cols, labels):
        plt.bar([i + offset for i in x], df[col].values, width=width, label=label)

    plt.xticks(list(x), df[x_col].astype(str), rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def classify_comparison(label: str) -> str:
    if label in MAIN_ENCODE_ORDER:
        return "main_encode"
    return "extreme_demo"


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\nRun phase4_stft_selected_tracks.py first."
        )

    print(f"Reading STFT summary CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "track_slug",
        "comparison_label",
        "stft_mae_db",
        "stft_rmse_db",
        "stft_bias_db",
        "stft_abs_p95_db",
        "stft_mae_20_200_db",
        "stft_mae_200_2000_db",
        "stft_mae_2000_8000_db",
        "stft_mae_8000_20000_db",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in STFT summary CSV: {missing}")

    df["comparison_group"] = df["comparison_label"].apply(classify_comparison)

    # Preserve clean ordering where possible
    df["track_slug"] = pd.Categorical(df["track_slug"], categories=TRACK_ORDER, ordered=True)

    main_mask = df["comparison_group"] == "main_encode"
    extreme_mask = df["comparison_group"] == "extreme_demo"

    main_df = df[main_mask].copy()
    extreme_df = df[extreme_mask].copy()

    if not main_df.empty:
        main_df["comparison_label"] = pd.Categorical(
            main_df["comparison_label"],
            categories=MAIN_ENCODE_ORDER,
            ordered=True,
        )
        main_df = main_df.sort_values(["track_slug", "comparison_label"])

    # ---------- Tables ----------
    all_rows_csv = TABLE_ROOT / "phase45_stft_all_rows.csv"
    df.to_csv(all_rows_csv, index=False)

    # Main encode summary by preset across tracks
    if not main_df.empty:
        preset_summary = (
            main_df.groupby("comparison_label", observed=False)
            .agg(
                n_rows=("track_slug", "count"),
                mean_stft_mae_db=("stft_mae_db", "mean"),
                mean_stft_rmse_db=("stft_rmse_db", "mean"),
                mean_stft_bias_db=("stft_bias_db", "mean"),
                mean_stft_abs_p95_db=("stft_abs_p95_db", "mean"),
                mean_band_20_200_db=("stft_mae_20_200_db", "mean"),
                mean_band_200_2000_db=("stft_mae_200_2000_db", "mean"),
                mean_band_2000_8000_db=("stft_mae_2000_8000_db", "mean"),
                mean_band_8000_20000_db=("stft_mae_8000_20000_db", "mean"),
            )
            .reset_index()
        )
        preset_summary_csv = TABLE_ROOT / "phase45_stft_main_encode_summary.csv"
        preset_summary.to_csv(preset_summary_csv, index=False)
    else:
        preset_summary = pd.DataFrame()

    # Main encode summary by track
    if not main_df.empty:
        track_summary = (
            main_df.groupby("track_slug", observed=False)
            .agg(
                n_rows=("comparison_label", "count"),
                mean_stft_mae_db=("stft_mae_db", "mean"),
                mean_stft_rmse_db=("stft_rmse_db", "mean"),
                mean_stft_bias_db=("stft_bias_db", "mean"),
                mean_stft_abs_p95_db=("stft_abs_p95_db", "mean"),
                mean_band_20_200_db=("stft_mae_20_200_db", "mean"),
                mean_band_200_2000_db=("stft_mae_200_2000_db", "mean"),
                mean_band_2000_8000_db=("stft_mae_2000_8000_db", "mean"),
                mean_band_8000_20000_db=("stft_mae_8000_20000_db", "mean"),
            )
            .reset_index()
        )
        track_summary_csv = TABLE_ROOT / "phase45_stft_track_summary.csv"
        track_summary.to_csv(track_summary_csv, index=False)
    else:
        track_summary = pd.DataFrame()

    # Extreme demo summary
    if not extreme_df.empty:
        extreme_summary = (
            extreme_df.groupby("comparison_label", observed=False)
            .agg(
                n_rows=("track_slug", "count"),
                mean_stft_mae_db=("stft_mae_db", "mean"),
                mean_stft_rmse_db=("stft_rmse_db", "mean"),
                mean_stft_bias_db=("stft_bias_db", "mean"),
                mean_stft_abs_p95_db=("stft_abs_p95_db", "mean"),
                mean_band_20_200_db=("stft_mae_20_200_db", "mean"),
                mean_band_200_2000_db=("stft_mae_200_2000_db", "mean"),
                mean_band_2000_8000_db=("stft_mae_2000_8000_db", "mean"),
                mean_band_8000_20000_db=("stft_mae_8000_20000_db", "mean"),
            )
            .reset_index()
            .sort_values("mean_stft_mae_db", ascending=False)
        )
        extreme_summary_csv = TABLE_ROOT / "phase45_stft_extreme_demo_summary.csv"
        extreme_summary.to_csv(extreme_summary_csv, index=False)
    else:
        extreme_summary = pd.DataFrame()

    # Ranked rows
    ranked_rows = df.sort_values("stft_mae_db", ascending=False).copy()
    ranked_rows_csv = TABLE_ROOT / "phase45_stft_ranked_rows.csv"
    ranked_rows.to_csv(ranked_rows_csv, index=False)

    # ---------- Figures ----------
    if not preset_summary.empty:
        preset_idx = preset_summary.set_index("comparison_label")

        save_bar_chart(
            preset_idx["mean_stft_mae_db"],
            "Mean STFT MAE by Main Encode Preset",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_main_encode_mean_stft_mae.png",
        )

        save_bar_chart(
            preset_idx["mean_stft_rmse_db"],
            "Mean STFT RMSE by Main Encode Preset",
            "Mean STFT RMSE (dB)",
            FIG_ROOT / "phase45_main_encode_mean_stft_rmse.png",
        )

        save_bar_chart(
            preset_idx["mean_stft_abs_p95_db"],
            "95th Percentile Absolute STFT Difference by Main Encode Preset",
            "Mean |Δ STFT| 95th percentile (dB)",
            FIG_ROOT / "phase45_main_encode_mean_stft_abs_p95.png",
        )

        save_grouped_band_chart(
            preset_summary,
            "comparison_label",
            [
                "mean_band_20_200_db",
                "mean_band_200_2000_db",
                "mean_band_2000_8000_db",
                "mean_band_8000_20000_db",
            ],
            "Band-Limited Mean STFT MAE by Main Encode Preset",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_main_encode_band_stft_mae.png",
        )

    if not track_summary.empty:
        track_idx = track_summary.set_index("track_slug")

        save_bar_chart(
            track_idx["mean_stft_mae_db"],
            "Mean STFT MAE by Track",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_track_mean_stft_mae.png",
            figsize=(11, 5),
        )

        save_bar_chart(
            track_idx["mean_stft_abs_p95_db"],
            "95th Percentile Absolute STFT Difference by Track",
            "Mean |Δ STFT| 95th percentile (dB)",
            FIG_ROOT / "phase45_track_mean_stft_abs_p95.png",
            figsize=(11, 5),
        )

        save_grouped_band_chart(
            track_summary,
            "track_slug",
            [
                "mean_band_20_200_db",
                "mean_band_200_2000_db",
                "mean_band_2000_8000_db",
                "mean_band_8000_20000_db",
            ],
            "Band-Limited Mean STFT MAE by Track",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_track_band_stft_mae.png",
        )

    if not extreme_summary.empty:
        extreme_idx = extreme_summary.set_index("comparison_label")

        save_bar_chart(
            extreme_idx["mean_stft_mae_db"],
            "Mean STFT MAE for Extreme Prison Song Encodes",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_extreme_demo_mean_stft_mae.png",
            figsize=(12, 5),
        )

        save_bar_chart(
            extreme_idx["mean_stft_abs_p95_db"],
            "95th Percentile Absolute STFT Difference for Extreme Prison Song Encodes",
            "Mean |Δ STFT| 95th percentile (dB)",
            FIG_ROOT / "phase45_extreme_demo_mean_stft_abs_p95.png",
            figsize=(12, 5),
        )

        save_grouped_band_chart(
            extreme_summary,
            "comparison_label",
            [
                "mean_band_20_200_db",
                "mean_band_200_2000_db",
                "mean_band_2000_8000_db",
                "mean_band_8000_20000_db",
            ],
            "Band-Limited Mean STFT MAE for Extreme Prison Song Encodes",
            "Mean STFT MAE (dB)",
            FIG_ROOT / "phase45_extreme_demo_band_stft_mae.png",
        )

    print("\nPhase 4.5 STFT summary complete.")
    print(f"Input CSV:              {INPUT_CSV}")
    print(f"Table output folder:    {TABLE_ROOT}")
    print(f"Figure output folder:   {FIG_ROOT}")

    if not preset_summary.empty:
        print(f"Main encode summary:    {TABLE_ROOT / 'phase45_stft_main_encode_summary.csv'}")
    if not track_summary.empty:
        print(f"Track summary:          {TABLE_ROOT / 'phase45_stft_track_summary.csv'}")
    if not extreme_summary.empty:
        print(f"Extreme demo summary:   {TABLE_ROOT / 'phase45_stft_extreme_demo_summary.csv'}")
    print(f"Ranked rows:            {TABLE_ROOT / 'phase45_stft_ranked_rows.csv'}")


if __name__ == "__main__":
    main()