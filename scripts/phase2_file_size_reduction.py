from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"

PRESETS = [
    ("mp3_128", "128 CBR"),
    ("mp3_192", "192 CBR"),
    ("mp3_320", "320 CBR"),
    ("mp3_v0", "V0"),
    ("mp3_v2", "V2"),
    ("mp3_v5", "V5"),
]


def discover_wav_files(audio_root: Path) -> list[Path]:
    return sorted([p for p in audio_root.rglob("*.wav") if p.is_file()])


def bytes_to_kb(n: int) -> float:
    return n / 1024.0


def bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    wav_files = discover_wav_files(AUDIO_ROOT)
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under: {AUDIO_ROOT}")

    detail_rows: list[dict[str, object]] = []

    for wav_path in wav_files:
        wav_rel = wav_path.relative_to(AUDIO_ROOT)
        wav_size_bytes = wav_path.stat().st_size

        for preset_folder, preset_label in PRESETS:
            encoded_path = ENCODED_ROOT / preset_folder / wav_rel.with_suffix(".mp3")

            if not encoded_path.exists():
                print(f"Missing encoded file: {encoded_path}")
                continue

            enc_size_bytes = encoded_path.stat().st_size

            size_ratio = enc_size_bytes / wav_size_bytes if wav_size_bytes > 0 else None
            percent_reduction = (
                100.0 * (1.0 - (enc_size_bytes / wav_size_bytes))
                if wav_size_bytes > 0 else None
            )

            detail_rows.append({
                "track": str(wav_rel),
                "preset_folder": preset_folder,
                "preset_label": preset_label,
                "wav_size_bytes": wav_size_bytes,
                "wav_size_kb": bytes_to_kb(wav_size_bytes),
                "wav_size_mb": bytes_to_mb(wav_size_bytes),
                "encoded_size_bytes": enc_size_bytes,
                "encoded_size_kb": bytes_to_kb(enc_size_bytes),
                "encoded_size_mb": bytes_to_mb(enc_size_bytes),
                "size_ratio_encoded_to_wav": size_ratio,
                "percent_reduction_from_wav": percent_reduction,
            })

    if not detail_rows:
        raise RuntimeError("No size comparison rows were generated.")

    detail_csv = TABLE_ROOT / "phase2_file_size_reduction_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    # Summary by preset
    summary_map: dict[str, dict[str, object]] = {}

    for row in detail_rows:
        preset_label = str(row["preset_label"])
        if preset_label not in summary_map:
            summary_map[preset_label] = {
                "preset_label": preset_label,
                "file_count": 0,
                "mean_wav_size_mb": 0.0,
                "mean_encoded_size_mb": 0.0,
                "mean_size_ratio_encoded_to_wav": 0.0,
                "mean_percent_reduction_from_wav": 0.0,
            }

        summary_map[preset_label]["file_count"] += 1
        summary_map[preset_label]["mean_wav_size_mb"] += float(row["wav_size_mb"])
        summary_map[preset_label]["mean_encoded_size_mb"] += float(row["encoded_size_mb"])
        summary_map[preset_label]["mean_size_ratio_encoded_to_wav"] += float(row["size_ratio_encoded_to_wav"])
        summary_map[preset_label]["mean_percent_reduction_from_wav"] += float(row["percent_reduction_from_wav"])

    summary_rows = []
    preset_order = [label for _, label in PRESETS]

    for preset_label in preset_order:
        if preset_label not in summary_map:
            continue

        item = summary_map[preset_label]
        n = int(item["file_count"])

        summary_rows.append({
            "preset_label": preset_label,
            "file_count": n,
            "mean_wav_size_mb": item["mean_wav_size_mb"] / n,
            "mean_encoded_size_mb": item["mean_encoded_size_mb"] / n,
            "mean_size_ratio_encoded_to_wav": item["mean_size_ratio_encoded_to_wav"] / n,
            "mean_percent_reduction_from_wav": item["mean_percent_reduction_from_wav"] / n,
        })

    summary_csv = TABLE_ROOT / "phase2_file_size_reduction_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDone.")
    print(f"Detailed CSV: {detail_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Tracks found: {len(wav_files)}")
    print(f"Rows written: {len(detail_rows)}")


if __name__ == "__main__":
    main()