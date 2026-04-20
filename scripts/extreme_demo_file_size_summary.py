from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
DEMO_ROOT = PROJECT_ROOT / "output" / "extreme_demo" / "Prison_Song"
DEMO_AUDIO_ROOT = DEMO_ROOT / "audio"
TABLE_ROOT = DEMO_ROOT / "tables"

SEARCH_TERM = "Prison Song"


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


def bytes_to_kb(n: int) -> float:
    return n / 1024.0


def bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    source_wav = find_target_wav(AUDIO_ROOT, SEARCH_TERM)

    if not DEMO_AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"Demo audio folder not found: {DEMO_AUDIO_ROOT}\nRun extreme_degrade_prison_song.py first."
        )

    demo_files = sorted([p for p in DEMO_AUDIO_ROOT.iterdir() if p.is_file()])
    if not demo_files:
        raise FileNotFoundError(f"No demo audio files found in: {DEMO_AUDIO_ROOT}")

    wav_size_bytes = source_wav.stat().st_size
    rows = []

    for demo_path in demo_files:
        demo_size_bytes = demo_path.stat().st_size

        ratio = demo_size_bytes / wav_size_bytes if wav_size_bytes > 0 else None
        percent_reduction = 100.0 * (1.0 - ratio) if ratio is not None else None

        rows.append({
            "source_wav": str(source_wav),
            "demo_file": demo_path.name,
            "demo_path": str(demo_path),
            "wav_size_bytes": wav_size_bytes,
            "wav_size_kb": bytes_to_kb(wav_size_bytes),
            "wav_size_mb": bytes_to_mb(wav_size_bytes),
            "demo_size_bytes": demo_size_bytes,
            "demo_size_kb": bytes_to_kb(demo_size_bytes),
            "demo_size_mb": bytes_to_mb(demo_size_bytes),
            "size_ratio_demo_to_wav": ratio,
            "percent_reduction_from_wav": percent_reduction,
        })

    csv_path = TABLE_ROOT / "Prison_Song__extreme_demo_file_size_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Source WAV: {source_wav}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()