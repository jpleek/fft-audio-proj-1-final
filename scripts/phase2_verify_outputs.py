from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
LOG_ROOT = PROJECT_ROOT / "output" / "logs"
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VERIFY_CSV = TABLE_ROOT / f"phase2_verification_{RUN_TIMESTAMP}.csv"
VERIFY_LOG = LOG_ROOT / f"phase2_verification_{RUN_TIMESTAMP}.txt"

WAV_GLOB = "*.wav"
MP3_GLOB = "*.mp3"

PRESET_FOLDERS = [
    "mp3_128",
    "mp3_192",
    "mp3_320",
    "mp3_v0",
    "mp3_v2",
    "mp3_v5",
]


def discover_wav_files(audio_root: Path) -> list[Path]:
    return sorted([p for p in audio_root.rglob(WAV_GLOB) if p.is_file()])


def relative_audio_subpath(wav_path: Path) -> Path:
    return wav_path.relative_to(AUDIO_ROOT)


def expected_mp3_relpath(wav_path: Path) -> Path:
    return relative_audio_subpath(wav_path).with_suffix(".mp3")


def count_mp3s_in_folder(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob(MP3_GLOB) if p.is_file())


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio root not found: {AUDIO_ROOT}")

    if not ENCODED_ROOT.exists():
        raise FileNotFoundError(f"Encoded root not found: {ENCODED_ROOT}")

    wav_files = discover_wav_files(AUDIO_ROOT)
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under: {AUDIO_ROOT}")

    verification_rows: list[dict[str, str]] = []
    missing_count = 0
    present_count = 0

    # Build expected set of relative mp3 paths from WAVs
    expected_rel_mp3s = {expected_mp3_relpath(wav) for wav in wav_files}

    preset_counts: dict[str, int] = {}
    preset_missing: dict[str, int] = {}
    preset_extra: dict[str, int] = {}

    for preset in PRESET_FOLDERS:
        preset_dir = ENCODED_ROOT / preset
        preset_counts[preset] = count_mp3s_in_folder(preset_dir)

        actual_rel_mp3s = set()
        if preset_dir.exists():
            actual_rel_mp3s = {
                p.relative_to(preset_dir)
                for p in preset_dir.rglob(MP3_GLOB)
                if p.is_file()
            }

        preset_missing[preset] = len(expected_rel_mp3s - actual_rel_mp3s)
        preset_extra[preset] = len(actual_rel_mp3s - expected_rel_mp3s)

    for wav_path in wav_files:
        rel_wav = relative_audio_subpath(wav_path)
        rel_mp3 = rel_wav.with_suffix(".mp3")

        row = {
            "source_wav": str(rel_wav),
        }

        all_present = True

        for preset in PRESET_FOLDERS:
            mp3_path = ENCODED_ROOT / preset / rel_mp3
            exists = mp3_path.exists()
            row[preset] = "OK" if exists else "MISSING"

            if exists:
                present_count += 1
            else:
                missing_count += 1
                all_present = False

        row["all_presets_present"] = "True" if all_present else "False"
        verification_rows.append(row)

    fieldnames = ["source_wav", *PRESET_FOLDERS, "all_presets_present"]

    with VERIFY_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(verification_rows)

    total_wavs = len(wav_files)
    total_expected_mp3s = total_wavs * len(PRESET_FOLDERS)

    with VERIFY_LOG.open("w", encoding="utf-8") as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("Phase 2.5 Verification Report\n")
        log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
        log_file.write(f"Project root: {PROJECT_ROOT}\n")
        log_file.write(f"Audio root: {AUDIO_ROOT}\n")
        log_file.write(f"Encoded root: {ENCODED_ROOT}\n")
        log_file.write("=" * 80 + "\n\n")

        log_file.write(f"Total source WAV files: {total_wavs}\n")
        log_file.write(f"Expected MP3 outputs:  {total_expected_mp3s}\n")
        log_file.write(f"Present MP3 outputs:   {present_count}\n")
        log_file.write(f"Missing MP3 outputs:   {missing_count}\n\n")

        log_file.write("Per-preset summary:\n")
        for preset in PRESET_FOLDERS:
            log_file.write(
                f"  {preset}: count={preset_counts[preset]}, "
                f"missing={preset_missing[preset]}, extra={preset_extra[preset]}\n"
            )

        log_file.write("\nMissing file details:\n")
        any_missing = False
        for row in verification_rows:
            if row["all_presets_present"] == "False":
                any_missing = True
                log_file.write(f"  {row['source_wav']}\n")
                for preset in PRESET_FOLDERS:
                    if row[preset] == "MISSING":
                        log_file.write(f"    - missing in {preset}\n")

        if not any_missing:
            log_file.write("  None\n")

    print("\nVerification complete.")
    print(f"Verification CSV: {VERIFY_CSV}")
    print(f"Verification log: {VERIFY_LOG}")
    print(f"Total source WAV files: {total_wavs}")
    print(f"Expected MP3 outputs:  {total_expected_mp3s}")
    print(f"Present MP3 outputs:   {present_count}")
    print(f"Missing MP3 outputs:   {missing_count}")

    print("\nPer-preset summary:")
    for preset in PRESET_FOLDERS:
        print(
            f"  {preset}: count={preset_counts[preset]}, "
            f"missing={preset_missing[preset]}, extra={preset_extra[preset]}"
        )


if __name__ == "__main__":
    main()