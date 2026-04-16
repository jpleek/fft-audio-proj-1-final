from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# =========================
# User-configurable settings
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
LOG_ROOT = PROJECT_ROOT / "output" / "logs"
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"

SUMMARY_CSV = TABLE_ROOT / f"phase2_encoding_summary_{RUN_TIMESTAMP}.csv"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG = LOG_ROOT / f"phase2_encode_log_{RUN_TIMESTAMP}.txt"

# Set to True to skip files that already exist
SKIP_EXISTING = True

# Standardize audio during encoding
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 2

# WAV search pattern
WAV_GLOB = "*.wav"


@dataclass(frozen=True)
class EncodePreset:
    folder_name: str
    ffmpeg_args: list[str]
    label: str


PRESETS = [
    EncodePreset(
        folder_name="mp3_128",
        ffmpeg_args=["-c:a", "libmp3lame", "-b:a", "128k"],
        label="MP3 CBR 128 kbps",
    ),
    EncodePreset(
        folder_name="mp3_192",
        ffmpeg_args=["-c:a", "libmp3lame", "-b:a", "192k"],
        label="MP3 CBR 192 kbps",
    ),
    EncodePreset(
        folder_name="mp3_320",
        ffmpeg_args=["-c:a", "libmp3lame", "-b:a", "320k"],
        label="MP3 CBR 320 kbps",
    ),
    EncodePreset(
        folder_name="mp3_v0",
        ffmpeg_args=["-c:a", "libmp3lame", "-q:a", "0"],
        label="MP3 VBR q0",
    ),
    EncodePreset(
        folder_name="mp3_v2",
        ffmpeg_args=["-c:a", "libmp3lame", "-q:a", "2"],
        label="MP3 VBR q2",
    ),
    EncodePreset(
        folder_name="mp3_v5",
        ffmpeg_args=["-c:a", "libmp3lame", "-q:a", "5"],
        label="MP3 VBR q5",
    ),
]


def ensure_directories() -> None:
    """Create required output directories if they do not already exist."""
    ENCODED_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def check_external_tools() -> None:
    """Verify ffmpeg and ffprobe are available on PATH."""
    missing = []
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            missing.append(tool)

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing required external tool(s): {missing_str}. "
            "Install ffmpeg so both ffmpeg and ffprobe are available."
        )


def discover_wav_files(audio_root: Path) -> list[Path]:
    """Find all WAV files recursively under the audio root."""
    wav_files = sorted(audio_root.rglob(WAV_GLOB))
    return [p for p in wav_files if p.is_file()]


def relative_audio_subpath(wav_path: Path) -> Path:
    """
    Return the WAV path relative to AUDIO_ROOT.
    Example:
      audio/Deftones/around the fur/song.wav
      -> Deftones/around the fur/song.wav
    """
    return wav_path.relative_to(AUDIO_ROOT)


def output_mp3_path(wav_path: Path, preset: EncodePreset) -> Path:
    """
    Build the mirrored MP3 output path under output/encoded/<preset>/...
    """
    rel = relative_audio_subpath(wav_path)
    rel_mp3 = rel.with_suffix(".mp3")
    return ENCODED_ROOT / preset.folder_name / rel_mp3


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture output."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def probe_audio_file(path: Path) -> dict[str, str]:
    """
    Probe an audio file with ffprobe and return a small metadata dictionary.
    Uses CSV-friendly strings for easy writing later.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bit_rate"
        ":format=duration,size,format_name",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    result = run_command(cmd)

    if result.returncode != 0:
        return {
            "probe_ok": "False",
            "probe_error": result.stderr.strip(),
            "codec_name": "",
            "sample_rate": "",
            "channels": "",
            "bit_rate": "",
            "duration": "",
            "size": "",
            "format_name": "",
        }

    parsed: dict[str, str] = {
        "probe_ok": "True",
        "probe_error": "",
        "codec_name": "",
        "sample_rate": "",
        "channels": "",
        "bit_rate": "",
        "duration": "",
        "size": "",
        "format_name": "",
    }

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()

    return parsed


def encode_one_file(wav_path: Path, preset: EncodePreset) -> tuple[bool, Path, str]:
    """
    Encode one WAV file to one MP3 preset.
    Returns:
      (success, output_path, stderr_or_message)
    """
    out_path = output_mp3_path(wav_path, preset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_EXISTING and out_path.exists():
        return True, out_path, "Skipped existing file"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(wav_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        *preset.ffmpeg_args,
        str(out_path),
    ]

    result = run_command(cmd)

    if result.returncode != 0:
        return False, out_path, result.stderr.strip()

    return True, out_path, result.stderr.strip()


def write_log_header(log_handle, wav_count: int) -> None:
    """Write a clean run header to the log file."""
    log_handle.write("=" * 80 + "\n")
    log_handle.write("Phase 2 Audio Encoding Run\n")
    log_handle.write(f"Timestamp: {datetime.now().isoformat()}\n")
    log_handle.write(f"Project root: {PROJECT_ROOT}\n")
    log_handle.write(f"Audio root: {AUDIO_ROOT}\n")
    log_handle.write(f"Encoded root: {ENCODED_ROOT}\n")
    log_handle.write(f"WAV files found: {wav_count}\n")
    log_handle.write(f"Presets: {', '.join(p.folder_name for p in PRESETS)}\n")
    log_handle.write("=" * 80 + "\n\n")


def main() -> None:
    ensure_directories()
    check_external_tools()

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio root not found: {AUDIO_ROOT}")

    wav_files = discover_wav_files(AUDIO_ROOT)

    # one-file test first
    #wav_files = wav_files[:1]

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under: {AUDIO_ROOT}")

    summary_rows: list[dict[str, str]] = []

    with RUN_LOG.open("w", encoding="utf-8") as log_file:
        write_log_header(log_file, len(wav_files))

        for wav_index, wav_path in enumerate(wav_files, start=1):
            rel_wav = relative_audio_subpath(wav_path)
            print(f"[{wav_index}/{len(wav_files)}] Processing: {rel_wav}")
            log_file.write(f"[SOURCE] {rel_wav}\n")

            source_probe = probe_audio_file(wav_path)

            for preset in PRESETS:
                success, mp3_path, message = encode_one_file(wav_path, preset)
                rel_mp3 = mp3_path.relative_to(PROJECT_ROOT)

                if success:
                    mp3_probe = probe_audio_file(mp3_path)
                    status = "OK"
                    print(f"  -> {preset.folder_name}: {rel_mp3}")
                    log_file.write(f"  [OK] {preset.folder_name} -> {rel_mp3}\n")
                else:
                    mp3_probe = {
                        "probe_ok": "False",
                        "probe_error": "",
                        "codec_name": "",
                        "sample_rate": "",
                        "channels": "",
                        "bit_rate": "",
                        "duration": "",
                        "size": "",
                        "format_name": "",
                    }
                    status = "FAILED"
                    print(f"  -> {preset.folder_name}: FAILED")
                    log_file.write(f"  [FAILED] {preset.folder_name}\n")
                    log_file.write(message + "\n")

                summary_rows.append({
                    "source_wav": str(rel_wav),
                    "source_abs_path": str(wav_path),
                    "preset_folder": preset.folder_name,
                    "preset_label": preset.label,
                    "output_mp3": str(rel_mp3),
                    "status": status,
                    "skip_existing": str(SKIP_EXISTING),
                    "target_sample_rate": str(TARGET_SAMPLE_RATE),
                    "target_channels": str(TARGET_CHANNELS),
                    "source_codec": source_probe.get("codec_name", ""),
                    "source_sample_rate": source_probe.get("sample_rate", ""),
                    "source_channels": source_probe.get("channels", ""),
                    "source_bit_rate": source_probe.get("bit_rate", ""),
                    "source_duration_sec": source_probe.get("duration", ""),
                    "source_size_bytes": source_probe.get("size", ""),
                    "output_codec": mp3_probe.get("codec_name", ""),
                    "output_sample_rate": mp3_probe.get("sample_rate", ""),
                    "output_channels": mp3_probe.get("channels", ""),
                    "output_bit_rate": mp3_probe.get("bit_rate", ""),
                    "output_duration_sec": mp3_probe.get("duration", ""),
                    "output_size_bytes": mp3_probe.get("size", ""),
                    "ffmpeg_message": message.replace("\n", " ").strip(),
                })

            log_file.write("\n")

    fieldnames = [
        "source_wav",
        "source_abs_path",
        "preset_folder",
        "preset_label",
        "output_mp3",
        "status",
        "skip_existing",
        "target_sample_rate",
        "target_channels",
        "source_codec",
        "source_sample_rate",
        "source_channels",
        "source_bit_rate",
        "source_duration_sec",
        "source_size_bytes",
        "output_codec",
        "output_sample_rate",
        "output_channels",
        "output_bit_rate",
        "output_duration_sec",
        "output_size_bytes",
        "ffmpeg_message",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    success_count = sum(1 for row in summary_rows if row["status"] == "OK")
    fail_count = sum(1 for row in summary_rows if row["status"] == "FAILED")

    print("\nDone.")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Run log:     {RUN_LOG}")
    print(f"Successful encodes: {success_count}")
    print(f"Failed encodes:     {fail_count}")


if __name__ == "__main__":
    main()