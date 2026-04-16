from __future__ import annotations

import csv
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "extreme_demo"

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
            f'More than one WAV matched "{search_term}". Narrow the SEARCH_TERM or hardcode the path.'
        )

    return matches[0]


def run_ffmpeg(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed.\nCommand: {' '.join(cmd)}\n\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    source_wav = find_target_wav(AUDIO_ROOT, SEARCH_TERM)
    rel_wav = source_wav.relative_to(AUDIO_ROOT)

    song_slug = "Prison_Song"
    demo_root = OUTPUT_ROOT / song_slug
    audio_out = demo_root / "audio"
    manifest_csv = demo_root / "extreme_demo_manifest.csv"

    audio_out.mkdir(parents=True, exist_ok=True)

    print(f"Source WAV: {source_wav}")
    print(f"Output dir: {audio_out}")

    outputs = []

    # 1) Reference-ish compressed version
    out_128 = audio_out / "Prison_Song__mp3_128.mp3"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        str(out_128),
    ])
    outputs.append(("mp3_128", out_128, "Baseline compressed comparison"))

    # 2) Brutal MP3: mono + reduced sample rate + low bitrate
    out_32 = audio_out / "Prison_Song__mp3_32_mono_22050.mp3"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-ac", "1",
        "-ar", "22050",
        "-c:a", "libmp3lame",
        "-b:a", "32k",
        str(out_32),
    ])
    outputs.append(("mp3_32_mono_22050", out_32, "Heavy lossy compression"))

    # 3) Extremely brutal MP3
    out_16 = audio_out / "Prison_Song__mp3_16_mono_8000.mp3"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-ac", "1",
        "-ar", "8000",
        "-c:a", "libmp3lame",
        "-b:a", "16k",
        str(out_16),
    ])
    outputs.append(("mp3_16_mono_8000", out_16, "Very extreme lossy compression"))

    # 4) 8-bit unsigned PCM WAV at 8 kHz mono
    out_u8_8k = audio_out / "Prison_Song__wav_u8_mono_8000.wav"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-ac", "1",
        "-ar", "8000",
        "-c:a", "pcm_u8",
        str(out_u8_8k),
    ])
    outputs.append(("wav_u8_mono_8000", out_u8_8k, "Low-resolution PCM"))

    # 5) 8-bit unsigned PCM WAV at 4 kHz mono
    out_u8_4k = audio_out / "Prison_Song__wav_u8_mono_4000.wav"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-ac", "1",
        "-ar", "4000",
        "-c:a", "pcm_u8",
        str(out_u8_4k),
    ])
    outputs.append(("wav_u8_mono_4000", out_u8_4k, "Severely bandwidth-limited PCM"))

    # 6) Low-pass + 8-bit + 8 kHz mono
    out_lowpass = audio_out / "Prison_Song__wav_lowpass3500_u8_mono_8000.wav"
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(source_wav),
        "-af", "lowpass=f=3500",
        "-ac", "1",
        "-ar", "8000",
        "-c:a", "pcm_u8",
        str(out_lowpass),
    ])
    outputs.append(("wav_lowpass3500_u8_mono_8000", out_lowpass, "Low-pass filtered and bit-depth reduced"))

    # Write manifest
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_wav", "relative_source_wav", "variant_label", "output_file", "notes"])
        for label, out_path, notes in outputs:
            writer.writerow([
                str(source_wav),
                str(rel_wav),
                label,
                str(out_path),
                notes,
            ])

    print("\nDone.")
    print(f"Manifest CSV: {manifest_csv}")
    print("Created files:")
    for label, out_path, _ in outputs:
        print(f"  {label}: {out_path}")


if __name__ == "__main__":
    main()