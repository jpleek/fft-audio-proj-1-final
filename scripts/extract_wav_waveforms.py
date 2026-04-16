from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# User settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "figures_wav_only"

TARGET_SR = 44100
MONO = True

# Plot full waveform if None, otherwise only first N seconds
PLOT_SECONDS = None

# If you want a quick test run, set to an integer like 2 or 5
LIMIT_TO_FIRST_N_TRACKS = None

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================

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


def relative_audio_subpath(wav_path: Path) -> Path:
    return wav_path.relative_to(AUDIO_ROOT)


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


def save_wav_waveform_plot(
    x: np.ndarray,
    sr: int,
    title: str,
    out_path: Path,
) -> None:
    if PLOT_SECONDS is None:
        x_plot = x
        t = np.arange(len(x_plot)) / sr
        subtitle = "Full track waveform"
    else:
        n = min(len(x), int(PLOT_SECONDS * sr))
        x_plot = x[:n]
        t = np.arange(len(x_plot)) / sr
        subtitle = f"First {PLOT_SECONDS:.2f} s"

    plt.figure(figsize=(12, 4.5))
    plt.plot(t, x_plot, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{title}\n{subtitle}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    wav_files = discover_wav_files(AUDIO_ROOT)
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under: {AUDIO_ROOT}")

    if LIMIT_TO_FIRST_N_TRACKS is not None:
        wav_files = wav_files[:LIMIT_TO_FIRST_N_TRACKS]

    for idx, wav_path in enumerate(wav_files, start=1):
        rel_wav = relative_audio_subpath(wav_path)
        track_stem = build_track_stem(rel_wav)

        print(f"[{idx}/{len(wav_files)}] Processing: {rel_wav}")

        audio, sr = load_audio_ffmpeg(wav_path, target_sr=TARGET_SR, mono=MONO)

        waveform_dir = OUTPUT_ROOT / track_stem / "waveform"
        waveform_dir.mkdir(parents=True, exist_ok=True)

        out_path = waveform_dir / f"{track_stem}__wav_waveform.png"

        save_wav_waveform_plot(
            x=audio,
            sr=sr,
            title=str(rel_wav),
            out_path=out_path,
        )

    print("\nDone.")
    print(f"WAV-only waveform figures saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()