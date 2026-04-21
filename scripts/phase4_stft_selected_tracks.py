from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT / "audio"
ENCODED_ROOT = PROJECT_ROOT / "output" / "encoded"
EXTREME_ROOT = PROJECT_ROOT / "output" / "extreme_demo" / "Prison_Song" / "audio"
FIG_ROOT = PROJECT_ROOT / "output" / "figures" / "phase4_stft_selected_tracks"
TABLE_ROOT = PROJECT_ROOT / "output" / "tables"

TARGET_SR = 44100
MONO = True
EPS = 1e-12

STFT_NPERSEG = 2048
STFT_NOVERLAP = 1536
STFT_NFFT = 2048

TRACK_TARGETS = [
    {
        "search_term": "My Own Summer (Shove It)",
        "slug": "my_own_summer_shove_it",
        "include_main_encodes": True,
        "include_extreme_demo": False,
    },
    {
        "search_term": "How I Could Just Kill a Man",
        "slug": "how_i_could_just_kill_a_man",
        "include_main_encodes": True,
        "include_extreme_demo": False,
    },
    {
        "search_term": "Sugar",
        "slug": "sugar",
        "include_main_encodes": True,
        "include_extreme_demo": False,
    },
    {
        "search_term": "Prison Song",
        "slug": "prison_song",
        "include_main_encodes": True,
        "include_extreme_demo": True,
    },
]

PRESETS = [
    {"folder": "mp3_128", "short_label": "128 CBR", "long_label": "MP3 128 kbps CBR"},
    {"folder": "mp3_192", "short_label": "192 CBR", "long_label": "MP3 192 kbps CBR"},
    {"folder": "mp3_320", "short_label": "320 CBR", "long_label": "MP3 320 kbps CBR"},
    {"folder": "mp3_v0", "short_label": "V0", "long_label": "MP3 VBR q0"},
    {"folder": "mp3_v2", "short_label": "V2", "long_label": "MP3 VBR q2"},
    {"folder": "mp3_v5", "short_label": "V5", "long_label": "MP3 VBR q5"},
]


def find_unique_wav(audio_root: Path, search_term: str) -> Path:
    matches = sorted(
        [p for p in audio_root.rglob("*.wav") if search_term.lower() in p.name.lower()]
    )

    if not matches:
        raise FileNotFoundError(
            f'No WAV file found under {audio_root} containing "{search_term}" in the filename.'
        )

    if len(matches) > 1:
        print(f'Multiple matches found for "{search_term}":')
        for m in matches:
            print(f"  {m}")
        raise RuntimeError(
            f'More than one WAV matched "{search_term}". Narrow the search term.'
        )

    return matches[0]


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
    return audio, target_sr


def stft_db(
    x: np.ndarray,
    sr: int,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    nfft: int = STFT_NFFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Zxx = stft(
        x,
        fs=sr,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(noverlap, max(0, len(x) - 1)),
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx)
    Sdb = 20 * np.log10(mag + EPS)
    return f, t, Sdb


def stft_difference_metrics(ref_db: np.ndarray, test_db: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    diff = test_db - ref_db

    out = {
        "stft_mae_db": float(np.mean(np.abs(diff))),
        "stft_rmse_db": float(np.sqrt(np.mean(diff**2))),
        "stft_bias_db": float(np.mean(diff)),
        "stft_abs_p95_db": float(np.percentile(np.abs(diff), 95)),
    }

    bands = [
        ("stft_mae_20_200_db", 20, 200),
        ("stft_mae_200_2000_db", 200, 2000),
        ("stft_mae_2000_8000_db", 2000, 8000),
        ("stft_mae_8000_20000_db", 8000, 20000),
    ]

    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(np.abs(diff[mask, :]))) if np.any(mask) else np.nan

    return out


def plot_spectrogram(
    t: np.ndarray,
    f: np.ndarray,
    Sdb: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
) -> None:
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, Sdb, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.yscale("log")
    plt.ylim(20, 20000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_difference_spectrogram(
    t: np.ndarray,
    f: np.ndarray,
    diff_db: np.ndarray,
    title: str,
    out_path: Path,
    lim: float = 20.0,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, diff_db, shading="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    plt.yscale("log")
    plt.ylim(20, 20000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Δ Magnitude vs WAV (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def encoded_path_for(wav_rel: Path, preset_folder: str) -> Path:
    return ENCODED_ROOT / preset_folder / wav_rel.with_suffix(".mp3")


def collect_main_encode_files(source_wav: Path) -> list[tuple[str, Path]]:
    wav_rel = source_wav.relative_to(AUDIO_ROOT)
    out = []

    for preset in PRESETS:
        enc_path = encoded_path_for(wav_rel, preset["folder"])
        if not enc_path.exists():
            print(f"Missing encoded file for {preset['short_label']}: {enc_path}")
            continue
        out.append((preset["short_label"], enc_path))

    return out


def collect_extreme_demo_files(extreme_root: Path) -> list[tuple[str, Path]]:
    if not extreme_root.exists():
        print(f"Extreme demo folder not found: {extreme_root}")
        return []

    allowed = {".mp3", ".wav"}
    files = sorted([p for p in extreme_root.iterdir() if p.is_file() and p.suffix.lower() in allowed])
    return [(p.stem, p) for p in files]


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    for target in TRACK_TARGETS:
        source_wav = find_unique_wav(AUDIO_ROOT, target["search_term"])
        wav_rel = source_wav.relative_to(AUDIO_ROOT)
        slug = target["slug"]

        print(f"\nProcessing track: {wav_rel}")

        track_root = FIG_ROOT / slug
        source_dir = track_root / "source_stft"
        encode_dir = track_root / "encode_stft"
        diff_dir = track_root / "difference_stft"

        for d in [source_dir, encode_dir, diff_dir]:
            d.mkdir(parents=True, exist_ok=True)

        wav_audio, sr = load_audio_ffmpeg(source_wav, target_sr=TARGET_SR, mono=MONO)
        wav_f, wav_t, wav_Sdb = stft_db(wav_audio, sr)

        plot_spectrogram(
            wav_t,
            wav_f,
            wav_Sdb,
            f"STFT Spectrogram\nWAV source: {source_wav.name}",
            source_dir / f"{slug}__wav_source_stft.png",
        )

        compare_files: list[tuple[str, Path]] = []

        if target["include_main_encodes"]:
            compare_files.extend(collect_main_encode_files(source_wav))

        if target["include_extreme_demo"]:
            compare_files.extend(collect_extreme_demo_files(EXTREME_ROOT))

        seen_labels = set()

        for label, path in compare_files:
            if label in seen_labels:
                continue
            seen_labels.add(label)

            print(f"  Comparing: {label} -> {path}")

            test_audio, sr_test = load_audio_ffmpeg(path, target_sr=TARGET_SR, mono=MONO)
            if sr_test != sr:
                raise RuntimeError(f"Sample rate mismatch for {path}: {sr_test} vs {sr}")

            min_len = min(len(wav_audio), len(test_audio))
            wav_use = wav_audio[:min_len]
            test_use = test_audio[:min_len]

            f_ref, t_ref, Sdb_ref = stft_db(wav_use, sr)
            f_test, t_test, Sdb_test = stft_db(test_use, sr)

            min_frames = min(Sdb_ref.shape[1], Sdb_test.shape[1])
            Sdb_ref = Sdb_ref[:, :min_frames]
            Sdb_test = Sdb_test[:, :min_frames]
            t_plot = t_ref[:min_frames]
            f_plot = f_ref

            diff_db = Sdb_test - Sdb_ref
            metrics = stft_difference_metrics(Sdb_ref, Sdb_test, f_plot)

            safe_label = (
                label.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(":", "_")
            )

            plot_spectrogram(
                t_plot,
                f_plot,
                Sdb_test,
                f"STFT Spectrogram\n{label} | {source_wav.name}",
                encode_dir / f"{slug}__{safe_label}__stft.png",
            )

            plot_difference_spectrogram(
                t_plot,
                f_plot,
                diff_db,
                f"STFT Difference from WAV\n{label} | {source_wav.name}",
                diff_dir / f"{slug}__{safe_label}__difference_from_wav_stft.png",
                lim=20.0,
            )

            rows.append({
                "track_slug": slug,
                "track_file": str(wav_rel),
                "comparison_label": label,
                "comparison_file": str(path),
                "duration_sec_analyzed": min_len / sr,
                "sample_rate_hz": sr,
                **metrics,
            })

    if not rows:
        raise RuntimeError("No STFT rows were generated.")

    csv_path = TABLE_ROOT / "phase4_stft_selected_tracks_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPhase 4 STFT analysis complete.")
    print(f"Figure root: {FIG_ROOT}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()