"""Benchmark generated timing against human timing on held-out ranked songs.

The songs come from the held-out test shards of the training dataset (project-riz/osu-beatmaps),
so the model has not seen them. Only songs whose human timing has a single BPM are used, so a
correct result is exactly one red line at the right BPM and phase.

Usage:
    python timing_benchmark.py --name v32 --num-songs 20
    python timing_benchmark.py --name v32-temp09 --num-songs 20 timing_temperature=0.9

Anything after the options is passed to inference.py as Hydra overrides. Songs are downloaded
once into --data-dir, and runs that already have a generated .osu are reused. inference.py
needs ffmpeg on PATH to read the audio.

Metrics, per song and averaged:
    bpm_exact   generated timing is a single red line within 0.5 BPM of the human BPM
    red_lines   number of red lines generated (the human maps have exactly one)
    median_err  median signed distance from each human beat to the nearest generated beat,
                in ms (positive = generated beat is later)
    mae         mean absolute value of that distance, in ms
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATASET_REPO = "project-riz/osu-beatmaps"
# V32 trains on compressed shards [0, 72) and evaluates on [72, 74)
FIRST_HELD_OUT_SHARD = "compressed/data-000072.tar"


# --- Timing parsing and scoring ---------------------------------------------------------

def red_lines(osu_text: str) -> list[tuple[float, float]]:
    """Uninherited timing points as (time_ms, beat_length_ms), sorted by time."""
    lines = osu_text.splitlines()
    try:
        i = next(k for k, line in enumerate(lines) if line.strip() == "[TimingPoints]") + 1
    except StopIteration:
        return []
    reds = []
    while i < len(lines) and not lines[i].strip().startswith("["):
        parts = lines[i].strip().split(",")
        if len(parts) >= 2:
            time, beat_length = float(parts[0]), float(parts[1])
            uninherited = len(parts) < 7 or parts[6].strip() == "1"
            if uninherited and beat_length > 0:
                reds.append((time, beat_length))
        i += 1
    return sorted(reds)


def beats_from_red_lines(reds: list[tuple[float, float]], end_ms: float) -> np.ndarray:
    beats = []
    for k, (start, length) in enumerate(reds):
        stop = reds[k + 1][0] if k + 1 < len(reds) else end_ms
        beats.extend(np.arange(start, stop - 1, length))
    return np.array(beats)


def signed_beat_errors(true_beats: np.ndarray, generated: list[tuple[float, float]]) -> np.ndarray:
    """For each true beat, the signed distance to the nearest beat of the generated grid."""
    errors = []
    for t in true_beats:
        governing = [r for r in generated if r[0] <= t + 1]
        start, length = governing[-1] if governing else generated[0]
        d = (t - start) % length
        errors.append(-d if d <= length / 2 else length - d)
    return np.array(errors)


@dataclass
class SongScore:
    song: str
    human_bpm: float
    generated_bpms: list[float]
    bpm_exact: bool
    red_lines: int
    median_err_ms: float
    mae_ms: float


def score_song(song: str, human_osu: str, generated_osu: str, audio_length_ms: float) -> SongScore | None:
    human = red_lines(human_osu)
    generated = red_lines(generated_osu)
    if len(human) != 1 or not generated:
        return None
    beats = beats_from_red_lines(human, audio_length_ms - 1000)
    beats = beats[beats >= 0]
    errors = signed_beat_errors(beats, generated)
    human_bpm = 60000 / human[0][1]
    generated_bpms = sorted({round(60000 / r[1], 2) for r in generated})
    return SongScore(
        song=song,
        human_bpm=round(human_bpm, 2),
        generated_bpms=generated_bpms,
        bpm_exact=len(generated) == 1 and abs(generated_bpms[0] - human_bpm) < 0.5,
        red_lines=len(generated),
        median_err_ms=float(np.median(errors)),
        mae_ms=float(np.abs(errors).mean()),
    )


def summarize(scores: list[SongScore]) -> dict:
    exact = [s for s in scores if s.bpm_exact]
    return {
        "songs": len(scores),
        "bpm_exact": len(exact),
        "red_lines_total": sum(s.red_lines for s in scores),
        "median_err_ms_on_exact": float(np.median([s.median_err_ms for s in exact])) if exact else None,
        "mae_ms_on_exact": float(np.mean([s.mae_ms for s in exact])) if exact else None,
    }


# --- Data -------------------------------------------------------------------------------

def first_beatmap_with_content(meta: dict) -> dict | None:
    return next((b for b in meta.get("beatmaps") or [] if b.get("content")), None)


def audio_extension(data: bytes) -> str:
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF":
        return "wav"
    return "mp3"


def fetch_songs(data_dir: Path, num_songs: int) -> list[Path]:
    """Download held-out single-BPM songs (original audio + beatmaps) into data_dir."""
    existing = sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists()) if data_dir.exists() else []
    if len(existing) >= num_songs:
        return existing[:num_songs]

    from datasets import load_dataset
    from huggingface_hub import list_repo_files

    def stream(path: str):
        return load_dataset(DATASET_REPO, data_files=[path], streaming=True, split="train").decode(False)

    first_held_out_key = next(iter(stream(FIRST_HELD_OUT_SHARD)))["__key__"]
    shards = sorted(f for f in list_repo_files(DATASET_REPO, repo_type="dataset") if f.startswith("original/"))
    data_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in existing}

    for shard in reversed(shards):
        print(f"Reading {shard} ...")
        any_held_out = False
        for row in stream(shard):
            key = row["__key__"]
            if key < first_held_out_key:
                continue
            any_held_out = True
            if key in have:
                continue
            beatmap = first_beatmap_with_content(row["json"])
            if beatmap is None or len(red_lines(beatmap["content"])) != 1:
                continue
            audio = row["mp3"]["bytes"]
            song_dir = data_dir / key
            song_dir.mkdir(exist_ok=True)
            (song_dir / f"original.{audio_extension(audio)}").write_bytes(audio)
            (song_dir / "meta.json").write_text(json.dumps(row["json"]), encoding="utf-8")
            have.add(key)
            if len(have) >= num_songs:
                return sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists())[:num_songs]
        if not any_held_out:
            break
    songs = sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists())
    print(f"Only found {len(songs)} held-out single-BPM songs")
    return songs


# --- Generation -------------------------------------------------------------------------

def generate_timing(audio: Path, out_dir: Path, overrides: list[str], seed: int) -> Path | None:
    existing = list(out_dir.glob("*.osu"))
    if existing:
        return existing[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "inference.py",
        f"audio_path={audio}",
        f"output_path={out_dir}",
        "gamemode=0", "difficulty=5", "year=2023",
        "output_type=[TIMING]",
        f"seed={seed}",
        f"hydra.run.dir={out_dir / 'hydra'}",
        *overrides,
    ]
    with open(out_dir / "inference.log", "w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"  inference failed, see {out_dir / 'inference.log'}")
        return None
    existing = list(out_dir.glob("*.osu"))
    return existing[0] if existing else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Name of this run; results go to <runs-dir>/<name>")
    parser.add_argument("--num-songs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "benchmarks" / "timing_data")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "benchmarks" / "timing_runs")
    args, overrides = parser.parse_known_args()

    songs = fetch_songs(args.data_dir, args.num_songs)
    run_dir = args.runs_dir / args.name
    scores: list[SongScore] = []
    for i, song_dir in enumerate(songs, 1):
        audio = next(song_dir.glob("original.*"))
        meta = json.loads((song_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"[{i}/{len(songs)}] {song_dir.name}")
        generated = generate_timing(audio, run_dir / song_dir.name, overrides, args.seed)
        if generated is None:
            continue
        score = score_song(
            song_dir.name,
            first_beatmap_with_content(meta)["content"],
            generated.read_text(encoding="utf-8"),
            float(meta["audio_length"]) * 1000,
        )
        if score is None:
            continue
        scores.append(score)
        print(f"  human {score.human_bpm} BPM, generated {score.generated_bpms}  "
              f"exact={'yes' if score.bpm_exact else 'no'}  median_err={score.median_err_ms:+.1f} ms  "
              f"mae={score.mae_ms:.1f} ms")

    summary = summarize(scores)
    print()
    print(f"{args.name}: BPM exact on {summary['bpm_exact']}/{summary['songs']} songs, "
          f"{summary['red_lines_total']} red lines generated (human: {summary['songs']})")
    if summary["bpm_exact"]:
        print(f"  on exact songs: median beat error {summary['median_err_ms_on_exact']:+.1f} ms, "
              f"mean absolute error {summary['mae_ms_on_exact']:.1f} ms")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({
        "overrides": overrides,
        "seed": args.seed,
        "summary": summary,
        "songs": [asdict(s) for s in scores],
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
