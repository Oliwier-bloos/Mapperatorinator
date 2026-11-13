"""Make one osu!mania map from a song in one command, choosing the best of several candidates.

Steps:
1. Convert the audio to MP3 if needed (osu! expects MP3 or OGG).
2. Timing: generate it several times and keep the version most runs agree on (or use --timing).
   Red lines after the music fades out are dropped. --double-bpm doubles every BPM, for songs
   whose tempo is ambiguous: the extra grid lines leave room for in-between notes.
3. Generate candidate maps in parallel, one per seed x temperature.
4. Score every candidate: star rating, how many notes are finer than 1/4, how often beat patterns
   repeat (structure), density spikes and invalid notes.
5. Keep the best one under --max-stars and write it as an .osz next to the song.

Example:
    python mania_generate.py "C:/Users/me/Downloads/song.m4a" --max-stars 7
    python mania_generate.py song.mp3 --tags style/chordjack skillset/speedjack \\
        --avoid-tags style/jumpstream style/handstream

Anything unrecognised is passed to inference.py as a Hydra override. ffmpeg must be on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mania_benchmark import parse_notes, pattern_repetition, pattern_stats, star_rating
from timing_benchmark import red_lines

REPO_ROOT = Path(__file__).resolve().parent
SECTION_MS = 6000


# --- Timing -------------------------------------------------------------------------------

def timing_signature(osu_text: str) -> tuple:
    """Red lines rounded so that runs which agree compare equal."""
    return tuple((round(t / 5) * 5, round(60000 / length, 1)) for t, length in red_lines(osu_text))


def consensus_timing(osu_texts: list[str]) -> str:
    """The timing most runs agree on; ties go to the earliest run."""
    signatures = [timing_signature(t) for t in osu_texts]
    best, _ = Counter(signatures).most_common(1)[0]
    return osu_texts[signatures.index(best)]


def music_end_ms(samples: np.ndarray, sample_rate: int, window_s: float = 0.5, quiet_db: float = 35.0) -> int:
    """End of the last window that is within `quiet_db` of the loudest window."""
    n = max(1, int(window_s * sample_rate))
    count = len(samples) // n
    if count == 0:
        return int(len(samples) * 1000 / sample_rate)
    rms = np.sqrt((samples[:count * n].reshape(count, n) ** 2).mean(axis=1)) + 1e-12
    loud = np.nonzero(20 * np.log10(rms / rms.max()) > -quiet_db)[0]
    return int((loud[-1] + 1) * window_s * 1000) if len(loud) else int(count * window_s * 1000)


def _split_timing(osu_text: str) -> tuple[str, list[str], str]:
    head, rest = osu_text.split("[TimingPoints]", 1)
    body, _, tail = rest.partition("\n[")
    points = [line for line in body.strip().splitlines() if line.strip()]
    return head, points, ("\n[" + tail) if tail else ""


def _join_timing(head: str, points: list[str], tail: str) -> str:
    return head + "[TimingPoints]\n" + "\n".join(points) + "\n" + tail


def trim_timing_after(osu_text: str, end_ms: float) -> str:
    """Drop timing points from `end_ms` on, keeping at least the first red line."""
    head, points, tail = _split_timing(osu_text)
    kept = [p for p in points if float(p.split(",")[0]) < end_ms] or points[:1]
    return _join_timing(head, kept, tail)


def double_bpm(osu_text: str) -> str:
    """Double the BPM of every red line."""
    head, points, tail = _split_timing(osu_text)
    out = []
    for p in points:
        parts = p.split(",")
        if len(parts) >= 7 and parts[6].strip() == "1" and float(parts[1]) > 0:
            parts[1] = repr(float(parts[1]) / 2)
        out.append(",".join(parts))
    return _join_timing(head, out, tail)


# --- Scoring ------------------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    path: str
    stars: float
    notes_per_s: float
    finer_than_quarter: float  # share of gaps between notes shorter than 1/4 beat
    repetition: float
    spike: float               # densest 6 s section / median section
    holds: float
    chords_3plus: float
    jack_rate: float
    invalid: int


def finer_than_quarter(osu_text: str, keys: int) -> float:
    reds = red_lines(osu_text)
    times = sorted({n.time for n in parse_notes(osu_text, keys)})
    if not reds or len(times) < 2:
        return 0.0
    finer = 0
    for a, b in zip(times, times[1:]):
        governing = [r for r in reds if r[0] <= a + 1]
        beat = (governing[-1] if governing else reds[0])[1]
        finer += (b - a) < beat / 4 - 3
    return finer / (len(times) - 1)


def density_spike(osu_text: str, keys: int) -> float:
    times = [n.time for n in parse_notes(osu_text, keys)]
    if len(times) < 2:
        return 1.0
    start, end = times[0], times[-1]
    counts = [sum(1 for t in times if s <= t < s + SECTION_MS) for s in range(start, end - SECTION_MS + 1, SECTION_MS)]
    counts = [c for c in counts if c > 0]
    if len(counts) < 3:
        return 1.0
    return max(counts) / float(np.median(counts))


def describe(name: str, path: Path, keys: int) -> Candidate:
    text = path.read_text(encoding="utf-8")
    notes = parse_notes(text, keys)
    stats = pattern_stats(notes, keys)
    length_s = max(1e-3, (notes[-1].time - notes[0].time) / 1000) if notes else 1e-3
    return Candidate(
        name=name, path=str(path), stars=star_rating(text), notes_per_s=len(notes) / length_s,
        finer_than_quarter=finer_than_quarter(text, keys), repetition=pattern_repetition(text, keys),
        spike=density_spike(text, keys), holds=stats["holds"], chords_3plus=stats["chords_3plus"],
        jack_rate=stats["jack_rate"], invalid=stats["invalid"],
    )


def select(candidates: list[Candidate], max_stars: float, max_spike: float = 1.35,
           near: float = 1.0) -> tuple[Candidate, str]:
    """Pick a candidate: valid and under the cap, preferably within `near` stars of it and
    without a density spike, then the one whose patterns repeat most (best structure)."""
    pool = [c for c in candidates if c.invalid == 0] or candidates
    under = [c for c in pool if c.stars <= max_stars + 0.05]
    if not under:
        best = min(pool, key=lambda c: c.stars)
        return best, f"every candidate is over {max_stars} stars; picked the easiest"
    close = [c for c in under if c.stars >= max_stars - near]
    if not close:
        best = max(under, key=lambda c: c.stars)
        return best, f"no candidate within {near} stars of the cap; picked the hardest under it"
    calm = [c for c in close if c.spike <= max_spike]
    group = calm or close
    best = max(group, key=lambda c: (round(c.repetition, 2), c.stars))
    reason = "best pattern structure under the cap"
    if not calm:
        reason += f" (every candidate has a density spike above {max_spike}x)"
    return best, reason


# --- Running inference --------------------------------------------------------------------

def run_inference(overrides: list[str], log_path: Path) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "inference.py", *overrides], cwd=REPO_ROOT, stdout=log,
                            stderr=subprocess.STDOUT, env=os.environ | {"PYTHONUTF8": "1"})


def run_parallel(jobs: list[tuple[str, list[str], Path]], parallel: int) -> dict[str, Path | None]:
    """Run inference jobs (name, overrides, out_dir) at most `parallel` at a time."""
    results: dict[str, Path | None] = {}
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, Path]] = []
    while pending or running:
        while pending and len(running) < parallel:
            name, overrides, out_dir = pending.pop(0)
            out_dir.mkdir(parents=True, exist_ok=True)
            existing = list(out_dir.glob("beatmap*.osu"))
            if existing:
                results[name] = existing[0]
                continue
            proc = run_inference(overrides + [f"output_path={out_dir}", f"hydra.run.dir={out_dir / 'hydra'}"],
                                 out_dir / "inference.log")
            running.append((name, proc, out_dir))
        for item in list(running):
            name, proc, out_dir = item
            if proc.poll() is not None:
                running.remove(item)
                found = list(out_dir.glob("beatmap*.osu"))
                results[name] = found[0] if proc.returncode == 0 and found else None
                print(f"  {name}: {'done' if results[name] else 'failed, see ' + str(out_dir / 'inference.log')}", flush=True)
        if running:
            time.sleep(1)
    return results


def hydra_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(hydra_string(v) for v in values) + "]"


# --- Packaging ----------------------------------------------------------------------------

def package_osz(out: Path, audio: Path, osu_path: Path, version: str) -> None:
    text = osu_path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^AudioFilename:.*$", "AudioFilename: audio.mp3", text)
    text = re.sub(r"(?m)^Version:.*$", f"Version:{version}", text)
    title = re.search(r"(?m)^Title:(.*)$", text).group(1).strip()
    artist = re.search(r"(?m)^Artist:(.*)$", text).group(1).strip()
    name = safe_filename(f"{artist} - {title} (Mapperatorinator) [{version}].osu")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(audio, "audio.mp3")
        z.writestr(name, text)


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def title_from_filename(path: Path) -> str:
    # Drop a trailing download id such as "song-816623178"
    return re.sub(r"[-_ ]\d{6,}$", "", path.stem) or path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--keys", type=int, default=4)
    parser.add_argument("--max-stars", type=float, default=7.0, help="Star rating cap for the chosen map")
    parser.add_argument("--difficulties", type=float, nargs="+", default=None,
                        help="Difficulties requested from the model (default: 1 and 0.5 below --max-stars). "
                             "How many stars a request gives varies by song, so trying two helps.")
    parser.add_argument("--tags", nargs="*", default=[], help="Style tags, e.g. style/chordjack skillset/speedjack")
    parser.add_argument("--avoid-tags", nargs="*", default=[], help="Tags to steer away from")
    parser.add_argument("--holds", type=float, default=0.05, help="Hold note ratio")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7],
                        help="Sampling temperatures; lower repeats patterns more (0.9 felt random)")
    parser.add_argument("--column-temperature", type=float, default=0.5, help="Sampling temperature for mania columns")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="CFG on difficulty and tags (1 = off)")
    parser.add_argument("--parallel", type=int, default=3, help="Inference processes at once")
    parser.add_argument("--timing", type=Path, default=None, help="Use the timing of this .osu instead of generating it")
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--double-bpm", action="store_true", help="Double the BPM of the timing")
    parser.add_argument("--title", default=None)
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--out", type=Path, default=None, help="Folder for the .osz (default: next to the audio)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Default: mania_maps/<title>")
    args, extra = parser.parse_known_args()

    title = args.title or title_from_filename(args.audio)
    work = args.work_dir or REPO_ROOT / "mania_maps" / safe_filename(title)
    work.mkdir(parents=True, exist_ok=True)
    difficulties = args.difficulties or [args.max_stars - 1.0, args.max_stars - 0.5]

    # 1. Audio
    audio = work / "audio.mp3"
    if not audio.exists():
        if args.audio.suffix.lower() == ".mp3":
            shutil.copyfile(args.audio, audio)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.audio), "-vn", "-ar", "44100", "-ac", "2",
                            "-c:a", "libmp3lame", "-b:a", "320k", str(audio)], check=True)
    sys.path.insert(0, str(REPO_ROOT))
    from osuT5.osuT5.dataset.data_utils import load_audio_file
    end_ms = music_end_ms(load_audio_file(str(audio), 16000), 16000)
    print(f"{title}: music ends at {end_ms / 1000:.1f} s")

    common = [f"audio_path={audio}", "gamemode=3", f"keycount={args.keys}", "year=2024"]

    # 2. Timing
    timing_path = work / "timing.osu"
    if args.timing:
        timing_text = args.timing.read_text(encoding="utf-8")
    else:
        print(f"Generating timing {args.timing_runs} times ...")
        jobs = [(f"timing {s}", common + [f"difficulty={max(difficulties)}", "output_type=[TIMING]", f"seed={s}"],
                 work / f"timing_{s}") for s in range(args.timing_runs)]
        runs = [p.read_text(encoding="utf-8") for p in run_parallel(jobs, args.parallel).values() if p]
        if not runs:
            sys.exit("Timing generation failed")
        timing_text = consensus_timing(runs)
        agree = sum(timing_signature(t) == timing_signature(timing_text) for t in runs)
        print(f"  {agree}/{len(runs)} runs agree")
    timing_text = trim_timing_after(timing_text, end_ms)
    if args.double_bpm:
        timing_text = double_bpm(timing_text)
    timing_path.write_text(timing_text, encoding="utf-8")
    print("  timing: " + ", ".join(f"{bpm:g} BPM from {t / 1000:.1f} s" for t, bpm in timing_signature(timing_text)))

    # 3. Candidates
    map_overrides = common + [
        f"beatmap_path={timing_path}", "in_context=[TIMING]", "output_type=[MAP]",
        f"hold_note_ratio={args.holds}", "mania_max_full_chord_run=1", f"end_time={end_ms + 500}",
        f"title={hydra_string(title)}", f"artist={hydra_string(args.artist)}", "creator=Mapperatorinator",
        f"cfg_scale={args.cfg_scale}", "cfg_conditions=[difficulty]" if args.cfg_scale > 1 else "cfg_conditions=[]",
        f"mania_column_temperature={args.column_temperature}",
    ]
    if args.tags:
        map_overrides.append(f"descriptors={hydra_list(args.tags)}")
        if args.avoid_tags:
            # The negative prompt needs as many descriptors as the positive one
            avoid = (args.avoid_tags * len(args.tags))[:len(args.tags)]
            map_overrides.append(f"negative_descriptors={hydra_list(avoid)}")
    jobs = []
    for difficulty in difficulties:
        for temperature in args.temperatures:
            for seed in args.seeds:
                name = f"d{difficulty:g}_t{temperature:g}_s{seed}"
                jobs.append((name, map_overrides + [f"difficulty={difficulty}", f"seed={seed}", f"temperature={temperature}",
                                                    f"version={hydra_string(name)}"] + extra, work / f"cand_{name}"))
    print(f"Generating {len(jobs)} candidates ...")
    maps = run_parallel(jobs, args.parallel)

    # 4. Score and pick
    candidates = [describe(name, path, args.keys) for name, path in maps.items() if path]
    if not candidates:
        sys.exit("Every candidate failed")
    best, reason = select(candidates, args.max_stars)
    print()
    print(f"{'candidate':18} {'stars':>6} {'notes/s':>8} {'<1/4':>6} {'repeat':>7} {'spike':>6} {'holds':>6} {'3+ch':>5} {'jacks':>6}")
    for c in sorted(candidates, key=lambda c: c.stars):
        mark = "  <- picked" if c is best else ""
        print(f"{c.name:18} {c.stars:6.2f} {c.notes_per_s:8.1f} {c.finer_than_quarter:6.0%} {c.repetition:7.0%} "
              f"{c.spike:6.2f} {c.holds:6.0%} {c.chords_3plus:5.0%} {c.jack_rate:6.0%}{mark}")
    print(f"Picked {best.name}: {reason}")

    # 5. Package
    out_dir = args.out or args.audio.parent
    version = f"{args.keys}K {best.stars:.2f}"
    osz = out_dir / f"{safe_filename(title)} (Mapperatorinator {version}).osz"
    if osz.exists():
        osz = osz.with_name(osz.stem + f" {best.name}.osz")
    package_osz(osz, audio, Path(best.path), version)
    (work / "candidates.json").write_text(json.dumps({"picked": best.name, "reason": reason,
                                                      "candidates": [asdict(c) for c in candidates]}, indent=1))
    print(f"Wrote {osz}")


if __name__ == "__main__":
    main()
