"""Make one osu!mania map from a song in one command, choosing the best of several candidates.

Steps:
1. Convert the audio to MP3 if needed (osu! expects MP3 or OGG).
2. Timing: generate it several times and keep the version most runs agree on (or use --timing).
   Red lines after the music fades out are dropped. --double-bpm doubles every BPM, for songs
   whose tempo is ambiguous: the extra grid lines leave room for in-between notes.
3. Generate candidate maps in parallel, one per seed x temperature.
4. Score every candidate: star rating, how many notes are finer than 1/4, how often beat patterns
   repeat (structure), density spikes and invalid notes.
5. Keep the best one under --max-stars.
6. Find sections much harder than the rest of that map (bursts of quarter-beat jacks, or
   density spikes) and regenerate just those sections with a few more seeds, keeping a version
   without the spike. --no-fix-spikes skips this.
7. Write the map as an .osz next to the song.

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
    monotone: float = 0.0       # share of notes inside 8+ gaps of the same rhythm (streams of any snap)
    rhythm_changes: float = 0.0  # share of consecutive gaps that differ


# Style presets: default tags, tags to steer away from, and what "more of this style" means
STYLES = {
    "any": {"tags": [], "avoid": []},
    "tech": {"tags": ["skillset/tech", "tech/complex snap"], "avoid": ["streams/stamina", "style/jumpstream"]},
    "jacks": {"tags": ["style/chordjack", "skillset/speedjack"], "avoid": ["style/jumpstream", "style/handstream"]},
}


def style_score(c: Candidate, style: str) -> float:
    if style == "tech":
        return c.rhythm_changes - c.monotone
    if style == "jacks":
        return c.jack_rate
    return c.repetition


def rhythm_stats(osu_text: str, keys: int, run_length: int = 8) -> tuple[float, float]:
    """(monotone, rhythm_changes): share of notes in runs of `run_length`+ equal gaps, and share
    of consecutive gaps that differ. Gaps are compared at 1/24 beat resolution."""
    reds = red_lines(osu_text)
    times = sorted({n.time for n in parse_notes(osu_text, keys)})
    if not reds or len(times) < 3:
        return 0.0, 0.0
    gaps = []
    for a, b in zip(times, times[1:]):
        governing = [r for r in reds if r[0] <= a + 1]
        beat = (governing[-1] if governing else reds[0])[1]
        gaps.append(round((b - a) / beat * 24))
    changes = sum(x != y for x, y in zip(gaps, gaps[1:])) / (len(gaps) - 1)
    in_runs, current = 0, 1
    for x, y in zip(gaps, gaps[1:] + [None]):
        if x == y:
            current += 1
        else:
            if current >= run_length:
                in_runs += current
            current = 1
    return in_runs / len(gaps), changes


def loudest_window(samples: np.ndarray, sample_rate: int, seconds: float, end_ms: float) -> tuple[int, int]:
    """Start and end (ms) of the loudest `seconds` long stretch before `end_ms`."""
    block = int(0.5 * sample_rate)
    count = min(len(samples) // block, int(end_ms / 500))
    width = int(seconds * 2)
    if count <= width:
        return 0, int(min(end_ms, seconds * 1000))
    energy = (samples[:count * block].reshape(count, block) ** 2).mean(axis=1)
    sums = np.convolve(energy, np.ones(width), mode="valid")
    start = int(np.argmax(sums)) * 500
    return start, start + int(seconds * 1000)


def pick_difficulty(previews: list[tuple[float, Candidate]], min_stars: float, max_stars: float,
                    style: str) -> tuple[float, str]:
    """The difficulty request whose preview lands in the star range with the most of the style;
    if none lands in range, the one closest to it."""
    in_range = [(d, c) for d, c in previews if min_stars - 0.05 <= c.stars <= max_stars + 0.05]
    if in_range:
        d, c = max(in_range, key=lambda dc: (round(style_score(dc[1], style), 2), dc[1].stars))
        return d, f"{c.stars:.2f} stars in the preview"
    middle = (min_stars + max_stars) / 2
    d, c = min(previews, key=lambda dc: abs(dc[1].stars - middle))
    return d, f"closest to the range ({c.stars:.2f} stars in the preview)"


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


@dataclass
class Spike:
    start: int        # ms
    end: int          # ms
    fast_jacks: int   # most quarter-beat (or faster) same-column repeats in one of its sections
    notes: int        # most notes in one of its sections
    reason: str


def section_stats(osu_text: str, keys: int, section_ms: int = SECTION_MS) -> list[tuple[int, int, int]]:
    """(start ms, notes, fast jacks) for consecutive sections from the first note.

    A fast jack is a note hit on the same column as the previous note in that column, at most a
    quarter beat later.
    """
    reds = red_lines(osu_text)
    notes = parse_notes(osu_text, keys)
    if not reds or not notes:
        return []

    def beat_at(t):
        governing = [r for r in reds if r[0] <= t + 1]
        return (governing[-1] if governing else reds[0])[1]

    jack_times = []
    for column in range(keys):
        times = [n.time for n in notes if n.column == column]
        jack_times += [b for a, b in zip(times, times[1:]) if b - a <= beat_at(a) / 4 + 2]
    note_times = np.array([n.time for n in notes])
    jack_times = np.array(sorted(jack_times))
    stats = []
    for s in range(notes[0].time, notes[-1].time + 1, section_ms):
        count = int(((note_times >= s) & (note_times < s + section_ms)).sum())
        jacks = int(((jack_times >= s) & (jack_times < s + section_ms)).sum())
        stats.append((s, count, jacks))
    return stats


def find_spikes(osu_text: str, keys: int, jack_floor: int = 20, jack_ratio: float = 3.0,
                density_ratio: float = 1.35, section_ms: int = SECTION_MS) -> list[Spike]:
    """Sections far harder than the rest of the map, merged into regions.

    Flags a section when its fast jacks reach `jack_floor` and `jack_ratio` times the map's median,
    or its note count is over `density_ratio` times the median. Measuring against the map itself
    keeps a map that is speedjack all the way through from counting as one long spike.
    """
    stats = section_stats(osu_text, keys, section_ms)
    active = [st for st in stats if st[1] > 0]
    if len(active) < 3:
        return []
    median_notes = float(np.median([n for _, n, _ in active]))
    median_jacks = float(np.median([j for _, _, j in active]))
    jack_limit = max(jack_floor, jack_ratio * median_jacks)

    flagged = []
    for start, count, jacks in stats:
        reasons = []
        if jacks >= jack_limit:
            reasons.append("jack burst")
        if count > density_ratio * median_notes:
            reasons.append("density")
        if reasons:
            flagged.append((start, count, jacks, reasons))

    spikes: list[Spike] = []
    for start, count, jacks, reasons in flagged:
        if spikes and start <= spikes[-1].end:
            last = spikes[-1]
            last.end = start + section_ms
            last.fast_jacks = max(last.fast_jacks, jacks)
            last.notes = max(last.notes, count)
            last.reason = " + ".join(sorted(set(last.reason.split(" + ") + reasons)))
        else:
            spikes.append(Spike(start, start + section_ms, jacks, count, " + ".join(reasons)))
    return spikes


def describe(name: str, path: Path, keys: int) -> Candidate:
    text = path.read_text(encoding="utf-8")
    notes = parse_notes(text, keys)
    stats = pattern_stats(notes, keys)
    length_s = max(1e-3, (notes[-1].time - notes[0].time) / 1000) if notes else 1e-3
    monotone, changes = rhythm_stats(text, keys)
    return Candidate(
        name=name, path=str(path), stars=star_rating(text), notes_per_s=len(notes) / length_s,
        finer_than_quarter=finer_than_quarter(text, keys), repetition=pattern_repetition(text, keys),
        spike=density_spike(text, keys), holds=stats["holds"], chords_3plus=stats["chords_3plus"],
        jack_rate=stats["jack_rate"], invalid=stats["invalid"], monotone=monotone, rhythm_changes=changes,
    )


def select(candidates: list[Candidate], max_stars: float, min_stars: float | None = None,
           max_spike: float = 1.35, style: str = "any") -> tuple[Candidate, str]:
    """Pick a candidate: valid and under the cap, preferably at least `min_stars` (default one star
    under the cap) and without a density spike, then the one with the most of the style (for "any":
    the patterns that repeat most), then the most pattern repetition."""
    if min_stars is None:
        min_stars = max_stars - 1.0
    pool = [c for c in candidates if c.invalid == 0] or candidates
    under = [c for c in pool if c.stars <= max_stars + 0.05]
    if not under:
        best = min(pool, key=lambda c: c.stars)
        return best, f"every candidate is over {max_stars} stars; picked the easiest"
    close = [c for c in under if c.stars >= min_stars - 0.05]
    if not close:
        best = max(under, key=lambda c: c.stars)
        return best, f"no candidate between {min_stars} and {max_stars} stars; picked the hardest under the cap"
    calm = [c for c in close if c.spike <= max_spike]
    group = calm or close
    best = max(group, key=lambda c: (round(style_score(c, style), 2), round(c.repetition, 2), c.stars))
    reason = "best pattern structure under the cap" if style == "any" else f"most {style} in the star range"
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
            proc = run_inference(overrides + [f"output_path={hydra_string(str(out_dir))}",
                                              f"hydra.run.dir={hydra_string(str(out_dir / 'hydra'))}"],
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
    """Quote a value for a Hydra override.

    Inside quotes Hydra keeps backslashes literally (so Windows paths pass through as they are),
    except before a quote: there they must be doubled, and the quote itself escaped.
    """
    escaped = re.sub(r'(\\*)"', lambda m: m.group(1) * 2 + '\\"', value)
    escaped = re.sub(r"(\\+)$", lambda m: m.group(1) * 2, escaped)
    return '"' + escaped + '"'


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(hydra_string(v) for v in values) + "]"


def with_overrides(overrides: list[str], **values) -> list[str]:
    """Replace (or add) Hydra overrides; Hydra rejects a key given twice."""
    kept = [o for o in overrides if o.split("=", 1)[0] not in values]
    return kept + [f"{key}={value}" for key, value in values.items()]


def mmss(ms: float) -> str:
    return f"{int(ms // 60000)}:{int(ms // 1000) % 60:02d}"


def fix_spikes(map_path: Path, overrides: list[str], work: Path, keys: int, seeds: list[int], parallel: int,
               min_stars: float, max_stars: float, max_fixes: int) -> tuple[Path, list[dict]]:
    """Regenerate each spike region of a map with several seeds and keep a version without it.

    Returns the final map and a log of what happened to each spike.
    """
    log = []
    current = map_path
    for i, spike in enumerate(find_spikes(current.read_text(encoding="utf-8"), keys)[:max_fixes]):
        print(f"Spike at {mmss(spike.start)}-{mmss(spike.end)}: {spike.reason} "
              f"({spike.fast_jacks} fast jacks, {spike.notes} notes in 6 s). Regenerating it with {len(seeds)} seeds ...")
        base = work / f"fix{i}_base.osu"
        shutil.copyfile(current, base)
        jobs = [(f"fix {i} seed {s}",
                 with_overrides(overrides, beatmap_path=hydra_string(str(base)), add_to_beatmap="true",
                                start_time=spike.start, end_time=spike.end, seed=s),
                 work / f"fix{i}_s{s}") for s in seeds]
        options = []
        for name, path in run_parallel(jobs, parallel).items():
            if not path:
                continue
            text = path.read_text(encoding="utf-8")
            remaining = [sp for sp in find_spikes(text, keys) if sp.start < spike.end and sp.end > spike.start]
            jacks = sum(j for s, _, j in section_stats(text, keys) if spike.start <= s < spike.end)
            stars = star_rating(text)
            options.append({"name": name, "path": path, "resolved": not remaining, "fast_jacks": jacks, "stars": stars,
                            "in_range": min_stars - 0.05 <= stars <= max_stars + 0.05})
        resolved = [o for o in options if o["resolved"]]
        entry = {"start": spike.start, "end": spike.end, "reason": spike.reason, "fast_jacks": spike.fast_jacks,
                 "notes": spike.notes, "fixed_with": None}
        if resolved:
            pick = max(resolved, key=lambda o: (o["in_range"], -o["fast_jacks"]))
            current = pick["path"]
            entry["fixed_with"] = pick["name"]
            print(f"  fixed with {pick['name']}: {pick['fast_jacks']} fast jacks there now, map {pick['stars']:.2f} stars")
        else:
            print("  no seed removed it; kept the original section")
        log.append(entry)
    return current, log


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
    parser.add_argument("--min-stars", type=float, default=None, help="Preferred lowest star rating (default: 1 under the cap)")
    parser.add_argument("--difficulties", type=float, nargs="+", default=None,
                        help="Difficulties requested from the model (default: 1 and 0.5 below --max-stars). "
                             "How many stars a request gives varies by song, so trying two helps.")
    parser.add_argument("--style", choices=sorted(STYLES), default="any",
                        help="tech: complex snaps and changing rhythm, few long streams; jacks: chordjack and "
                             "speedjack; any: let the music decide. Sets default tags and how the best try is picked.")
    parser.add_argument("--tags", nargs="*", default=None, help="Style tags (default: from --style), e.g. style/chordjack")
    parser.add_argument("--avoid-tags", nargs="*", default=None, help="Tags to steer away from (default: from --style)")
    parser.add_argument("--preview-seconds", type=float, default=60,
                        help="First try each difficulty request on the loudest this-many seconds and generate the "
                             "full song only with the best one (0 = generate every request in full; slower)")
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
    parser.add_argument("--no-fix-spikes", action="store_true",
                        help="Don't regenerate sections much harder than the rest of the map")
    parser.add_argument("--fix-seeds", type=int, nargs="+", default=[3, 4, 5], help="Seeds to try on each spike")
    parser.add_argument("--max-fixes", type=int, default=3, help="Most spike regions to regenerate")
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
    samples = load_audio_file(str(audio), 16000)
    end_ms = music_end_ms(samples, 16000)
    print(f"{title}: music ends at {end_ms / 1000:.1f} s")
    started = time.time()

    def elapsed() -> str:
        return f"[{(time.time() - started) / 60:.1f} min]"

    common = [f"audio_path={hydra_string(str(audio))}", "gamemode=3", f"keycount={args.keys}", "year=2024"]

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
    print("  timing: " + ", ".join(f"{bpm:g} BPM from {t / 1000:.1f} s" for t, bpm in timing_signature(timing_text))
          + f" {elapsed()}")

    # 3. Candidates
    map_overrides = common + [
        f"beatmap_path={hydra_string(str(timing_path))}", "in_context=[TIMING]", "output_type=[MAP]",
        f"hold_note_ratio={args.holds}", "mania_max_full_chord_run=1", f"end_time={end_ms + 500}",
        f"title={hydra_string(title)}", f"artist={hydra_string(args.artist)}", "creator=Mapperatorinator",
        f"cfg_scale={args.cfg_scale}", "cfg_conditions=[difficulty]" if args.cfg_scale > 1 else "cfg_conditions=[]",
        f"mania_column_temperature={args.column_temperature}",
    ]
    tags = args.tags if args.tags is not None else STYLES[args.style]["tags"]
    avoid_tags = args.avoid_tags if args.avoid_tags is not None else STYLES[args.style]["avoid"]
    if tags:
        map_overrides.append(f"descriptors={hydra_list(tags)}")
        if avoid_tags:
            # The negative prompt needs as many descriptors as the positive one
            avoid = (avoid_tags * len(tags))[:len(tags)]
            map_overrides.append(f"negative_descriptors={hydra_list(avoid)}")
    min_stars = args.min_stars if args.min_stars is not None else args.max_stars - 1.0

    def job(name: str, difficulty: float, seed: int, temperature: float, **more) -> tuple[str, list[str], Path]:
        overrides = with_overrides(map_overrides + extra, difficulty=difficulty, seed=seed, temperature=temperature,
                                   version=hydra_string(name), **more)
        return name, overrides, work / name

    # Preview: try each difficulty request on the loudest stretch, then generate the full song only with the best
    if args.preview_seconds > 0 and len(difficulties) > 1:
        start, stop = loudest_window(samples, 16000, args.preview_seconds, end_ms)
        print(f"Previewing {len(difficulties)} difficulty requests on {mmss(start)}-{mmss(stop)} ...")
        preview_jobs = [job(f"preview_d{d:g}", d, args.seeds[0], args.temperatures[0], start_time=start, end_time=stop)
                        for d in difficulties]
        results = run_parallel(preview_jobs, args.parallel)  # in finishing order, so match by name
        previews = [(d, describe(name, results[name], args.keys))
                    for d, (name, _, _) in zip(difficulties, preview_jobs) if results.get(name)]
        if previews:
            for d, c in previews:
                print(f"  difficulty {d:g}: {c.stars:.2f} stars, {c.notes_per_s:.1f} notes/s, "
                      f"{c.monotone:.0%} in long same-rhythm runs, {c.rhythm_changes:.0%} rhythm changes")
            chosen, why = pick_difficulty(previews, min_stars, args.max_stars, args.style)
            difficulties = [chosen]
            print(f"  using difficulty {chosen:g}: {why} {elapsed()}")

    jobs = [job(f"cand_d{d:g}_t{t:g}_s{s}", d, s, t) for d in difficulties for t in args.temperatures for s in args.seeds]
    print(f"Generating {len(jobs)} full-length candidates ...")
    maps = run_parallel(jobs, args.parallel)

    # 4. Score and pick
    candidates = [describe(name, path, args.keys) for name, path in maps.items() if path]
    if not candidates:
        sys.exit("Every candidate failed")
    best, reason = select(candidates, args.max_stars, args.min_stars, style=args.style)
    print()
    print(f"{'candidate':22} {'stars':>6} {'notes/s':>8} {'<1/4':>6} {'repeat':>7} {'streamy':>8} {'changes':>8} "
          f"{'spike':>6} {'holds':>6} {'3+ch':>5} {'jacks':>6}")
    for c in sorted(candidates, key=lambda c: c.stars):
        mark = "  <- picked" if c is best else ""
        print(f"{c.name:22} {c.stars:6.2f} {c.notes_per_s:8.1f} {c.finer_than_quarter:6.0%} {c.repetition:7.0%} "
              f"{c.monotone:8.0%} {c.rhythm_changes:8.0%} {c.spike:6.2f} {c.holds:6.0%} {c.chords_3plus:5.0%} "
              f"{c.jack_rate:6.0%}{mark}")
    print(f"Picked {best.name}: {reason} {elapsed()}")

    # 5. Fix sections much harder than the rest of the map
    final_path, fixes = Path(best.path), []
    if not args.no_fix_spikes:
        overrides = next(o for name, o, _ in jobs if name == best.name)
        final_path, fixes = fix_spikes(final_path, overrides, work, args.keys, args.fix_seeds, args.parallel,
                                       min_stars, args.max_stars, args.max_fixes)
        if not fixes:
            print("No spikes: no section is much harder than the rest of the map")
    final = describe(best.name, final_path, args.keys)

    # 6. Package
    out_dir = args.out or args.audio.parent
    version = f"{args.keys}K {final.stars:.2f}"
    osz = out_dir / f"{safe_filename(title)} (Mapperatorinator {version}).osz"
    if osz.exists():
        osz = osz.with_name(osz.stem + f" {best.name}.osz")
    package_osz(osz, audio, final_path, version)
    (work / "candidates.json").write_text(json.dumps({
        "picked": best.name, "reason": reason, "spike_fixes": fixes, "final": asdict(final),
        "candidates": [asdict(c) for c in candidates]}, indent=1, default=str))
    print(f"Final map: {final.stars:.2f} stars, {final.notes_per_s:.1f} notes/s, {final.repetition:.0%} repetition, "
          f"{final.monotone:.0%} in long same-rhythm runs, {final.rhythm_changes:.0%} rhythm changes {elapsed()}")
    print(f"Wrote {osz}")


if __name__ == "__main__":
    main()
