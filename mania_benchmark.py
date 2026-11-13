"""Benchmark generated osu!mania maps against human maps of the same songs.

Songs come from the held-out test shards of the training dataset (project-riz/osu-beatmaps), so
the model has not seen them. Each song needs at least one ranked human map with the requested key
count. Per song, the human map closest to --target-sr is the reference: the model gets its star
rating, year and hold note ratio, generates a map for the same time window, and the two maps are
compared over that window.

Usage:
    python mania_benchmark.py --name v32
    python mania_benchmark.py --name v32-human-timing --human-timing
    python mania_benchmark.py --name v32-cfg cfg_scale=1.5 "cfg_conditions=[hold_note_ratio,difficulty]"

Anything after the options is passed to inference.py as Hydra overrides. Songs are downloaded once
into --data-dir, and songs that already have a generated map in the run directory are reused, so
use a new --name when you change settings. inference.py needs ffmpeg on PATH to read the audio.

Metrics (generated vs. human reference, over the same window):
    sr_error        absolute star rating difference (rosu-pp)
    onset_f1        F1 of chord timestamps matched within 25 ms
    human_pair_f1   the same F1 between the reference and the set's other human map closest in
                    star rating, as a rough ceiling: two valid maps of one song differ too
    holds           share of hold notes
    chords_3plus    share of chords with 3 or more notes
    jack_rate       share of notes on a column that the previous chord also used
    repetition      share of beats whose note pattern repeats or mirrors one of the previous 8
    invalid         notes overlapping or stacked in the same column (should be 0)
    bpm_correct     every generated red line in the window is within 0.5 BPM of the human BPM
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from timing_benchmark import iter_shard, red_lines

REPO_ROOT = Path(__file__).resolve().parent
# V32 trains on compressed shards [0, 72) and evaluates on [72, 74)
HELD_OUT_SHARDS = ["compressed/data-000073.tar", "compressed/data-000072.tar"]
ONSET_TOLERANCE_MS = 25


# --- Parsing and scoring ----------------------------------------------------------------

@dataclass
class Note:
    time: int
    column: int
    end: int  # equals time for normal notes


def parse_notes(osu_text: str, keys: int, start: float = -np.inf, end: float = np.inf) -> list[Note]:
    """Mania notes whose start time lies in [start, end], sorted by time and column."""
    if "[HitObjects]" not in osu_text:
        return []
    notes = []
    for line in osu_text.split("[HitObjects]", 1)[1].strip().splitlines():
        parts = line.split(",")
        if len(parts) < 5:
            continue
        x, time, kind = float(parts[0]), int(float(parts[2])), int(parts[3])
        if not start <= time <= end:
            continue
        column = min(keys - 1, max(0, int(x * keys / 512)))
        note_end = int(parts[5].split(":", 1)[0]) if kind & 128 and len(parts) > 5 else time
        notes.append(Note(time, column, note_end))
    return sorted(notes, key=lambda n: (n.time, n.column))


def onset_f1(reference: list[int], generated: list[int], tolerance: float = ONSET_TOLERANCE_MS) -> float:
    """F1 of one-to-one matches between two sets of timestamps."""
    a, b = sorted(set(reference)), sorted(set(generated))
    i = j = hits = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    precision, recall = hits / max(1, len(b)), hits / max(1, len(a))
    return 2 * precision * recall / max(1e-9, precision + recall)


def pattern_stats(notes: list[Note], keys: int) -> dict:
    chords: dict[int, set[int]] = {}
    for n in notes:
        chords.setdefault(n.time, set()).add(n.column)
    times = sorted(chords)
    sizes = [len(chords[t]) for t in times]
    jacks = sum(len(chords[times[k]] & chords[times[k - 1]]) for k in range(1, len(times)))
    invalid = 0
    for column in range(keys):
        in_column = [n for n in notes if n.column == column]
        for prev, cur in zip(in_column, in_column[1:]):
            if cur.time <= prev.end:
                invalid += 1
    holds = [n for n in notes if n.end > n.time]
    return {
        "notes": len(notes),
        "holds": len(holds) / max(1, len(notes)),
        "chords_3plus": sum(1 for s in sizes if s >= 3) / max(1, len(sizes)),
        "jack_rate": jacks / max(1, len(notes)),
        "invalid": invalid,
    }


def pattern_repetition(osu_text: str, keys: int, start: float = -np.inf, end: float = np.inf,
                       lookback: int = 8) -> float:
    """Share of beats whose note pattern repeats, or mirrors, one of the previous `lookback` beats.

    A beat's pattern is the set of (position in the beat at 1/8 resolution, columns hit). Human
    maps repeat and mirror motifs with the music; randomly placed notes rarely do. A map a player
    praised scored 59%, one that felt like "randomized notes" scored 13%, and ranked human 4K maps
    around 4 stars average 17%.
    """
    reds = red_lines(osu_text)
    notes = parse_notes(osu_text, keys, start, end)
    if not reds or not notes:
        return 0.0
    beats: dict[int, dict[int, set[int]]] = {}
    for n in notes:
        governing = [r for r in reds if r[0] <= n.time + 1]
        offset, beat_length = governing[-1] if governing else reds[0]
        position = (n.time - offset) / beat_length
        beat = int(np.floor(position + 1e-3))
        slot = int(round((position - beat) * 8)) % 8
        beats.setdefault(beat, {}).setdefault(slot, set()).add(n.column)
    patterns = {b: frozenset((s, frozenset(c)) for s, c in slots.items()) for b, slots in beats.items()}

    def mirrored(pattern):
        return frozenset((s, frozenset(keys - 1 - c for c in cols)) for s, cols in pattern)

    repeated = 0
    for b, pattern in patterns.items():
        recent = [patterns[k] for k in range(b - lookback, b) if k in patterns]
        if pattern in recent or mirrored(pattern) in recent:
            repeated += 1
    return repeated / len(patterns)


def hold_ratio(osu_text: str, keys: int) -> float:
    notes = parse_notes(osu_text, keys)
    return sum(1 for n in notes if n.end > n.time) / max(1, len(notes))


def trimmed(osu_text: str, start: float, end: float) -> str:
    """The beatmap with only the hit objects that start in [start, end]."""
    head, objects = osu_text.split("[HitObjects]", 1)
    keep = [line for line in objects.strip().splitlines()
            if line.count(",") >= 4 and start <= float(line.split(",")[2]) <= end]
    return head + "[HitObjects]\n" + "\n".join(keep) + "\n"


def star_rating(osu_text: str) -> float:
    import rosu_pp_py as rosu

    with tempfile.NamedTemporaryFile("w", suffix=".osu", delete=False, encoding="utf-8") as f:
        f.write(osu_text)
    try:
        return float(rosu.Difficulty().calculate(rosu.Beatmap(path=f.name)).stars)
    finally:
        Path(f.name).unlink()


def bpm_correct(human_osu: str, generated_osu: str, start: float, end: float) -> bool:
    """Whether both maps have the same BPM (within 0.5) everywhere in the window."""
    human, generated = red_lines(human_osu), red_lines(generated_osu)
    if not human or not generated:
        return False

    def bpm_at(reds, t):
        governing = [r for r in reds if r[0] <= t]
        return 60000 / (governing[-1] if governing else reds[0])[1]

    # The BPM only changes at red lines, so check just after each change on either side.
    # 50 ms of slack so red lines placed a few ms apart don't count as a mismatch.
    checks = [start + 50] + [t + 50 for t, _ in human + generated if start <= t <= end - 50]
    return all(abs(bpm_at(generated, t) - bpm_at(human, t)) < 0.5 for t in checks)


@dataclass
class SongResult:
    song: str
    reference_beatmap_id: int
    human_sr: float
    generated_sr: float
    onset_f1: float
    human_pair_f1: float | None
    bpm_correct: bool
    human: dict
    generated: dict


def score_song(song: str, reference_id: int, human_osu: str, generated_osu: str, other_human_osu: str | None,
               start: float, end: float, keys: int) -> SongResult:
    human_notes = parse_notes(human_osu, keys, start, end)
    generated_notes = parse_notes(generated_osu, keys, start, end)
    pair_f1 = None
    if other_human_osu is not None:
        pair_f1 = onset_f1([n.time for n in human_notes], [n.time for n in parse_notes(other_human_osu, keys, start, end)])
    human_stats = pattern_stats(human_notes, keys) | {"repetition": pattern_repetition(human_osu, keys, start, end)}
    generated_stats = pattern_stats(generated_notes, keys) | {"repetition": pattern_repetition(generated_osu, keys, start, end)}
    return SongResult(
        song=song,
        reference_beatmap_id=reference_id,
        human_sr=star_rating(trimmed(human_osu, start, end)),
        generated_sr=star_rating(trimmed(generated_osu, start, end)),
        onset_f1=onset_f1([n.time for n in human_notes], [n.time for n in generated_notes]),
        human_pair_f1=pair_f1,
        bpm_correct=bpm_correct(human_osu, generated_osu, start, end),
        human=human_stats,
        generated=generated_stats,
    )


def summarize(results: list[SongResult]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "songs": len(results),
        "sr_error": mean([abs(r.generated_sr - r.human_sr) for r in results]),
        "onset_f1": mean([r.onset_f1 for r in results]),
        "human_pair_f1": mean([r.human_pair_f1 for r in results]),
        "hold_ratio_error": mean([abs(r.generated["holds"] - r.human["holds"]) for r in results]),
        "bpm_correct": sum(r.bpm_correct for r in results),
        "invalid_notes": sum(r.generated["invalid"] for r in results),
    }
    for key in ["notes", "holds", "chords_3plus", "jack_rate", "repetition"]:
        summary[f"human_{key}"] = mean([r.human[key] for r in results])
        summary[f"generated_{key}"] = mean([r.generated[key] for r in results])
    return summary


# --- Data -------------------------------------------------------------------------------

def mania_maps(meta: dict, keys: int) -> list[dict]:
    return [b for b in meta.get("beatmaps") or []
            if str(b.get("mode")) == "3" and float(b.get("diff_size") or 0) == keys and b.get("content")
            and str(b.get("approved")) in ("1", "2")]


def fetch_songs(data_dir: Path, num_songs: int, keys: int) -> list[Path]:
    """Download held-out songs that have ranked human maps with `keys` keys."""
    def saved():
        return sorted(p for p in data_dir.iterdir() if (p / "meta.json").exists()) if data_dir.exists() else []

    if len(saved()) >= num_songs:
        return saved()[:num_songs]
    data_dir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in saved()}
    for shard in HELD_OUT_SHARDS:
        print(f"Reading {shard} ...")
        for key, sample in iter_shard(shard):
            if key in have or "json" not in sample or "opus" not in sample:
                continue
            meta = json.loads(sample["json"])
            maps = mania_maps(meta, keys)
            if not maps:
                continue
            song_dir = data_dir / key
            song_dir.mkdir(exist_ok=True)
            # Opus in an Ogg container; inference.py accepts it as .ogg
            (song_dir / "audio.ogg").write_bytes(sample["opus"])
            for beatmap in maps:
                (song_dir / f"human_{beatmap['beatmap_id']}.osu").write_text(beatmap["content"], encoding="utf-8")
            (song_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            have.add(key)
            if len(have) >= num_songs:
                return saved()[:num_songs]
    print(f"Only found {len(saved())} held-out songs with {keys}K maps")
    return saved()


# --- Generation -------------------------------------------------------------------------

@dataclass
class Plan:
    reference_beatmap_id: int
    other_beatmap_id: int | None
    difficulty: float
    year: int
    hold_note_ratio: float
    start: int
    end: int


def make_plan(song_dir: Path, keys: int, target_sr: float) -> Plan:
    meta = json.loads((song_dir / "meta.json").read_text(encoding="utf-8"))
    maps = mania_maps(meta, keys)
    reference = min(maps, key=lambda b: abs(float(b["difficultyrating"]) - target_sr))
    others = [b for b in maps if b["beatmap_id"] != reference["beatmap_id"]]
    other = min(others, key=lambda b: abs(float(b["difficultyrating"]) - float(reference["difficultyrating"]))) if others else None
    length_ms = float(meta["audio_length"]) * 1000
    start, end = (20000, 80000) if length_ms >= 90000 else (10000, max(20000, int(length_ms) - 10000))
    date = str(reference.get("approved_date") or reference.get("submit_date"))[:19]
    reference_text = (song_dir / f"human_{reference['beatmap_id']}.osu").read_text(encoding="utf-8")
    return Plan(
        reference_beatmap_id=int(reference["beatmap_id"]),
        other_beatmap_id=int(other["beatmap_id"]) if other else None,
        difficulty=round(float(reference["difficultyrating"]), 2),
        year=datetime.fromisoformat(date).year,
        hold_note_ratio=round(hold_ratio(reference_text, keys), 3),
        start=start,
        end=end,
    )


def generate(song_dir: Path, plan: Plan, out_dir: Path, keys: int, seed: int, human_timing: bool,
             overrides: list[str]) -> Path | None:
    existing = list(out_dir.glob("beatmap*.osu"))
    if existing:
        return existing[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "inference.py",
        f"audio_path={song_dir / 'audio.ogg'}",
        f"output_path={out_dir}",
        "gamemode=3", f"keycount={keys}",
        f"difficulty={plan.difficulty}", f"year={plan.year}", f"hold_note_ratio={plan.hold_note_ratio}",
        f"start_time={plan.start}", f"end_time={plan.end}",
        f"seed={seed}",
        f"hydra.run.dir={out_dir / 'hydra'}",
    ]
    if human_timing:
        command += [f"beatmap_path={song_dir / f'human_{plan.reference_beatmap_id}.osu'}",
                    "in_context=[TIMING]", "output_type=[MAP,SV]"]
    command += overrides
    env = os.environ | {"PYTHONUTF8": "1"}
    with open(out_dir / "inference.log", "w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        print(f"  inference failed, see {out_dir / 'inference.log'}")
        return None
    existing = list(out_dir.glob("beatmap*.osu"))
    return existing[0] if existing else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Name of this run; results go to <runs-dir>/<name>")
    parser.add_argument("--num-songs", type=int, default=14)
    parser.add_argument("--keys", type=int, default=4)
    parser.add_argument("--target-sr", type=float, default=4.0, help="Pick the human map closest to this star rating")
    parser.add_argument("--human-timing", action="store_true", help="Give the model the human map's timing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None, help="Default: benchmarks/mania<keys>k_data")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Default: benchmarks/mania<keys>k_runs")
    args, overrides = parser.parse_known_args()
    data_dir = args.data_dir or REPO_ROOT / "benchmarks" / f"mania{args.keys}k_data"
    runs_dir = args.runs_dir or REPO_ROOT / "benchmarks" / f"mania{args.keys}k_runs"

    songs = fetch_songs(data_dir, args.num_songs, args.keys)
    run_dir = runs_dir / args.name
    results: list[SongResult] = []
    for i, song_dir in enumerate(songs, 1):
        plan = make_plan(song_dir, args.keys, args.target_sr)
        print(f"[{i}/{len(songs)}] {song_dir.name}: {plan.difficulty} stars, "
              f"{plan.hold_note_ratio:.0%} holds, {plan.start // 1000}-{plan.end // 1000} s", flush=True)
        generated = generate(song_dir, plan, run_dir / song_dir.name, args.keys, args.seed, args.human_timing, overrides)
        if generated is None:
            continue
        other = (song_dir / f"human_{plan.other_beatmap_id}.osu").read_text(encoding="utf-8") if plan.other_beatmap_id else None
        result = score_song(
            song_dir.name, plan.reference_beatmap_id,
            (song_dir / f"human_{plan.reference_beatmap_id}.osu").read_text(encoding="utf-8"),
            generated.read_text(encoding="utf-8"), other, plan.start, plan.end, args.keys,
        )
        results.append(result)
        h, g = result.human, result.generated
        print(f"  stars {result.human_sr:.2f} -> {result.generated_sr:.2f}  notes {h['notes']} -> {g['notes']}  "
              f"holds {h['holds']:.0%} -> {g['holds']:.0%}  3+ chords {h['chords_3plus']:.0%} -> {g['chords_3plus']:.0%}  "
              f"jacks {h['jack_rate']:.0%} -> {g['jack_rate']:.0%}  onset F1 {result.onset_f1:.2f}  "
              f"BPM {'ok' if result.bpm_correct else 'wrong'}  invalid {g['invalid']}", flush=True)

    s = summarize(results)
    print()
    print(f"{args.name}: {s['songs']} songs (human -> generated)")
    if results:
        print(f"  star rating error   {s['sr_error']:.2f}")
        print(f"  onset F1            {s['onset_f1']:.2f}" + (f"   (two human maps: {s['human_pair_f1']:.2f})" if s["human_pair_f1"] is not None else ""))
        print(f"  notes               {s['human_notes']:.0f} -> {s['generated_notes']:.0f}")
        print(f"  holds               {s['human_holds']:.0%} -> {s['generated_holds']:.0%}   (mean error {s['hold_ratio_error']:.0%})")
        print(f"  3+ note chords      {s['human_chords_3plus']:.0%} -> {s['generated_chords_3plus']:.0%}")
        print(f"  jacks               {s['human_jack_rate']:.0%} -> {s['generated_jack_rate']:.0%}")
        print(f"  pattern repetition  {s['human_repetition']:.0%} -> {s['generated_repetition']:.0%}")
        print(f"  BPM correct         {s['bpm_correct']}/{s['songs']}")
        print(f"  invalid notes       {s['invalid_notes']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps({
        "overrides": overrides,
        "human_timing": args.human_timing,
        "seed": args.seed,
        "keys": args.keys,
        "summary": s,
        "songs": [asdict(r) for r in results],
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
