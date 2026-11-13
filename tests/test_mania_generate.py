import numpy as np
import pytest

from mania_generate import (
    Candidate,
    consensus_timing,
    density_spike,
    double_bpm,
    finer_than_quarter,
    hydra_list,
    music_end_ms,
    select,
    title_from_filename,
    trim_timing_after,
)
from mania_benchmark import pattern_repetition
from timing_benchmark import red_lines

X = {0: 64, 1: 192, 2: 320, 3: 448}


def osu(timing: str, objects: list[str] = ()) -> str:
    return ("osu file format v14\n\n[General]\nAudioFilename: audio.mp3\nMode: 3\n\n"
            f"[TimingPoints]\n{timing}\n\n[HitObjects]\n" + "\n".join(objects) + "\n")


def note(col, t):
    return f"{X[col]},192,{t},1,0,0:0:0:0:"


RED_120 = "1000,500,4,2,0,100,1,0"


def test_consensus_picks_the_majority():
    a, b = osu(RED_120), osu("1000,400,4,2,0,100,1,0")
    assert consensus_timing([b, a, a]) == a


def test_music_end_detects_fade_to_silence():
    sr = 1000
    samples = np.concatenate([np.sin(np.arange(5 * sr)), np.zeros(5 * sr)])
    assert 5000 <= music_end_ms(samples, sr) <= 5500


def test_trim_timing_after_end():
    text = osu(f"{RED_120}\n84512,8721,4,2,0,100,1,0\n93235,333,4,2,0,100,1,0")
    assert red_lines(trim_timing_after(text, 84000)) == [(1000.0, 500.0)]


def test_trim_keeps_first_red_line():
    assert red_lines(trim_timing_after(osu(RED_120), 0)) == [(1000.0, 500.0)]


def test_double_bpm_only_changes_red_lines():
    text = osu(f"{RED_120}\n2000,-50,4,2,0,100,0,0")
    doubled = double_bpm(text)
    assert red_lines(doubled) == [(1000.0, 250.0)]
    assert "2000,-50,4,2,0,100,0,0" in doubled


def test_finer_than_quarter():
    quarters = osu(RED_120, [note(i % 4, 1000 + 125 * i) for i in range(40)])
    eighths = osu(RED_120, [note(i % 4, 1000 + 62 * i) for i in range(40)])
    assert finer_than_quarter(quarters, 4) == 0
    assert finer_than_quarter(eighths, 4) == 1


def test_density_spike():
    even = [note(i % 4, 1000 + 250 * i) for i in range(200)]
    assert density_spike(osu(RED_120, even), 4) == pytest.approx(1.0, abs=0.1)
    burst = even + [note((i + 2) % 4, 30125 + 250 * i) for i in range(24)]
    assert density_spike(osu(RED_120, burst), 4) > 1.35


def test_pattern_repetition():
    motif = [note(0, 1000 + 500 * b) for b in range(32)] + [note(2, 1250 + 500 * b) for b in range(32)]
    assert pattern_repetition(osu(RED_120, motif), 4) > 0.9
    mirrored = [note(0 if b % 2 else 3, 1000 + 500 * b) for b in range(32)]
    assert pattern_repetition(osu(RED_120, mirrored), 4) > 0.9
    rng = np.random.default_rng(0)
    random_notes = [note(int(c), 1000 + 125 * i) for i, c in enumerate(rng.integers(0, 4, 256))]
    assert pattern_repetition(osu(RED_120, random_notes), 4) < 0.2


def cand(name, stars, repetition=0.3, spike=1.1, invalid=0):
    return Candidate(name, "", stars, 20, 0, repetition, spike, 0, 0, 0, invalid)


def test_select_prefers_structure_within_the_band():
    best, _ = select([cand("a", 6.9, 0.15), cand("b", 6.5, 0.40), cand("c", 7.3, 0.6)], 7.0)
    assert best.name == "b"


def test_select_skips_spiky_candidates():
    best, _ = select([cand("a", 6.9, 0.5, spike=1.6), cand("b", 6.6, 0.3)], 7.0)
    assert best.name == "b"


def test_select_takes_hardest_under_cap_when_none_near():
    best, reason = select([cand("a", 4.0, 0.9), cand("b", 5.2, 0.1)], 7.0)
    assert best.name == "b" and "hardest" in reason


def test_select_when_all_over_cap():
    best, reason = select([cand("a", 7.4), cand("b", 7.9)], 7.0)
    assert best.name == "a" and "over" in reason


def test_select_avoids_invalid():
    best, _ = select([cand("a", 6.9, 0.9, invalid=2), cand("b", 6.8, 0.2)], 7.0)
    assert best.name == "b"


def test_helpers():
    from pathlib import Path
    assert title_from_filename(Path("tick-tack-819830653.m4a")) == "tick-tack"
    assert title_from_filename(Path("SHAKE!THE!CITY!-816623178.m4a")) == "SHAKE!THE!CITY!"
    assert hydra_list(["style/LN density", 'a"b']) == '["style/LN density","a\\"b"]'
