import numpy as np
import pytest

from mania_generate import (
    Candidate,
    consensus_timing,
    density_spike,
    double_bpm,
    find_spikes,
    finer_than_quarter,
    hydra_list,
    hydra_string,
    loudest_window,
    music_end_ms,
    pick_difficulty,
    rhythm_stats,
    section_stats,
    select,
    style_score,
    title_from_filename,
    trim_timing_after,
    with_overrides,
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


def test_select_respects_min_stars():
    candidates = [cand("a", 6.2, 0.9), cand("b", 6.6, 0.3), cand("c", 6.9, 0.2)]
    best, _ = select(candidates, 7.0, min_stars=6.5)
    assert best.name == "b"
    best, _ = select(candidates, 7.0)
    assert best.name == "a"


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


# --- Spike detection ------------------------------------------------------------------------
# 120 BPM: a beat is 500 ms, a quarter beat 125 ms, a 6 s section holds 48 quarter-beat steps.

def stream(start, steps):
    """Rolling 1/4 stream: each column comes back every beat, so no fast jacks."""
    return [note(i % 4, start + 125 * i) for i in range(steps)]


def jacky(start, steps):
    """Each column hit twice in a row at 1/4: a fast jack every other note."""
    return [note((i // 2) % 4, start + 125 * i) for i in range(steps)]


def test_section_stats_counts_fast_jacks():
    stats = section_stats(osu(RED_120, jacky(1000, 48)), 4)
    assert stats[0] == (1000, 48, 24)


def test_jack_burst_is_flagged():
    objects = stream(1000, 48 * 4) + jacky(1000 + 6000 * 4, 48) + stream(1000 + 6000 * 5, 48 * 4)
    spikes = find_spikes(osu(RED_120, objects), 4)
    assert len(spikes) == 1
    assert (spikes[0].start, spikes[0].end) == (25000, 31000)
    assert spikes[0].reason == "jack burst" and spikes[0].fast_jacks >= 20


def test_speedjack_all_the_way_is_not_a_spike():
    assert find_spikes(osu(RED_120, jacky(1000, 48 * 8)), 4) == []


def test_density_spike_is_flagged():
    chords = [n for i in range(48) for n in (note(i % 2, 25000 + 125 * i), note(2 + i % 2, 25000 + 125 * i))]
    objects = stream(1000, 48 * 4) + chords + stream(31000, 48 * 4)
    spikes = find_spikes(osu(RED_120, objects), 4)
    assert len(spikes) == 1 and spikes[0].reason == "density"


def test_adjacent_spike_sections_merge():
    objects = stream(1000, 48 * 4) + jacky(25000, 96) + stream(37000, 48 * 4)
    spikes = find_spikes(osu(RED_120, objects), 4)
    assert [(s.start, s.end) for s in spikes] == [(25000, 37000)]


def test_with_overrides_replaces_keys():
    result = with_overrides(["seed=0", "end_time=9", "keycount=4"], seed=3, start_time=1)
    assert result == ["end_time=9", "keycount=4", "seed=3", "start_time=1"]


@pytest.mark.parametrize("value", [
    r"C:\Users\me\mania_maps\Rude Buster (Camellia Remix)\timing.osu",
    "C:/Users/me/Downloads/SHAKE!THE!CITY! [x], y=z.mp3",
    'a title with "quotes"',
    "ends with a backslash\\",
    'backslash before quote\\"x',
])
def test_quoted_values_parse_in_hydra(value):
    from hydra.core.override_parser.overrides_parser import OverridesParser

    override = OverridesParser.create().parse_overrides([f"key={hydra_string(value)}"])[0]
    assert override.value() == value


# --- Style-aware picking and preview ----------------------------------------------------------

def test_rhythm_stats_stream_vs_changing_rhythm():
    stream_map = osu(RED_120, stream(1000, 64))
    assert rhythm_stats(stream_map, 4) == (pytest.approx(1.0), 0.0)
    # Gaps cycle 1/4, 1/6, 1/2, 1/3: the rhythm changes on every note and never runs
    t, notes = 1000, []
    for i in range(64):
        notes.append(note(i % 4, int(t)))
        t += [125, 500 / 6, 250, 500 / 3][i % 4]
    monotone, changes = rhythm_stats(osu(RED_120, notes), 4)
    assert monotone == 0 and changes == pytest.approx(1.0)


def test_style_scores():
    c = Candidate("x", "", 6.8, 20, 0.2, 0.3, 1.1, 0, 0.1, 0.25, 0, monotone=0.4, rhythm_changes=0.3)
    assert style_score(c, "tech") == pytest.approx(-0.1)
    assert style_score(c, "jacks") == 0.25
    assert style_score(c, "any") == 0.3


def test_select_tech_prefers_changing_rhythm_over_repetition():
    streamy = Candidate("streamy", "", 6.8, 20, 0.1, 0.5, 1.1, 0, 0.1, 0.1, 0, monotone=0.8, rhythm_changes=0.1)
    techy = Candidate("techy", "", 6.7, 20, 0.3, 0.2, 1.1, 0, 0.1, 0.1, 0, monotone=0.3, rhythm_changes=0.3)
    assert select([streamy, techy], 7.0, 6.5, style="tech")[0].name == "techy"
    assert select([streamy, techy], 7.0, 6.5, style="any")[0].name == "streamy"


def test_loudest_window_finds_the_loud_part():
    sr = 1000
    samples = np.concatenate([0.1 * np.ones(20 * sr), np.ones(10 * sr), 0.1 * np.ones(30 * sr)])
    start, stop = loudest_window(samples, sr, 10, 60000)
    assert (start, stop) == (20000, 30000)


def test_loudest_window_short_song():
    assert loudest_window(np.ones(5000), 1000, 60, 5000) == (0, 5000)


def test_pick_difficulty_prefers_in_range_then_style():
    previews = [(6.0, cand("a", 6.3, 0.4)), (6.5, cand("b", 6.6, 0.2)), (7.0, cand("c", 6.9, 0.5))]
    d, _ = pick_difficulty(previews, 6.5, 7.0, "any")
    assert d == 7.0
    d, why = pick_difficulty([(6.0, cand("a", 5.5)), (6.5, cand("b", 6.0))], 6.5, 7.0, "any")
    assert d == 6.5 and "closest" in why
