import json

import pytest

from mania_benchmark import (
    Note,
    bpm_correct,
    hold_ratio,
    make_plan,
    onset_f1,
    parse_notes,
    pattern_stats,
    score_song,
    star_rating,
    summarize,
    trimmed,
)

X = {0: 64, 1: 192, 2: 320, 3: 448}  # 4K column centers


def osu(timing: str, objects: list[str]) -> str:
    return ("osu file format v14\n\n[General]\nAudioFilename: audio.ogg\nMode: 3\n\n"
            "[Metadata]\nTitle:t\nArtist:a\nCreator:c\nVersion:v\n\n"
            "[Difficulty]\nHPDrainRate:8\nCircleSize:4\nOverallDifficulty:8\nApproachRate:5\n"
            "SliderMultiplier:1.4\nSliderTickRate:1\n\n"
            f"[TimingPoints]\n{timing}\n\n[HitObjects]\n" + "\n".join(objects) + "\n")


def note(col, t):
    return f"{X[col]},192,{t},1,0,0:0:0:0:"


def hold(col, t, end):
    return f"{X[col]},192,{t},128,0,{end}:0:0:0:0:"


TIMING_120 = "1000,500,4,2,0,100,1,0"


def test_parse_notes_columns_holds_and_window():
    text = osu(TIMING_120, [note(0, 1000), hold(2, 1500, 2000), note(3, 9000)])
    assert parse_notes(text, 4) == [Note(1000, 0, 1000), Note(1500, 2, 2000), Note(9000, 3, 9000)]
    assert parse_notes(text, 4, 1200, 5000) == [Note(1500, 2, 2000)]


def test_hold_ratio():
    assert hold_ratio(osu(TIMING_120, [note(0, 1000), hold(1, 1500, 2000)]), 4) == 0.5


@pytest.mark.parametrize("generated, expected", [
    ([1000, 1500, 2000], 1.0),
    ([1020, 1480, 2010], 1.0),     # within 25 ms
    ([1000, 1500], 0.8),           # recall 2/3, precision 1
    ([1100, 1600, 2100], 0.0),     # all 100 ms late
])
def test_onset_f1(generated, expected):
    assert onset_f1([1000, 1500, 2000], generated) == pytest.approx(expected)


def test_pattern_stats_chords_and_jacks():
    notes = [Note(1000, 0, 1000), Note(1000, 1, 1000), Note(1000, 2, 1000),  # 3-note chord
             Note(1250, 2, 1250),                                             # jack on column 2
             Note(1500, 3, 1800)]                                             # hold
    stats = pattern_stats(notes, 4)
    assert stats["notes"] == 5
    assert stats["chords_3plus"] == pytest.approx(1 / 3)
    assert stats["jack_rate"] == pytest.approx(1 / 5)
    assert stats["holds"] == pytest.approx(1 / 5)
    assert stats["invalid"] == 0


def test_pattern_stats_counts_overlaps():
    notes = [Note(1000, 1, 2000), Note(1500, 1, 1500), Note(2500, 1, 2500), Note(2500, 2, 2500)]
    assert pattern_stats(notes, 4)["invalid"] == 1


@pytest.mark.parametrize("generated_timing, expected", [
    ("1000,500,4,2,0,100,1,0", True),
    ("1000,500,4,2,0,100,1,0\n30000,500,4,2,0,100,1,0", True),  # extra red line, same BPM
    ("1000,250,4,2,0,100,1,0", False),                          # double tempo
    ("1000,500,4,2,0,100,1,0\n30000,496,4,2,0,100,1,0", False),  # drifts to 121 BPM
])
def test_bpm_correct(generated_timing, expected):
    human = osu(TIMING_120, [])
    assert bpm_correct(human, osu(generated_timing, []), 20000, 80000) is expected


def test_bpm_correct_ignores_sections_outside_the_window():
    generated = osu("1000,250,4,2,0,100,1,0\n15000,500,4,2,0,100,1,0\n90000,250,4,2,0,100,1,0", [])
    assert bpm_correct(osu(TIMING_120, []), generated, 20000, 80000)


def test_bpm_correct_follows_human_tempo_changes():
    human = osu("1000,500,4,2,0,100,1,0\n40000,400,4,2,0,100,1,0", [])  # 120 then 150 BPM
    assert bpm_correct(human, osu("1000,500,4,2,0,100,1,0\n40000,400,4,2,0,100,1,0", []), 20000, 80000)
    assert not bpm_correct(human, osu("1000,500,4,2,0,100,1,0", []), 20000, 80000)


def test_trimmed_keeps_window_objects():
    text = osu(TIMING_120, [note(0, 1000), note(1, 25000), note(2, 90000)])
    assert parse_notes(trimmed(text, 20000, 80000), 4) == [Note(25000, 1, 25000)]


def test_star_rating_increases_with_density():
    sparse = osu(TIMING_120, [note(i % 4, 1000 + 500 * i) for i in range(100)])
    dense = osu(TIMING_120, [note(i % 4, 1000 + 125 * i) for i in range(400)])
    assert 0 < star_rating(sparse) < star_rating(dense)


def test_score_and_summary():
    human = osu(TIMING_120, [note(i % 4, 20000 + 250 * i) for i in range(100)])
    generated = osu(TIMING_120, [note((i + 1) % 4, 20000 + 250 * i) for i in range(100)])
    result = score_song("s", 1, human, generated, None, 20000, 80000, 4)
    assert result.onset_f1 == 1.0 and result.bpm_correct
    assert result.generated_sr == pytest.approx(result.human_sr, abs=0.05)
    summary = summarize([result])
    assert summary["songs"] == 1 and summary["bpm_correct"] == 1 and summary["invalid_notes"] == 0
    assert summary["human_pair_f1"] is None


def test_make_plan_picks_reference_and_other(tmp_path):
    song = tmp_path / "000001"
    song.mkdir()
    beatmaps = []
    for bid, sr in [(11, 2.1), (12, 3.9), (13, 5.2)]:
        objects = [hold(0, 1000, 1400), note(1, 2000)] if bid == 12 else [note(0, 1000)]
        (song / f"human_{bid}.osu").write_text(osu(TIMING_120, objects), encoding="utf-8")
        beatmaps.append({"beatmap_id": bid, "difficultyrating": sr, "mode": "3", "diff_size": "4",
                         "approved": "1", "approved_date": "2021-05-01 12:00:00", "content": "x"})
    (song / "meta.json").write_text(json.dumps({"audio_length": 150.0, "beatmaps": beatmaps}), encoding="utf-8")
    plan = make_plan(song, 4, target_sr=4.0)
    assert plan.reference_beatmap_id == 12 and plan.other_beatmap_id == 13
    assert plan.difficulty == 3.9 and plan.year == 2021 and plan.hold_note_ratio == 0.5
    assert (plan.start, plan.end) == (20000, 80000)
