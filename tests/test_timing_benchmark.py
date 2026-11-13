import numpy as np
import pytest

from timing_benchmark import beats_from_red_lines, red_lines, score_song, signed_beat_errors, summarize


def osu(timing_points: str) -> str:
    return f"osu file format v14\n\n[General]\nMode: 0\n\n[TimingPoints]\n{timing_points}\n\n[HitObjects]\n"


HUMAN_120 = osu("1000,500,4,2,0,100,1,0")


def test_red_lines_skips_inherited_points():
    text = osu("1000,500,4,2,0,100,1,0\n2000,-50,4,2,0,100,0,0\n9000,400,4,2,0,100,1,0")
    assert red_lines(text) == [(1000.0, 500.0), (9000.0, 400.0)]


def test_red_lines_without_timing_section():
    assert red_lines("osu file format v14\n\n[HitObjects]\n") == []


def test_beats_follow_each_red_line():
    beats = beats_from_red_lines([(0.0, 500.0), (2000.0, 250.0)], 3000)
    np.testing.assert_allclose(beats, [0, 500, 1000, 1500, 2000, 2250, 2500, 2750])


@pytest.mark.parametrize("shift, expected", [(0, 0), (12, 12), (-7, -7)])
def test_signed_error_sign(shift, expected):
    # Positive means the generated beat is later than the human beat
    errors = signed_beat_errors(np.array([1000.0, 1500.0, 2000.0]), [(1000.0 + shift, 500.0)])
    np.testing.assert_allclose(errors, expected)


def test_exact_match_scores_zero():
    score = score_song("s", HUMAN_120, osu("1000,500,4,2,0,100,1,0"), 60000)
    assert score.bpm_exact and score.red_lines == 1
    assert score.median_err_ms == 0 and score.mae_ms == 0


def test_offset_error_is_measured():
    score = score_song("s", HUMAN_120, osu("1020,500,4,2,0,100,1,0"), 60000)
    assert score.bpm_exact
    assert score.median_err_ms == pytest.approx(20)


def test_double_tempo_is_not_exact():
    score = score_song("s", HUMAN_120, osu("1000,250,4,2,0,100,1,0"), 60000)
    assert not score.bpm_exact
    assert score.generated_bpms == [240.0]


def test_extra_bpm_changes_are_not_exact():
    generated = osu("1000,500,4,2,0,100,1,0\n20000,502,4,2,0,100,1,0")
    score = score_song("s", HUMAN_120, generated, 60000)
    assert not score.bpm_exact and score.red_lines == 2


def test_multi_bpm_human_maps_are_skipped():
    human = osu("1000,500,4,2,0,100,1,0\n9000,400,4,2,0,100,1,0")
    assert score_song("s", human, HUMAN_120, 60000) is None


def test_summary():
    scores = [
        score_song("a", HUMAN_120, osu("1010,500,4,2,0,100,1,0"), 60000),
        score_song("b", HUMAN_120, osu("1000,250,4,2,0,100,1,0"), 60000),
    ]
    summary = summarize(scores)
    assert summary["songs"] == 2 and summary["bpm_exact"] == 1 and summary["red_lines_total"] == 2
    assert summary["median_err_ms_on_exact"] == pytest.approx(10)
