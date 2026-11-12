"""Tests for the per-token-type sampling temperatures (timing, mania column, taiko hit).

The V32 tokenizer and training config are the real ones, so these tests check the rules
against the token order V32 actually uses. They need network access the first time to
download `tokenizer.json` from the Hugging Face model repo.
"""
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from slider import Beatmap

from osuT5.osuT5.dataset.osu_parser import OsuParser
from osuT5.osuT5.event import ContextType, Event, EventType
from osuT5.osuT5.inference.logit_processors import (
    ConditionalTemperatureLogitsWarper,
    get_beat_type_tokens,
    get_mania_column_tokens,
    get_mania_type_tokens,
    get_scroll_speed_tokens,
    get_snapping_tokens,
)
from osuT5.osuT5.inference.server import build_logits_processors
from osuT5.osuT5.tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_T = 0.9
TIMING_T = 0.1
MANIA_T = 0.5
TAIKO_T = 0.7


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("OliBomby/Mapperatorinator-v32")


@pytest.fixture(scope="module")
def train_args():
    import config  # noqa: F401  registers the inference config schema
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs" / "inference"), version_base="1.1"):
        cfg = compose(config_name="v32")
    return OmegaConf.to_object(cfg).train


def make_warper(tokenizer, types_first, timing_context=False):
    return ConditionalTemperatureLogitsWarper(
        BASE_T, TIMING_T, MANIA_T, TAIKO_T, types_first,
        get_beat_type_tokens(tokenizer),
        get_mania_type_tokens(tokenizer),
        get_scroll_speed_tokens(tokenizer),
        snapping_tokens=get_snapping_tokens(tokenizer),
        mania_column_tokens=get_mania_column_tokens(tokenizer),
        timing_context=timing_context,
    )


def tok(tokenizer, event_type, value=0):
    return tokenizer.encode(Event(event_type, value))


def random_scores(tokenizer, batch=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, tokenizer.vocab_size_out, generator=generator) * 3


def non_column_mask(tokenizer):
    """All tokens except mania columns, which time-first models give their own temperature."""
    mask = torch.ones(tokenizer.vocab_size_out, dtype=torch.bool)
    mask[tokenizer.event_start[EventType.MANIA_COLUMN]:tokenizer.event_end[EventType.MANIA_COLUMN]] = False
    return mask


def assert_scaled(tokenizer, out, scores, temperature):
    mask = non_column_mask(tokenizer)
    torch.testing.assert_close(out[..., mask], scores[..., mask] / temperature)


# --- Assumptions about V32 -------------------------------------------------------------

def test_v32_is_time_first(train_args):
    assert train_args.data.types_first is False


# --- Time-first (V32) rules ------------------------------------------------------------

@pytest.mark.parametrize("last_tokens", [
    [EventType.TIME_SHIFT],
    [EventType.SNAPPING, EventType.SCROLL_SPEED],  # would be a taiko hit in the map pass
])
def test_timing_pass_uses_timing_temperature(tokenizer, last_tokens):
    warper = make_warper(tokenizer, types_first=False, timing_context=True)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[tokenizer.sos_id] + [tok(tokenizer, t, 1) for t in last_tokens]])
    # Every token, mania columns included: the timing pass only contains timing events
    torch.testing.assert_close(warper(input_ids, scores), scores / TIMING_T)


def test_map_pass_uses_base_temperature(tokenizer):
    warper = make_warper(tokenizer, types_first=False, timing_context=False)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[tokenizer.sos_id, tok(tokenizer, EventType.TIME_SHIFT, 50)]])
    assert_scaled(tokenizer, warper(input_ids, scores), scores, BASE_T)


def test_mania_columns_resampled_within_their_category(tokenizer):
    warper = make_warper(tokenizer, types_first=False)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[tokenizer.sos_id, tok(tokenizer, EventType.SNAPPING, 4)]])
    out = warper(input_ids, scores)

    start, end = tokenizer.event_start[EventType.MANIA_COLUMN], tokenizer.event_end[EventType.MANIA_COLUMN]
    base_probs = torch.softmax(scores / BASE_T, dim=-1)
    out_probs = torch.softmax(out, dim=-1)

    # Probability of picking some column is unchanged
    torch.testing.assert_close(out_probs[:, start:end].sum(), base_probs[:, start:end].sum())
    # Which column is picked follows the mania column temperature
    torch.testing.assert_close(
        out_probs[:, start:end] / out_probs[:, start:end].sum(),
        torch.softmax(scores[:, start:end] / MANIA_T, dim=-1),
    )
    # Every other token keeps its base-temperature probability
    outside = torch.ones(scores.shape[-1], dtype=torch.bool)
    outside[start:end] = False
    torch.testing.assert_close(out_probs[:, outside], base_probs[:, outside])


def test_mania_category_survives_masked_columns(tokenizer):
    warper = make_warper(tokenizer, types_first=False)
    scores = random_scores(tokenizer)
    start, end = tokenizer.event_start[EventType.MANIA_COLUMN], tokenizer.event_end[EventType.MANIA_COLUMN]
    scores[:, start:end] = -torch.inf
    out = warper(torch.tensor([[tokenizer.sos_id]]), scores)
    assert not torch.isnan(out).any()
    assert torch.isneginf(out[:, start:end]).all()


def test_taiko_hit_after_snap_and_scroll_speed(tokenizer):
    warper = make_warper(tokenizer, types_first=False)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[
        tokenizer.sos_id,
        tok(tokenizer, EventType.TIME_SHIFT, 50),
        tok(tokenizer, EventType.SNAPPING, 4),
        tok(tokenizer, EventType.SCROLL_SPEED, 100),
    ]])
    assert_scaled(tokenizer, warper(input_ids, scores), scores, TAIKO_T)


def test_std_slider_velocity_is_not_a_taiko_hit(tokenizer):
    warper = make_warper(tokenizer, types_first=False)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[
        tokenizer.sos_id,
        tok(tokenizer, EventType.TIME_SHIFT, 50),
        tok(tokenizer, EventType.SNAPPING, 4),
        tok(tokenizer, EventType.POS, 100),
        tok(tokenizer, EventType.POS_REFINE, 3),
        tok(tokenizer, EventType.SCROLL_SPEED, 100),
    ]])
    assert_scaled(tokenizer, warper(input_ids, scores), scores, BASE_T)


def test_rules_are_applied_per_row(tokenizer):
    warper = make_warper(tokenizer, types_first=False)
    scores = random_scores(tokenizer, batch=2)
    taiko_row = [tok(tokenizer, EventType.SNAPPING, 4), tok(tokenizer, EventType.SCROLL_SPEED, 100)]
    other_row = [tok(tokenizer, EventType.SNAPPING, 4), tok(tokenizer, EventType.TIME_SHIFT, 60)]
    out = warper(torch.tensor([other_row, taiko_row]), scores)
    assert_scaled(tokenizer, out[0], scores[0], BASE_T)
    assert_scaled(tokenizer, out[1], scores[1], TAIKO_T)


# --- Types-first (V28, V29, V31) rules are unchanged -----------------------------------

def test_types_first_timing_rule(tokenizer):
    warper = make_warper(tokenizer, types_first=True)
    scores = random_scores(tokenizer)
    after_beat = torch.tensor([[tokenizer.sos_id, tok(tokenizer, EventType.BEAT)]])
    after_circle = torch.tensor([[tokenizer.sos_id, tok(tokenizer, EventType.CIRCLE)]])
    torch.testing.assert_close(warper(after_beat, scores), scores / TIMING_T)
    torch.testing.assert_close(warper(after_circle, scores), scores / BASE_T)


def test_types_first_ignores_timing_context(tokenizer):
    warper = make_warper(tokenizer, types_first=True, timing_context=True)
    scores = random_scores(tokenizer)
    after_circle = torch.tensor([[tokenizer.sos_id, tok(tokenizer, EventType.CIRCLE)]])
    torch.testing.assert_close(warper(after_circle, scores), scores / BASE_T)


def test_types_first_mania_column_rule(tokenizer):
    warper = make_warper(tokenizer, types_first=True)
    scores = random_scores(tokenizer)
    input_ids = torch.tensor([[
        tok(tokenizer, EventType.HOLD_NOTE),
        tok(tokenizer, EventType.TIME_SHIFT, 50),
        tok(tokenizer, EventType.SNAPPING, 4),
    ]])
    torch.testing.assert_close(warper(input_ids, scores), scores / MANIA_T)


def test_no_special_temperatures_is_plain_temperature(tokenizer):
    warper = ConditionalTemperatureLogitsWarper(
        BASE_T, BASE_T, BASE_T, BASE_T, False,
        get_beat_type_tokens(tokenizer), get_mania_type_tokens(tokenizer), get_scroll_speed_tokens(tokenizer),
        snapping_tokens=get_snapping_tokens(tokenizer), mania_column_tokens=get_mania_column_tokens(tokenizer),
    )
    scores = random_scores(tokenizer)
    torch.testing.assert_close(warper(torch.tensor([[tokenizer.sos_id]]), scores), scores / BASE_T)


def test_rejects_non_positive_temperature(tokenizer):
    with pytest.raises(ValueError):
        ConditionalTemperatureLogitsWarper(0.9, 0.0, 0.9, 0.9, False, (), (), ())


# --- Wiring through build_logits_processors --------------------------------------------

@pytest.mark.parametrize("context_type, expected", [
    (ContextType.TIMING, TIMING_T),
    (ContextType.MAP, BASE_T),
    (None, BASE_T),
])
def test_build_logits_processors_passes_context_type(tokenizer, context_type, expected):
    processors = build_logits_processors(
        tokenizer, 1.0, 0, False, BASE_T, TIMING_T, MANIA_T, TAIKO_T, 0, "cpu",
        context_type=context_type,
    )
    warpers = [p for p in processors if isinstance(p, ConditionalTemperatureLogitsWarper)]
    assert len(warpers) == 1
    assert warpers[0].temperature == expected


# --- The token order the rules rely on, checked with the real V32 parser --------------

OSU_TEMPLATE = """osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: {mode}
LetterboxInBreaks: 0
WidescreenStoryboard: 1

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 16
TimelineZoom: 1

[Metadata]
Title:test
TitleUnicode:test
Artist:test
ArtistUnicode:test
Creator:test
Version:test
Source:
Tags:
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:{cs}
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
0,0,"bg.jpg",0,0

[TimingPoints]
1000,500,4,1,0,100,1,0
3000,-50,4,1,0,100,0,0

[HitObjects]
{hit_objects}
"""

STD_OBJECTS = """256,192,1000,5,0,0:0:0:0:
300,192,1500,1,2,0:0:0:0:
100,100,2000,2,0,B|200:100|200:200,1,140,2|0,0:0|0:0,0:0:0:0:
100,300,3000,6,8,P|200:350|300:300,1,140,0|0,0:0|0:0,0:0:0:0:"""

TAIKO_OBJECTS = """256,192,1000,5,0,0:0:0:0:
256,192,1250,1,2,0:0:0:0:
256,192,1500,1,8,0:0:0:0:
256,192,3000,1,0,0:0:0:0:
256,192,3250,1,4,0:0:0:0:"""

MANIA_OBJECTS = """64,192,1000,1,0,0:0:0:0:
192,192,1250,1,0,0:0:0:0:
320,192,1500,128,0,2000:0:0:0:0:
448,192,3000,1,0,0:0:0:0:"""


def parse(train_args, tokenizer, tmp_path, mode, cs, hit_objects, types_first=None):
    path = tmp_path / f"mode{mode}.osu"
    path.write_text(OSU_TEMPLATE.format(mode=mode, cs=cs, hit_objects=hit_objects), encoding="utf-8")
    parser = OsuParser(train_args, tokenizer)
    if types_first is not None:
        parser.types_first = types_first
    events, _ = parser.parse(Beatmap.from_path(path))
    return [e.type for e in events]


@pytest.mark.parametrize("types_first", [False, True])
def test_taiko_hitsound_follows_snap_and_scroll_speed(train_args, tokenizer, tmp_path, types_first):
    types = parse(train_args, tokenizer, tmp_path, mode=1, cs=5, hit_objects=TAIKO_OBJECTS, types_first=types_first)
    hitsounds = [i for i, t in enumerate(types) if t == EventType.HITSOUND]
    assert len(hitsounds) == 5
    for i in hitsounds:
        assert types[i - 2:i] == [EventType.SNAPPING, EventType.SCROLL_SPEED]


@pytest.mark.parametrize("types_first", [False, True])
def test_std_scroll_speed_never_follows_snap(train_args, tokenizer, tmp_path, types_first):
    types = parse(train_args, tokenizer, tmp_path, mode=0, cs=4, hit_objects=STD_OBJECTS, types_first=types_first)
    scroll_speeds = [i for i, t in enumerate(types) if t == EventType.SCROLL_SPEED]
    assert len(scroll_speeds) == 2  # one per slider head
    for i in scroll_speeds:
        assert types[i - 1] != EventType.SNAPPING


def test_mania_columns_are_tokens(train_args, tokenizer, tmp_path):
    types = parse(train_args, tokenizer, tmp_path, mode=3, cs=4, hit_objects=MANIA_OBJECTS)
    assert types.count(EventType.MANIA_COLUMN) >= 4
    # Time-first: a mania note's type token comes after its column, so the old look-back rule can't see it
    first_column = types.index(EventType.MANIA_COLUMN)
    assert EventType.CIRCLE not in types[:first_column]
