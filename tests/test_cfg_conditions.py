"""Tests for choosing which conditions classifier-free guidance strengthens (cfg_conditions)."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from osuT5.osuT5.inference.processor import (
    CFG_CONDITIONS,
    GenerationConfig,
    Processor,
    negative_generation_config,
)
from osuT5.osuT5.tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

MANIA = GenerationConfig(gamemode=3, difficulty=4.0, year=2023, keycount=4, hold_note_ratio=0.3,
                         scroll_speed_ratio=0.1, mapper_id=123)


def test_default_negative_prompt_is_unchanged():
    negative = negative_generation_config(MANIA)
    assert negative.difficulty == 4.0 and negative.hold_note_ratio == 0.3 and negative.year == 2023
    assert negative.keycount == 4
    assert negative.mapper_id is None  # mapper only kept when descriptors are used


def test_mapper_kept_with_descriptors():
    config = GenerationConfig(gamemode=0, mapper_id=123, descriptors=["style/clean"],
                              negative_descriptors=["style/messy"])
    negative = negative_generation_config(config)
    assert negative.mapper_id == 123
    assert negative.descriptors == ["style/messy"]


@pytest.mark.parametrize("name", CFG_CONDITIONS)
def test_selected_condition_becomes_unknown(name):
    negative = negative_generation_config(MANIA, [name])
    assert getattr(negative, name) is None
    for other in set(CFG_CONDITIONS) - {name}:
        assert getattr(negative, other) == getattr(MANIA, other)


def test_rejects_conditions_without_unknown_token():
    with pytest.raises(ValueError):
        negative_generation_config(MANIA, ["keycount"])


@pytest.fixture(scope="module")
def processor():
    import config  # noqa: F401  registers the inference config schema
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs" / "inference"), version_base="1.1"):
        cfg = compose(config_name="v32", overrides=["cfg_scale=2", "cfg_conditions=[hold_note_ratio,difficulty]"])
    args = OmegaConf.to_object(cfg)
    tokenizer = Tokenizer.from_pretrained("OliBomby/Mapperatorinator-v32", subfolder="gamemode=3")
    return Processor(args, None, tokenizer)


def test_processor_negative_prompt_uses_unknown_tokens(processor):
    tokenizer = processor.tokenizer
    positive = processor.get_class_vector(MANIA, 120000)[0].tolist()
    negative = processor.get_class_vector(negative_generation_config(MANIA, processor.cfg_conditions), 120000)[0].tolist()
    changed = [(p, n) for p, n in zip(positive, negative) if p != n]
    assert len(positive) == len(negative)
    assert sorted(n for _, n in changed) == sorted([tokenizer.hold_note_ratio_unk, tokenizer.diff_unk])


def test_processor_rejects_bad_condition_names():
    import config  # noqa: F401
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs" / "inference"), version_base="1.1"):
        cfg = compose(config_name="v32", overrides=["cfg_conditions=[keycount]"])
    args = OmegaConf.to_object(cfg)
    with pytest.raises(ValueError):
        Processor(args, None, Tokenizer.from_pretrained("OliBomby/Mapperatorinator-v32"))
