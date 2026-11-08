"""The ablation configs must differ from the main config in exactly one setting."""

from pathlib import Path

import pytest

from cktn_electra.config import PretrainConfig, load_pretrain_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "pretrain"


def _flatten(config: PretrainConfig) -> dict:
    data = config.to_dict()
    sampler = data.pop("sampler")
    data.pop("variant")
    return {**data, **{f"sampler.{k}": v for k, v in sampler.items()}}


def test_main_config_equals_defaults():
    assert load_pretrain_config(CONFIG_DIR / "main.json") == PretrainConfig()


@pytest.mark.parametrize(
    "name, key, value",
    [
        ("no_script_aware", "sampler.script_aware", False),
        ("no_linear_lambda", "lambda_schedule", "constant"),
        ("no_vocab_aug", "base_model", "google/rembert"),
    ],
)
def test_ablation_changes_one_field(name, key, value):
    main = _flatten(load_pretrain_config(CONFIG_DIR / "main.json"))
    ablation = _flatten(load_pretrain_config(CONFIG_DIR / f"{name}.json"))
    diff = {k for k in main if main[k] != ablation[k]}
    assert diff == {key}
    assert ablation[key] == value


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        PretrainConfig.from_dict({"not_a_field": 1})
