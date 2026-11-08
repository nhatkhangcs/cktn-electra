import pytest

from cktn_electra.config import PretrainConfig
from cktn_electra.schedule import ConstantLambdaScheduler, LinearLambdaScheduler, build_lambda_scheduler
from cktn_electra.text import (
    SCRIPT_KHMER,
    SCRIPT_LATIN,
    SCRIPT_OTHER,
    SCRIPT_SHARED,
    is_noise_surface,
    script_profile,
    token_surface,
)


@pytest.mark.parametrize(
    "surface, expected",
    [
        ("noọng", SCRIPT_LATIN),
        ("bruk", SCRIPT_LATIN),
        ("ខ្មែរ", SCRIPT_KHMER),
        ("123", SCRIPT_SHARED),
        (",", SCRIPT_SHARED),
        ("ខa", SCRIPT_OTHER),  # mixed Khmer + Latin
        ("слово", SCRIPT_OTHER),  # Cyrillic
    ],
)
def test_script_profile(surface, expected):
    assert script_profile(surface) == expected


@pytest.mark.parametrize("surface, noisy", [(",", True), ("12a", True), ("%%", True), ("", True), ("nhà", False), ("ភាសា", False)])
def test_is_noise_surface(surface, noisy):
    assert is_noise_surface(surface) is noisy


def test_token_surface_strips_markers():
    assert token_surface("▁nhà") == "nhà"
    assert token_surface("##ing") == "ing"


def test_linear_lambda_schedule_matches_paper():
    sched = LinearLambdaScheduler(lambda_max=50.0, zero_until_epoch=2, ramp_until_epoch=3, steps_per_epoch=100)
    assert sched.get_lambda(0) == 0.0
    assert sched.get_lambda(200) == 0.0  # end of epoch 2
    assert sched.get_lambda(250) == pytest.approx(25.0)  # middle of epoch 3
    assert sched.get_lambda(300) == 50.0
    assert sched.get_lambda(599) == 50.0


def test_build_scheduler_variants():
    assert isinstance(build_lambda_scheduler(PretrainConfig(), 10), LinearLambdaScheduler)
    constant = build_lambda_scheduler(PretrainConfig(lambda_schedule="constant"), 10)
    assert isinstance(constant, ConstantLambdaScheduler)
    assert constant.get_lambda(0) == 50.0


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PretrainConfig(lambda_schedule="cosine")
    with pytest.raises(ValueError):
        PretrainConfig(lambda_zero_until_epoch=3, lambda_ramp_until_epoch=2)
