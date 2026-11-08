"""Schedules for the discriminator loss weight lambda(t)."""

from __future__ import annotations

from typing import Protocol

from cktn_electra.config import PretrainConfig


class LambdaScheduler(Protocol):
    def get_lambda(self, global_step: int) -> float: ...


class LinearLambdaScheduler:
    """Linear generator-discriminator schedule.

    lambda = 0 up to the end of ``zero_until_epoch``, then rises linearly to
    ``lambda_max`` at the end of ``ramp_until_epoch``, and stays there.
    With the paper defaults (2, 3) the discriminator is off for epochs 1-2,
    ramps up during epoch 3 and is fully weighted for epochs 4-6.
    """

    def __init__(
        self,
        lambda_max: float,
        zero_until_epoch: int,
        ramp_until_epoch: int,
        steps_per_epoch: int,
    ):
        self.lambda_max = lambda_max
        self.warmup_steps = zero_until_epoch * steps_per_epoch
        self.ramp_end_steps = ramp_until_epoch * steps_per_epoch

    def get_lambda(self, global_step: int) -> float:
        if global_step <= self.warmup_steps:
            return 0.0
        if global_step >= self.ramp_end_steps:
            return self.lambda_max
        progress = (global_step - self.warmup_steps) / (self.ramp_end_steps - self.warmup_steps)
        return self.lambda_max * progress


class ConstantLambdaScheduler:
    """Fixed lambda from the first update (the "no linear lambda" ablation)."""

    def __init__(self, lambda_max: float):
        self.lambda_max = lambda_max

    def get_lambda(self, global_step: int) -> float:
        del global_step
        return self.lambda_max


def build_lambda_scheduler(config: PretrainConfig, steps_per_epoch: int) -> LambdaScheduler:
    if config.lambda_schedule == "constant":
        return ConstantLambdaScheduler(config.lambda_max)
    return LinearLambdaScheduler(
        lambda_max=config.lambda_max,
        zero_until_epoch=config.lambda_zero_until_epoch,
        ramp_until_epoch=config.lambda_ramp_until_epoch,
        steps_per_epoch=steps_per_epoch,
    )
