"""Lightweight fixed-step solvers for Flow-Planner inference.

The official Flow-Planner path solves a Flow Matching ODE over action tokens.
Diffusion DPM-Solver++ is not directly applicable because it assumes a VP
noise schedule and logSNR parameterization.  This module keeps the same Flow
Matching velocity field and exposes deployment-oriented fixed-step solvers:
Euler, midpoint, AB2, and a DEIS-like AB3 multistep update.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


VelocityFn = Callable[..., torch.Tensor]
MAX_EVAL_TIME = 1.0 - 1e-5


def _as_solver_time(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Match flow_matching.ODESolver: pass scalar ODE time into VelocityModel."""

    value = min(float(value), MAX_EVAL_TIME)
    return torch.tensor(value, device=device, dtype=dtype)


def _velocity(velocity_model: VelocityFn, x: torch.Tensor, t_value: float, model_extra: Dict) -> torch.Tensor:
    t = _as_solver_time(t_value, x.device, x.dtype)
    return velocity_model(x=x, t=t, **model_extra)


@torch.no_grad()
def euler_sample(velocity_model: VelocityFn, x_init: torch.Tensor, steps: int, **model_extra) -> torch.Tensor:
    """First-order explicit Euler integration from t=0 to t=1."""

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    x = x_init
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = step * dt
        x = x + dt * _velocity(velocity_model, x, t, model_extra)
    return x


@torch.no_grad()
def midpoint_sample(velocity_model: VelocityFn, x_init: torch.Tensor, steps: int, **model_extra) -> torch.Tensor:
    """Second-order midpoint integration from t=0 to t=1."""

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    x = x_init
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = step * dt
        k1 = _velocity(velocity_model, x, t, model_extra)
        k2 = _velocity(velocity_model, x + 0.5 * dt * k1, t + 0.5 * dt, model_extra)
        x = x + dt * k2
    return x


@torch.no_grad()
def ab2_sample(velocity_model: VelocityFn, x_init: torch.Tensor, steps: int, **model_extra) -> torch.Tensor:
    """DPM-style second-order multistep integration using Adams-Bashforth.

    This has one model evaluation per step after the Euler bootstrap, so it is
    cheaper than midpoint at the same step count.  It is a Flow-Matching
    multistep acceleration path, not the diffusion-specific DPM-Solver++ formula.
    """

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if steps == 1:
        return euler_sample(velocity_model, x_init, steps=1, **model_extra)

    x = x_init
    dt = 1.0 / float(steps)

    v_prev = _velocity(velocity_model, x, 0.0, model_extra)
    x = x + dt * v_prev

    for step in range(1, steps):
        t = step * dt
        v_cur = _velocity(velocity_model, x, t, model_extra)
        x = x + dt * (1.5 * v_cur - 0.5 * v_prev)
        v_prev = v_cur

    return x


@torch.no_grad()
def ab3_sample(velocity_model: VelocityFn, x_init: torch.Tensor, steps: int, **model_extra) -> torch.Tensor:
    """DEIS-like third-order multistep integration on a uniform time grid.

    DEIS-style acceleration extrapolates previous model outputs.  On this
    uniform Flow Matching grid, the implementation is the third-order
    Adams-Bashforth update after two bootstrap steps.
    """

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if steps <= 2:
        return ab2_sample(velocity_model, x_init, steps=steps, **model_extra)

    x = x_init
    dt = 1.0 / float(steps)

    v_nm2 = _velocity(velocity_model, x, 0.0, model_extra)
    x = x + dt * v_nm2

    v_nm1 = _velocity(velocity_model, x, dt, model_extra)
    x = x + dt * (1.5 * v_nm1 - 0.5 * v_nm2)

    for step in range(2, steps):
        t = step * dt
        v_n = _velocity(velocity_model, x, t, model_extra)
        x = x + (dt / 12.0) * (23.0 * v_n - 16.0 * v_nm1 + 5.0 * v_nm2)
        v_nm2, v_nm1 = v_nm1, v_n

    return x


def sample_flow_ode(
    velocity_model: VelocityFn,
    x_init: torch.Tensor,
    steps: int,
    solver: str = "midpoint",
    **model_extra,
) -> torch.Tensor:
    """Dispatch fixed-step Flow Matching sampling."""

    solver = solver.lower()
    if solver == "euler":
        return euler_sample(velocity_model, x_init, steps, **model_extra)
    if solver in {"midpoint", "rk2"}:
        return midpoint_sample(velocity_model, x_init, steps, **model_extra)
    if solver in {"dpm", "ab2", "multistep"}:
        return ab2_sample(velocity_model, x_init, steps, **model_extra)
    if solver in {"deis", "deis3", "ab3"}:
        return ab3_sample(velocity_model, x_init, steps, **model_extra)
    raise ValueError(f"Unsupported Flow ODE solver: {solver}")
