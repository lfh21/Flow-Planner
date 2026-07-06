"""Lightweight fixed-step solvers for Flow-Planner inference.

The official Flow-Planner path solves a Flow Matching ODE over action tokens.
Diffusion DPM-Solver++ is not directly applicable because it assumes a VP
noise schedule and logSNR parameterization.  This module keeps the same Flow
Matching velocity field and exposes deployment-oriented fixed-step solvers:
Euler, midpoint, AB2, and a DEIS-like AB3 multistep update.

It also includes a Sana-style Flow-DPM-Solver path for x_start/data-prediction
models.  That path follows DPM-Solver++'s data-prediction multistep update with
the rectified-flow schedule alpha = 1 - sigma, rather than black-box velocity
integration.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch


VelocityFn = Callable[..., torch.Tensor]
MAX_EVAL_TIME = 1.0 - 1e-5
MIN_SIGMA = 1e-4


def _as_solver_time(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Match flow_matching.ODESolver: pass scalar ODE time into VelocityModel."""

    value = min(float(value), MAX_EVAL_TIME)
    return torch.tensor(value, device=device, dtype=dtype)


def _velocity(velocity_model: VelocityFn, x: torch.Tensor, t_value: float, model_extra: Dict) -> torch.Tensor:
    t = _as_solver_time(t_value, x.device, x.dtype)
    return velocity_model(x=x, t=t, **model_extra)


def _model_time(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _as_solver_time(value, device, dtype).unsqueeze(0)


def _x_start_prediction(model_fn: Callable[..., torch.Tensor], x: torch.Tensor, t_value: float, use_cfg: bool, cfg_weight: float, model_extra: Dict) -> torch.Tensor:
    """Run Flow-Planner decoder as a data/x_start prediction model."""

    t = _model_time(t_value, x.device, x.dtype)
    model_x = x.repeat(2, *[1] * (x.dim() - 1)) if use_cfg else x
    pred = model_fn(model_x, t, **model_extra)
    if use_cfg:
        pred_cond, pred_uncond = torch.chunk(pred, 2)
        pred = (1 - cfg_weight) * pred_uncond + cfg_weight * pred_cond
    return pred


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _flow_dpm_schedule(steps: int, device: torch.device, dtype: torch.dtype, shift: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build descending rectified-flow DPM-Solver++ schedule.

    sigma decreases from nearly 1 to 0.  We keep a tiny terminal sigma for the
    internal update to avoid division by zero, then force the final alpha to 1.
    """

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    sigma = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    sigma = _shift_sigma(sigma, shift)
    sigma_internal = sigma.clamp(min=MIN_SIGMA, max=1.0 - MIN_SIGMA)
    alpha = 1.0 - sigma_internal
    lamb = torch.log(alpha) - torch.log(sigma_internal)
    model_t = (1.0 - sigma_internal).clamp(min=0.0, max=MAX_EVAL_TIME)
    return sigma_internal, alpha, lamb, model_t


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


@torch.no_grad()
def flow_dpm_solver_sample(
    model_fn: Callable[..., torch.Tensor],
    x_init: torch.Tensor,
    steps: int,
    use_cfg: bool,
    cfg_weight: float,
    model_pred_type: str = "x_start",
    timestep_shift: float = 1.0,
    **model_extra,
) -> torch.Tensor:
    """Sana-style Flow-DPM-Solver for rectified-flow data prediction.

    This implements the data-prediction DPM-Solver++ multistep update under the
    rectified-flow schedule alpha = 1 - sigma.  Flow-Planner's official
    checkpoint predicts x_start, so ``model_fn`` is used directly as x_theta.
    """

    if model_pred_type != "x_start":
        raise ValueError(
            "flow_dpm_solver currently requires model_pred_type='x_start'. "
            f"Got {model_pred_type!r}."
        )

    sigma, alpha, lamb, model_t = _flow_dpm_schedule(
        steps=steps,
        device=x_init.device,
        dtype=x_init.dtype,
        shift=float(timestep_shift),
    )

    x = x_init
    x0_prev = _x_start_prediction(
        model_fn=model_fn,
        x=x,
        t_value=float(model_t[0].item()),
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
        model_extra=model_extra,
    )

    h = lamb[1] - lamb[0]
    x = (sigma[1] / sigma[0]) * x - alpha[1] * (torch.exp(-h) - 1.0) * x0_prev
    if steps == 1:
        return x0_prev

    x0_cur = _x_start_prediction(
        model_fn=model_fn,
        x=x,
        t_value=float(model_t[1].item()),
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
        model_extra=model_extra,
    )

    for idx in range(2, steps + 1):
        h_i = lamb[idx] - lamb[idx - 1]
        h_prev = lamb[idx - 1] - lamb[idx - 2]
        r_i = h_prev / h_i
        d_i = (1.0 + 0.5 / r_i) * x0_cur - (0.5 / r_i) * x0_prev
        x = (sigma[idx] / sigma[idx - 1]) * x - alpha[idx] * (torch.exp(-h_i) - 1.0) * d_i

        if idx < steps:
            x0_prev = x0_cur
            x0_cur = _x_start_prediction(
                model_fn=model_fn,
                x=x,
                t_value=float(model_t[idx].item()),
                use_cfg=use_cfg,
                cfg_weight=cfg_weight,
                model_extra=model_extra,
            )

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
