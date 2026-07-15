#!/usr/bin/env python3
"""Visualize Flow-Planner fixed-step generation trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str


METHODS = [
    MethodSpec("torchdiffeq", "TorchDiffEq ref", "#2563eb"),
    MethodSpec("euler", "Euler", "#ea580c"),
    MethodSpec("midpoint", "Midpoint", "#9333ea"),
    MethodSpec("ab2", "AB2", "#16a34a"),
    MethodSpec("deis", "DEIS", "#0891b2"),
    MethodSpec("flow_dpm", "Flow-DPM", "#dc2626"),
]


def load_eval_module(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = repo_root / "scripts" / "eval_ade_fde_flowplanner.py"
    spec = importlib.util.spec_from_file_location("eval_ade_fde_flowplanner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load eval module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def velocity_at(velocity_model: Callable, x: torch.Tensor, t_value: float, model_extra: dict) -> torch.Tensor:
    t = torch.tensor(min(float(t_value), 1.0 - 1e-5), device=x.device, dtype=x.dtype)
    return velocity_model(x=x, t=t, **model_extra)


@torch.no_grad()
def euler_states(velocity_model: Callable, x_init: torch.Tensor, steps: int, model_extra: dict) -> list[torch.Tensor]:
    x = x_init.clone()
    states = [x.detach().clone()]
    dt = 1.0 / float(steps)
    for step in range(steps):
        x = x + dt * velocity_at(velocity_model, x, step * dt, model_extra)
        states.append(x.detach().clone())
    return states


@torch.no_grad()
def midpoint_states(velocity_model: Callable, x_init: torch.Tensor, steps: int, model_extra: dict) -> list[torch.Tensor]:
    x = x_init.clone()
    states = [x.detach().clone()]
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = step * dt
        k1 = velocity_at(velocity_model, x, t, model_extra)
        k2 = velocity_at(velocity_model, x + 0.5 * dt * k1, t + 0.5 * dt, model_extra)
        x = x + dt * k2
        states.append(x.detach().clone())
    return states


@torch.no_grad()
def ab2_states(velocity_model: Callable, x_init: torch.Tensor, steps: int, model_extra: dict) -> list[torch.Tensor]:
    if steps == 1:
        return euler_states(velocity_model, x_init, steps, model_extra)
    x = x_init.clone()
    states = [x.detach().clone()]
    dt = 1.0 / float(steps)
    v_prev = velocity_at(velocity_model, x, 0.0, model_extra)
    x = x + dt * v_prev
    states.append(x.detach().clone())
    for step in range(1, steps):
        v_cur = velocity_at(velocity_model, x, step * dt, model_extra)
        x = x + dt * (1.5 * v_cur - 0.5 * v_prev)
        v_prev = v_cur
        states.append(x.detach().clone())
    return states


@torch.no_grad()
def deis_states(velocity_model: Callable, x_init: torch.Tensor, steps: int, model_extra: dict) -> list[torch.Tensor]:
    if steps <= 2:
        return ab2_states(velocity_model, x_init, steps, model_extra)
    x = x_init.clone()
    states = [x.detach().clone()]
    dt = 1.0 / float(steps)
    v_nm2 = velocity_at(velocity_model, x, 0.0, model_extra)
    x = x + dt * v_nm2
    states.append(x.detach().clone())
    v_nm1 = velocity_at(velocity_model, x, dt, model_extra)
    x = x + dt * (1.5 * v_nm1 - 0.5 * v_nm2)
    states.append(x.detach().clone())
    for step in range(2, steps):
        v_n = velocity_at(velocity_model, x, step * dt, model_extra)
        x = x + (dt / 12.0) * (23.0 * v_n - 16.0 * v_nm1 + 5.0 * v_nm2)
        v_nm2, v_nm1 = v_nm1, v_n
        states.append(x.detach().clone())
    return states


@torch.no_grad()
def flow_dpm_states(
    model_fn: Callable,
    x_init: torch.Tensor,
    steps: int,
    use_cfg: bool,
    cfg_weight: float,
    model_extra: dict,
) -> list[torch.Tensor]:
    from flow_planner.model.flow_planner_model.flow_utils.flow_solver import (
        _flow_dpm_schedule,
        _x_start_prediction,
    )

    sigma, alpha, lamb, model_t = _flow_dpm_schedule(
        steps=steps,
        device=x_init.device,
        dtype=x_init.dtype,
        shift=1.0,
    )
    x = x_init.clone()
    states = [x.detach().clone()]
    x0_prev = _x_start_prediction(model_fn, x, float(model_t[0].item()), use_cfg, cfg_weight, model_extra)
    h = lamb[1] - lamb[0]
    x = (sigma[1] / sigma[0]) * x - alpha[1] * (torch.exp(-h) - 1.0) * x0_prev
    states.append(x.detach().clone())
    if steps == 1:
        return states

    x0_cur = _x_start_prediction(model_fn, x, float(model_t[1].item()), use_cfg, cfg_weight, model_extra)
    for idx in range(2, steps + 1):
        h_i = lamb[idx] - lamb[idx - 1]
        h_prev = lamb[idx - 1] - lamb[idx - 2]
        r_i = h_prev / h_i
        d_i = (1.0 + 0.5 / r_i) * x0_cur - (0.5 / r_i) * x0_prev
        x = (sigma[idx] / sigma[idx - 1]) * x - alpha[idx] * (torch.exp(-h_i) - 1.0) * d_i
        states.append(x.detach().clone())
        if idx < steps:
            x0_prev = x0_cur
            x0_cur = _x_start_prediction(model_fn, x, float(model_t[idx].item()), use_cfg, cfg_weight, model_extra)
    return states


def collect_flow_states(model, batch, steps: int, seed: int, use_cfg: bool, cfg_weight: float):
    from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel

    batch_size = batch.ego_current.shape[0]
    if use_cfg:
        cfg_flags = torch.cat(
            [
                torch.ones((batch_size,), device=model.device),
                torch.zeros((batch_size,), device=model.device),
            ],
            dim=0,
        ).to(torch.int32)
    else:
        cfg_flags = torch.ones((batch_size,), device=model.device).to(torch.int32)

    model_inputs, _ = model.prepare_model_input(cfg_flags, batch, use_cfg, is_training=False)
    encoder_inputs = model.extract_encoder_inputs(model_inputs)
    encoder_outputs = model.encoder(**encoder_inputs)
    decoder_model_extra = model.extract_decoder_inputs(encoder_outputs, model_inputs)

    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    x_init = torch.randn(
        (
            batch_size,
            model.action_num,
            model.planner_params["action_len"],
            model.planner_params["state_dim"],
        ),
        device=model.device,
        generator=generator,
    )

    velocity_func = model.flow_ode.translation_funcs[(model.model_type, "velocity")]
    velocity_model = VelocityModel(
        model.decoder,
        model.flow_ode.path,
        velocity_func,
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
    )

    all_states = {
        "torchdiffeq": midpoint_states(velocity_model, x_init, steps, decoder_model_extra),
        "euler": euler_states(velocity_model, x_init, steps, decoder_model_extra),
        "midpoint": midpoint_states(velocity_model, x_init, steps, decoder_model_extra),
        "ab2": ab2_states(velocity_model, x_init, steps, decoder_model_extra),
        "deis": deis_states(velocity_model, x_init, steps, decoder_model_extra),
        "flow_dpm": flow_dpm_states(
            model.decoder,
            x_init,
            steps,
            use_cfg,
            cfg_weight,
            dict(decoder_model_extra),
        ),
    }
    return all_states


def token_state_to_xy(model, state: torch.Tensor) -> np.ndarray:
    from flow_planner.model.model_utils.traj_tool import assemble_actions

    traj = assemble_actions(
        state,
        model.planner_params["future_len"],
        model.planner_params["action_len"],
        model.planner_params["action_overlap"],
        model.planner_params["state_dim"],
        model.assemble_method,
    )
    traj = model.data_processor.state_postprocess(traj)
    return traj[0, 0, :, :2].detach().cpu().numpy()


def draw_grid(model, all_states, gt_xy: np.ndarray, output: Path, sample_index: int, steps: int) -> None:
    trajectories = {}
    all_xy = [gt_xy]
    for spec in METHODS:
        method_trajs = [token_state_to_xy(model, state) for state in all_states[spec.key]]
        trajectories[spec.key] = method_trajs
        all_xy.extend(method_trajs)

    stacked = np.concatenate(all_xy, axis=0)
    finite = np.isfinite(stacked).all(axis=1)
    if finite.any():
        x_min, y_min = stacked[finite].min(axis=0)
        x_max, y_max = stacked[finite].max(axis=0)
    else:
        x_min, y_min, x_max, y_max = -10.0, -10.0, 10.0, 10.0
    margin = max(5.0, 0.08 * max(x_max - x_min, y_max - y_min, 1.0))
    x_lim = (x_min - margin, x_max + margin)
    y_lim = (y_min - margin, y_max + margin)

    fig, axes = plt.subplots(
        len(METHODS),
        steps + 1,
        figsize=(2.2 * (steps + 1), 2.35 * len(METHODS)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for col, title in enumerate(["init"] + [f"step {idx:02d}" for idx in range(1, steps + 1)]):
        axes[0, col].set_title(title, fontsize=10)

    for row, spec in enumerate(METHODS):
        for col in range(steps + 1):
            ax = axes[row, col]
            pred = trajectories[spec.key][col]
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=1.2, linestyle="--", alpha=0.75)
            ax.plot(pred[:, 0], pred[:, 1], color=spec.color, linewidth=1.4)
            ax.scatter(gt_xy[0, 0], gt_xy[0, 1], color="black", s=10, marker="o")
            ax.scatter(pred[-1, 0], pred[-1, 1], color=spec.color, s=14, marker="x")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid(True, color="#e5e7eb", linewidth=0.5)
            ax.tick_params(labelsize=6, length=2)
            if col == 0:
                ax.set_ylabel(spec.label, fontsize=10)

    fig.suptitle(
        f"Flow-Planner {steps}-step generation, sample {sample_index} "
        "(black dashed = GT ego future)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default="/home/liangfh/project/nuplan-devkit/Flow-Planner")
    parser.add_argument("--config_file", default="checkpoints/flow-planner-hf/model_config.yaml")
    parser.add_argument("--ckpt_file", default="checkpoints/flow-planner-hf/model.pth")
    parser.add_argument("--data_dir", default="/mnt/d/nuplan-v1.1_val/data/processed_1w")
    parser.add_argument("--data_list", default="diffusion_planner_training_1w.json")
    parser.add_argument("--sample_index", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--no_use_cfg", dest="use_cfg", action="store_false")
    parser.add_argument("--cfg_weight", type=float, default=None)
    parser.add_argument("--output", default="results/visualizations/flowplanner_solvers_10step_sample3.png")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    eval_mod = load_eval_module(repo_root)
    seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg, model = eval_mod.load_flowplanner(
        str(repo_root / args.config_file),
        str(repo_root / args.ckpt_file),
        device,
        args.steps,
        "torchdiffeq",
    )
    if args.cfg_weight is None:
        args.cfg_weight = float(cfg.model.cfg_weight)

    dataset_args = argparse.Namespace(
        data_dir=args.data_dir,
        data_list=str(repo_root / args.data_list)
        if not Path(args.data_list).is_absolute()
        else args.data_list,
        diffusionplanner_1w_adapter=True,
        num=-1,
    )
    dataset, total = eval_mod.build_dataset(dataset_args, cfg)
    if args.sample_index < 0 or args.sample_index >= total:
        raise IndexError(f"sample_index {args.sample_index} out of range for dataset size {total}")

    loader = DataLoader(
        Subset(dataset, [args.sample_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=eval_mod.collect_batch,
    )
    batch = next(iter(loader)).to(device)
    with torch.inference_mode():
        all_states = collect_flow_states(model, batch, args.steps, args.seed, args.use_cfg, args.cfg_weight)

    gt_xy = batch.ego_future[0, :, :2].detach().cpu().numpy()
    output = Path(args.output)
    output = repo_root / output if not output.is_absolute() else output
    draw_grid(model, all_states, gt_xy, output, args.sample_index, args.steps)
    print(f"[viz] wrote {output}")


if __name__ == "__main__":
    main()
