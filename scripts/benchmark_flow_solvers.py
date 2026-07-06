#!/usr/bin/env python3
"""Benchmark Flow-Planner ODE solvers against a reference sampler."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


DEFAULT_SOLVERS = ["euler", "midpoint", "ab2", "deis"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_module(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    module_path = repo_root / "scripts" / "eval_ade_fde_flowplanner.py"
    spec = importlib.util.spec_from_file_location("eval_ade_fde_flowplanner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load eval module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_solver(model, solver: str) -> None:
    model.flow_ode.sample_params["sample_solver"] = solver


def timed_forward(model, batch, solver: str, seed: int, use_cfg: bool, cfg_weight: float, device: str):
    set_solver(model, solver)
    seed_all(seed)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    pred = model(batch, mode="inference", use_cfg=use_cfg, cfg_weight=cfg_weight)
    if device == "cuda":
        torch.cuda.synchronize()
    return pred, (time.perf_counter() - start) * 1000.0


def summarize_latency(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
    }


def empty_accumulator() -> dict:
    return {
        "latencies_ms": [],
        "abs_diff_sum": 0.0,
        "sq_diff_sum": 0.0,
        "diff_count": 0,
        "max_abs_diff": 0.0,
        "allclose_1e_2": True,
        "allclose_1e_1": True,
        "ade": [],
        "fde": [],
    }


def update_quality(acc: dict, pred: torch.Tensor, gt_xy: torch.Tensor) -> None:
    pred_xy = pred[:, 0, :, :2]
    dist = torch.norm(pred_xy - gt_xy, dim=-1)
    acc["ade"].append(dist.mean(dim=-1).detach().cpu())
    acc["fde"].append(dist[:, -1].detach().cpu())


def update_diff(acc: dict, pred: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (pred - ref).detach()
    abs_diff = diff.abs()
    acc["abs_diff_sum"] += float(abs_diff.sum().item())
    acc["sq_diff_sum"] += float((diff * diff).sum().item())
    acc["diff_count"] += int(diff.numel())
    acc["max_abs_diff"] = max(acc["max_abs_diff"], float(abs_diff.max().item()))
    acc["allclose_1e_2"] = acc["allclose_1e_2"] and bool(torch.allclose(pred, ref, atol=1e-2, rtol=1e-2))
    acc["allclose_1e_1"] = acc["allclose_1e_1"] and bool(torch.allclose(pred, ref, atol=1e-1, rtol=1e-1))


def finalize_solver(acc: dict, ref_acc: dict | None = None) -> dict:
    ade = torch.cat(acc["ade"]) if acc["ade"] else torch.empty(0)
    fde = torch.cat(acc["fde"]) if acc["fde"] else torch.empty(0)
    result = {
        "latency": summarize_latency(acc["latencies_ms"]),
        "ade_mean": float(ade.mean().item()) if ade.numel() else 0.0,
        "fde_mean": float(fde.mean().item()) if fde.numel() else 0.0,
        "ade_median": float(ade.median().item()) if ade.numel() else 0.0,
        "fde_median": float(fde.median().item()) if fde.numel() else 0.0,
    }
    if ref_acc is None:
        result["output_diff_vs_reference"] = {
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "rmse_diff": 0.0,
            "allclose_1e_2": True,
            "allclose_1e_1": True,
        }
        return result

    count = max(acc["diff_count"], 1)
    result["output_diff_vs_reference"] = {
        "max_abs_diff": float(acc["max_abs_diff"]),
        "mean_abs_diff": float(acc["abs_diff_sum"] / count),
        "rmse_diff": float((acc["sq_diff_sum"] / count) ** 0.5),
        "allclose_1e_2": bool(acc["allclose_1e_2"]),
        "allclose_1e_1": bool(acc["allclose_1e_1"]),
    }
    result["ade_delta_vs_reference"] = result["ade_mean"] - float(torch.cat(ref_acc["ade"]).mean().item())
    result["fde_delta_vs_reference"] = result["fde_mean"] - float(torch.cat(ref_acc["fde"]).mean().item())
    return result


def markdown_table(results: dict, reference_solver: str, solvers: list[str]) -> str:
    rows = [
        "| Solver | NFE | Mean latency ms | Median latency ms | P95 ms | Speedup vs ref | Mean abs diff | RMSE diff | Max abs diff | ADE mean | FDE mean | ADE delta | FDE delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ref_mean = results["solvers"][reference_solver]["latency"]["mean_ms"]
    ordered = [reference_solver] + solvers
    for solver in ordered:
        item = results["solvers"][solver]
        lat = item["latency"]
        diff = item["output_diff_vs_reference"]
        speedup = ref_mean / lat["mean_ms"] if lat["mean_ms"] else 0.0
        rows.append(
            "| {solver} | {nfe} | {mean:.2f} | {median:.2f} | {p95:.2f} | {speedup:.2f}x | "
            "{mad:.6f} | {rmse:.6f} | {maxdiff:.6f} | {ade:.4f} | {fde:.4f} | {aded:.4f} | {fded:.4f} |".format(
                solver=solver,
                nfe=item["nfe"],
                mean=lat["mean_ms"],
                median=lat["median_ms"],
                p95=lat["p95_ms"],
                speedup=speedup,
                mad=diff["mean_abs_diff"],
                rmse=diff["rmse_diff"],
                maxdiff=diff["max_abs_diff"],
                ade=item["ade_mean"],
                fde=item["fde_mean"],
                aded=item.get("ade_delta_vs_reference", 0.0),
                fded=item.get("fde_delta_vs_reference", 0.0),
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default="/home/liangfh/project/nuplan-devkit/Flow-Planner")
    parser.add_argument("--config_file", default="checkpoints/flow-planner-hf/model_config.yaml")
    parser.add_argument("--ckpt_file", default="checkpoints/flow-planner-hf/model.pth")
    parser.add_argument("--data_dir", default="/mnt/d/nuplan-v1.1_val/data/processed_1w")
    parser.add_argument("--data_list", default="diffusion_planner_training_1w.json")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference_solver", default="torchdiffeq")
    parser.add_argument("--solvers", nargs="+", default=DEFAULT_SOLVERS)
    parser.add_argument("--sample_steps", type=int, default=4)
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--no_use_cfg", dest="use_cfg", action="store_false")
    parser.add_argument("--cfg_weight", type=float, default=None)
    parser.add_argument("--output_json", default="results/benchmark_flow_solvers_bs1_num100.json")
    parser.add_argument("--output_md", default="results/benchmark_flow_solvers_bs1_num100.md")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    eval_mod = load_eval_module(repo_root)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seed_all(args.seed)
    cfg, model = eval_mod.load_flowplanner(
        str(repo_root / args.config_file),
        str(repo_root / args.ckpt_file),
        device,
        args.sample_steps,
        args.reference_solver,
    )
    if args.cfg_weight is None:
        args.cfg_weight = float(cfg.model.cfg_weight)

    dataset_args = argparse.Namespace(
        data_dir=args.data_dir,
        data_list=str(repo_root / args.data_list)
        if not Path(args.data_list).is_absolute()
        else args.data_list,
        diffusionplanner_1w_adapter=True,
        num=args.num,
    )
    seed_all(args.seed)
    dataset, total = eval_mod.build_dataset(dataset_args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=eval_mod.collect_batch,
    )

    solvers = [solver for solver in args.solvers if solver != args.reference_solver]
    all_solver_names = [args.reference_solver] + solvers
    accumulators = {solver: empty_accumulator() for solver in all_solver_names}

    nfe = {
        "torchdiffeq": 2 * args.sample_steps,
        "euler": args.sample_steps,
        "midpoint": 2 * args.sample_steps,
        "rk2": 2 * args.sample_steps,
        "ab2": args.sample_steps,
        "dpm": args.sample_steps,
        "deis": args.sample_steps,
        "deis3": args.sample_steps,
        "ab3": args.sample_steps,
    }

    print(f"[benchmark] device = {device}")
    if device == "cuda":
        print(f"[benchmark] gpu = {torch.cuda.get_device_name(0)}")
    print(f"[benchmark] torch = {torch.__version__}")
    print(f"[benchmark] dataset = {len(dataset)} / {total}, batch_size = 1")
    print(f"[benchmark] sample_steps = {args.sample_steps}")
    print(f"[benchmark] reference = {args.reference_solver}")
    print(f"[benchmark] solvers = {', '.join(solvers)}")
    print(f"[benchmark] warmup = {args.warmup}")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            gt_xy = batch.ego_future[:, :, :2].to(device)
            pair_seed = args.seed + batch_idx

            ref_pred, ref_ms = timed_forward(
                model, batch, args.reference_solver, pair_seed, args.use_cfg, args.cfg_weight, device
            )
            if batch_idx >= args.warmup:
                accumulators[args.reference_solver]["latencies_ms"].append(ref_ms)
            update_quality(accumulators[args.reference_solver], ref_pred, gt_xy)

            ordered_solvers = solvers if batch_idx % 2 == 0 else list(reversed(solvers))
            for solver in ordered_solvers:
                pred, elapsed_ms = timed_forward(
                    model, batch, solver, pair_seed, args.use_cfg, args.cfg_weight, device
                )
                if batch_idx >= args.warmup:
                    accumulators[solver]["latencies_ms"].append(elapsed_ms)
                update_quality(accumulators[solver], pred, gt_xy)
                update_diff(accumulators[solver], pred, ref_pred)

            if (batch_idx + 1) % 10 == 0:
                print(f"[benchmark] processed {batch_idx + 1}/{len(dataset)} samples")

    results = {
        "num_samples": int(len(dataset)),
        "batch_size": 1,
        "warmup": int(args.warmup),
        "sample_steps": int(args.sample_steps),
        "reference_solver": args.reference_solver,
        "solvers": {},
    }
    ref_acc = accumulators[args.reference_solver]
    for solver in all_solver_names:
        results["solvers"][solver] = finalize_solver(
            accumulators[solver],
            None if solver == args.reference_solver else ref_acc,
        )
        results["solvers"][solver]["nfe"] = int(nfe.get(solver, -1))

    table = markdown_table(results, args.reference_solver, solvers)
    print("\n" + table)

    if args.output_json:
        output_json = repo_root / args.output_json if not Path(args.output_json).is_absolute() else Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"[benchmark] wrote json to {output_json}")
    if args.output_md:
        output_md = repo_root / args.output_md if not Path(args.output_md).is_absolute() else Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table)
        print(f"[benchmark] wrote markdown to {output_md}")


if __name__ == "__main__":
    main()
