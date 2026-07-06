#!/usr/bin/env python3
"""
Open-loop ADE/FDE evaluation for Flow-Planner checkpoints.

This mirrors the Diffusion-Planner scripts/eval_ade_fde.py workflow, but uses
Flow-Planner's own NuPlanDataset/model stack and a temporary adapter for the
existing Diffusion-Planner processed_1w data format:

  Diffusion-Planner 1w:
    ego_past    [21, 7]   = x, y, heading, vx, vy, ax, ay
    ego_current [10]

  Flow-Planner official config expects:
    ego_past    [21, 14]  = x, y, cos, sin, vx, vy, ax, ay, ...
    ego_current [16]

The adapter pads missing ego features with zeros. This is intended for a
compatibility smoke/open-loop check, not as a claim that the representation is
identical to Flow-Planner's official preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_planner.data.dataset.nuplan import NuPlanDataset  # noqa: E402
from flow_planner.data.utils.collect import collect_batch  # noqa: E402


class DiffusionPlanner1wAdapter(Dataset):
    """Pad Diffusion-Planner ego features to Flow-Planner expected dimensions."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        ego_past = sample.ego_past
        if ego_past.shape[-1] == 7:
            padded = torch.zeros(ego_past.shape[:-1] + (14,), dtype=ego_past.dtype)
            padded[..., 0:2] = ego_past[..., 0:2]
            padded[..., 2] = torch.cos(ego_past[..., 2])
            padded[..., 3] = torch.sin(ego_past[..., 2])
            padded[..., 4:8] = ego_past[..., 3:7]
            sample.ego_past = padded
        elif ego_past.shape[-1] != 14:
            raise ValueError(f"Unsupported ego_past dim: {ego_past.shape}")

        ego_current = sample.ego_current
        if ego_current.shape[-1] == 10:
            padded = torch.zeros(ego_current.shape[:-1] + (16,), dtype=ego_current.dtype)
            padded[..., 0:10] = ego_current
            sample.ego_current = padded
        elif ego_current.shape[-1] != 16:
            raise ValueError(f"Unsupported ego_current dim: {ego_current.shape}")

        return sample


def ensure_runtime_config(cfg: Any, device: str) -> Any:
    """Fill runtime-only OmegaConf keys that are absent in HF model_config.yaml."""

    cfg.device = device
    if "data" not in cfg:
        cfg.data = {
            "dataset": {
                "train": {
                    "future_downsampling_method": "uniform",
                    "predicted_neighbor_num": cfg.model.neighbor_pred_num,
                }
            }
        }

    cfg.model.device = device
    cfg.model.model_decoder.device = device
    cfg.model.flow_ode.time_sampler.device = device
    if "core" in cfg:
        cfg.core.device = device
    return cfg


def load_flowplanner(config_file: str, ckpt_file: str, device: str, sample_steps: int | None):
    cfg = OmegaConf.load(config_file)
    cfg = ensure_runtime_config(cfg, device)
    if sample_steps is not None:
        cfg.model.flow_ode.sample_steps = int(sample_steps)

    model = instantiate(cfg.model).to(device)
    state = torch.load(ckpt_file, map_location=device, weights_only=True)
    if isinstance(state, dict) and "ema_state_dict" in state:
        state = state["ema_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, model


def build_dataset(args: argparse.Namespace, cfg: Any):
    base = NuPlanDataset(
        args.data_dir,
        args.data_list,
        past_neighbor_num=cfg.model.neighbor_num,
        predicted_neighbor_num=cfg.model.neighbor_pred_num,
        future_len=cfg.model.future_len,
        future_downsampling_method=cfg.data.dataset.train.future_downsampling_method,
    )
    total = len(base)
    dataset: Dataset = DiffusionPlanner1wAdapter(base) if args.diffusionplanner_1w_adapter else base

    if args.num > 0 and args.num < total:
        indices = random.sample(range(total), args.num)
        dataset = Subset(dataset, indices)

    return dataset, total


def percentile(values: np.ndarray, pct: float) -> float:
    return float(np.percentile(values, pct))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="checkpoints/flow-planner-hf/model_config.yaml")
    parser.add_argument("--ckpt_file", default="checkpoints/flow-planner-hf/model.pth")
    parser.add_argument("--data_dir", default="/mnt/d/nuplan-v1.1_val/data/processed_1w")
    parser.add_argument("--data_list", default="diffusion_planner_training_1w.json")
    parser.add_argument("--num", type=int, default=200, help="-1 means full dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--cfg_weight", type=float, default=None)
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--diffusionplanner_1w_adapter", action="store_true", default=True)
    parser.add_argument("--no_diffusionplanner_1w_adapter", dest="diffusionplanner_1w_adapter", action="store_false")
    parser.add_argument("--use_cfg", dest="use_cfg", action="store_true", default=True)
    parser.add_argument("--no_use_cfg", dest="use_cfg", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg, model = load_flowplanner(args.config_file, args.ckpt_file, device, args.sample_steps)
    if args.cfg_weight is None:
        args.cfg_weight = float(cfg.model.cfg_weight)

    dataset, total = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
        collate_fn=collect_batch,
    )

    print(f"[eval] config = {args.config_file}")
    print(f"[eval] ckpt = {args.ckpt_file}")
    print(f"[eval] device = {device}")
    if device == "cuda":
        print(f"[eval] gpu = {torch.cuda.get_device_name(0)}")
    print(f"[eval] torch = {torch.__version__}")
    print(f"[eval] use_cfg = {args.use_cfg}  cfg_weight = {args.cfg_weight}")
    print(f"[eval] sample_steps = {cfg.model.flow_ode.sample_steps}")
    print(f"[eval] adapter = {args.diffusionplanner_1w_adapter}")
    print(f"[eval] dataset size = {len(dataset)} / {total}")
    print(f"[eval] batch_size = {args.batch_size}")

    ade_list = []
    fde_list = []
    latencies_ms = []
    total_samples = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            batch = batch.to(device)
            gt_xy = batch.ego_future[:, :, :2].to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            pred = model(batch, mode="inference", use_cfg=args.use_cfg, cfg_weight=args.cfg_weight)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            pred_xy = pred[:, 0, :, :2]
            if pred_xy.shape[1] != gt_xy.shape[1]:
                raise RuntimeError(f"Prediction/GT length mismatch: {pred_xy.shape} vs {gt_xy.shape}")

            dist = torch.norm(pred_xy - gt_xy, dim=-1)
            ade = dist.mean(dim=-1)
            fde = dist[:, -1]

            ade_list.append(ade.detach().cpu())
            fde_list.append(fde.detach().cpu())

            if batch_idx >= args.warmup_batches:
                latencies_ms.append((end - start) * 1000.0)
                total_samples += int(pred.shape[0])

    ade_all = torch.cat(ade_list)
    fde_all = torch.cat(fde_list)

    metrics = {
        "num_samples": int(ade_all.numel()),
        "ade_mean": float(ade_all.mean().item()),
        "ade_std": float(ade_all.std().item()) if ade_all.numel() > 1 else 0.0,
        "ade_median": float(ade_all.median().item()),
        "fde_mean": float(fde_all.mean().item()),
        "fde_std": float(fde_all.std().item()) if fde_all.numel() > 1 else 0.0,
        "fde_median": float(fde_all.median().item()),
    }

    print("\n" + "=" * 56)
    print(f"  samples      : {metrics['num_samples']}")
    print(f"  ADE mean/std : {metrics['ade_mean']:.4f} +/- {metrics['ade_std']:.4f} m")
    print(f"  FDE mean/std : {metrics['fde_mean']:.4f} +/- {metrics['fde_std']:.4f} m")
    print(f"  ADE median   : {metrics['ade_median']:.4f} m")
    print(f"  FDE median   : {metrics['fde_median']:.4f} m")

    if latencies_ms:
        lat = np.asarray(latencies_ms, dtype=np.float64)
        per_sample = lat / float(args.batch_size)
        latency_metrics = {
            "warmup_batches": int(args.warmup_batches),
            "steady_batches": int(len(latencies_ms)),
            "steady_samples": int(total_samples),
            "batch_latency_mean_ms": float(lat.mean()),
            "batch_latency_median_ms": float(np.median(lat)),
            "batch_latency_p95_ms": percentile(lat, 95),
            "batch_latency_p99_ms": percentile(lat, 99),
            "batch_latency_max_ms": float(lat.max()),
            "per_sample_latency_mean_ms": float(per_sample.mean()),
            "per_sample_latency_median_ms": float(np.median(per_sample)),
        }
        metrics.update(latency_metrics)

        total_time_s = float(lat.sum() / 1000.0)
        print("-" * 56)
        print(f"  latency warmup       : {args.warmup_batches} batches")
        print(f"  latency steady       : {len(latencies_ms)} batches / {total_samples} samples")
        print(f"  batch latency mean   : {lat.mean():.2f} ms")
        print(f"  batch latency median : {np.median(lat):.2f} ms")
        print(f"  batch latency p95    : {np.percentile(lat, 95):.2f} ms")
        print(f"  batch latency p99    : {np.percentile(lat, 99):.2f} ms")
        print(f"  batch latency max    : {lat.max():.2f} ms")
        print(f"  per-sample mean      : {per_sample.mean():.2f} ms")
        print(f"  per-sample median    : {np.median(per_sample):.2f} ms")
        if total_time_s > 0:
            throughput = total_samples / total_time_s
            metrics["throughput_samples_per_s"] = float(throughput)
            print(f"  throughput           : {throughput:.1f} samples/s")
    else:
        print("-" * 56)
        print("  latency: no steady batches collected")
    print("=" * 56)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"[eval] wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
