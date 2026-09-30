#!/usr/bin/env python
"""Warm up the tri-input conditioner without loading the diffusion model."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import PairedSatelliteDataset
from src.local_spectral_checker import CheckerConfig
from src.tri_input_conditioner import TriInputConditioner
from src.tri_input_data import (
    RawValueConversion,
    load_raw_band_stats,
    save_raw_band_stats,
    validate_sentinel2_l2a_band_names,
)
from src.tri_input_runtime import masked_preview_l1, tri_diagnostic_metrics
from src.utils import (
    atomic_torch_save,
    configure_logging,
    load_yaml_config,
    resolve_project_path,
    save_json,
    save_yaml,
    seed_everything,
    tensor_to_pil,
    worker_init_fn,
)


LOGGER = logging.getLogger("train_tri_input_warmup")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("A", "B", "a", "b"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init_conditioner", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device(value)


def _make_conditioner(
    config: dict[str, Any],
    stage: str,
    initial_path: Path | None,
    *,
    require_same_stage: bool = False,
) -> tuple[TriInputConditioner, Any, dict[str, Any]]:
    tri = config.get("tri_input")
    if not isinstance(tri, dict) or not bool(tri.get("enabled", False)):
        raise ValueError("Warmup requires tri_input.enabled=true")
    layout = validate_sentinel2_l2a_band_names(tri.get("band_names"))
    conversion = RawValueConversion.from_config(tri.get("raw_value_conversion"))
    if not tri.get("raw_stats_path"):
        raise ValueError("tri_input.raw_stats_path must point to train-only statistics")
    stats_path = resolve_project_path(tri["raw_stats_path"], PROJECT_ROOT)
    stats = load_raw_band_stats(
        stats_path,
        expected_band_names=layout.band_names,
        expected_conversion=conversion,
    )
    checker_mapping = dict(tri.get("checker") or {})
    checker_mapping["enabled"] = stage == "B"
    conditioner = TriInputConditioner(
        raw_mean=stats.mean,
        raw_std=stats.std,
        surface_band_indices=layout.surface_indices,
        components=int(tri.get("components", 8)),
        scale=int(config.get("scale", 4)),
        checker_config=CheckerConfig.from_mapping(checker_mapping),
    )
    if initial_path is not None:
        initial = TriInputConditioner.from_pretrained(initial_path)
        if tuple(initial.config["surface_band_indices"]) != layout.surface_indices:
            raise ValueError("Initial conditioner uses a different explicit surface-band mapping")
        if not torch.equal(initial.raw_mean.cpu(), conditioner.raw_mean.cpu()) or not torch.equal(
            initial.raw_std.cpu(), conditioner.raw_std.cpu()
        ):
            raise ValueError("Initial conditioner raw statistics differ from the configured train statistics")
        if require_same_stage and initial.artifact_metadata.get("training_stage") != f"warmup_{stage}":
            raise ValueError(
                f"--resume requires warmup_{stage}, found "
                f"{initial.artifact_metadata.get('training_stage')!r}; use --init_conditioner for a stage transition"
            )
        expected, actual = set(conditioner.state_dict()), set(initial.state_dict())
        if expected != actual:
            raise RuntimeError(
                "Initial conditioner structure differs from the requested warmup structure: "
                f"missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}"
            )
        conditioner.load_state_dict(initial.state_dict(), strict=True)
    metadata = {
        "tri_input_enabled": True,
        "training_stage": f"warmup_{stage}",
        "band_names": list(layout.band_names),
        "raw_value_conversion": conversion.as_dict(),
        "raw_stats_source": str(stats_path),
        "checker": checker_mapping,
        "scientific_scope": "preview reconstruction and internal heuristic checker diagnostics",
    }
    conditioner.artifact_metadata.update(metadata)
    return conditioner, stats, metadata


def _build_dataset(config: dict[str, Any], output_dir: Path) -> PairedSatelliteDataset:
    tri = config["tri_input"]
    train = (tri.get("data") or {}).get("train")
    if not isinstance(train, dict) or not train.get("raw_ms_dir") or not train.get("unmixing_dir"):
        raise ValueError("tri_input.data.train raw_ms_dir/unmixing_dir must be explicit")
    if float(config.get("synthetic_replay_probability", 0.0)) != 0.0:
        raise ValueError(
            "Tri-input warmup uses the configured train_lr_subdir (including LR_bicubic) "
            "without random replay; set synthetic_replay_probability=0"
        )
    return PairedSatelliteDataset(
        data_root=config["data_root"],
        split="train",
        lr_subdir=config["train_lr_subdir"],
        gt_subdir=config.get("gt_subdir", "GT"),
        gt_crop_size=int(config.get("gt_crop_size", 512)),
        scale=int(config.get("scale", 4)),
        training=True,
        strict_pairs=bool(config.get("strict_pairs", False)),
        invalid_log_path=output_dir / "invalid_train_pairs.csv",
        prompt_mode=str(config.get("prompt_mode", "fixed")),
        metadata_path=config.get("metadata_path"),
        prompt_dropout_probability=0.0,
        augment=bool(config.get("augment", True)),
        tri_input_enabled=True,
        raw_ms_dir=train["raw_ms_dir"],
        unmixing_dir=train["unmixing_dir"],
        raw_band_names=tri.get("band_names"),
        raw_value_conversion=tri.get("raw_value_conversion"),
        aux_manifest_path=train.get("manifest_path"),
        aux_recursive=bool(train.get("recursive", False)),
        synthetic_aux_policy="error",
    )


def _save(
    output: Path,
    conditioner: TriInputConditioner,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    stats: Any,
    metadata: dict[str, Any],
    data_generator: torch.Generator,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    conditioner.save_pretrained(output, artifact_metadata=metadata)
    save_raw_band_stats(stats, output / "raw_band_stats.json")
    atomic_torch_save(optimizer.state_dict(), output / "optimizer.pt")
    numpy_state = np.random.get_state()
    trainer_state: dict[str, Any] = {
        "global_step": step,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "data_generator_state": data_generator.get_state(),
    }
    if torch.cuda.is_available():
        trainer_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    atomic_torch_save(trainer_state, output / "trainer_state.pt")
    save_yaml(config, output / "training_config.yaml")


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.init_conditioner is not None and args.resume is not None:
        raise ValueError("Use either --init_conditioner (weight-only stage transition) or --resume")
    if args.max_steps <= 0 or args.learning_rate <= 0 or args.checkpointing_steps <= 0:
        raise ValueError("max_steps, learning_rate, and checkpointing_steps must be positive")
    config = load_yaml_config(args.config)
    stage = args.stage.upper()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    initial_path = args.resume or args.init_conditioner
    initial_path = initial_path.expanduser().resolve() if initial_path is not None else None
    conditioner, stats, metadata = _make_conditioner(
        config, stage, initial_path, require_same_stage=args.resume is not None
    )
    dataset = _build_dataset(config, output_dir)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=(args.num_workers if args.num_workers is not None else int(config.get("num_workers", 0))),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    device = _device(args.device)
    conditioner.to(device).train()
    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=args.learning_rate)
    global_step = 0
    if args.resume is not None:
        optimizer_path = initial_path / "optimizer.pt"
        state_path = initial_path / "trainer_state.pt"
        if not optimizer_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(
                f"Same-stage resume requires {optimizer_path} and {state_path}"
            )
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        global_step = int(state["global_step"])
        torch.set_rng_state(state["torch_rng_state"])
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state:
            numpy_state = state["numpy_rng_state"]
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if "data_generator_state" in state:
            generator.set_state(state["data_generator_state"])

    progress = tqdm(total=args.max_steps, initial=min(global_step, args.max_steps), desc=f"warmup-{stage}")
    last_batch: dict[str, Any] | None = None
    diagnostic_names = (
        "checker_valid_windows",
        "checker_accepted_fraction",
        "checker_mean_shift_hr",
        "checker_internal_gain",
    )
    diagnostic_sums = {name: 0.0 for name in diagnostic_names}
    diagnostic_steps = 0
    fallback_reasons: Counter[str] = Counter()
    while global_step < args.max_steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            rgb = batch["lr"].to(device, dtype=torch.float32)
            gt = batch["gt"].to(device, dtype=torch.float32)
            output = conditioner(
                rgb,
                batch["raw_ms"].to(device, dtype=torch.float32),
                batch["unmixing"].to(device, dtype=torch.float32),
                batch["raw_valid"].to(device).bool(),
                batch["aux_present"].to(device).bool(),
            )
            loss = masked_preview_l1(output.preview, gt, output.active_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite warmup preview loss at step {global_step}")
            loss.backward()
            optimizer.step()
            global_step += 1
            progress.update(1)
            metrics = tri_diagnostic_metrics(output)
            for name in diagnostic_names:
                diagnostic_sums[name] += float(metrics[name].detach())
            diagnostic_steps += 1
            fallback_reasons.update(output.checker_diagnostics.fallback_reasons)
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                accepted=f"{float(metrics['checker_accepted_fraction']):.3f}",
                gain=f"{float(metrics['checker_internal_gain']):.3g}",
            )
            last_batch = {"preview": output.preview.detach().cpu(), "gt": gt.detach().cpu()}
            if global_step % args.checkpointing_steps == 0:
                _save(
                    output_dir / f"checkpoint-{global_step:05d}",
                    conditioner,
                    optimizer,
                    global_step,
                    config,
                    stats,
                    metadata,
                    generator,
                )
            if global_step >= args.max_steps:
                break
    progress.close()
    final_dir = output_dir / "final"
    _save(
        final_dir,
        conditioner,
        optimizer,
        global_step,
        config,
        stats,
        metadata,
        generator,
    )
    if last_batch is not None:
        preview = tensor_to_pil(last_batch["preview"][0])
        target = tensor_to_pil(last_batch["gt"][0])
        canvas = Image.new("RGB", (preview.width * 2, preview.height))
        canvas.paste(preview, (0, 0))
        canvas.paste(target, (preview.width, 0))
        canvas.save(output_dir / "last_preview_gt.png")
    save_json(
        {
            "stage": stage,
            "global_step": global_step,
            "checker_enabled": stage == "B",
            "checker_diagnostics_mean": {
                name: diagnostic_sums[name] / max(1, diagnostic_steps)
                for name in diagnostic_names
            },
            "checker_fallback_reasons": dict(sorted(fallback_reasons.items())),
            "note": "checker scores/support are internal heuristic diagnostics, not calibrated truth confidence",
        },
        output_dir / "warmup_summary.json",
    )
    LOGGER.info("Warmup stage %s completed at step %d: %s", stage, global_step, final_dir)


if __name__ == "__main__":
    main()
