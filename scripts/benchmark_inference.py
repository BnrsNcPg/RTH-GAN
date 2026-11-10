from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import canonical_config_hash, environment_lines, file_sha256, git_commit, read_yaml, utc_timestamp, write_json
from scripts.model_profiling import build_profile_generator, inference_scope, parameter_counts


def main():
    parser = argparse.ArgumentParser(description="Benchmark complete generator inference, including dense PINN prior generation, without DataLoader time.")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--untrained", action="store_true", help="Explicit random-weight engineering diagnostic only")
    parser.add_argument("--method-config", type=Path, default=Path("configs/methods/rthgan_manuscript.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/rthgan_manuscript_profile_512.yaml"))
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("results/complexity"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Formal inference benchmarking requires a CUDA GPU")
    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("Use at least one warm-up and two measured iterations")
    method = read_yaml(args.method_config)
    if args.input_size < 1:
        raise ValueError("input-size must be positive")
    requested_device = torch.device(args.device)
    torch.cuda.set_device(requested_device.index if requested_device.index is not None else torch.cuda.current_device())
    torch.manual_seed(args.seed)
    model_config = read_yaml(args.model_config)
    generator = build_profile_generator(method, model_config, args.device, args.checkpoint)
    sample = torch.rand(1, 3, args.input_size, args.input_size, device=args.device)
    if method["method"] != "rthgan_manuscript":
        sample = sample * 2 - 1
    with torch.inference_mode():
        for _ in range(args.warmup):
            generator(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        latencies = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            generator(sample)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
    mean_latency = statistics.fmean(latencies)
    payload = {
        "schema_version": 2,
        "method": method["method"],
        "display_name": method["display_name"],
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "weights": "random_initialization" if args.untrained else "checkpoint",
        "evaluation_scope": "untrained_engineering_diagnostic" if args.untrained else "checkpoint_benchmark",
        "model_config_hash": canonical_config_hash(model_config) if args.untrained else None,
        "seed": args.seed,
        "device": str(torch.cuda.current_device()),
        **parameter_counts(generator),
        **inference_scope(generator),
        "input_shape": [1, 3, args.input_size, args.input_size],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": statistics.pstdev(latencies),
        "fps": 1000.0 / mean_latency,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2,
        "environment": environment_lines(args.device),
        "git_commit": git_commit(),
        "timestamp": utc_timestamp(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{method['method']}_inference.json"
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
