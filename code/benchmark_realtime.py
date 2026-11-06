#!/usr/bin/env python3
"""
benchmark_realtime.py

Real-time benchmarking and profiling suite for the 5,000-neuron Drosophila subnetwork
on NVIDIA RTX 3050 GPU.

Evaluates:
- Wall-clock simulation duration vs. simulated biological time
- Real-time ratio (t_sim / t_wall, target >= 1.0x)
- Step latency distribution (mean, p50, p95, p99 in microseconds)
- GPU VRAM consumption (peak allocated and cached)
- Biological firing rates of descending walking command neurons and feeding motor neurons
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure code/ is in import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import torch

from realtime_subnet_engine import build_engine


def setup_experiment_rates(engine, experiment: str, custom_rate: float = None):
    """
    Configure Poisson activation rates for the specified biological experiment.
    """
    rates = torch.zeros(engine.num_neurons, dtype=torch.float32, device=engine.device)
    stimulated_indices = []

    if experiment == "walk":
        rate = custom_rate if custom_rate is not None else 100.0  # 100 Hz
        rates[engine.p9_indices] = rate
        stimulated_indices = engine.p9_indices
        print(f"  Experiment: 'walk' | P9 inputs ({len(engine.p9_indices)} neurons) stimulated at {rate} Hz", flush=True)

    elif experiment == "taste_sugar":
        rate = custom_rate if custom_rate is not None else 200.0  # 200 Hz
        rates[engine.sugar_indices] = rate
        stimulated_indices = engine.sugar_indices
        print(f"  Experiment: 'taste_sugar' | Sugar GRNs ({len(engine.sugar_indices)} neurons) stimulated at {rate} Hz", flush=True)

    elif experiment == "taste_bitter":
        rate = custom_rate if custom_rate is not None else 200.0  # 200 Hz
        rates[engine.bitter_indices] = rate
        stimulated_indices = engine.bitter_indices
        print(f"  Experiment: 'taste_bitter' | Bitter GRNs ({len(engine.bitter_indices)} neurons) stimulated at {rate} Hz", flush=True)

    elif experiment == "walk_and_taste":
        walk_rate = 100.0
        sugar_rate = 200.0
        rates[engine.p9_indices] = walk_rate
        rates[engine.sugar_indices] = sugar_rate
        stimulated_indices = torch.cat([engine.p9_indices, engine.sugar_indices])
        print(f"  Experiment: 'walk_and_taste' | P9 ({walk_rate} Hz) + Sugar ({sugar_rate} Hz)", flush=True)

    elif experiment == "idle":
        print("  Experiment: 'idle' | Spontaneous baseline (no external stimulation)", flush=True)

    else:
        raise ValueError(f"Unknown experiment: {experiment}. Choices: walk, taste_sugar, taste_bitter, walk_and_taste, idle")

    engine.set_stimulated_neurons(stimulated_indices)
    return rates


def run_benchmark(
    t_run: float = 1.0,
    experiment: str = "walk_and_taste",
    subnetwork_dir: str = "data/subnetwork_5k",
    device: str = "cuda",
    block_steps: int = 100,
    use_cuda_graph: bool = True,
):
    print("=" * 75, flush=True)
    print("Drosophila 5,000-Neuron Subnetwork Real-Time Benchmark", flush=True)
    print(f"Target duration: {t_run:.2f} s | Experiment: {experiment} | Device: {device} | CUDA Graph: {use_cuda_graph}", flush=True)
    print("=" * 75, flush=True)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"GPU Hardware: {device_name} (Total VRAM: {total_vram_mb:.0f} MB)", flush=True)
    else:
        print("Running on CPU", flush=True)

    # 1. Load engine
    engine, meta = build_engine(subnetwork_dir=subnetwork_dir, device=device)
    engine.reset_state()

    total_steps = int(round((t_run * 1000.0) / engine.dt))
    num_blocks = total_steps // block_steps
    rem_steps = total_steps % block_steps

    rates = setup_experiment_rates(engine, experiment)

    # 2. Warmup & CUDA Graph Capture
    print(f"\nWarm-up & Graph capture phase (block size = {block_steps} steps = {block_steps*engine.dt:.1f} ms)...", flush=True)
    if use_cuda_graph and engine.device.type == "cuda":
        engine.capture_cuda_graph(block_size=block_steps)
    engine.step_block(rates, block_steps=block_steps, use_cuda_graph=use_cuda_graph)
    engine.reset_state()
    if device == "cuda":
        torch.cuda.synchronize()

    # 3. Timed Simulation
    print(f"\nExecuting {total_steps:,} simulation timesteps ({t_run} simulated seconds)...", flush=True)
    block_latencies_us = []

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for b in range(num_blocks):
        t_b0 = time.perf_counter()
        engine.step_block(rates, block_steps=block_steps, use_cuda_graph=use_cuda_graph)
        if device == "cuda":
            torch.cuda.synchronize()
        t_b1 = time.perf_counter()
        dt_step_us = ((t_b1 - t_b0) * 1e6) / block_steps
        block_latencies_us.append(dt_step_us)

    if rem_steps > 0:
        engine.step_block(rates, block_steps=rem_steps, use_cuda_graph=False)
        if device == "cuda":
            torch.cuda.synchronize()

    t_end = time.perf_counter()
    wall_duration = t_end - t_start

    # 4. Metrics & Profiling
    realtime_ratio = t_run / max(wall_duration, 1e-6)
    latencies = np.array(block_latencies_us)
    mean_lat_us = float(np.mean(latencies))
    p50_lat_us = float(np.percentile(latencies, 50))
    p95_lat_us = float(np.percentile(latencies, 95))
    p99_lat_us = float(np.percentile(latencies, 99))

    vram_alloc_mb = 0.0
    vram_cached_mb = 0.0
    if device == "cuda":
        vram_alloc_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        vram_cached_mb = torch.cuda.max_memory_reserved() / 1024 / 1024

    # 5. Biological Firing Analysis
    rates_summary = engine.get_group_firing_rates(t_run)

    print("\n" + "=" * 75, flush=True)
    print("BENCHMARK PERFORMANCE RESULTS", flush=True)
    print("=" * 75, flush=True)
    print(f"Simulated Biological Time : {t_run:.2f} s ({total_steps:,} steps @ dt={engine.dt} ms)", flush=True)
    print(f"Wall-Clock Elapsed Time   : {wall_duration:.4f} s", flush=True)
    print(f"Real-Time Throughput Ratio: {realtime_ratio:.2f}x  {'[REAL-TIME ACHIEVED]' if realtime_ratio >= 1.0 else '[BELOW REAL-TIME]'}", flush=True)
    print(f"Mean Latency per Step     : {mean_lat_us:.2f} us (Threshold: <= 100.0 us)", flush=True)
    print(f"Latency Percentiles (us)  : p50: {p50_lat_us:.2f} | p95: {p95_lat_us:.2f} | p99: {p99_lat_us:.2f}", flush=True)
    if device == "cuda":
        print(f"Peak VRAM Allocated       : {vram_alloc_mb:.2f} MB (Budget: 500 MB)", flush=True)
        print(f"Peak VRAM Reserved/Cached : {vram_cached_mb:.2f} MB", flush=True)
    print("-" * 75, flush=True)

    print("\nBIOLOGICAL CIRCUIT ACTIVITY", flush=True)
    print("-" * 75, flush=True)
    for grp in ["p9_input", "sugar_grn", "bitter_grn", "descending", "motor_mn9", "sez_interneuron"]:
        if grp in rates_summary:
            info = rates_summary[grp]
            print(f"{grp:<22}: Mean: {info['mean_rate_hz']:>6.2f} Hz | Max: {info['max_rate_hz']:>6.2f} Hz | Active: {info['active_neurons']}/{info['total_neurons']}", flush=True)

    print("\nKey Descending Locomotion Neurons:", flush=True)
    for lbl, r in rates_summary.get("descending_neurons_detail", {}).items():
        print(f"  - {lbl:<16}: {r:>6.2f} Hz", flush=True)

    print("\nProboscis Feeding Motor Neurons:", flush=True)
    for lbl, r in rates_summary.get("mn9_feeding_motor_detail", {}).items():
        print(f"  - {lbl:<16}: {r:>6.2f} Hz", flush=True)
    print("=" * 75, flush=True)

    return {
        "t_run": t_run,
        "wall_duration": wall_duration,
        "realtime_ratio": realtime_ratio,
        "mean_lat_us": mean_lat_us,
        "p50_lat_us": p50_lat_us,
        "p99_lat_us": p99_lat_us,
        "vram_alloc_mb": vram_alloc_mb,
        "rates": rates_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 5,000-neuron Drosophila subnetwork.")
    parser.add_argument("--t_run", type=float, default=1.0, help="Simulated duration in seconds (default: 1.0)")
    parser.add_argument(
        "--experiment",
        type=str,
        default="walk_and_taste",
        choices=["walk", "taste_sugar", "taste_bitter", "walk_and_taste", "idle"],
        help="Experiment protocol",
    )
    parser.add_argument("--subnetwork-dir", type=str, default="data/subnetwork_5k", help="Subnetwork directory")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/cpu)")
    parser.add_argument("--block-steps", type=int, default=100, help="Step block size (default: 100)")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA Graph capture")

    args = parser.parse_args()

    run_benchmark(
        t_run=args.t_run,
        experiment=args.experiment,
        subnetwork_dir=args.subnetwork_dir,
        device=args.device,
        block_steps=args.block_steps,
        use_cuda_graph=not args.no_cuda_graph,
    )
