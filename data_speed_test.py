#!/usr/bin/env python3
"""Comprehensive data loading speed test for A4BS_250h dataset."""

import time
from pathlib import Path
import argparse
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

from utils.dataset import build_a4bs250h_dataloader
from train_minimal import build_random_dataloader


def time_dataloader(dataloader: DataLoader, max_batches: int = 100) -> Dict[str, float]:
    """Time a dataloader for max_batches and return statistics."""
    times = []
    total_samples = 0

    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        batch_start = time.time()
        # Simulate minimal processing (just move to device)
        batch = {k: v.to('cpu') for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch_time = time.time() - batch_start

        times.append(batch_time)
        total_samples += batch['mixture'].shape[0]

    total_time = time.time() - start_time

    if times:
        avg_batch_time = sum(times) / len(times)
        min_batch_time = min(times)
        max_batch_time = max(times)
        throughput = total_samples / total_time  # samples/sec
    else:
        avg_batch_time = min_batch_time = max_batch_time = throughput = 0.0

    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'min_batch_time': min_batch_time,
        'max_batch_time': max_batch_time,
        'throughput_samples_per_sec': throughput,
        'total_samples': total_samples,
        'num_batches': len(times)
    }


def run_speed_test(dataset_root: str, batch_size: int, num_workers: int, max_batches: int = 100) -> Dict[str, Dict[str, float]]:
    """Run speed test for both real and random data."""
    results = {}

    print(f"\n=== Testing batch_size={batch_size}, num_workers={num_workers} ===")

    # Test real dataset
    print("Testing real dataset...")
    try:
        real_loader = build_a4bs250h_dataloader(
            root=dataset_root,
            split='train',
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=False
        )
        results['real'] = time_dataloader(real_loader, max_batches)
        print(".3f")
    except Exception as e:
        print(f"Real dataset test failed: {e}")
        results['real'] = {'error': str(e)}

    # Test random dataset
    print("Testing random dataset...")
    random_loader = build_random_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        split='train'
    )
    results['random'] = time_dataloader(random_loader, max_batches)
    print(".3f")

    return results


def main():
    parser = argparse.ArgumentParser(description="Data loading speed test")
    parser.add_argument("--dataset-root", type=str, default="/mnt/kkl/A4BS_250h", help="Dataset root")
    parser.add_argument("--max-batches", type=int, default=50, help="Max batches to test")
    args = parser.parse_args()

    print("Data Loading Speed Test for A4BS_250h")
    print("=" * 50)

    # Test configurations
    configs = [
        (8, 0),    # small batch, no workers
        (8, 4),    # small batch, some workers
        (16, 8),   # medium batch, medium workers
        (16, 16),  # medium batch, many workers
        (32, 16),  # large batch, many workers
    ]

    all_results = {}

    for batch_size, num_workers in configs:
        results = run_speed_test(args.dataset_root, batch_size, num_workers, args.max_batches)
        all_results[f"bs{batch_size}_nw{num_workers}"] = results

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print("<10")
    print("-" * 60)

    for config, results in all_results.items():
        print("<10")
        for data_type, stats in results.items():
            if isinstance(stats, dict) and 'error' not in stats:
                print("10.3f")
            else:
                print("10")

    print("\nAnalysis:")
    print("- If real dataset is much slower than random, I/O is the bottleneck")
    print("- If both are similar, bottleneck is elsewhere (CPU preprocessing, GPU transfer, etc.)")
    print("- Compare throughput: higher is better")
    print("- Look at batch time variance: high variance indicates I/O issues")


if __name__ == "__main__":
    main()