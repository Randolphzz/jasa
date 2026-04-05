#!/usr/bin/env python3
"""Profile training step by step to identify bottlenecks."""

import time
import torch
from torch.utils.data import DataLoader

from models.waveform_training_entry import WaveformTrainingEntry, WaveformTrainingEntryConfig
from train_minimal import build_random_dataloader


def profile_training_step(entry: WaveformTrainingEntry, num_batches: int = 10):
    """Profile individual components of the training step."""
    dataloader = build_random_dataloader(batch_size=16, num_workers=4, split='train')

    times = {
        'data_loading': [],
        'to_device': [],
        'adapter': [],
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }

    for batch_idx, waveform_batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        start_total = time.time()

        # Data loading (already done by dataloader)
        start_to_device = time.time()
        if isinstance(waveform_batch, dict):
            waveform_batch = {k: v.to(entry.device) for k, v in waveform_batch.items()}
        else:
            waveform_batch = waveform_batch.to(entry.device)
        times['to_device'].append(time.time() - start_to_device)

        # Adapter (STFT)
        start_adapter = time.time()
        spectral_batch = entry.adapter(waveform_batch)
        spectral_batch = spectral_batch.to(device=entry.device)
        times['adapter'].append(time.time() - start_adapter)

        # Forward pass
        start_forward = time.time()
        entry.training_step.train(True)
        entry.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            step_output = entry.training_step(spectral_batch)
        times['forward'].append(time.time() - start_forward)

        # Backward pass
        start_backward = time.time()
        step_output.total_loss.backward()
        times['backward'].append(time.time() - start_backward)

        # Optimizer step
        start_optimizer = time.time()
        entry.optimizer.step()
        times['optimizer'].append(time.time() - start_optimizer)

        total_time = time.time() - start_total
        times['total'].append(total_time)
        times['data_loading'].append(start_to_device - start_total)  # Time spent in dataloader

    # Calculate averages
    results = {}
    for key, values in times.items():
        results[key] = {
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

    return results


def main():
    print("Profiling training step components...")

    entry = WaveformTrainingEntry(WaveformTrainingEntryConfig(
        dataset_root='/mnt/kkl/A4BS_250h',
        batch_size=16,
        num_workers=16,
        gpus='0,1'
    ))

    results = profile_training_step(entry, num_batches=20)

    print("\nTiming breakdown (seconds per batch):")
    print("-" * 50)
    for component, stats in results.items():
        print("20")

    print("\nAnalysis:")
    print("- data_loading: Time spent waiting for dataloader")
    print("- to_device: CPU->GPU transfer")
    print("- adapter: STFT computation")
    print("- forward: Model forward pass")
    print("- backward: Gradient computation")
    print("- optimizer: Parameter updates")

    total_avg = results['total']['avg']
    print(".3f")
    print(".1f")


if __name__ == "__main__":
    main()