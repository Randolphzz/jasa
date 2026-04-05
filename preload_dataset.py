#!/usr/bin/env python3
"""Preload dataset to memory-mapped tensors for faster training."""

import os
import torch
from pathlib import Path
from utils.dataset import A4BS250hDataset


def preload_dataset_to_memory(dataset: A4BS250hDataset, cache_dir: str = "./cache") -> str:
    """Preload entire dataset to memory-mapped tensors."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    print(f"Preloading {len(dataset)} samples to {cache_path}...")

    # Preload all data
    all_data = []
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"Loaded {i}/{len(dataset)} samples...")
        sample = dataset[i]
        all_data.append({
            'mixture': sample['mixture'],
            'bc': sample['bc'],
            'clean': sample['clean'],
            'bc_channel': sample['bc_channel']
        })

    # Save as memory-mapped tensors
    mixture_tensors = torch.stack([s['mixture'] for s in all_data])
    bc_tensors = torch.stack([s['bc'] for s in all_data])
    clean_tensors = torch.stack([s['clean'] for s in all_data])
    bc_channels = torch.tensor([s['bc_channel'] for s in all_data])

    # Memory map to disk
    mixture_file = cache_path / "mixture.dat"
    bc_file = cache_path / "bc.dat"
    clean_file = cache_path / "clean.dat"
    bc_channel_file = cache_path / "bc_channel.dat"

    torch.save(mixture_tensors, mixture_file)
    torch.save(bc_tensors, bc_file)
    torch.save(clean_tensors, clean_file)
    torch.save(bc_channels, bc_channel_file)

    print(f"Preloaded data saved to {cache_path}")
    return str(cache_path)


class MemoryDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset for fast loading."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

        # Load memory-mapped tensors
        self.mixture = torch.load(self.cache_dir / "mixture.dat", map_location='cpu')
        self.bc = torch.load(self.cache_dir / "bc.dat", map_location='cpu')
        self.clean = torch.load(self.cache_dir / "clean.dat", map_location='cpu')
        self.bc_channels = torch.load(self.cache_dir / "bc_channel.dat", map_location='cpu')

    def __len__(self):
        return len(self.mixture)

    def __getitem__(self, idx):
        return {
            'mixture': self.mixture[idx],
            'bc': self.bc[idx],
            'clean': self.clean[idx],
            'bc_channel': self.bc_channels[idx]
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/kkl/A4BS_250h")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default="./dataset_cache")
    args = parser.parse_args()

    dataset = A4BS250hDataset(args.dataset_root, args.split)
    cache_path = preload_dataset_to_memory(dataset, args.cache_dir)

    # Test loading
    mem_dataset = MemoryDataset(cache_path)
    print(f"Memory dataset loaded: {len(mem_dataset)} samples")

    # Test speed
    import time
    start = time.time()
    for i in range(100):
        sample = mem_dataset[i]
    elapsed = time.time() - start
    print(".6f")