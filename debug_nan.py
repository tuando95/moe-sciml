#!/usr/bin/env python3
"""Debug NaN issues in training."""

import torch
from pathlib import Path
from src.utils.config import Config
from src.training.trainer import create_data_loaders

# Load config
config = Config(Path('configs/quick_test.yml'))

# Create data loaders
print("Loading data...")
train_loader, val_loader, test_loader = create_data_loaders(
    config, 'multi_scale_oscillators', force_regenerate=False
)

# Check first batch
print("\nChecking first batch...")
batch = next(iter(train_loader))

for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Contains NaN: {torch.isnan(value).any().item()}")
        print(f"  Contains Inf: {torch.isinf(value).any().item()}")
        print(f"  Min: {value.min().item():.6f}")
        print(f"  Max: {value.max().item():.6f}")
        print(f"  Mean: {value.mean().item():.6f}")
        print(f"  Std: {value.std().item():.6f}")

# Load cached data directly
cache_path = Path('data/cache/synthetic/multi_scale_oscillators_train_64fd2f86.pt')
if cache_path.exists():
    print(f"\nLoading cached data from {cache_path}")
    data = torch.load(cache_path)
    
    trajectories = data['trajectories']
    print(f"\nTrajectories shape: {trajectories.shape}")
    print(f"Contains NaN: {torch.isnan(trajectories).any().item()}")
    print(f"Contains Inf: {torch.isinf(trajectories).any().item()}")
    print(f"Min: {trajectories.min().item():.6f}")
    print(f"Max: {trajectories.max().item():.6f}")
    print(f"Mean: {trajectories.mean().item():.6f}")
    print(f"Std: {trajectories.std().item():.6f}")
    
    # Check for outliers
    outliers = (trajectories.abs() > 100).sum().item()
    print(f"Points > 100: {outliers} ({outliers / trajectories.numel() * 100:.2f}%)")