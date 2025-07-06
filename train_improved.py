#!/usr/bin/env python3
"""Train AME-ODE with improved configuration and initialization."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path

from src.utils.config import Config
from src.training.trainer import AMEODETrainer, create_data_loaders
from src.evaluation.metrics import AMEODEMetrics
from src.evaluation.visualization import AMEODEVisualizer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train AME-ODE with improved settings')
    parser.add_argument('--config', type=str, default='configs/improved_ame.yml',
                        help='Path to configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick-test', action='store_true',
                        help='Use quick test config instead')
    
    args = parser.parse_args()
    
    # Override config if quick test requested
    if args.quick_test:
        args.config = 'configs/quick_test.yml'
    
    # Load configuration
    config = Config(Path(args.config))
    
    # Add improved initialization settings if not present
    if 'use_improved_init' not in config.model:
        config._config['model']['use_improved_init'] = True
    if 'expert_init_strategy' not in config.model:
        config._config['model']['expert_init_strategy'] = 'mixed'
    if 'gating_init_strategy' not in config.model:
        config._config['model']['gating_init_strategy'] = 'uniform'
    
    # Set random seed
    set_seed(args.seed)
    
    # Create data loaders
    print(f"Loading {args.system} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.system, force_regenerate=False
    )
    
    # Initialize trainer
    print("\nInitializing AME-ODE with improved settings...")
    trainer = AMEODETrainer(config, device=args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nModel initialized with {total_params:,} parameters")
    print(f"Number of experts: {trainer.model.n_experts}")
    print(f"Expert architecture: {config.model['expert_architecture']}")
    print(f"Gating temperature: {trainer.model.temperature}")
    
    # Training
    print("\nStarting training...")
    print(f"Optimizer: {config.training['optimizer']}")
    print(f"Learning rate: {config.training['learning_rate']}")
    print(f"Scheduler: {config.training.get('scheduler', {}).get('type', 'none')}")
    print(f"Regularization weights:")
    for key, value in config.training['regularization'].items():
        print(f"  {key}: {value}")
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Initialize metrics calculator
    metrics_calc = AMEODEMetrics(config.to_dict())
    
    # Evaluate
    trainer.model.eval()
    test_losses = []
    all_metrics = []
    
    with torch.no_grad():
        for batch in test_loader:
            trajectory = batch['trajectory'].to(trainer.device)
            times = batch['times'].to(trainer.device)
            x0 = batch['initial_condition'].to(trainer.device)
            
            # Ensure times is 1D
            if times.dim() > 1:
                times = times[0]
            
            # Forward pass
            pred_traj, info = trainer.model(x0, times)
            
            # Compute loss
            losses = trainer.loss_fn(pred_traj, trajectory, info, trainer.model)
            test_losses.append(losses['reconstruction'].item())
            
            # Compute metrics
            metrics = metrics_calc.compute_all_metrics(
                pred_traj, trajectory, info, times
            )
            all_metrics.append(metrics)
    
    # Aggregate results
    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    
    print(f"\nTest Reconstruction Loss: {avg_test_loss:.6f} ± {std_test_loss:.6f}")
    
    # Print other metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        print("\nAdditional Metrics:")
        for key, value in avg_metrics.items():
            if key != 'trajectory_mse':  # Already printed as test loss
                print(f"  {key}: {value:.6f}")
    
    # Compare with baseline
    baseline_single_mse = 0.529915  # From your results
    improvement = (baseline_single_mse - avg_test_loss) / baseline_single_mse * 100
    print(f"\nImprovement over Single Neural ODE: {improvement:.1f}%")
    
    # Save final checkpoint
    checkpoint_path = Path(config.logging['checkpoint_dir']) / 'final_model.pt'
    print(f"\nSaving final model to {checkpoint_path}")
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config.to_dict(),
        'test_mse': avg_test_loss,
        'metrics': avg_metrics if all_metrics else {},
        'epoch': trainer.current_epoch,
    }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()