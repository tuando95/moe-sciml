#!/usr/bin/env python3
"""Main training script for AME-ODE."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path

from src.utils.config import Config
from src.training.trainer import AMEODETrainer, create_data_loaders
from src.evaluation.metrics import AMEODEMetrics, PerformanceProfiler
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
    parser = argparse.ArgumentParser(description='Train AME-ODE model')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate model (requires --resume)')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of synthetic datasets (ignore cache)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(Path(args.config))
    
    # Set random seed
    set_seed(args.seed if args.seed else config['seed'])
    
    # Create data loaders
    print(f"Loading {args.system} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.system, force_regenerate=args.force_regenerate
    )
    
    # Initialize trainer
    trainer = AMEODETrainer(config, device=args.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    if args.evaluate:
        # Evaluation mode
        if not args.resume:
            raise ValueError("--evaluate requires --resume to specify model checkpoint")
        
        print("Running evaluation...")
        evaluate_model(trainer, test_loader, config)
    else:
        # Training mode
        print("Starting training...")
        trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        evaluate_model(trainer, test_loader, config)


def evaluate_model(trainer, test_loader, config):
    """Comprehensive model evaluation."""
    device = trainer.device
    
    # Initialize metrics and visualization
    metrics_calculator = AMEODEMetrics(config.to_dict())
    visualizer = AMEODEVisualizer(save_dir=Path(config.logging['log_dir']) / 'visualizations')
    profiler = PerformanceProfiler()
    
    # Evaluation metrics
    all_metrics = []
    
    # Sample trajectories for visualization
    sample_batch = next(iter(test_loader))
    sample_traj = sample_batch['trajectory'].to(device)
    sample_times = sample_batch['times'].to(device)
    sample_x0 = sample_batch['initial_condition'].to(device)
    
    # Model inference - use fast inference for evaluation
    trainer.model.eval()
    with torch.no_grad():
        # For visualization we need the info, so use regular forward
        pred_traj, model_info = trainer.model(sample_x0, sample_times)
        
        # But also benchmark fast inference
        import time
        start_time = time.time()
        fast_pred_traj = trainer.model.fast_inference(sample_x0, sample_times)
        fast_time = time.time() - start_time
        
        start_time = time.time()
        _, _ = trainer.model(sample_x0, sample_times)
        regular_time = time.time() - start_time
        
        print(f"\nInference Speed Comparison:")
        print(f"  Regular inference: {regular_time:.4f}s")
        print(f"  Fast inference: {fast_time:.4f}s")
        print(f"  Speedup: {regular_time/fast_time:.2f}x")
    
    # Compute metrics
    batch_metrics = metrics_calculator.compute_all_metrics(
        pred_traj, sample_traj, model_info, sample_times
    )
    all_metrics.append(batch_metrics)
    
    # Performance profiling
    perf_metrics = profiler.profile_forward_pass(
        trainer.model, sample_x0[:10], sample_times, n_runs=10
    )
    memory_metrics = profiler.profile_memory_usage(
        trainer.model, sample_x0[:10], sample_times
    )
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Phase portraits
    for i in range(min(3, trainer.model.n_experts)):
        visualizer.plot_phase_portraits(
            trainer.model, expert_idx=i, save_name=f'phase_portrait_expert_{i}'
        )
    visualizer.plot_phase_portraits(
        trainer.model, save_name='phase_portrait_mixture'
    )
    
    # Routing heatmap
    visualizer.plot_routing_heatmap(
        trainer.model, save_name='routing_heatmap'
    )
    
    # Trajectory comparison
    visualizer.plot_trajectory_comparison(
        sample_traj, pred_traj, sample_times, save_name='trajectory_comparison'
    )
    
    # Expert usage evolution
    if 'routing_weights' in model_info and model_info['routing_weights'].numel() > 0:
        visualizer.plot_expert_usage_evolution(
            model_info['routing_weights'], sample_times, save_name='expert_usage'
        )
    
    # Training history
    visualizer.plot_loss_landscape(
        trainer.metrics_history, save_name='training_history'
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\nTrajectory Metrics:")
    print(f"  MSE: {avg_metrics.get('trajectory_mse', 0):.6f}")
    print(f"  RMSE: {avg_metrics.get('trajectory_rmse', 0):.6f}")
    print(f"  Hausdorff Distance: {avg_metrics.get('hausdorff_distance', 0):.6f}")
    
    print("\nComputational Efficiency:")
    print(f"  Mean Active Experts: {avg_metrics.get('mean_active_experts', 0):.2f}")
    print(f"  Routing Sparsity: {avg_metrics.get('routing_sparsity', 0):.2%}")
    print(f"  Forward Pass Time: {perf_metrics['mean_forward_time']:.4f}s")
    print(f"  Throughput: {perf_metrics['throughput']:.2f} samples/s")
    print(f"  Peak Memory: {memory_metrics['peak_memory_mb']:.2f} MB")
    
    print("\nExpert Specialization:")
    print(f"  Routing Entropy Rate: {avg_metrics.get('routing_entropy_rate', 0):.4f}")
    if 'expert_specialization_mi' in avg_metrics:
        print(f"  Mutual Information: {avg_metrics['expert_specialization_mi']:.4f}")
    
    print("\nModel Information:")
    print(f"  Total Parameters: {memory_metrics['model_params']:,}")
    print(f"  Number of Experts: {trainer.model.n_experts}")
    print(f"  State Dimension: {trainer.model.state_dim}")
    
    print("="*50)


if __name__ == '__main__':
    main()