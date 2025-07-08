#!/usr/bin/env python3
"""Simple and fast evaluation of ablation checkpoints."""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ame_ode import AMEODE
from src.utils.config import Config
from src.training.trainer import create_data_loaders


def evaluate_checkpoint(checkpoint_path: Path, test_loader, device: torch.device) -> Dict[str, Any]:
    """Evaluate a single checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model
    model = AMEODE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    # Evaluate
    total_mse = 0.0
    total_samples = 0
    inference_times = []
    
    print(f"  Batch size: {test_loader.batch_size}")
    print(f"  Total batches: {len(test_loader)}")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                print(f"  Processing batch {i+1}/{len(test_loader)}...")
                
            trajectory = batch['trajectory'].to(device)
            times = batch['times'].to(device)
            x0 = batch['initial_condition'].to(device)
            
            # Ensure times is 1D
            if times.dim() > 1:
                times_1d = times[0]
            else:
                times_1d = times
            
            # Time the inference
            start_time = time.time()
            pred_trajectory, info = model(x0, times_1d)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Compute MSE
            mse = torch.mean((pred_trajectory - trajectory) ** 2).item()
            batch_size = x0.shape[0]
            total_mse += mse * batch_size
            total_samples += batch_size
            
            # Only process first few batches for quick evaluation
            if i >= 10:  # Process only 10 batches for speed
                print("  (Evaluating on subset for speed)")
                break
    
    # Compute metrics
    avg_mse = total_mse / total_samples
    avg_inference_time = np.mean(inference_times)
    
    metrics = {
        'trajectory_mse': avg_mse,
        'total_params': total_params,
        'inference_time_per_batch': avg_inference_time,
        'inference_time_per_traj': avg_inference_time / test_loader.batch_size,
        'n_experts': config['model'].get('n_experts', 4),
        'temperature': config['model'].get('temperature', 1.0),
        'training_epochs': checkpoint.get('epoch', 'N/A')
    }
    
    # Add routing info if available
    if 'routing_weights' in info and info['routing_weights'].numel() > 0:
        weights = info['routing_weights']
        # Average over time and batch
        avg_weights = weights.mean(dim=(0, 1)).cpu().numpy()
        metrics['expert_usage'] = avg_weights.tolist()
        
        # Routing entropy
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        metrics['routing_entropy'] = entropy.item()
        
        # Active experts
        threshold = model.expert_threshold
        active_experts = (weights > threshold).float().sum(dim=-1).mean()
        metrics['mean_active_experts'] = active_experts.item()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ablation checkpoints')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints_ablation'),
                        help='Directory containing ablation checkpoints')
    parser.add_argument('--baseline-checkpoint', type=Path, default=Path('checkpoints_test/best_model.pt'),
                        help='Path to baseline model checkpoint')
    parser.add_argument('--config', type=Path, default=Path('configs/ablation/base_multiscale.yml'),
                        help='Base configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='System name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=Path, default=Path('ablation_results/quick_evaluation.json'),
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config and create test loader
    config = Config(args.config)
    print(f"\nLoading test data for {args.system}...")
    _, _, test_loader = create_data_loaders(config, args.system)
    print(f"Test dataset size: {len(test_loader.dataset)} trajectories")
    print(f"Batch size: {test_loader.batch_size}")
    
    results = {}
    
    # Evaluate baseline
    if args.baseline_checkpoint.exists():
        print(f"\nEvaluating baseline: {args.baseline_checkpoint}")
        baseline_metrics = evaluate_checkpoint(args.baseline_checkpoint, test_loader, device)
        results['baseline'] = {
            'checkpoint': str(args.baseline_checkpoint),
            'metrics': baseline_metrics
        }
        print(f"  MSE: {baseline_metrics['trajectory_mse']:.6f}")
    
    # Evaluate ablations
    ablation_dirs = [d for d in args.checkpoint_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(ablation_dirs)} ablation checkpoints")
    
    for ablation_dir in ablation_dirs:
        checkpoint_path = ablation_dir / 'best_model.pt'
        if checkpoint_path.exists():
            print(f"\nEvaluating {ablation_dir.name}...")
            try:
                metrics = evaluate_checkpoint(checkpoint_path, test_loader, device)
                results[ablation_dir.name] = {
                    'checkpoint': str(checkpoint_path),
                    'metrics': metrics
                }
                print(f"  MSE: {metrics['trajectory_mse']:.6f}")
                if 'mean_active_experts' in metrics:
                    print(f"  Active experts: {metrics['mean_active_experts']:.2f}")
            except Exception as e:
                print(f"  Error: {e}")
                results[ablation_dir.name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline_mse = results.get('baseline', {}).get('metrics', {}).get('trajectory_mse', None)
    
    # Sort by MSE
    sorted_results = []
    for name, data in results.items():
        if 'metrics' in data and name != 'baseline':
            sorted_results.append((name, data['metrics']))
    sorted_results.sort(key=lambda x: x[1]['trajectory_mse'])
    
    print(f"\n{'Configuration':<25} {'MSE':<12} {'vs Baseline':<15} {'Active Experts':<15}")
    print("-" * 70)
    
    if baseline_mse:
        print(f"{'BASELINE':<25} {baseline_mse:<12.6f} {'---':<15} {'---':<15}")
        print("-" * 70)
    
    for name, metrics in sorted_results:
        mse = metrics['trajectory_mse']
        change = ((mse - baseline_mse) / baseline_mse * 100) if baseline_mse else 0
        active = metrics.get('mean_active_experts', 'N/A')
        active_str = f"{active:.2f}" if isinstance(active, (int, float)) else active
        
        print(f"{name:<25} {mse:<12.6f} {change:+.1f}%{'':<10} {active_str:<15}")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Save text report
    report_path = args.output.parent / 'quick_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"AME-ODE Ablation Study Quick Evaluation\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        if baseline_mse:
            f.write(f"Baseline MSE: {baseline_mse:.6f}\n\n")
        
        f.write("Results (sorted by MSE):\n")
        f.write("-"*60 + "\n")
        for name, metrics in sorted_results:
            f.write(f"\n{name}:\n")
            f.write(f"  MSE: {metrics['trajectory_mse']:.6f}\n")
            if baseline_mse:
                change = (metrics['trajectory_mse'] - baseline_mse) / baseline_mse * 100
                f.write(f"  vs Baseline: {change:+.1f}%\n")
            f.write(f"  Parameters: {metrics['total_params']:,}\n")
            f.write(f"  Active Experts: {metrics.get('mean_active_experts', 'N/A')}\n")
            f.write(f"  Temperature: {metrics.get('temperature', 'N/A')}\n")
    
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()