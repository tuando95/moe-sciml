#!/usr/bin/env python3
"""Run ablation studies for AME-ODE."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import copy
import json
from typing import Dict, Any, List

from src.utils.config import Config
from src.training.trainer import AMEODETrainer, create_data_loaders
from src.evaluation.metrics import AMEODEMetrics


def run_single_ablation(
    base_config: Config,
    ablation_config: Dict[str, Any],
    system_name: str,
    device: str,
    save_dir: Path,
) -> Dict[str, float]:
    """Run a single ablation experiment."""
    # Create modified config
    config = copy.deepcopy(base_config)
    config_dict = config.to_dict()
    
    # Apply ablation modifications
    for key_path, value in ablation_config.items():
        keys = key_path.split('.')
        current = config_dict
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    # Update config
    config._config = config_dict
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, system_name)
    
    # Train model
    trainer = AMEODETrainer(config, device=device)
    trainer.train(train_loader, val_loader, num_epochs=50)  # Shorter for ablations
    
    # Evaluate
    metrics_calculator = AMEODEMetrics(config.to_dict())
    test_metrics = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            trajectory = batch['trajectory'].to(device)
            times = batch['times'].to(device)
            x0 = batch['initial_condition'].to(device)
            
            pred_trajectory, model_info = trainer.model(x0, times)
            
            batch_metrics = metrics_calculator.compute_all_metrics(
                pred_trajectory, trajectory, model_info, times
            )
            test_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in test_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in test_metrics])
    
    # Save results
    results = {
        'ablation_config': ablation_config,
        'metrics': avg_metrics,
        'best_val_loss': trainer.best_val_loss,
    }
    
    ablation_name = '_'.join([f"{k.split('.')[-1]}_{v}" for k, v in ablation_config.items()])
    results_path = save_dir / f"{ablation_name}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return avg_metrics


def run_ablation_study(
    config_path: Path,
    ablation_type: str,
    system_name: str,
    device: str,
    output_dir: Path,
):
    """Run a complete ablation study."""
    # Load base configuration
    base_config = Config(config_path)
    
    # Define ablation configurations
    ablation_configs = get_ablation_configs(ablation_type)
    
    # Create output directory
    ablation_dir = output_dir / ablation_type
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline
    print(f"\nRunning baseline for {ablation_type}...")
    baseline_metrics = run_single_ablation(
        base_config, {}, system_name, device, ablation_dir
    )
    
    # Run ablations
    all_results = {'baseline': baseline_metrics}
    
    for i, ablation_config in enumerate(ablation_configs):
        print(f"\nRunning ablation {i+1}/{len(ablation_configs)}: {ablation_config}")
        
        try:
            metrics = run_single_ablation(
                base_config, ablation_config, system_name, device, ablation_dir
            )
            all_results[str(ablation_config)] = metrics
        except Exception as e:
            print(f"Ablation failed: {e}")
            all_results[str(ablation_config)] = {'error': str(e)}
    
    # Save summary
    summary_path = ablation_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print_ablation_summary(ablation_type, all_results)


def get_ablation_configs(ablation_type: str) -> List[Dict[str, Any]]:
    """Get configurations for different ablation studies."""
    if ablation_type == 'routing_mechanism':
        return [
            # Input-only routing
            {'model.gating_architecture.use_history': False},
            # History-only routing
            {'model.gating_architecture.use_state': False},
            # Static routing (temperature → ∞)
            {'model.temperature': 100.0},
            # Hard routing (temperature → 0)
            {'model.temperature': 0.1},
        ]
    
    elif ablation_type == 'expert_initialization':
        return [
            # Random initialization
            {'model.expert_initialization': 'random'},
            # Timescale only
            {'model.expert_initialization': 'timescale'},
            # Stability only
            {'model.expert_initialization': 'stability'},
        ]
    
    elif ablation_type == 'integration_scheme':
        return [
            # Fixed-step Euler
            {'integration.method': 'euler', 'integration.adaptive_step': False},
            # Fixed-step RK4
            {'integration.method': 'rk4', 'integration.adaptive_step': False},
            # Adaptive without routing awareness
            {'integration.routing_aware_step': False},
        ]
    
    elif ablation_type == 'n_experts':
        return [
            {'model.n_experts': 1},
            {'model.n_experts': 2},
            {'model.n_experts': 6},
            {'model.n_experts': 8},
        ]
    
    elif ablation_type == 'regularization':
        return [
            # No routing regularization
            {'training.regularization.route_weight': 0.0},
            # No diversity loss
            {'training.regularization.diversity_weight': 0.0},
            # No smoothness loss
            {'training.regularization.smoothness_weight': 0.0},
            # No balance loss
            {'training.regularization.balance_weight': 0.0},
            # All regularization off
            {
                'training.regularization.route_weight': 0.0,
                'training.regularization.diversity_weight': 0.0,
                'training.regularization.smoothness_weight': 0.0,
                'training.regularization.balance_weight': 0.0,
            },
        ]
    
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")


def print_ablation_summary(ablation_type: str, results: Dict[str, Any]):
    """Print summary of ablation results."""
    print("\n" + "="*60)
    print(f"ABLATION STUDY SUMMARY: {ablation_type}")
    print("="*60)
    
    # Key metrics to compare
    key_metrics = ['trajectory_mse', 'mean_active_experts', 'routing_entropy_rate']
    
    # Print table header
    print(f"\n{'Configuration':<30} " + " ".join(f"{m:<15}" for m in key_metrics))
    print("-" * (30 + 16 * len(key_metrics)))
    
    # Print results
    baseline_metrics = results.get('baseline', {})
    for config_name, metrics in results.items():
        if isinstance(metrics, dict) and 'error' not in metrics:
            row = f"{config_name[:30]:<30} "
            for metric in key_metrics:
                value = metrics.get(metric, 0)
                # Show relative change from baseline
                if config_name != 'baseline' and metric in baseline_metrics:
                    baseline_val = baseline_metrics[metric]
                    if baseline_val > 0:
                        rel_change = (value - baseline_val) / baseline_val * 100
                        row += f"{value:.4f} ({rel_change:+.1f}%) "
                    else:
                        row += f"{value:.4f} (N/A) "
                else:
                    row += f"{value:.4f}         "
            print(row)
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run AME-ODE ablation studies')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to base configuration file')
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['routing_mechanism', 'expert_initialization', 
                                'integration_scheme', 'n_experts', 'regularization'],
                        help='Type of ablation study to run')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run ablation study
    run_ablation_study(
        Path(args.config),
        args.ablation,
        args.system,
        args.device,
        Path(args.output_dir)
    )


if __name__ == '__main__':
    main()