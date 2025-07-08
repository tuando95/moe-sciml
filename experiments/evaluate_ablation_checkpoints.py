#!/usr/bin/env python3
"""Evaluate trained models from ablation study checkpoints."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ame_ode import AMEODE
from src.data.synthetic_systems import SyntheticDataGenerator
from src.evaluation.metrics import AMEODEMetrics
from src.utils.config import Config
from src.evaluation.visualization import AMEODEVisualizer


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint and extract model state and config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> AMEODE:
    """Create and load model from checkpoint."""
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Extract expert architecture
    expert_arch = model_config.get('expert_architecture', {})
    expert_hidden_dims = [expert_arch.get('width', 128)] * expert_arch.get('depth', 5)
    
    # Extract gating architecture
    gating_arch = model_config.get('gating_architecture', {})
    gating_hidden_dims = [gating_arch.get('width', 64)] * gating_arch.get('depth', 3)
    
    # Extract history embedding
    history_config = model_config.get('history_embedding', {})
    history_dim = history_config.get('hidden_dim', 64)
    
    # Create model
    model = AMEODE(
        state_dim=checkpoint.get('state_dim', 4),
        n_experts=model_config.get('n_experts', 4),
        expert_hidden_dims=expert_hidden_dims,
        gating_hidden_dims=gating_hidden_dims,
        history_dim=history_dim,
        temperature=model_config.get('temperature', 1.0),
        expert_threshold=model_config.get('expert_threshold', 0.01),
        device=device
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_model(
    model: AMEODE,
    test_data: Dict[str, torch.Tensor],
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """Evaluate a single model on test data."""
    model.eval()
    
    # Move data to device
    x0 = test_data['initial_conditions'].to(device)
    true_trajectories = test_data['trajectories'].to(device)
    times = test_data['times'].to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        pred_trajectories, info = model(x0, times[0])  # Use first time series
    inference_time = time.time() - start_time
    
    # Compute basic trajectory metrics
    mse = torch.mean((pred_trajectories - true_trajectories) ** 2).item()
    rmse = torch.sqrt(torch.mean((pred_trajectories - true_trajectories) ** 2)).item()
    mae = torch.mean(torch.abs(pred_trajectories - true_trajectories)).item()
    
    metrics = {
        'trajectory_mse': mse,
        'trajectory_rmse': rmse,
        'trajectory_mae': mae
    }
    
    # Add additional metrics
    metrics['inference_time'] = inference_time
    metrics['inference_time_per_traj'] = inference_time / x0.shape[0]  # Use actual batch size
    
    # Expert usage statistics
    if 'routing_weights' in info:
        weights = info['routing_weights']  # Shape: (time, batch, n_experts)
        
        # Average weights over time and batch
        avg_weights = weights.mean(dim=(0, 1)).cpu().numpy()
        metrics['expert_usage'] = avg_weights.tolist()
        
        # Routing entropy
        eps = 1e-8
        entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
        metrics['routing_entropy'] = entropy.item()
        
        # Number of active experts (weight > threshold)
        threshold = model.expert_threshold
        active_experts = (weights > threshold).float().sum(dim=-1).mean()
        metrics['mean_active_experts'] = active_experts.item()
        
        # Expert usage variance (load balancing)
        usage_variance = weights.mean(dim=(0, 1)).var()
        metrics['expert_usage_variance'] = usage_variance.item()
    
    return metrics


def load_cached_test_data(
    cache_dir: Path,
    system_name: str,
    config_path: Path = Path('configs/ablation/base_multiscale.yml')
) -> Dict[str, torch.Tensor]:
    """Load cached test dataset based on configuration."""
    # Load config to get data parameters
    config = Config(config_path)
    
    # Find the system config
    system_config = None
    for sys_cfg in config.data.synthetic_systems:
        if sys_cfg['name'] == system_name:
            system_config = sys_cfg
            break
    
    if system_config is None:
        raise ValueError(f"System {system_name} not found in config")
    
    # Create data generator with full config structure
    generator_config = {
        'data': {
            'synthetic_systems': [system_config],
            'noise': config.data.noise,
            'train_val_test_split': config.data.get('train_val_test_split', [0.6, 0.2, 0.2]),
            'augmentation': config.data.get('augmentation', {})
        },
        'cache_dir': str(cache_dir)
    }
    
    data_generator = SyntheticDataGenerator(generator_config, cache_dir=str(cache_dir))
    
    # Generate datasets for each split to get the cached data
    train_data = data_generator.generate_dataset(system_name, split='train')
    val_data = data_generator.generate_dataset(system_name, split='val')
    test_data = data_generator.generate_dataset(system_name, split='test')
    
    # Return test data
    print(f"Loaded test data: {test_data['initial_conditions'].shape[0]} trajectories")
    return test_data


def evaluate_all_checkpoints(
    checkpoint_dir: Path,
    baseline_checkpoint: Path,
    system_name: str,
    device: torch.device,
    cache_dir: Path = Path('cache')
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all ablation checkpoints and baseline."""
    results = {}
    
    # Load cached test data once
    print(f"\nLoading cached test data for {system_name}...")
    test_data = load_cached_test_data(cache_dir, system_name)
    
    # First, evaluate baseline
    print("\nEvaluating baseline model...")
    if baseline_checkpoint.exists():
        checkpoint = load_checkpoint(baseline_checkpoint, device)
        model = create_model_from_checkpoint(checkpoint, device)
        
        baseline_metrics = evaluate_model(model, test_data, device)
        results['baseline'] = {
            'checkpoint': str(baseline_checkpoint),
            'metrics': baseline_metrics
        }
        print(f"Baseline MSE: {baseline_metrics['trajectory_mse']:.6f}")
    else:
        print(f"Baseline checkpoint not found: {baseline_checkpoint}")
        results['baseline'] = None
    
    # Evaluate each ablation
    ablation_configs = []
    for config_dir in checkpoint_dir.iterdir():
        if config_dir.is_dir():
            best_model_path = config_dir / 'best_model.pt'
            if best_model_path.exists():
                ablation_configs.append((config_dir.name, best_model_path))
    
    print(f"\nFound {len(ablation_configs)} ablation checkpoints to evaluate")
    
    for config_name, checkpoint_path in tqdm(ablation_configs, desc="Evaluating ablations"):
        try:
            checkpoint = load_checkpoint(checkpoint_path, device)
            model = create_model_from_checkpoint(checkpoint, device)
            
            # Use same test data for all models
            metrics = evaluate_model(model, test_data, device)
            
            results[config_name] = {
                'checkpoint': str(checkpoint_path),
                'metrics': metrics,
                'training_epochs': checkpoint.get('epoch', 'unknown'),
                'best_val_loss': checkpoint.get('best_val_loss', None)
            }
            
        except Exception as e:
            print(f"\nError evaluating {config_name}: {e}")
            results[config_name] = {
                'checkpoint': str(checkpoint_path),
                'error': str(e)
            }
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS - CHECKPOINT EVALUATION")
    print("="*100)
    
    # Get baseline metrics
    baseline_metrics = results.get('baseline', {}).get('metrics', {})
    baseline_mse = baseline_metrics.get('trajectory_mse', None)
    
    # Prepare table data
    table_data = []
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        
        if 'error' in result:
            table_data.append({
                'config': config_name,
                'status': 'ERROR',
                'error': result['error']
            })
        else:
            metrics = result['metrics']
            row = {
                'config': config_name,
                'mse': metrics['trajectory_mse'],
                'active_experts': metrics.get('mean_active_experts', 'N/A'),
                'routing_entropy': metrics.get('routing_entropy', 'N/A'),
                'inference_time': metrics.get('inference_time_per_traj', 'N/A') * 1000,  # ms
                'training_epochs': result.get('training_epochs', 'N/A')
            }
            
            # Calculate relative change from baseline
            if baseline_mse is not None:
                row['mse_change'] = (metrics['trajectory_mse'] - baseline_mse) / baseline_mse * 100
            
            table_data.append(row)
    
    # Sort by MSE
    table_data.sort(key=lambda x: x.get('mse', float('inf')))
    
    # Print header
    print(f"\n{'Configuration':<25} {'MSE':<12} {'vs Baseline':<12} {'Active Exp':<12} {'Entropy':<10} {'Time(ms)':<10}")
    print("-" * 95)
    
    # Print baseline first
    if baseline_metrics:
        print(f"{'BASELINE':<25} {baseline_mse:<12.6f} {'---':<12} "
              f"{baseline_metrics.get('mean_active_experts', 'N/A'):<12} "
              f"{baseline_metrics.get('routing_entropy', 'N/A'):<10.3f} "
              f"{baseline_metrics.get('inference_time_per_traj', 0)*1000:<10.2f}")
        print("-" * 95)
    
    # Print ablations
    for row in table_data:
        if 'error' in row:
            print(f"{row['config']:<25} {'ERROR: ' + row['error'][:60]}")
        else:
            mse_str = f"{row['mse']:.6f}"
            change_str = f"{row['mse_change']:+.1f}%" if 'mse_change' in row else "N/A"
            active_str = f"{row['active_experts']:.2f}" if isinstance(row['active_experts'], (int, float)) else "N/A"
            entropy_str = f"{row['routing_entropy']:.3f}" if isinstance(row['routing_entropy'], (int, float)) else "N/A"
            time_str = f"{row['inference_time']:.2f}" if isinstance(row['inference_time'], (int, float)) else "N/A"
            
            print(f"{row['config']:<25} {mse_str:<12} {change_str:<12} {active_str:<12} {entropy_str:<10} {time_str:<10}")
    
    print("="*100)
    
    # Find best configuration
    best_configs = [row for row in table_data if 'mse' in row]
    if best_configs:
        best = min(best_configs, key=lambda x: x['mse'])
        print(f"\nBest configuration: {best['config']} (MSE: {best['mse']:.6f})")
        if 'mse_change' in best:
            print(f"Improvement over baseline: {-best['mse_change']:.1f}%")


def save_detailed_results(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Save detailed results to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def generate_report(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate a comprehensive text report of ablation results."""
    report_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    report_lines.append("="*80)
    report_lines.append("AME-ODE ABLATION STUDY REPORT")
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Get baseline metrics
    baseline = results.get('baseline', {})
    baseline_metrics = baseline.get('metrics', {})
    baseline_mse = baseline_metrics.get('trajectory_mse', None)
    
    # Summary Statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*40)
    
    # Calculate statistics
    mse_values = []
    active_experts_values = []
    entropy_values = []
    
    for config, result in results.items():
        if config != 'baseline' and 'metrics' in result:
            metrics = result['metrics']
            mse_values.append(metrics['trajectory_mse'])
            if 'mean_active_experts' in metrics:
                active_experts_values.append(metrics['mean_active_experts'])
            if 'routing_entropy' in metrics:
                entropy_values.append(metrics['routing_entropy'])
    
    if mse_values:
        report_lines.append(f"Number of configurations evaluated: {len(mse_values)}")
        report_lines.append(f"Baseline MSE: {baseline_mse:.6f}" if baseline_mse else "Baseline: Not available")
        report_lines.append("")
        report_lines.append("Trajectory MSE Statistics:")
        report_lines.append(f"  Mean: {np.mean(mse_values):.6f}")
        report_lines.append(f"  Std:  {np.std(mse_values):.6f}")
        report_lines.append(f"  Min:  {np.min(mse_values):.6f}")
        report_lines.append(f"  Max:  {np.max(mse_values):.6f}")
        report_lines.append("")
        
        if active_experts_values:
            report_lines.append("Active Experts Statistics:")
            report_lines.append(f"  Mean: {np.mean(active_experts_values):.2f}")
            report_lines.append(f"  Std:  {np.std(active_experts_values):.2f}")
            report_lines.append("")
        
        if entropy_values:
            report_lines.append("Routing Entropy Statistics:")
            report_lines.append(f"  Mean: {np.mean(entropy_values):.3f}")
            report_lines.append(f"  Std:  {np.std(entropy_values):.3f}")
            report_lines.append("")
    
    # Detailed Results
    report_lines.append("="*80)
    report_lines.append("DETAILED RESULTS BY CONFIGURATION")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Sort configurations by MSE
    sorted_configs = []
    for config, result in results.items():
        if config != 'baseline' and 'metrics' in result:
            sorted_configs.append((config, result))
    sorted_configs.sort(key=lambda x: x[1]['metrics']['trajectory_mse'])
    
    # Baseline first
    if baseline_metrics:
        report_lines.append("BASELINE MODEL")
        report_lines.append("-"*40)
        report_lines.append(f"Checkpoint: {baseline.get('checkpoint', 'N/A')}")
        report_lines.append(f"Trajectory MSE: {baseline_mse:.6f}")
        report_lines.append(f"Active Experts: {baseline_metrics.get('mean_active_experts', 'N/A')}")
        report_lines.append(f"Routing Entropy: {baseline_metrics.get('routing_entropy', 'N/A')}")
        report_lines.append(f"Inference Time/Traj: {baseline_metrics.get('inference_time_per_traj', 0)*1000:.2f} ms")
        report_lines.append("")
    
    # Each configuration
    for rank, (config_name, result) in enumerate(sorted_configs, 1):
        metrics = result['metrics']
        report_lines.append(f"{rank}. {config_name.upper()}")
        report_lines.append("-"*40)
        report_lines.append(f"Checkpoint: {result.get('checkpoint', 'N/A')}")
        report_lines.append(f"Training Epochs: {result.get('training_epochs', 'N/A')}")
        report_lines.append(f"Best Val Loss: {result.get('best_val_loss', 'N/A')}")
        report_lines.append("")
        report_lines.append("Metrics:")
        report_lines.append(f"  Trajectory MSE: {metrics['trajectory_mse']:.6f}")
        
        if baseline_mse:
            change = (metrics['trajectory_mse'] - baseline_mse) / baseline_mse * 100
            report_lines.append(f"  vs Baseline: {change:+.1f}%")
        
        report_lines.append(f"  Active Experts: {metrics.get('mean_active_experts', 'N/A')}")
        report_lines.append(f"  Routing Entropy: {metrics.get('routing_entropy', 'N/A')}")
        report_lines.append(f"  Expert Usage Variance: {metrics.get('expert_usage_variance', 'N/A')}")
        report_lines.append(f"  Inference Time/Traj: {metrics.get('inference_time_per_traj', 0)*1000:.2f} ms")
        
        if 'expert_usage' in metrics:
            report_lines.append(f"  Expert Usage Distribution: {[f'{u:.3f}' for u in metrics['expert_usage']]}")
        
        report_lines.append("")
    
    # Best configurations
    report_lines.append("="*80)
    report_lines.append("TOP PERFORMING CONFIGURATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if sorted_configs:
        # Best by MSE
        best_mse = sorted_configs[0]
        report_lines.append(f"Best MSE: {best_mse[0]} ({best_mse[1]['metrics']['trajectory_mse']:.6f})")
        
        # Most efficient (fewest active experts)
        efficient_configs = [(c, r) for c, r in sorted_configs if 'mean_active_experts' in r['metrics']]
        if efficient_configs:
            most_efficient = min(efficient_configs, key=lambda x: x[1]['metrics']['mean_active_experts'])
            report_lines.append(f"Most Efficient: {most_efficient[0]} ({most_efficient[1]['metrics']['mean_active_experts']:.2f} active experts)")
        
        # Best balance (good MSE with fewer experts)
        if baseline_mse and efficient_configs:
            # Score = normalized MSE + normalized active experts
            balance_scores = []
            for config, result in efficient_configs:
                mse = result['metrics']['trajectory_mse']
                experts = result['metrics']['mean_active_experts']
                mse_score = mse / baseline_mse  # Lower is better
                expert_score = experts / 4.0  # Assuming 4 is max experts
                balance_score = mse_score + 0.5 * expert_score  # Weight efficiency
                balance_scores.append((config, balance_score, mse, experts))
            
            best_balance = min(balance_scores, key=lambda x: x[1])
            report_lines.append(f"Best Balance: {best_balance[0]} (MSE: {best_balance[2]:.6f}, Experts: {best_balance[3]:.2f})")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {output_path}")


def create_visualization_plots(results: Dict[str, Dict[str, Any]], output_dir: Path):
    """Create visualization plots for ablation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Prepare data
    configs = []
    mse_values = []
    active_experts = []
    entropy_values = []
    inference_times = []
    
    baseline_mse = results.get('baseline', {}).get('metrics', {}).get('trajectory_mse', None)
    
    for config, result in results.items():
        if config != 'baseline' and 'metrics' in result:
            metrics = result['metrics']
            configs.append(config)
            mse_values.append(metrics['trajectory_mse'])
            active_experts.append(metrics.get('mean_active_experts', np.nan))
            entropy_values.append(metrics.get('routing_entropy', np.nan))
            inference_times.append(metrics.get('inference_time_per_traj', 0) * 1000)
    
    # 1. MSE Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, mse_values)
    
    # Color bars based on performance relative to baseline
    if baseline_mse:
        colors = ['green' if mse < baseline_mse else 'red' for mse in mse_values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        ax.axhline(y=baseline_mse, color='blue', linestyle='--', label=f'Baseline ({baseline_mse:.6f})')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Trajectory MSE')
    ax.set_title('Trajectory MSE by Configuration')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Active Experts vs MSE Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(active_experts, mse_values, s=100, alpha=0.6)
    
    # Add labels
    for i, config in enumerate(configs):
        ax.annotate(config, (active_experts[i], mse_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    if baseline_mse:
        baseline_experts = results.get('baseline', {}).get('metrics', {}).get('mean_active_experts', None)
        if baseline_experts:
            ax.scatter(baseline_experts, baseline_mse, s=200, c='red', 
                      marker='*', label='Baseline', zorder=5)
    
    ax.set_xlabel('Mean Active Experts')
    ax.set_ylabel('Trajectory MSE')
    ax.set_title('Efficiency vs Accuracy Trade-off')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Multi-metric Radar Chart
    from math import pi
    
    # Select subset of metrics for radar chart
    metrics_to_plot = ['mse', 'active_experts', 'entropy', 'inference_time']
    metric_labels = ['MSE\n(lower better)', 'Active Experts\n(lower better)', 
                    'Routing Entropy\n(higher better)', 'Inference Time\n(lower better)']
    
    # Normalize metrics to 0-1 scale
    normalized_data = {}
    for config, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            normalized_data[config] = [
                1 - (metrics['trajectory_mse'] / max(mse_values)) if mse_values else 0,  # Invert so higher is better
                1 - (metrics.get('mean_active_experts', 4) / 4),  # Assume max 4 experts
                metrics.get('routing_entropy', 0) / max(entropy_values) if entropy_values else 0,
                1 - (metrics.get('inference_time_per_traj', 0) * 1000 / max(inference_times)) if inference_times else 0
            ]
    
    # Create radar chart for top 5 configs + baseline
    top_configs = sorted([(c, r) for c, r in results.items() if 'metrics' in r], 
                        key=lambda x: x[1]['metrics']['trajectory_mse'])[:6]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    num_vars = len(metrics_to_plot)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    for config, result in top_configs:
        if config in normalized_data:
            values = normalized_data[config]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=config)
            ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison (Top Configurations)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data matrix
    metric_names = ['MSE', 'Active Experts', 'Routing Entropy', 'Inference Time (ms)']
    data_matrix = []
    config_labels = []
    
    for config, result in sorted(results.items(), key=lambda x: x[1].get('metrics', {}).get('trajectory_mse', float('inf'))):
        if 'metrics' in result:
            metrics = result['metrics']
            config_labels.append(config)
            data_matrix.append([
                metrics['trajectory_mse'],
                metrics.get('mean_active_experts', np.nan),
                metrics.get('routing_entropy', np.nan),
                metrics.get('inference_time_per_traj', 0) * 1000
            ])
    
    # Normalize each metric to 0-1 for better visualization
    data_matrix = np.array(data_matrix)
    normalized_matrix = np.zeros_like(data_matrix)
    for i in range(data_matrix.shape[1]):
        col = data_matrix[:, i]
        valid = ~np.isnan(col)
        if np.any(valid):
            min_val = np.nanmin(col)
            max_val = np.nanmax(col)
            if max_val > min_val:
                normalized_matrix[:, i] = (col - min_val) / (max_val - min_val)
                # Invert MSE, active experts, and inference time (lower is better)
                if i in [0, 1, 3]:
                    normalized_matrix[:, i] = 1 - normalized_matrix[:, i]
    
    im = ax.imshow(normalized_matrix.T, aspect='auto', cmap='RdYlGn')
    
    # Set ticks
    ax.set_xticks(np.arange(len(config_labels)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.set_yticklabels(metric_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance (higher is better)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(config_labels)):
        for j in range(len(metric_names)):
            text = ax.text(i, j, f'{data_matrix[i, j]:.3f}', 
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Performance Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models from ablation checkpoints')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints_ablation'),
                        help='Directory containing ablation checkpoints')
    parser.add_argument('--baseline-checkpoint', type=Path, default=Path('checkpoints_test/best_model.pt'),
                        help='Path to baseline model checkpoint')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='System name for data generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--output', type=Path, default=Path('ablation_results/checkpoint_evaluation.json'),
                        help='Output file for detailed results')
    parser.add_argument('--cache-dir', type=Path, default=Path('cache'),
                        help='Cache directory for datasets')
    parser.add_argument('--config', type=Path, default=Path('configs/ablation/base_multiscale.yml'),
                        help='Base configuration file to get data parameters')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = evaluate_all_checkpoints(
        args.checkpoint_dir,
        args.baseline_checkpoint,
        args.system,
        device,
        args.cache_dir
    )
    
    # Print comparison
    print_comparison_table(results)
    
    # Save detailed results
    save_detailed_results(results, args.output)
    
    # Generate report
    report_path = args.output.parent / 'ablation_report.txt'
    generate_report(results, report_path)
    
    # Create visualizations
    viz_dir = args.output.parent / 'ablation_visualizations'
    create_visualization_plots(results, viz_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate statistics across ablations
    mse_values = []
    for config, result in results.items():
        if config != 'baseline' and 'metrics' in result:
            mse_values.append(result['metrics']['trajectory_mse'])
    
    if mse_values:
        print(f"Number of successful evaluations: {len(mse_values)}")
        print(f"Mean MSE across ablations: {np.mean(mse_values):.6f}")
        print(f"Std MSE across ablations: {np.std(mse_values):.6f}")
        print(f"Best MSE: {np.min(mse_values):.6f}")
        print(f"Worst MSE: {np.max(mse_values):.6f}")
        
    print(f"\nResults saved to:")
    print(f"  - JSON: {args.output}")
    print(f"  - Report: {report_path}")
    print(f"  - Visualizations: {viz_dir}/")


if __name__ == '__main__':
    main()