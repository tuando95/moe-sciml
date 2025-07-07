#!/usr/bin/env python3
"""Run ablation studies for AME-ODE using configuration files."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
import subprocess
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def load_config_with_includes(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file, handling !include directives."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Simple include processing
    if '!include' in content:
        base_dir = config_path.parent
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith('!include'):
                include_file = line.strip().split()[1]
                include_path = base_dir / include_file
                with open(include_path, 'r') as inc_f:
                    included_content = inc_f.read()
                processed_lines.append(f"# Included from {include_file}")
                processed_lines.append(included_content)
            else:
                processed_lines.append(line)
        
        content = '\n'.join(processed_lines)
    
    return yaml.safe_load(content)


def run_single_experiment(config_path: Path, gpu_id: int = 0) -> Dict[str, Any]:
    """Run a single experiment using the training script."""
    # Set GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Extract experiment name from config path
    exp_name = config_path.stem
    
    # Run training
    print(f"\nRunning experiment: {exp_name} on GPU {gpu_id}")
    start_time = time.time()
    
    # Get system name from config
    config = load_config_with_includes(config_path)
    system_name = 'multi_scale_oscillators'  # Default
    if 'data' in config and 'synthetic_systems' in config['data']:
        for sys_config in config['data']['synthetic_systems']:
            if sys_config.get('enabled', False):
                system_name = sys_config['name']
                break
    
    cmd = [
        'python', 'train_progressive.py',
        '--config', str(config_path),
        '--system', system_name,
        '--device', 'cuda'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        training_time = time.time() - start_time
        
        # Parse output to extract final metrics
        metrics = parse_training_output(result.stdout)
        metrics['training_time'] = training_time
        metrics['status'] = 'success'
        
        return {
            'config': str(config_path),
            'experiment': exp_name,
            'metrics': metrics,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'config': str(config_path),
            'experiment': exp_name,
            'status': 'failed',
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def parse_training_output(output: str) -> Dict[str, float]:
    """Parse training output to extract final metrics."""
    metrics = {}
    
    # Look for final test metrics in the output
    lines = output.split('\n')
    
    for line in lines:
        # Look for test reconstruction loss
        if 'Test Reconstruction Loss:' in line:
            try:
                # Format: "Test Reconstruction Loss: 0.123456 ± 0.001234"
                parts = line.split(':')[1].strip().split('±')
                metrics['trajectory_mse'] = float(parts[0].strip())
                if len(parts) > 1:
                    metrics['test_mse_std'] = float(parts[1].strip())
            except:
                pass
        
        # Look for best validation reconstruction
        if 'Val Recon:' in line and '(best:' in line:
            try:
                # Format: "Val Recon: 0.1234 (best: 0.1234)"
                val_part = line.split('Val Recon:')[1].split('(best:')[0]
                metrics['val_reconstruction'] = float(val_part.strip())
            except:
                pass
        
        # Look for routing entropy
        if 'Routing Entropy:' in line:
            try:
                metrics['routing_entropy'] = float(line.split(':')[1].strip())
            except:
                pass
        
        # Look for expert usage variance
        if 'Expert Usage Var:' in line:
            try:
                metrics['expert_usage_var'] = float(line.split(':')[1].strip())
            except:
                pass
        
        # Look for number of active experts
        if 'Active experts' in line:
            try:
                # Extract from training output if available
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'experts:' and i + 1 < len(parts):
                        metrics['mean_active_experts'] = float(parts[i + 1])
            except:
                pass
    
    # Also extract best validation loss if available
    for line in lines:
        if 'Best validation reconstruction loss:' in line:
            try:
                metrics['best_val_reconstruction'] = float(line.split(':')[1].strip())
            except:
                pass
    
    return metrics


def run_ablation_category(
    category: str,
    config_dir: Path,
    num_gpus: int = 1,
    output_dir: Path = Path('ablation_results')
) -> Dict[str, Any]:
    """Run all ablations in a category."""
    # Find all config files for this category
    if category == 'all':
        config_files = list(config_dir.glob('*.yml'))
    else:
        config_files = list(config_dir.glob(f'{category}*.yml'))
    
    if not config_files:
        print(f"No config files found for category: {category}")
        return {}
    
    print(f"\nFound {len(config_files)} configurations for {category}")
    for cf in config_files:
        print(f"  - {cf.name}")
    
    # Create output directory
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments in parallel if multiple GPUs available
    results = {}
    
    if num_gpus > 1:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            # Submit jobs
            future_to_config = {}
            for i, config_file in enumerate(config_files):
                gpu_id = i % num_gpus
                future = executor.submit(run_single_experiment, config_file, gpu_id)
                future_to_config[future] = config_file
            
            # Collect results
            for future in as_completed(future_to_config):
                config_file = future_to_config[future]
                try:
                    result = future.result()
                    results[config_file.stem] = result
                    
                    # Save individual result
                    result_path = category_dir / f"{config_file.stem}_result.json"
                    with open(result_path, 'w') as f:
                        # Remove stdout/stderr from saved file to keep it readable
                        save_result = {k: v for k, v in result.items() if k not in ['stdout', 'stderr']}
                        json.dump(save_result, f, indent=2)
                        
                except Exception as e:
                    print(f"Error running {config_file}: {e}")
                    results[config_file.stem] = {'status': 'error', 'error': str(e)}
    else:
        # Run sequentially
        for config_file in config_files:
            result = run_single_experiment(config_file, 0)
            results[config_file.stem] = result
            
            # Save individual result
            result_path = category_dir / f"{config_file.stem}_result.json"
            with open(result_path, 'w') as f:
                save_result = {k: v for k, v in result.items() if k not in ['stdout', 'stderr']}
                json.dump(save_result, f, indent=2)
    
    # Save summary
    summary_path = category_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_category_summary(category, results)
    
    return results


def print_category_summary(category: str, results: Dict[str, Any]):
    """Print summary of ablation results for a category."""
    print("\n" + "="*80)
    print(f"ABLATION CATEGORY SUMMARY: {category}")
    print("="*80)
    
    # Find baseline
    baseline_name = 'base_multiscale'
    if baseline_name not in results:
        # Try to find the most basic configuration as baseline
        for name in ['experts_k4', 'temp_1.0', 'base']:
            if name in results:
                baseline_name = name
                break
    
    # Key metrics to compare
    key_metrics = ['trajectory_mse', 'mean_active_experts', 'routing_entropy', 'training_time']
    
    # Print table header
    print(f"\n{'Configuration':<30} " + " ".join(f"{m:<18}" for m in key_metrics))
    print("-" * (30 + 19 * len(key_metrics)))
    
    # Get baseline metrics
    baseline_metrics = {}
    if baseline_name in results and results[baseline_name].get('status') == 'success':
        baseline_metrics = results[baseline_name].get('metrics', {})
    
    # Print results sorted by name
    sorted_configs = sorted(results.keys())
    for config_name in sorted_configs:
        result = results[config_name]
        
        if result.get('status') == 'success':
            metrics = result.get('metrics', {})
            row = f"{config_name[:30]:<30} "
            
            for metric in key_metrics:
                value = metrics.get(metric, 0)
                
                # Format value
                if metric == 'training_time':
                    value_str = f"{value:.1f}s"
                else:
                    value_str = f"{value:.4f}"
                
                # Show relative change from baseline
                if config_name != baseline_name and metric in baseline_metrics:
                    baseline_val = baseline_metrics[metric]
                    if baseline_val > 0:
                        rel_change = (value - baseline_val) / baseline_val * 100
                        row += f"{value_str} ({rel_change:+.1f}%)"
                    else:
                        row += f"{value_str} (N/A)"
                else:
                    row += f"{value_str}"
                
                row += " " * (19 - len(row.split()[-1]))
            
            print(row)
        else:
            print(f"{config_name[:30]:<30} FAILED: {result.get('error', 'Unknown error')}")
    
    print("="*80)


def compare_ablation_categories(results_dir: Path):
    """Compare results across different ablation categories."""
    print("\n" + "="*80)
    print("CROSS-CATEGORY COMPARISON")
    print("="*80)
    
    # Load all summaries
    all_results = {}
    for category_dir in results_dir.iterdir():
        if category_dir.is_dir():
            summary_file = category_dir / 'summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    all_results[category_dir.name] = json.load(f)
    
    # Find best configuration in each category
    print("\nBest configurations by trajectory MSE:")
    print("-" * 60)
    
    best_configs = []
    for category, results in all_results.items():
        best_mse = float('inf')
        best_config = None
        
        for config_name, result in results.items():
            if result.get('status') == 'success':
                mse = result.get('metrics', {}).get('trajectory_mse', float('inf'))
                if mse < best_mse:
                    best_mse = mse
                    best_config = config_name
        
        if best_config:
            best_configs.append({
                'category': category,
                'config': best_config,
                'mse': best_mse,
                'metrics': results[best_config].get('metrics', {})
            })
            print(f"{category:<20} {best_config:<30} MSE: {best_mse:.6f}")
    
    # Overall best
    if best_configs:
        overall_best = min(best_configs, key=lambda x: x['mse'])
        print(f"\nOverall best: {overall_best['config']} from {overall_best['category']} (MSE: {overall_best['mse']:.6f})")


def main():
    parser = argparse.ArgumentParser(description='Run AME-ODE ablation studies using config files')
    parser.add_argument('--category', type=str, default='all',
                        choices=['all', 'routing', 'experts', 'temp', 'reg'],
                        help='Category of ablations to run')
    parser.add_argument('--config-dir', type=Path, default=Path('configs/ablation'),
                        help='Directory containing ablation config files')
    parser.add_argument('--output-dir', type=Path, default=Path('ablation_results'),
                        help='Output directory for results')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use for parallel execution')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results across categories')
    
    args = parser.parse_args()
    
    if args.compare:
        # Just compare existing results
        compare_ablation_categories(args.output_dir)
    else:
        # Run ablations
        if args.category == 'all':
            # Run all categories
            categories = ['routing', 'experts', 'temp', 'reg']
            all_results = {}
            
            for category in categories:
                print(f"\n{'='*80}")
                print(f"Running {category} ablations...")
                print('='*80)
                
                results = run_ablation_category(
                    category,
                    args.config_dir,
                    args.num_gpus,
                    args.output_dir
                )
                all_results[category] = results
            
            # Final comparison
            compare_ablation_categories(args.output_dir)
        else:
            # Run single category
            run_ablation_category(
                args.category,
                args.config_dir,
                args.num_gpus,
                args.output_dir
            )


if __name__ == '__main__':
    main()