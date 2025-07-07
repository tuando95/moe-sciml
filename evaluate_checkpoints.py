#!/usr/bin/env python3
"""Evaluate all models from saved checkpoints with optimized inference."""

import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

def print_detailed_results(results_file):
    """Print detailed metrics from results file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 80)
    
    # Print results for each model
    for model_key, model_data in sorted(results.items()):
        if isinstance(model_data, dict) and 'test_metrics' in model_data:
            print(f"\n{model_data.get('model', model_key.upper())}")
            print("-" * 60)
            print(f"Parameters: {model_data.get('total_params', 'N/A'):,}")
            print(f"Test Loss: {model_data.get('test_loss', 'N/A'):.6f}")
            
            # Print all available metrics
            if 'test_metrics' in model_data:
                print("\nDetailed Metrics:")
                for metric_name, metric_data in sorted(model_data['test_metrics'].items()):
                    if isinstance(metric_data, dict) and 'mean' in metric_data:
                        print(f"  {metric_name}:")
                        print(f"    Mean: {metric_data['mean']:.6f}")
                        print(f"    Std:  {metric_data['std']:.6f}")
                        print(f"    Min:  {metric_data['min']:.6f}")
                        print(f"    Max:  {metric_data['max']:.6f}")
                    else:
                        print(f"  {metric_name}: {metric_data}")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find best model for each metric
    metrics_summary = {}
    for model_key, model_data in results.items():
        if 'test_metrics' in model_data:
            for metric_name, metric_data in model_data['test_metrics'].items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    if metric_name not in metrics_summary:
                        metrics_summary[metric_name] = []
                    metrics_summary[metric_name].append({
                        'model': model_data.get('model', model_key),
                        'value': metric_data['mean']
                    })
    
    # Print best model for each metric
    for metric_name, model_values in sorted(metrics_summary.items()):
        if model_values:
            # Lower is better for most metrics
            best = min(model_values, key=lambda x: x['value'])
            print(f"\nBest {metric_name}: {best['model']} ({best['value']:.6f})")
            
            # Print ranking
            ranked = sorted(model_values, key=lambda x: x['value'])
            for i, item in enumerate(ranked):
                print(f"  {i+1}. {item['model']}: {item['value']:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate models from checkpoints')
    parser.add_argument('--config', type=str, default='configs/quick_test.yml',
                        help='Configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_test',
                        help='Directory containing checkpoints')
    parser.add_argument('--fast-inference', action='store_true',
                        help='Use fast inference mode for AME-ODE')
    
    args = parser.parse_args()
    
    # Run the comparison script with checkpoint evaluation
    cmd = [
        sys.executable, 'compare_baselines.py',
        '--config', args.config,
        '--system', args.system,
        '--use-checkpoints',
        '--checkpoint-dir', args.checkpoint_dir
    ]
    
    if args.fast_inference:
        cmd.append('--fast-inference')
    
    print("Evaluating all models from checkpoints...")
    if args.fast_inference:
        print("Using fast inference mode for AME-ODE")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        
        # Find and display the latest results file
        results_dir = Path('results') / 'baselines'
        result_files = list(results_dir.glob(f"checkpoint_evaluation_{args.system}_*.json"))
        if result_files:
            latest_results = max(result_files, key=lambda p: p.stat().st_mtime)
            print_detailed_results(latest_results)
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()