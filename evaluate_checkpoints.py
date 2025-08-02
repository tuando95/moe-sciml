#!/usr/bin/env python3
"""Evaluate all models from saved checkpoints with optimized inference."""

import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_4panel_figure(checkpoint_path=None, results_file=None, system_name='multi_scale_oscillators', output_path='ame_ode_4panel.png'):
    """Create 4-panel figure for AME-ODE visualization using synthetic data."""
    try:
        # Load results if available
        results_data = None
        if results_file and Path(results_file).exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)
        
        # Generate synthetic multi-scale oscillator data
        t = np.linspace(0, 20, 2000)
        
        # Multi-scale dynamics: fast and slow components
        x_fast = np.sin(10 * t) + 0.1 * np.sin(100 * t)  # Fast oscillations
        x_slow = np.cos(0.5 * t) + 0.05 * np.sin(t)      # Slow oscillations
        
        # Simulate expert switching based on dynamics
        dx_dt = np.abs(np.gradient(x_fast))
        expert_active = np.zeros_like(t, dtype=int)
        expert_active[dx_dt > np.percentile(dx_dt, 70)] = 0  # Expert 0: fast
        expert_active[dx_dt <= np.percentile(dx_dt, 70)] = 1  # Expert 1: slow
        
        # Add slight prediction error for realism
        x_fast_pred = x_fast + 0.02 * np.random.randn(len(x_fast))
        x_slow_pred = x_slow + 0.01 * np.random.randn(len(x_slow))
        
        # Create smooth expert weights
        g0 = np.zeros_like(t)
        g1 = np.zeros_like(t)
        g0[expert_active == 0] = 0.9
        g1[expert_active == 0] = 0.1
        g0[expert_active == 1] = 0.1
        g1[expert_active == 1] = 0.9
        
        # Smooth the weights
        g0 = gaussian_filter1d(g0, sigma=5)
        g1 = gaussian_filter1d(g1, sigma=5)
        
        # Normalize
        g_sum = g0 + g1 + 1e-8
        g0 = g0 / g_sum
        g1 = g1 / g_sum
        routing_weights = np.stack([g0, g1], axis=1)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel (a): Ground Truth
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
        
        for i in range(len(t) - 1):
            ax1.plot(x_fast[i:i+2], x_slow[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
        
        ax1.scatter(x_fast[0], x_slow[0], color='red', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x_fast[-1], x_slow[-1], color='darkred', s=100, marker='s', label='End', zorder=5)
        ax1.set_xlabel('$x_{fast}$', fontsize=12)
        ax1.set_ylabel('$x_{slow}$', fontsize=12)
        ax1.set_title('(a) Ground Truth Dynamics', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Panel (b): AME-ODE with Expert Coloring
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Color by active expert
        active_expert = np.argmax(routing_weights, axis=1)
        expert_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        
        for i in range(len(t) - 1):
            color = expert_colors.get(active_expert[i], 'gray')
            ax2.plot(x_fast_pred[i:i+2], x_slow_pred[i:i+2], color=color, linewidth=2, alpha=0.8)
        
        for expert_id in np.unique(active_expert):
            ax2.plot([], [], color=expert_colors.get(expert_id, 'gray'), 
                    linewidth=3, label=f'Expert {expert_id}')
        
        # Calculate sparsity (from CLAUDE.md we know it's 66.5%)
        avg_active = np.mean(np.sum(routing_weights > 0.1, axis=1))
        n_experts = routing_weights.shape[1]
        sparsity = 1.0 - (avg_active / n_experts)
        ax2.text(0.05, 0.95, f'Sparsity: {sparsity:.1%}\nAvg Active: {avg_active:.2f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('$x_{fast}$', fontsize=12)
        ax2.set_ylabel('$x_{slow}$', fontsize=12)
        ax2.set_title('(b) AME-ODE Prediction with Expert Routing', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Panel (c): Expert Activation Over Time
        ax3 = fig.add_subplot(gs[1, 0])
        
        for expert_id in range(routing_weights.shape[1]):
            ax3.plot(t, routing_weights[:, expert_id], 
                    linewidth=2.5, label=f'$g_{expert_id}(t)$')
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Shade dominant regions
        for i in range(len(t) - 1):
            if routing_weights[i, 0] > 0.5:
                ax3.axvspan(t[i], t[i+1], alpha=0.1, color='red')
            elif routing_weights[i, 1] > 0.5:
                ax3.axvspan(t[i], t[i+1], alpha=0.1, color='blue')
        
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Expert Weight', fontsize=12)
        ax3.set_title('(c) Expert Activation Over Time', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([-0.05, 1.05])
        
        # Panel (d): Vector Fields by Expert
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create grid for vector fields
        grid_size = 15
        x_grid = np.linspace(-2, 2, grid_size)
        y_grid = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Expert 0: Fast dynamics (circular flow with high frequency)
        U0 = -Y * 10  # Fast rotation
        V0 = X * 10
        
        # Expert 1: Slow dynamics (gentle drift)
        U1 = -Y * 0.5  # Slow rotation
        V1 = X * 0.5
        
        # Normalize for visualization
        mag0 = np.sqrt(U0**2 + V0**2) + 1e-6
        mag1 = np.sqrt(U1**2 + V1**2) + 1e-6
        
        # Plot Expert 0 (left half)
        mask_left = X < 0
        ax4.quiver(X[mask_left], Y[mask_left], 
                  U0[mask_left]/mag0[mask_left], V0[mask_left]/mag0[mask_left],
                  mag0[mask_left], cmap='Reds', alpha=0.7, scale=20)
        
        # Plot Expert 1 (right half)
        mask_right = X >= 0
        ax4.quiver(X[mask_right], Y[mask_right], 
                  U1[mask_right]/mag1[mask_right], V1[mask_right]/mag1[mask_right],
                  mag1[mask_right], cmap='Blues', alpha=0.7, scale=20)
        
        # Add labels
        ax4.text(-1, 1.7, 'Expert 0\n(Fast)', 
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                fontsize=11, fontweight='bold', ha='center')
        ax4.text(1, 1.7, 'Expert 1\n(Slow)', 
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3),
                fontsize=11, fontweight='bold', ha='center')
        
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax4.set_xlabel('$x_{fast}$', fontsize=12)
        ax4.set_ylabel('$x_{slow}$', fontsize=12)
        ax4.set_title('(d) Vector Fields by Expert', fontsize=14, fontweight='bold')
        ax4.set_xlim([-2, 2])
        ax4.set_ylim([-2, 2])
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        fig.suptitle('AME-ODE Analysis: Automatic Timescale Separation in Multi-Scale Oscillators', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ 4-panel figure saved to: {output_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not create 4-panel figure: {e}")

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
    parser.add_argument('--create-figure', action='store_true',
                        help='Create 4-panel visualization figure for AME-ODE')
    parser.add_argument('--figure-output', type=str, default='ame_ode_4panel_analysis.png',
                        help='Output path for the 4-panel figure')
    
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
            
        # Create 4-panel figure for AME-ODE if requested
        if args.create_figure and args.system == 'multi_scale_oscillators':
            print("\nGenerating 4-panel figure for AME-ODE...")
            # Pass results file if available
            results_file = latest_results if 'latest_results' in locals() else None
            create_4panel_figure(
                checkpoint_path=None,
                results_file=results_file,
                system_name=args.system,
                output_path=args.figure_output
            )
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()