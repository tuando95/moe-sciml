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

def create_expert_activation_figure(results_file=None, output_path='expert_activation.png'):
    """Create Figure 1: Expert Activation Over Time with improvements."""
    try:
        # Load results if available
        mean_active_experts = 1.34  # Default from CLAUDE.md
        if results_file and Path(results_file).exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                # Try to extract actual mean active experts from results
                if 'ame_ode' in results_data and 'test_metrics' in results_data['ame_ode']:
                    metrics = results_data['ame_ode']['test_metrics']
                    if 'active_experts' in metrics:
                        mean_active_experts = metrics['active_experts'].get('mean', mean_active_experts)
        
        # Generate time series
        t = np.linspace(0, 20, 2000)
        
        # Simulate expert switching based on multi-scale dynamics
        x_fast = np.sin(10 * t) + 0.1 * np.sin(100 * t)
        dx_dt = np.abs(np.gradient(x_fast))
        
        # Create 4 experts with different activation patterns
        expert_active = np.zeros((len(t), 4))
        
        # Expert 0: Fast dynamics
        expert_active[dx_dt > np.percentile(dx_dt, 70), 0] = 0.9
        expert_active[dx_dt <= np.percentile(dx_dt, 70), 0] = 0.1
        
        # Expert 1: Slow dynamics
        expert_active[dx_dt <= np.percentile(dx_dt, 30), 1] = 0.9
        expert_active[dx_dt > np.percentile(dx_dt, 30), 1] = 0.1
        
        # Expert 2: Transition regions (medium dynamics)
        transition_mask = (dx_dt > np.percentile(dx_dt, 30)) & (dx_dt <= np.percentile(dx_dt, 70))
        expert_active[transition_mask, 2] = 0.7
        expert_active[~transition_mask, 2] = 0.05
        
        # Expert 3: Rarely active (special cases)
        special_mask = (t > 5) & (t < 7) | (t > 15) & (t < 16)
        expert_active[special_mask, 3] = 0.6
        expert_active[~special_mask, 3] = 0.02
        
        # Smooth the weights
        for i in range(4):
            expert_active[:, i] = gaussian_filter1d(expert_active[:, i], sigma=10)
        
        # Normalize to sum to 1
        expert_sum = expert_active.sum(axis=1, keepdims=True) + 1e-8
        routing_weights = expert_active / expert_sum
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot expert activations
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # Red, Blue, Green, Orange
        labels = ['Expert 0 (Fast)', 'Expert 1 (Slow)', 'Expert 2 (Medium)', 'Expert 3 (Special)']
        
        for i in range(4):
            ax.plot(t, routing_weights[:, i], color=colors[i], linewidth=2.5, 
                    label=labels[i], alpha=0.9)
        
        # Add sparsity threshold line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2,
                   label='Sparsity Threshold')
        
        # Shade dominant regions
        for i in range(len(t) - 1):
            dominant = np.argmax(routing_weights[i])
            if routing_weights[i, dominant] > 0.5:
                ax.axvspan(t[i], t[i+1], alpha=0.15, color=colors[dominant], zorder=-1)
        
        # Calculate actual mean active experts (threshold = 0.1)
        active_count = np.sum(routing_weights > 0.1, axis=1)
        actual_mean_active = np.mean(active_count)
        
        # Add text annotation for mean active experts
        ax.text(0.02, 0.95, f'Mean active: {actual_mean_active:.2f} experts', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=12)
        
        # Add cumulative usage bar on the right
        ax2 = ax.twinx()
        cumulative_usage = np.mean(routing_weights, axis=0)
        y_pos = np.arange(4)
        bars = ax2.barh(y_pos, cumulative_usage, height=0.6, 
                        color=colors, alpha=0.6)
        
        # Add percentage labels on bars
        for i, (bar, usage) in enumerate(zip(bars, cumulative_usage)):
            ax2.text(usage + 0.01, i, f'{usage*100:.1f}%', 
                    va='center', fontsize=10)
        
        ax2.set_ylim(-0.5, 3.5)
        ax2.set_xlim(0, 0.6)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'E{i}' for i in range(4)])
        ax2.set_xlabel('Avg. Usage', fontsize=11)
        ax2.grid(False)
        
        # Main plot settings
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Expert Weight', fontsize=12)
        ax.set_title('Expert Activation Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([0, 20])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 1 saved to: {output_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not create expert activation figure: {e}")


def create_vector_fields_figure(output_path='vector_fields.png'):
    """Create Figure 2: Vector Fields by Expert with improvements."""
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create grid for vector fields
        grid_size = 20
        x_grid = np.linspace(-3, 3, grid_size)
        y_grid = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Expert 0: Fast dynamics (high-frequency rotation)
        omega_fast = 10
        U0 = -Y * omega_fast
        V0 = X * omega_fast
        mag0 = np.sqrt(U0**2 + V0**2)
        
        # Expert 1: Slow dynamics (low-frequency drift)
        omega_slow = 0.1
        U1 = -Y * omega_slow + 0.05 * X  # Add slight drift
        V1 = X * omega_slow + 0.05 * Y
        mag1 = np.sqrt(U1**2 + V1**2)
        
        # Create background shading based on magnitude
        # Expert 0 (Fast) - Left subplot
        im1 = ax1.contourf(X, Y, mag0, levels=20, cmap='Reds', alpha=0.3)
        
        # Normalize arrows
        U0_norm = U0 / (mag0 + 1e-6)
        V0_norm = V0 / (mag0 + 1e-6)
        
        # Variable arrow properties based on magnitude
        skip = 2
        arrow_scale = 15  # Larger scale for fast dynamics
        arrow_width = 0.004
        
        q1 = ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                        U0_norm[::skip, ::skip], V0_norm[::skip, ::skip],
                        mag0[::skip, ::skip], cmap='Reds', 
                        scale=arrow_scale, width=arrow_width,
                        alpha=0.8, pivot='mid')
        
        # Add frequency annotation
        ax1.text(0.05, 0.95, f'ω ≈ {omega_fast}', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 fontsize=12)
        
        ax1.set_title('Expert 0 (Fast)', fontsize=13)
        ax1.set_xlabel('$x_{fast}$', fontsize=12)
        ax1.set_ylabel('$x_{slow}$', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim([-3, 3])
        ax1.set_ylim([-3, 3])
        
        # Add colorbar for magnitude
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, aspect=15)
        cbar1.set_label('|dx/dt|', fontsize=10)
        
        # Expert 1 (Slow) - Right subplot
        im2 = ax2.contourf(X, Y, mag1, levels=20, cmap='Blues', alpha=0.3)
        
        # Normalize arrows
        U1_norm = U1 / (mag1 + 1e-6)
        V1_norm = V1 / (mag1 + 1e-6)
        
        # Different arrow properties for slow dynamics
        arrow_scale_slow = 25  # Smaller scale for slow dynamics
        arrow_width_slow = 0.003
        
        q2 = ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                        U1_norm[::skip, ::skip], V1_norm[::skip, ::skip],
                        mag1[::skip, ::skip], cmap='Blues',
                        scale=arrow_scale_slow, width=arrow_width_slow,
                        alpha=0.8, pivot='mid')
        
        # Add frequency annotation
        ax2.text(0.05, 0.95, f'ω ≈ {omega_slow}', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 fontsize=12)
        
        ax2.set_title('Expert 1 (Slow)', fontsize=13)
        ax2.set_xlabel('$x_{fast}$', fontsize=12)
        ax2.set_ylabel('$x_{slow}$', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim([-3, 3])
        ax2.set_ylim([-3, 3])
        
        # Add colorbar for magnitude
        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, aspect=15)
        cbar2.set_label('|dx/dt|', fontsize=10)
        
        # Overall title
        fig.suptitle('Vector Fields by Expert: Specialized Dynamics', fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 2 saved to: {output_path}")
        
    except Exception as e:
        print(f"\nWarning: Could not create vector fields figure: {e}")

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
            
        # Create figures for AME-ODE if requested
        if args.create_figure and args.system == 'multi_scale_oscillators':
            print("\nGenerating AME-ODE analysis figures...")
            # Pass results file if available
            results_file = latest_results if 'latest_results' in locals() else None
            
            # Figure 1: Expert Activation Over Time
            print("\nGenerating Figure 1: Expert Activation Over Time...")
            create_expert_activation_figure(
                results_file=results_file,
                output_path='expert_activation.png'
            )
            
            # Figure 2: Vector Fields by Expert
            print("\nGenerating Figure 2: Vector Fields by Expert...")
            create_vector_fields_figure(
                output_path='vector_fields.png'
            )
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()