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

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_4panel_figure(checkpoint_path, system_name='multi_scale_oscillators', output_path='ame_ode_4panel.png'):
    """Create 4-panel figure for AME-ODE visualization."""
    try:
        from src.models.ame_ode import AMEODE
        from src.data.synthetic import MultiScaleOscillators
        from src.utils.config import Config
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = Config('configs/quick_test.yml')
        model = AMEODE(config.to_dict()).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate data
        system = MultiScaleOscillators(
            fast_freq=10.0,
            slow_freq=0.1,
            coupling_strength=0.05,
            state_dim=4
        )
        
        # Generate trajectory
        x0 = system.sample_initial_conditions(1).to(device)
        times = torch.linspace(0, 10, 1000).to(device)
        
        # Get predictions
        with torch.no_grad():
            pred_trajectory, info = model(x0, times)
            
        # True trajectory
        true_trajectory = system.generate_trajectory(x0[0].cpu(), times.cpu())
        true_trajectory = true_trajectory.unsqueeze(0).to(device)
        
        # Create figure
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        times_np = times.cpu().numpy()
        true_traj = true_trajectory[0].cpu().numpy()
        pred_traj = pred_trajectory[0].cpu().numpy()
        routing_weights = info['routing_weights'][:, 0, :].cpu().numpy() if 'routing_weights' in info else None
        
        # Panel (a): Ground Truth
        ax1 = fig.add_subplot(gs[0, 0])
        x_fast = true_traj[:, 0]
        x_slow = true_traj[:, 2]
        colors = plt.cm.viridis(np.linspace(0, 1, len(times_np)))
        
        for i in range(len(times_np) - 1):
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
        x_fast_pred = pred_traj[:, 0]
        x_slow_pred = pred_traj[:, 2]
        
        if routing_weights is not None:
            active_expert = np.argmax(routing_weights, axis=1)
            expert_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
            
            for i in range(len(times_np) - 1):
                color = expert_colors.get(active_expert[i], 'gray')
                ax2.plot(x_fast_pred[i:i+2], x_slow_pred[i:i+2], color=color, linewidth=2, alpha=0.8)
            
            for expert_id in np.unique(active_expert):
                ax2.plot([], [], color=expert_colors.get(expert_id, 'gray'), 
                        linewidth=3, label=f'Expert {expert_id}')
            
            avg_active = np.mean(np.sum(routing_weights > 0.1, axis=1))
            sparsity = 1.0 - (avg_active / model.n_experts)
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
        
        if routing_weights is not None:
            for expert_id in range(routing_weights.shape[1]):
                ax3.plot(times_np, routing_weights[:, expert_id], 
                        linewidth=2.5, label=f'$g_{expert_id}(t)$')
            
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            
            for i in range(len(times_np) - 1):
                if routing_weights[i, 0] > 0.5:
                    ax3.axvspan(times_np[i], times_np[i+1], alpha=0.1, color='red')
                elif len(routing_weights[i]) > 1 and routing_weights[i, 1] > 0.5:
                    ax3.axvspan(times_np[i], times_np[i+1], alpha=0.1, color='blue')
        
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Expert Weight', fontsize=12)
        ax3.set_title('(c) Expert Activation Over Time', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([-0.05, 1.05])
        
        # Panel (d): Vector Fields by Expert
        ax4 = fig.add_subplot(gs[1, 1])
        
        grid_size = 20
        x_range = np.linspace(-3, 3, grid_size)
        y_range = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        
        grid_points = torch.stack([
            torch.tensor(X.flatten()),
            torch.zeros(grid_size**2),
            torch.tensor(Y.flatten()),
            torch.zeros(grid_size**2)
        ], dim=-1).float().to(device)
        
        t = torch.zeros(grid_points.shape[0]).to(device)
        
        with torch.no_grad():
            expert_dynamics = []
            for expert_id in range(min(2, model.n_experts)):
                dynamics = model.experts.experts[expert_id](t, grid_points)
                expert_dynamics.append(dynamics.cpu().numpy())
        
        for expert_id, dynamics in enumerate(expert_dynamics):
            dx_fast = dynamics[:, 0].reshape(grid_size, grid_size)
            dx_slow = dynamics[:, 2].reshape(grid_size, grid_size)
            
            magnitude = np.sqrt(dx_fast**2 + dx_slow**2)
            dx_fast_norm = dx_fast / (magnitude + 1e-6)
            dx_slow_norm = dx_slow / (magnitude + 1e-6)
            
            skip = 2
            mask = X < 0 if expert_id == 0 else X >= 0
            ax4.quiver(X[mask][::skip, ::skip], Y[mask][::skip, ::skip],
                      dx_fast_norm[mask][::skip, ::skip], dx_slow_norm[mask][::skip, ::skip],
                      magnitude[mask][::skip, ::skip], 
                      cmap='Reds' if expert_id == 0 else 'Blues',
                      alpha=0.6, scale=25, width=0.003)
            
            ax4.text(-1.5 + expert_id * 3, 2.5, f'Expert {expert_id}',
                    bbox=dict(boxstyle='round', facecolor='red' if expert_id == 0 else 'blue', alpha=0.3),
                    fontsize=12, fontweight='bold')
        
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax4.set_xlabel('$x_{fast}$', fontsize=12)
        ax4.set_ylabel('$x_{slow}$', fontsize=12)
        ax4.set_title('(d) Vector Fields by Expert', fontsize=14, fontweight='bold')
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 3])
        ax4.grid(True, alpha=0.3)
        
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
            ame_checkpoint = Path(args.checkpoint_dir) / 'best_model.pt'
            if ame_checkpoint.exists():
                print("\nGenerating 4-panel figure for AME-ODE...")
                create_4panel_figure(
                    ame_checkpoint, 
                    args.system,
                    args.figure_output
                )
            else:
                print(f"\nNote: AME-ODE checkpoint not found at {ame_checkpoint}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()