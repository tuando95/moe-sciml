#!/usr/bin/env python3
"""Create 4-panel figure for AME-ODE evaluation on multi-scale oscillators."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
import json
from typing import Dict, Any, Tuple, Optional
import seaborn as sns

from src.models.ame_ode import AMEODE
from src.data.synthetic import MultiScaleOscillators
from src.utils.config import Config
from src.evaluation.metrics import evaluate_model


def load_checkpoint_and_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> AMEODE:
    """Load AME-ODE model from checkpoint."""
    config = Config(config_path)
    model = AMEODE(config.to_dict()).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def generate_trajectory_data(
    model: AMEODE,
    system: MultiScaleOscillators,
    n_trajectories: int = 1,
    time_span: float = 10.0,
    n_time_points: int = 1000,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """Generate trajectory data for visualization."""
    # Generate initial conditions
    x0 = system.sample_initial_conditions(n_trajectories).to(device)
    
    # Time points
    times = torch.linspace(0, time_span, n_time_points).to(device)
    
    # Generate true trajectory
    true_trajectory = system.generate_trajectory(x0[0].cpu(), times.cpu())
    true_trajectory = true_trajectory.unsqueeze(0).to(device)  # Add batch dim
    
    # Get AME-ODE prediction with detailed info
    with torch.no_grad():
        pred_trajectory, info = model(x0, times)
        
    return {
        'times': times,
        'x0': x0,
        'true_trajectory': true_trajectory,
        'pred_trajectory': pred_trajectory,
        'routing_weights': info.get('routing_weights', None),
        'expert_dynamics': info.get('expert_dynamics', None),
    }


def create_4panel_figure(
    data: Dict[str, torch.Tensor],
    model: AMEODE,
    save_path: Path,
    figsize: Tuple[float, float] = (12, 10)
):
    """Create the 4-panel figure as specified."""
    # Set up the figure with custom layout
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    times = data['times'].cpu().numpy()
    true_traj = data['true_trajectory'][0].cpu().numpy()  # First trajectory
    pred_traj = data['pred_trajectory'][0].cpu().numpy()
    routing_weights = data['routing_weights'][:, 0, :].cpu().numpy() if data['routing_weights'] is not None else None
    
    # Assuming 4D state: [x_fast1, x_fast2, x_slow1, x_slow2]
    # For visualization, we'll use x_fast1 vs x_slow1
    x_fast = true_traj[:, 0]  # Fast component
    x_slow = true_traj[:, 2]  # Slow component
    x_fast_pred = pred_traj[:, 0]
    x_slow_pred = pred_traj[:, 2]
    
    # === Panel (a): Ground Truth Dynamics ===
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create time colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    # Plot trajectory with time gradient
    for i in range(len(times) - 1):
        ax1.plot(x_fast[i:i+2], x_slow[i:i+2], color=colors[i], linewidth=2, alpha=0.8)
    
    # Add start and end markers
    ax1.scatter(x_fast[0], x_slow[0], color='red', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x_fast[-1], x_slow[-1], color='darkred', s=100, marker='s', label='End', zorder=5)
    
    ax1.set_xlabel('$x_{fast}$', fontsize=12)
    ax1.set_ylabel('$x_{slow}$', fontsize=12)
    ax1.set_title('(a) Ground Truth Dynamics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=times[-1]))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, pad=0.02, aspect=15)
    cbar1.set_label('Time', fontsize=10)
    
    # === Panel (b): AME-ODE Prediction with Expert Coloring ===
    ax2 = fig.add_subplot(gs[0, 1])
    
    if routing_weights is not None:
        # Determine active expert at each time point
        active_expert = np.argmax(routing_weights, axis=1)
        
        # Color scheme: Expert 0 (fast) = red, Expert 1 (slow) = blue
        expert_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}  # Support up to 4 experts
        
        # Plot trajectory colored by active expert
        for i in range(len(times) - 1):
            color = expert_colors.get(active_expert[i], 'gray')
            ax2.plot(x_fast_pred[i:i+2], x_slow_pred[i:i+2], 
                    color=color, linewidth=2, alpha=0.8)
        
        # Add expert legend
        for expert_id in np.unique(active_expert):
            ax2.plot([], [], color=expert_colors.get(expert_id, 'gray'), 
                    linewidth=3, label=f'Expert {expert_id}')
        
        # Calculate sparsity
        avg_active_experts = np.mean(np.sum(routing_weights > 0.1, axis=1))
        sparsity = 1.0 - (avg_active_experts / model.n_experts)
        
        ax2.text(0.05, 0.95, f'Sparsity: {sparsity:.1%}\nAvg Active: {avg_active_experts:.2f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Fallback if routing weights not available
        ax2.plot(x_fast_pred, x_slow_pred, 'r-', linewidth=2, label='Prediction')
    
    ax2.set_xlabel('$x_{fast}$', fontsize=12)
    ax2.set_ylabel('$x_{slow}$', fontsize=12)
    ax2.set_title('(b) AME-ODE Prediction with Expert Routing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # === Panel (c): Expert Activation Over Time ===
    ax3 = fig.add_subplot(gs[1, 0])
    
    if routing_weights is not None:
        # Plot routing weights over time
        for expert_id in range(routing_weights.shape[1]):
            ax3.plot(times, routing_weights[:, expert_id], 
                    linewidth=2.5, label=f'$g_{expert_id}(t)$')
        
        # Highlight sparse switching behavior
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add shaded regions to show dominant expert
        for i in range(len(times) - 1):
            if routing_weights[i, 0] > 0.5:  # Expert 0 dominant
                ax3.axvspan(times[i], times[i+1], alpha=0.1, color='red')
            elif routing_weights[i, 1] > 0.5:  # Expert 1 dominant
                ax3.axvspan(times[i], times[i+1], alpha=0.1, color='blue')
    
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Expert Weight', fontsize=12)
    ax3.set_title('(c) Expert Activation Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_ylim([-0.05, 1.05])
    
    # === Panel (d): Vector Fields by Expert ===
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create grid for vector field
    grid_size = 20
    x_range = np.linspace(-3, 3, grid_size)
    y_range = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Prepare grid points for model
    grid_points = torch.stack([
        torch.tensor(X.flatten()),
        torch.zeros(grid_size**2),  # x_fast2
        torch.tensor(Y.flatten()),
        torch.zeros(grid_size**2)   # x_slow2
    ], dim=-1).float().to(next(model.parameters()).device)
    
    t = torch.zeros(grid_points.shape[0]).to(grid_points.device)
    
    # Get dynamics from each expert
    with torch.no_grad():
        # Get individual expert dynamics
        expert_dynamics = []
        for expert_id in range(min(2, model.n_experts)):  # Show first 2 experts
            dynamics = model.experts.experts[expert_id](t, grid_points)
            expert_dynamics.append(dynamics.cpu().numpy())
    
    # Plot vector fields side by side
    for expert_id, dynamics in enumerate(expert_dynamics):
        # Extract dx/dt for fast and slow components
        dx_fast = dynamics[:, 0].reshape(grid_size, grid_size)
        dx_slow = dynamics[:, 2].reshape(grid_size, grid_size)
        
        # Normalize for visualization
        magnitude = np.sqrt(dx_fast**2 + dx_slow**2)
        dx_fast_norm = dx_fast / (magnitude + 1e-6)
        dx_slow_norm = dx_slow / (magnitude + 1e-6)
        
        # Color based on expert
        color = 'red' if expert_id == 0 else 'blue'
        
        # Plot vector field
        skip = 2
        ax4.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  dx_fast_norm[::skip, ::skip], dx_slow_norm[::skip, ::skip],
                  magnitude[::skip, ::skip], cmap='RdBu' if expert_id == 0 else 'Blues',
                  alpha=0.6, scale=25, width=0.003)
        
        # Add label
        ax4.text(-2.5 + expert_id * 3, 2.5, f'Expert {expert_id}',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                fontsize=12, fontweight='bold')
    
    ax4.set_xlabel('$x_{fast}$', fontsize=12)
    ax4.set_ylabel('$x_{slow}$', fontsize=12)
    ax4.set_title('(d) Vector Fields by Expert', fontsize=14, fontweight='bold')
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 3])
    ax4.grid(True, alpha=0.3)
    
    # Add vertical separator line
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    # Overall title
    fig.suptitle('AME-ODE Analysis: Automatic Timescale Separation in Multi-Scale Oscillators', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 4-panel AME-ODE visualization figure')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints_test/best_ame_ode.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                        default='configs/quick_test.yml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, 
                        default='ame_ode_4panel_figure.png',
                        help='Output figure path')
    parser.add_argument('--time-span', type=float, default=10.0,
                        help='Time span for trajectory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model from checkpoint...")
    model = load_checkpoint_and_model(Path(args.checkpoint), Path(args.config), device)
    
    # Create synthetic system
    print("Creating multi-scale oscillator system...")
    system = MultiScaleOscillators(
        fast_freq=10.0,
        slow_freq=0.1,
        coupling_strength=0.05,
        state_dim=4
    )
    
    # Generate trajectory data
    print("Generating trajectory data...")
    data = generate_trajectory_data(
        model, system, 
        n_trajectories=1,
        time_span=args.time_span,
        n_time_points=1000,
        device=device
    )
    
    # Evaluate model performance
    print("\nEvaluating model performance...")
    with torch.no_grad():
        metrics = evaluate_model(
            model,
            data['true_trajectory'],
            data['pred_trajectory'],
            data['times']
        )
    
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Relative Error: {metrics.get('relative_error', 0):.4%}")
    
    if data['routing_weights'] is not None:
        weights = data['routing_weights'][:, 0, :].cpu().numpy()
        avg_active = np.mean(np.sum(weights > 0.1, axis=1))
        sparsity = 1.0 - (avg_active / model.n_experts)
        print(f"Average Active Experts: {avg_active:.2f}/{model.n_experts}")
        print(f"Sparsity: {sparsity:.1%}")
    
    # Create figure
    print("\nCreating 4-panel figure...")
    create_4panel_figure(data, model, Path(args.output))
    
    print("\nDone!")


if __name__ == '__main__':
    main()