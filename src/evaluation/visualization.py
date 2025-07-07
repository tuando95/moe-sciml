import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from pathlib import Path


class AMEODEVisualizer:
    """Visualization tools for AME-ODE analysis."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_phase_portraits(
        self,
        model: torch.nn.Module,
        state_bounds: Tuple[float, float] = (-5, 5),
        n_grid: int = 50,
        expert_idx: Optional[int] = None,
        save_name: Optional[str] = None,
    ):
        """Plot phase portraits for individual experts or mixture."""
        device = next(model.parameters()).device
        
        # Create grid on the correct device
        x_range = torch.linspace(state_bounds[0], state_bounds[1], n_grid, device=device)
        y_range = torch.linspace(state_bounds[0], state_bounds[1], n_grid, device=device)
        X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
        
        # Flatten grid points
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        
        # Add zeros for additional dimensions if needed
        if model.state_dim > 2:
            zeros = torch.zeros(grid_points.shape[0], model.state_dim - 2, device=device)
            grid_points = torch.cat([grid_points, zeros], dim=-1)
        
        t = torch.zeros(grid_points.shape[0], device=device)
        
        with torch.no_grad():
            if expert_idx is not None:
                # Single expert dynamics
                dynamics = model.experts.experts[expert_idx](t, grid_points)
                title = f"Expert {expert_idx} Phase Portrait"
            else:
                # Full mixture dynamics
                dynamics = model.compute_dynamics_for_viz(t, grid_points)
                title = "AME-ODE Mixture Phase Portrait"
        
        # Extract 2D dynamics
        dx = dynamics[:, 0].cpu().numpy().reshape(n_grid, n_grid)
        dy = dynamics[:, 1].cpu().numpy().reshape(n_grid, n_grid)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize arrows for better visualization
        magnitude = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / (magnitude + 1e-6)
        dy_norm = dy / (magnitude + 1e-6)
        
        # Quiver plot
        skip = 2  # Skip some points for clarity
        X_cpu = X.cpu().numpy()
        Y_cpu = Y.cpu().numpy()
        ax.quiver(
            X_cpu[::skip, ::skip], Y_cpu[::skip, ::skip],
            dx_norm[::skip, ::skip], dy_norm[::skip, ::skip],
            magnitude[::skip, ::skip],
            cmap='viridis',
            alpha=0.6,
        )
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.colorbar(ax.collections[0], ax=ax, label='|dx/dt|')
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing
    
    def plot_routing_heatmap(
        self,
        model: torch.nn.Module,
        state_bounds: Tuple[float, float] = (-5, 5),
        n_grid: int = 50,
        save_name: Optional[str] = None,
    ):
        """Plot heatmaps of expert routing weights across state space."""
        device = next(model.parameters()).device
        
        # Create grid on the correct device
        x_range = torch.linspace(state_bounds[0], state_bounds[1], n_grid, device=device)
        y_range = torch.linspace(state_bounds[0], state_bounds[1], n_grid, device=device)
        X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
        
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        if model.state_dim > 2:
            zeros = torch.zeros(grid_points.shape[0], model.state_dim - 2, device=device)
            grid_points = torch.cat([grid_points, zeros], dim=-1)
        
        t = torch.zeros(grid_points.shape[0], device=device)
        
        # Get routing weights
        with torch.no_grad():
            # Save original hidden states
            original_hidden = model.gating.hidden_states
            
            # Reset hidden states for visualization batch size
            model.gating.reset_history(batch_size=grid_points.shape[0])
            
            # Initial dynamics estimate
            uniform_weights = torch.ones(grid_points.shape[0], model.n_experts) / model.n_experts
            uniform_weights = uniform_weights.to(device)
            dx_dt = model.experts(t, grid_points, uniform_weights)
            
            # Routing weights
            weights, _ = model.gating(grid_points, dx_dt, t, update_history=False)
            weights = weights.cpu().numpy()
            
            # Restore original hidden states
            model.gating.hidden_states = original_hidden
        
        # Plot heatmaps
        n_experts = weights.shape[1]
        fig, axes = plt.subplots(2, (n_experts + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_experts):
            weight_map = weights[:, i].reshape(n_grid, n_grid)
            
            im = axes[i].imshow(
                weight_map,
                extent=[state_bounds[0], state_bounds[1], state_bounds[0], state_bounds[1]],
                origin='lower',
                cmap='hot',
                vmin=0,
                vmax=1,
            )
            axes[i].set_title(f'Expert {i} Weight')
            axes[i].set_xlabel('x₁')
            axes[i].set_ylabel('x₂')
            plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(n_experts, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Expert Routing Weights Across State Space')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing
    
    def plot_trajectory_comparison(
        self,
        true_trajectory: torch.Tensor,
        pred_trajectory: torch.Tensor,
        times: torch.Tensor,
        dims_to_plot: Optional[List[int]] = None,
        save_name: Optional[str] = None,
    ):
        """Compare true and predicted trajectories."""
        true_traj = true_trajectory.cpu().numpy()
        pred_traj = pred_trajectory.cpu().numpy()
        times_np = times.cpu().numpy()
        
        # Select dimensions to plot
        if dims_to_plot is None:
            dims_to_plot = list(range(min(3, true_traj.shape[-1])))
        
        n_dims = len(dims_to_plot)
        
        # Time series plots
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        
        for i, dim in enumerate(dims_to_plot):
            # Plot trajectories
            for j in range(min(5, true_traj.shape[1])):  # Plot up to 5 trajectories
                axes[i].plot(times_np, true_traj[:, j, dim], 'b-', alpha=0.5, label='True' if j == 0 else '')
                axes[i].plot(times_np, pred_traj[:, j, dim], 'r--', alpha=0.5, label='Predicted' if j == 0 else '')
            
            axes[i].set_ylabel(f'x_{dim+1}')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend()
        
        axes[-1].set_xlabel('Time')
        plt.suptitle('Trajectory Comparison')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}_timeseries.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing
        
        # Phase space plots for 2D/3D
        if true_traj.shape[-1] >= 2:
            fig = plt.figure(figsize=(12, 5))
            
            # 2D phase space
            ax1 = fig.add_subplot(121)
            for j in range(min(5, true_traj.shape[1])):
                ax1.plot(true_traj[:, j, 0], true_traj[:, j, 1], 'b-', alpha=0.5, label='True' if j == 0 else '')
                ax1.plot(pred_traj[:, j, 0], pred_traj[:, j, 1], 'r--', alpha=0.5, label='Predicted' if j == 0 else '')
            ax1.set_xlabel('x₁')
            ax1.set_ylabel('x₂')
            ax1.set_title('2D Phase Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 3D phase space if available
            if true_traj.shape[-1] >= 3:
                ax2 = fig.add_subplot(122, projection='3d')
                for j in range(min(3, true_traj.shape[1])):
                    ax2.plot(true_traj[:, j, 0], true_traj[:, j, 1], true_traj[:, j, 2], 
                            'b-', alpha=0.5, label='True' if j == 0 else '')
                    ax2.plot(pred_traj[:, j, 0], pred_traj[:, j, 1], pred_traj[:, j, 2], 
                            'r--', alpha=0.5, label='Predicted' if j == 0 else '')
                ax2.set_xlabel('x₁')
                ax2.set_ylabel('x₂')
                ax2.set_zlabel('x₃')
                ax2.set_title('3D Phase Space')
                ax2.legend()
            
            plt.tight_layout()
            
            if save_name and self.save_dir:
                plt.savefig(self.save_dir / f"{save_name}_phasespace.png", dpi=150, bbox_inches='tight')
            plt.close()  # Close figure instead of showing
    
    def plot_expert_usage_evolution(
        self,
        routing_weights: torch.Tensor,
        times: torch.Tensor,
        save_name: Optional[str] = None,
    ):
        """Plot evolution of expert usage over time."""
        # Handle empty routing weights
        if routing_weights.numel() == 0:
            print("Warning: routing_weights is empty, skipping expert usage plot")
            return
            
        weights = routing_weights.cpu().numpy()
        times_np = times.cpu().numpy()
        
        # Ensure times is 1D
        if times_np.ndim > 1:
            # If times has batch dimension, use the first batch
            times_np = times_np[0] if times_np.shape[0] > 1 else times_np.squeeze()
        
        # Ensure times is still 1D after squeezing
        if times_np.ndim != 1:
            times_np = times_np.flatten()
        
        n_experts = weights.shape[-1]
        
        # Average over batch
        avg_weights = weights.mean(axis=1)
        
        # Ensure dimensions match
        if len(times_np) != len(avg_weights):
            print(f"Warning: Time points ({len(times_np)}) don't match weight points ({len(avg_weights)})")
            # Try to fix by interpolating or truncating
            if len(times_np) > len(avg_weights):
                times_np = times_np[:len(avg_weights)]
            else:
                avg_weights = avg_weights[:len(times_np)]
        
        # Plot stacked area chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Stacked area plot
        ax1.stackplot(times_np, avg_weights.T, labels=[f'Expert {i}' for i in range(n_experts)], alpha=0.7)
        ax1.set_ylabel('Cumulative Weight')
        ax1.set_title('Expert Usage Evolution')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Individual line plots
        for i in range(n_experts):
            ax2.plot(times_np, avg_weights[:, i], label=f'Expert {i}', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Average Weight')
        ax2.set_title('Individual Expert Weights')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing
    
    def plot_loss_landscape(
        self,
        losses_history: Dict[str, List[float]],
        save_name: Optional[str] = None,
    ):
        """Plot training loss landscape."""
        epochs = range(len(losses_history['train_loss']))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(epochs, losses_history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, losses_history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Routing entropy
        if 'routing_entropy' in losses_history:
            axes[0, 1].plot(epochs, losses_history['routing_entropy'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Routing Entropy')
            axes[0, 1].set_title('Expert Selection Entropy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in losses_history:
            axes[1, 0].plot(epochs, losses_history['learning_rate'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss components
        loss_components = ['reconstruction', 'routing', 'diversity', 'smoothness', 'balance']
        for comp in loss_components:
            if f'{comp}_loss' in losses_history:
                axes[1, 1].plot(epochs, losses_history[f'{comp}_loss'], label=comp)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Value')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Loss Landscape')
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing
    
    def create_interactive_3d_trajectory(
        self,
        trajectory: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None,
        save_name: Optional[str] = None,
    ):
        """Create interactive 3D trajectory visualization with Plotly."""
        traj = trajectory.cpu().numpy()
        
        if traj.shape[-1] < 3:
            print("Need at least 3D trajectory for 3D visualization")
            return
        
        # Select first trajectory if batch
        if traj.ndim == 3:
            traj = traj[:, 0, :]
        
        # Create figure
        fig = go.Figure()
        
        # Add trajectory
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            marker=dict(
                size=3,
                color=np.arange(len(traj)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time Step"),
            ),
            line=dict(
                color='darkblue',
                width=2,
            ),
            name='Trajectory',
        ))
        
        # Add starting point
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
            ),
            name='Start',
        ))
        
        # Update layout
        fig.update_layout(
            title='3D Trajectory Visualization',
            scene=dict(
                xaxis_title='x₁',
                yaxis_title='x₂',
                zaxis_title='x₃',
            ),
            showlegend=True,
        )
        
        if save_name and self.save_dir:
            fig.write_html(self.save_dir / f"{save_name}.html")
        
        fig.show()
    
    def plot_expert_specialization_matrix(
        self,
        specialization_matrix: np.ndarray,
        state_labels: Optional[List[str]] = None,
        save_name: Optional[str] = None,
    ):
        """Plot expert specialization patterns."""
        n_states, n_experts = specialization_matrix.shape
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            specialization_matrix.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Expert Weight'},
            yticklabels=[f'Expert {i}' for i in range(n_experts)],
            xticklabels=state_labels if state_labels else [f'State {i}' for i in range(n_states)],
            ax=ax,
        )
        
        ax.set_title('Expert Specialization Patterns')
        ax.set_xlabel('State Space Regions')
        ax.set_ylabel('Experts')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / f"{save_name}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing


def create_training_animation(
    checkpoints_dir: Path,
    output_path: Path,
    system_bounds: Tuple[float, float] = (-5, 5),
    n_grid: int = 30,
):
    """Create animation showing evolution of expert dynamics during training."""
    # This would load checkpoints and create an animation
    # showing how the phase portraits evolve during training
    pass  # Implementation depends on checkpoint format