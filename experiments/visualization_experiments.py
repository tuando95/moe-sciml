#!/usr/bin/env python3
"""Visualization experiments for expert dynamics and learning in AME-ODE."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.ame_ode import AMEODE
from src.data.preprocessing import create_experimental_dataloaders
from src.evaluation.visualization import AMEODEVisualizer
from src.evaluation.metrics import compute_expert_specialization_matrix
from src.utils.config import Config


class ExpertDynamicsVisualizer:
    """Advanced visualization for expert dynamics and specialization."""
    
    def __init__(self, config_path: Path, model_checkpoint: Path, output_dir: Path):
        self.config = Config(config_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.device = torch.device(
            self.config.compute['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        self.model = AMEODE(self.config.to_dict()).to(self.device)
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Basic visualizer
        self.basic_viz = AMEODEVisualizer(save_dir=output_dir)
    
    def visualize_expert_vector_fields(
        self,
        bounds: Tuple[float, float] = (-5, 5),
        n_grid: int = 30,
        t_values: List[float] = [0.0, 0.5, 1.0],
    ):
        """Visualize vector fields for each expert at different times."""
        print("\nVisualizing expert vector fields...")
        
        for t_val in t_values:
            fig, axes = plt.subplots(
                2, (self.model.n_experts + 1) // 2,
                figsize=(15, 10)
            )
            axes = axes.flatten()
            
            # Create grid
            x_range = torch.linspace(bounds[0], bounds[1], n_grid)
            y_range = torch.linspace(bounds[0], bounds[1], n_grid)
            X, Y = torch.meshgrid(x_range, y_range, indexing='xy')
            
            # Prepare grid points
            grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
            if self.model.state_dim > 2:
                zeros = torch.zeros(grid_points.shape[0], self.model.state_dim - 2)
                grid_points = torch.cat([grid_points, zeros], dim=-1)
            
            grid_points = grid_points.to(self.device)
            t_tensor = torch.tensor(t_val).to(self.device)
            
            # Get dynamics from each expert
            with torch.no_grad():
                expert_dynamics = self.model.experts.get_individual_dynamics(
                    t_tensor, grid_points
                )  # (batch, n_experts, state_dim)
            
            # Plot each expert's vector field
            for expert_id in range(self.model.n_experts):
                ax = axes[expert_id]
                
                # Extract 2D dynamics
                dx = expert_dynamics[:, expert_id, 0].cpu().numpy().reshape(n_grid, n_grid)
                dy = expert_dynamics[:, expert_id, 1].cpu().numpy().reshape(n_grid, n_grid)
                
                # Normalize for visualization
                magnitude = np.sqrt(dx**2 + dy**2)
                dx_norm = dx / (magnitude + 1e-6)
                dy_norm = dy / (magnitude + 1e-6)
                
                # Quiver plot
                skip = 2
                im = ax.quiver(
                    X[::skip, ::skip], Y[::skip, ::skip],
                    dx_norm[::skip, ::skip], dy_norm[::skip, ::skip],
                    magnitude[::skip, ::skip],
                    cmap='viridis',
                    alpha=0.6,
                )
                
                ax.set_title(f'Expert {expert_id}')
                ax.set_xlabel('x₁')
                ax.set_ylabel('x₂')
                ax.set_aspect('equal')
                plt.colorbar(im, ax=ax, label='|dx/dt|')
            
            # Hide unused subplots
            for i in range(self.model.n_experts, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Expert Vector Fields at t={t_val}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'expert_vector_fields_t{t_val}.png', dpi=150)
            plt.close()
    
    def visualize_expert_specialization_regions(
        self,
        test_states: Optional[torch.Tensor] = None,
        n_samples: int = 5000,
    ):
        """Visualize which regions of state space each expert specializes in."""
        print("\nVisualizing expert specialization regions...")
        
        if test_states is None:
            # Generate test states covering the space
            test_states = torch.randn(n_samples, self.model.state_dim) * 3
        
        test_states = test_states.to(self.device)
        
        # Get expert specialization matrix
        specialization_matrix = compute_expert_specialization_matrix(
            self.model, test_states, self.model.n_experts
        )
        
        # Get dominant expert for each state
        dominant_experts = np.argmax(specialization_matrix, axis=1)
        
        # Dimensionality reduction for visualization
        if self.model.state_dim > 2:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            states_2d = pca.fit_transform(test_states.cpu().numpy())
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"PCA explained variance: {explained_var:.2%}")
        else:
            states_2d = test_states.cpu().numpy()[:, :2]
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.get_cmap('tab10', self.model.n_experts)
        
        for expert_id in range(self.model.n_experts):
            mask = dominant_experts == expert_id
            if mask.any():
                plt.scatter(
                    states_2d[mask, 0],
                    states_2d[mask, 1],
                    c=[colors(expert_id)],
                    label=f'Expert {expert_id}',
                    alpha=0.6,
                    s=20,
                )
        
        plt.xlabel('PC1' if self.model.state_dim > 2 else 'x₁')
        plt.ylabel('PC2' if self.model.state_dim > 2 else 'x₂')
        plt.title('Expert Specialization Regions in State Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_specialization_regions.png', dpi=150)
        plt.close()
        
        # Create heatmap of specialization strengths
        fig, axes = plt.subplots(
            2, (self.model.n_experts + 1) // 2,
            figsize=(15, 10)
        )
        axes = axes.flatten()
        
        for expert_id in range(self.model.n_experts):
            ax = axes[expert_id]
            
            # Create 2D histogram of specialization strength
            weights = specialization_matrix[:, expert_id]
            
            h, xedges, yedges = np.histogram2d(
                states_2d[:, 0], states_2d[:, 1],
                bins=30,
                weights=weights,
            )
            
            im = ax.imshow(
                h.T,
                origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='hot',
                aspect='auto',
            )
            
            ax.set_title(f'Expert {expert_id} Specialization')
            ax.set_xlabel('PC1' if self.model.state_dim > 2 else 'x₁')
            ax.set_ylabel('PC2' if self.model.state_dim > 2 else 'x₂')
            plt.colorbar(im, ax=ax, label='Weight')
        
        # Hide unused subplots
        for i in range(self.model.n_experts, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Expert Specialization Heatmaps', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_specialization_heatmaps.png', dpi=150)
        plt.close()
    
    def visualize_learning_dynamics(
        self,
        checkpoint_dir: Path,
        epochs_to_plot: Optional[List[int]] = None,
    ):
        """Visualize how expert specialization evolves during training."""
        print("\nVisualizing learning dynamics...")
        
        # Find all checkpoints
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if not checkpoints:
            print("No checkpoints found for learning dynamics visualization")
            return
        
        if epochs_to_plot is None:
            # Select evenly spaced epochs
            n_checkpoints = min(6, len(checkpoints))
            indices = np.linspace(0, len(checkpoints)-1, n_checkpoints, dtype=int)
            checkpoints_to_use = [checkpoints[i] for i in indices]
        else:
            checkpoints_to_use = [
                cp for cp in checkpoints 
                if any(f'epoch_{e}' in cp.name for e in epochs_to_plot)
            ]
        
        # Generate test grid
        grid_size = 20
        x = torch.linspace(-3, 3, grid_size)
        y = torch.linspace(-3, 3, grid_size)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        test_points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        if self.model.state_dim > 2:
            zeros = torch.zeros(test_points.shape[0], self.model.state_dim - 2)
            test_points = torch.cat([test_points, zeros], dim=-1)
        
        test_points = test_points.to(self.device)
        
        # Visualize for each checkpoint
        fig, axes = plt.subplots(
            2, 3,
            figsize=(15, 10)
        )
        axes = axes.flatten()
        
        for idx, checkpoint_path in enumerate(checkpoints_to_use[:6]):
            ax = axes[idx]
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get epoch number
            epoch = int(checkpoint_path.stem.split('_')[-1])
            
            # Get routing weights
            with torch.no_grad():
                t = torch.tensor(0.0).to(self.device)
                uniform_weights = torch.ones(test_points.shape[0], self.model.n_experts) / self.model.n_experts
                uniform_weights = uniform_weights.to(self.device)
                dx_dt = self.model.experts(t, test_points, uniform_weights)
                
                weights, _ = self.model.gating(test_points, dx_dt, t, update_history=False)
                dominant_expert = weights.argmax(dim=-1).cpu().numpy()
            
            # Plot
            dominant_expert = dominant_expert.reshape(grid_size, grid_size)
            im = ax.imshow(
                dominant_expert,
                origin='lower',
                extent=[-3, 3, -3, 3],
                cmap='tab10',
                vmin=0,
                vmax=self.model.n_experts-1,
            )
            
            ax.set_title(f'Epoch {epoch}')
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
        
        # Hide unused subplots
        for i in range(len(checkpoints_to_use), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Evolution of Expert Specialization During Training', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_dynamics.png', dpi=150)
        plt.close()
    
    def create_interactive_trajectory_visualization(
        self,
        test_loader: torch.utils.data.DataLoader,
        n_trajectories: int = 5,
    ):
        """Create interactive 3D visualization of trajectories with expert routing."""
        print("\nCreating interactive trajectory visualizations...")
        
        # Get sample trajectories
        batch = next(iter(test_loader))
        trajectory = batch['trajectory'][:n_trajectories].to(self.device)
        times = batch['times'].to(self.device)
        x0 = batch['initial_condition'][:n_trajectories].to(self.device)
        
        # Get predictions and routing info
        with torch.no_grad():
            pred_traj, info = self.model(x0, times)
        
        routing_weights = info['routing_weights'].cpu().numpy()  # (T, B, K)
        
        # Create interactive plot for each trajectory
        for traj_idx in range(n_trajectories):
            if self.model.state_dim < 3:
                continue  # Skip 2D trajectories
            
            # Extract trajectory
            true_traj = trajectory[:, traj_idx, :3].cpu().numpy()
            pred_traj_np = pred_traj[:, traj_idx, :3].cpu().numpy()
            weights = routing_weights[:, traj_idx, :]
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
                subplot_titles=('3D Trajectory', 'Expert Weights Over Time'),
            )
            
            # 3D trajectory
            fig.add_trace(
                go.Scatter3d(
                    x=true_traj[:, 0],
                    y=true_traj[:, 1],
                    z=true_traj[:, 2],
                    mode='lines',
                    name='True',
                    line=dict(color='blue', width=4),
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=pred_traj_np[:, 0],
                    y=pred_traj_np[:, 1],
                    z=pred_traj_np[:, 2],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=2),
                    marker=dict(size=3),
                ),
                row=1, col=1
            )
            
            # Expert weights
            time_points = np.arange(len(weights))
            for expert_id in range(self.model.n_experts):
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=weights[:, expert_id],
                        mode='lines',
                        name=f'Expert {expert_id}',
                        line=dict(width=2),
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f'Trajectory {traj_idx} Analysis',
                showlegend=True,
                height=600,
            )
            
            fig.update_xaxes(title_text="Time Step", row=1, col=2)
            fig.update_yaxes(title_text="Weight", row=1, col=2)
            
            # Save
            fig.write_html(self.output_dir / f'interactive_trajectory_{traj_idx}.html')
        
        print(f"Created {n_trajectories} interactive visualizations")
    
    def visualize_expert_activation_patterns(
        self,
        test_loader: torch.utils.data.DataLoader,
        n_samples: int = 100,
    ):
        """Visualize expert activation patterns using t-SNE."""
        print("\nVisualizing expert activation patterns...")
        
        # Collect expert activations
        all_activations = []
        all_labels = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in test_loader:
                if sample_count >= n_samples:
                    break
                
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Get routing weights
                _, info = self.model(x0, times)
                
                if 'routing_weights' in info:
                    weights = info['routing_weights']  # (T, B, K)
                    
                    # Flatten time and batch dimensions
                    weights_flat = weights.reshape(-1, self.model.n_experts)
                    
                    # Get dominant expert for labeling
                    dominant = weights_flat.argmax(dim=-1)
                    
                    all_activations.append(weights_flat.cpu())
                    all_labels.append(dominant.cpu())
                    
                    sample_count += weights_flat.shape[0]
        
        # Concatenate all activations
        activations = torch.cat(all_activations, dim=0).numpy()[:n_samples]
        labels = torch.cat(all_labels, dim=0).numpy()[:n_samples]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        activations_2d = tsne.fit_transform(activations)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.get_cmap('tab10', self.model.n_experts)
        
        for expert_id in range(self.model.n_experts):
            mask = labels == expert_id
            if mask.any():
                plt.scatter(
                    activations_2d[mask, 0],
                    activations_2d[mask, 1],
                    c=[colors(expert_id)],
                    label=f'Expert {expert_id}',
                    alpha=0.6,
                    s=20,
                )
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization of Expert Routing Patterns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_activation_tsne.png', dpi=150)
        plt.close()
    
    def create_expert_summary_report(self):
        """Create a comprehensive summary of expert behaviors."""
        print("\nCreating expert summary report...")
        
        # Analyze each expert
        expert_summaries = []
        
        for expert_id in range(self.model.n_experts):
            summary = {
                'expert_id': expert_id,
                'characteristics': [],
            }
            
            # Sample dynamics at different points
            test_points = torch.randn(1000, self.model.state_dim).to(self.device)
            t = torch.tensor(0.0).to(self.device)
            
            with torch.no_grad():
                dynamics = self.model.experts.experts[expert_id](t, test_points)
                
                # Compute characteristics
                avg_magnitude = torch.norm(dynamics, dim=-1).mean().item()
                max_magnitude = torch.norm(dynamics, dim=-1).max().item()
                
                # Estimate divergence (proxy for stability)
                eps = 1e-4
                perturbed = test_points + torch.randn_like(test_points) * eps
                dynamics_perturbed = self.model.experts.experts[expert_id](t, perturbed)
                
                divergence = torch.norm(dynamics_perturbed - dynamics, dim=-1).mean() / eps
                
                summary['characteristics'] = {
                    'avg_magnitude': avg_magnitude,
                    'max_magnitude': max_magnitude,
                    'estimated_divergence': divergence.item(),
                }
            
            expert_summaries.append(summary)
        
        # Save report
        report = {
            'n_experts': self.model.n_experts,
            'expert_summaries': expert_summaries,
        }
        
        with open(self.output_dir / 'expert_summary_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nEXPERT SUMMARY:")
        print("-" * 40)
        for summary in expert_summaries:
            print(f"\nExpert {summary['expert_id']}:")
            for key, value in summary['characteristics'].items():
                print(f"  {key}: {value:.4f}")
        print("-" * 40)


def run_visualization_experiments(
    config_path: Path,
    model_checkpoint: Path,
    system_name: str,
    checkpoint_dir: Optional[Path] = None,
    output_dir: Path = Path('visualization_results'),
):
    """Run all visualization experiments."""
    # Create visualizer
    visualizer = ExpertDynamicsVisualizer(config_path, model_checkpoint, output_dir)
    
    # Load test data
    config = Config(config_path).to_dict()
    _, _, test_loader, _ = create_experimental_dataloaders(config, system_name)
    
    print("="*60)
    print("VISUALIZATION EXPERIMENTS")
    print("="*60)
    
    # 1. Expert vector fields
    visualizer.visualize_expert_vector_fields()
    
    # 2. Expert specialization regions
    visualizer.visualize_expert_specialization_regions()
    
    # 3. Learning dynamics (if checkpoints available)
    if checkpoint_dir and checkpoint_dir.exists():
        visualizer.visualize_learning_dynamics(checkpoint_dir)
    
    # 4. Interactive trajectory visualization
    visualizer.create_interactive_trajectory_visualization(test_loader)
    
    # 5. Expert activation patterns
    visualizer.visualize_expert_activation_patterns(test_loader)
    
    # 6. Expert summary report
    visualizer.create_expert_summary_report()
    
    # 7. Basic visualizations
    print("\nGenerating basic visualizations...")
    visualizer.basic_viz.plot_phase_portraits(visualizer.model, save_name='phase_portrait_full')
    visualizer.basic_viz.plot_routing_heatmap(visualizer.model, save_name='routing_heatmap_full')
    
    print("\nVisualization experiments completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run visualization experiments')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system name')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory with training checkpoints for learning dynamics')
    parser.add_argument('--output-dir', type=str, default='visualization_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_visualization_experiments(
        Path(args.config),
        Path(args.checkpoint),
        args.system,
        Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        Path(args.output_dir)
    )