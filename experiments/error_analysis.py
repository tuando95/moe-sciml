#!/usr/bin/env python3
"""Error analysis and computational efficiency experiments for AME-ODE."""

import torch
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from src.models.ame_ode import AMEODE
from src.data.preprocessing import create_experimental_dataloaders
from src.evaluation.metrics import AMEODEMetrics, analyze_expert_diversity
from src.utils.config import Config


class ErrorAnalysisExperiment:
    """Comprehensive error analysis for AME-ODE."""
    
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
        
        # Initialize metrics
        self.metrics_calc = AMEODEMetrics(self.config.to_dict())
    
    def analyze_failure_cases(
        self,
        test_loader: torch.utils.data.DataLoader,
        threshold_percentile: float = 90,
    ) -> Dict[str, Any]:
        """Identify and analyze failure cases."""
        print("\nAnalyzing failure cases...")
        
        # Collect errors and metadata
        errors = []
        trajectories = []
        predictions = []
        metadata = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Predict
                pred_traj, info = self.model(x0, times)
                
                # Compute trajectory-wise error
                mse_per_traj = torch.mean((pred_traj - trajectory) ** 2, dim=(0, 2))
                
                for i in range(len(mse_per_traj)):
                    errors.append(mse_per_traj[i].item())
                    trajectories.append(trajectory[:, i].cpu())
                    predictions.append(pred_traj[:, i].cpu())
                    metadata.append({
                        'initial_norm': torch.norm(x0[i]).item(),
                        'final_norm': torch.norm(trajectory[-1, i]).item(),
                        'trajectory_length': len(times),
                        'routing_entropy': info['routing_entropy'].item() if 'routing_entropy' in info else 0,
                    })
        
        # Identify failure cases (high error)
        errors = np.array(errors)
        threshold = np.percentile(errors, threshold_percentile)
        failure_mask = errors > threshold
        
        print(f"Found {failure_mask.sum()} failure cases (top {100-threshold_percentile}%)")
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure_patterns(
            errors, metadata, failure_mask, trajectories, predictions
        )
        
        # Visualize failure cases
        self._visualize_failure_cases(
            trajectories, predictions, errors, failure_mask
        )
        
        return failure_analysis
    
    def _analyze_failure_patterns(
        self,
        errors: np.ndarray,
        metadata: List[Dict],
        failure_mask: np.ndarray,
        trajectories: List[torch.Tensor],
        predictions: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """Analyze patterns in failure cases."""
        analysis = {}
        
        # Extract metadata arrays
        initial_norms = np.array([m['initial_norm'] for m in metadata])
        final_norms = np.array([m['final_norm'] for m in metadata])
        traj_lengths = np.array([m['trajectory_length'] for m in metadata])
        routing_entropies = np.array([m['routing_entropy'] for m in metadata])
        
        # Compare failure vs success statistics
        analysis['initial_norm'] = {
            'failure_mean': initial_norms[failure_mask].mean(),
            'success_mean': initial_norms[~failure_mask].mean(),
            'failure_std': initial_norms[failure_mask].std(),
            'success_std': initial_norms[~failure_mask].std(),
        }
        
        analysis['final_norm'] = {
            'failure_mean': final_norms[failure_mask].mean(),
            'success_mean': final_norms[~failure_mask].mean(),
        }
        
        analysis['routing_entropy'] = {
            'failure_mean': routing_entropies[failure_mask].mean(),
            'success_mean': routing_entropies[~failure_mask].mean(),
        }
        
        # Identify common failure modes
        failure_modes = []
        
        # Mode 1: Explosive trajectories
        explosive_mask = final_norms > 10 * initial_norms
        if explosive_mask[failure_mask].any():
            failure_modes.append({
                'type': 'explosive_growth',
                'count': explosive_mask[failure_mask].sum(),
                'percentage': explosive_mask[failure_mask].mean() * 100,
            })
        
        # Mode 2: High routing instability
        unstable_routing = routing_entropies > routing_entropies.mean() + 2 * routing_entropies.std()
        if unstable_routing[failure_mask].any():
            failure_modes.append({
                'type': 'routing_instability',
                'count': unstable_routing[failure_mask].sum(),
                'percentage': unstable_routing[failure_mask].mean() * 100,
            })
        
        analysis['failure_modes'] = failure_modes
        
        # Save analysis
        with open(self.output_dir / 'failure_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _visualize_failure_cases(
        self,
        trajectories: List[torch.Tensor],
        predictions: List[torch.Tensor],
        errors: np.ndarray,
        failure_mask: np.ndarray,
    ):
        """Visualize worst failure cases."""
        # Get worst cases
        worst_indices = np.argsort(errors)[-6:]  # Top 6 worst
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, worst_idx in enumerate(worst_indices):
            ax = axes[idx]
            
            true_traj = trajectories[worst_idx].numpy()
            pred_traj = predictions[worst_idx].numpy()
            
            # Plot first two dimensions
            if true_traj.shape[1] >= 2:
                ax.plot(true_traj[:, 0], true_traj[:, 1], 'b-', 
                       label='True', linewidth=2, alpha=0.7)
                ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', 
                       label='Predicted', linewidth=2, alpha=0.7)
                ax.set_xlabel('x₁')
                ax.set_ylabel('x₂')
            else:
                # 1D trajectory
                t = np.arange(len(true_traj))
                ax.plot(t, true_traj[:, 0], 'b-', label='True', linewidth=2)
                ax.plot(t, pred_traj[:, 0], 'r--', label='Predicted', linewidth=2)
                ax.set_xlabel('Time')
                ax.set_ylabel('x')
            
            ax.set_title(f'Error: {errors[worst_idx]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Worst Failure Cases', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'worst_failure_cases.png', dpi=150)
        plt.close()
    
    def analyze_expert_collapse(self) -> Dict[str, Any]:
        """Analyze expert collapse and diversity."""
        print("\nAnalyzing expert diversity...")
        
        diversity_metrics = analyze_expert_diversity(self.model, n_samples=1000)
        
        # Check for expert collapse
        min_distance = diversity_metrics['min_expert_distance']
        collapse_threshold = 0.1
        
        analysis = {
            'expert_diversity': diversity_metrics,
            'collapse_detected': min_distance < collapse_threshold,
            'collapse_pairs': [],
        }
        
        # Find collapsed pairs
        distance_matrix = diversity_metrics['distance_matrix']
        n_experts = distance_matrix.shape[0]
        
        for i in range(n_experts):
            for j in range(i+1, n_experts):
                if distance_matrix[i, j] < collapse_threshold:
                    analysis['collapse_pairs'].append({
                        'experts': (i, j),
                        'distance': distance_matrix[i, j],
                    })
        
        # Visualize distance matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            distance_matrix,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Distance'},
        )
        plt.title('Expert Pairwise Distances')
        plt.xlabel('Expert ID')
        plt.ylabel('Expert ID')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_distance_matrix.png', dpi=150)
        plt.close()
        
        return analysis
    
    def analyze_routing_patterns(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """Analyze expert routing patterns and failures."""
        print("\nAnalyzing routing patterns...")
        
        # Collect routing statistics
        routing_data = {
            'weights': [],
            'transitions': [],
            'dominant_experts': [],
            'entropy': [],
        }
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Get routing information
                _, info = self.model(x0, times)
                
                if 'routing_weights' in info:
                    weights = info['routing_weights'].cpu().numpy()  # (T, B, K)
                    
                    # Analyze each trajectory
                    for b in range(weights.shape[1]):
                        traj_weights = weights[:, b, :]
                        
                        # Dominant expert at each time
                        dominant = np.argmax(traj_weights, axis=1)
                        routing_data['dominant_experts'].extend(dominant)
                        
                        # Count transitions
                        transitions = np.sum(np.diff(dominant) != 0)
                        routing_data['transitions'].append(transitions)
                        
                        # Routing entropy
                        entropy = -np.sum(traj_weights * np.log(traj_weights + 1e-8), axis=1).mean()
                        routing_data['entropy'].append(entropy)
                        
                        routing_data['weights'].append(traj_weights)
        
        # Analyze patterns
        analysis = {
            'avg_transitions': np.mean(routing_data['transitions']),
            'std_transitions': np.std(routing_data['transitions']),
            'avg_entropy': np.mean(routing_data['entropy']),
            'expert_usage': {},
        }
        
        # Expert usage statistics
        dominant_experts = np.array(routing_data['dominant_experts'])
        for i in range(self.model.n_experts):
            usage = (dominant_experts == i).mean() * 100
            analysis['expert_usage'][f'expert_{i}'] = usage
        
        # Visualize routing patterns
        self._visualize_routing_patterns(routing_data)
        
        return analysis
    
    def _visualize_routing_patterns(self, routing_data: Dict[str, Any]):
        """Visualize routing patterns."""
        # Sample trajectories for visualization
        n_samples = min(10, len(routing_data['weights']))
        sample_indices = np.random.choice(len(routing_data['weights']), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Routing weights over time for sample trajectories
        ax1 = axes[0]
        for idx in sample_indices:
            weights = routing_data['weights'][idx]
            time_steps = np.arange(len(weights))
            
            for expert_id in range(weights.shape[1]):
                ax1.plot(time_steps, weights[:, expert_id], alpha=0.5)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Routing Weight')
        ax1.set_title('Expert Routing Weights Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Transition histogram
        ax2 = axes[1]
        transitions = routing_data['transitions']
        ax2.hist(transitions, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Expert Transitions')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Expert Transitions per Trajectory')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'routing_patterns.png', dpi=150)
        plt.close()
    
    def computational_profiling(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """Profile computational performance in detail."""
        print("\nProfiling computational performance...")
        
        # Different batch sizes for scaling analysis
        batch_sizes = [1, 4, 16, 32, 64]
        
        profiling_results = {
            'batch_scaling': {},
            'component_timing': {},
            'memory_usage': {},
        }
        
        # Get sample data
        sample_batch = next(iter(test_loader))
        max_batch = min(64, len(sample_batch['trajectory']))
        
        # Test batch scaling
        for bs in batch_sizes:
            if bs > max_batch:
                continue
            
            x0 = sample_batch['initial_condition'][:bs].to(self.device)
            times = sample_batch['times'].to(self.device)
            
            # Time forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            timing_runs = []
            for _ in range(10):
                start = time.time()
                _, _ = self.model(x0, times)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                timing_runs.append(time.time() - start)
            
            profiling_results['batch_scaling'][bs] = {
                'mean_time': np.mean(timing_runs),
                'std_time': np.std(timing_runs),
                'throughput': bs / np.mean(timing_runs),
            }
        
        # Component-wise timing with hooks
        self._profile_components(sample_batch, profiling_results)
        
        # Memory profiling
        if torch.cuda.is_available():
            self._profile_memory(sample_batch, profiling_results)
        
        # Visualize results
        self._visualize_profiling(profiling_results)
        
        # Save results
        with open(self.output_dir / 'computational_profiling.json', 'w') as f:
            json.dump(profiling_results, f, indent=2)
        
        return profiling_results
    
    def _profile_components(self, sample_batch: Dict, results: Dict):
        """Profile individual component timing."""
        x0 = sample_batch['initial_condition'][:10].to(self.device)
        times = sample_batch['times'].to(self.device)
        
        # Hook-based profiling
        component_times = {
            'expert_forward': [],
            'gating_forward': [],
            'integration': [],
        }
        
        # Add hooks
        expert_hook_times = []
        gating_hook_times = []
        
        def expert_hook(module, input, output):
            expert_hook_times.append(time.time())
        
        def gating_hook(module, input, output):
            gating_hook_times.append(time.time())
        
        expert_handle = self.model.experts.register_forward_hook(expert_hook)
        gating_handle = self.model.gating.register_forward_hook(gating_hook)
        
        # Run forward pass
        start_time = time.time()
        expert_hook_times.clear()
        gating_hook_times.clear()
        
        _, _ = self.model(x0, times)
        
        total_time = time.time() - start_time
        
        # Remove hooks
        expert_handle.remove()
        gating_handle.remove()
        
        # Estimate component times
        if expert_hook_times:
            results['component_timing']['expert_percentage'] = len(expert_hook_times) * 0.001 / total_time * 100
        if gating_hook_times:
            results['component_timing']['gating_percentage'] = len(gating_hook_times) * 0.001 / total_time * 100
        
        results['component_timing']['total_time'] = total_time
    
    def _profile_memory(self, sample_batch: Dict, results: Dict):
        """Profile memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x0 = sample_batch['initial_condition'][:32].to(self.device)
        times = sample_batch['times'].to(self.device)
        
        # Baseline memory
        baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Forward pass
        _, _ = self.model(x0, times)
        
        # Peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        results['memory_usage'] = {
            'baseline_mb': baseline_memory,
            'peak_mb': peak_memory,
            'forward_pass_mb': peak_memory - baseline_memory,
        }
    
    def _visualize_profiling(self, profiling_results: Dict):
        """Visualize profiling results."""
        # Batch scaling plot
        if 'batch_scaling' in profiling_results:
            batch_sizes = list(profiling_results['batch_scaling'].keys())
            throughputs = [profiling_results['batch_scaling'][bs]['throughput'] 
                          for bs in batch_sizes]
            
            plt.figure(figsize=(8, 6))
            plt.plot(batch_sizes, throughputs, 'o-', markersize=8, linewidth=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.title('Throughput vs Batch Size')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'throughput_scaling.png', dpi=150)
            plt.close()


def run_error_analysis(
    config_path: Path,
    model_checkpoint: Path,
    system_name: str,
    output_dir: Path,
):
    """Run complete error analysis."""
    # Create experiment
    experiment = ErrorAnalysisExperiment(config_path, model_checkpoint, output_dir)
    
    # Load test data
    config = Config(config_path).to_dict()
    _, _, test_loader, _ = create_experimental_dataloaders(config, system_name)
    
    # Run analyses
    print("="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # 1. Failure case analysis
    failure_analysis = experiment.analyze_failure_cases(test_loader)
    print(f"\nFailure Analysis Summary:")
    print(f"  Initial norm (failure vs success): "
          f"{failure_analysis['initial_norm']['failure_mean']:.3f} vs "
          f"{failure_analysis['initial_norm']['success_mean']:.3f}")
    print(f"  Routing entropy (failure vs success): "
          f"{failure_analysis['routing_entropy']['failure_mean']:.3f} vs "
          f"{failure_analysis['routing_entropy']['success_mean']:.3f}")
    
    # 2. Expert collapse analysis
    collapse_analysis = experiment.analyze_expert_collapse()
    print(f"\nExpert Diversity Analysis:")
    print(f"  Mean expert distance: {collapse_analysis['expert_diversity']['mean_expert_distance']:.4f}")
    print(f"  Min expert distance: {collapse_analysis['expert_diversity']['min_expert_distance']:.4f}")
    print(f"  Collapse detected: {collapse_analysis['collapse_detected']}")
    
    # 3. Routing pattern analysis
    routing_analysis = experiment.analyze_routing_patterns(test_loader)
    print(f"\nRouting Pattern Analysis:")
    print(f"  Average transitions per trajectory: {routing_analysis['avg_transitions']:.2f}")
    print(f"  Average routing entropy: {routing_analysis['avg_entropy']:.3f}")
    print("  Expert usage distribution:")
    for expert, usage in routing_analysis['expert_usage'].items():
        print(f"    {expert}: {usage:.1f}%")
    
    # 4. Computational profiling
    profiling_results = experiment.computational_profiling(test_loader)
    print(f"\nComputational Profiling:")
    if 'batch_scaling' in profiling_results:
        for bs, metrics in profiling_results['batch_scaling'].items():
            print(f"  Batch size {bs}: {metrics['throughput']:.1f} samples/sec")
    
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run error analysis experiments')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system name')
    parser.add_argument('--output-dir', type=str, default='error_analysis_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_error_analysis(
        Path(args.config),
        Path(args.checkpoint),
        args.system,
        Path(args.output_dir)
    )