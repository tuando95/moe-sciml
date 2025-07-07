import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import mutual_info_score
import warnings


class AMEODEMetrics:
    """Comprehensive metrics for evaluating AME-ODE performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_to_compute = config['evaluation']['metrics']
    
    def compute_all_metrics(
        self,
        pred_trajectory: torch.Tensor,
        true_trajectory: torch.Tensor,
        model_info: Dict[str, Any],
        times: torch.Tensor,
        ground_truth_experts: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute all requested metrics."""
        metrics = {}
        
        # Trajectory accuracy
        if 'trajectory_mse' in self.metrics_to_compute:
            metrics['trajectory_mse'] = self.trajectory_mse(pred_trajectory, true_trajectory)
            metrics['trajectory_rmse'] = np.sqrt(metrics['trajectory_mse'])
        
        # Computational efficiency
        if 'computational_efficiency' in self.metrics_to_compute:
            efficiency_metrics = self.computational_efficiency(model_info)
            metrics.update(efficiency_metrics)
        
        # Expert specialization
        if 'expert_specialization' in self.metrics_to_compute and ground_truth_experts is not None and 'routing_weights' in model_info:
            metrics['expert_specialization_mi'] = self.expert_specialization(
                model_info['routing_weights'], ground_truth_experts
            )
        
        # Long-term stability
        if 'long_term_stability' in self.metrics_to_compute:
            metrics['lyapunov_error'] = self.long_term_stability(
                pred_trajectory, true_trajectory, times
            )
        
        # Phase space geometry
        if 'phase_space_geometry' in self.metrics_to_compute:
            metrics['hausdorff_distance'] = self.phase_space_geometry(
                pred_trajectory, true_trajectory
            )
        
        # Energy conservation (if applicable)
        if 'energy_conservation' in self.metrics_to_compute:
            metrics['energy_drift'] = self.energy_conservation(
                pred_trajectory, true_trajectory
            )
        
        # Routing stability
        if 'routing_stability' in self.metrics_to_compute and 'routing_weights' in model_info:
            metrics['routing_entropy_rate'] = self.routing_stability(
                model_info['routing_weights']
            )
        
        return metrics
    
    def trajectory_mse(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> float:
        """Mean squared error between trajectories."""
        return torch.mean((pred - true) ** 2).item()
    
    def computational_efficiency(
        self,
        model_info: Dict[str, Any],
    ) -> Dict[str, float]:
        """Metrics related to computational efficiency."""
        metrics = {}
        
        # Active experts per timestep
        if 'routing_weights' in model_info:
            weights = model_info['routing_weights']
            threshold = self.config['model']['expert_threshold']
            active_experts = (weights > threshold).float().sum(dim=-1).mean()
            metrics['mean_active_experts'] = active_experts.item()
            
            # Sparsity ratio
            sparsity = 1.0 - active_experts / weights.shape[-1]
            metrics['routing_sparsity'] = sparsity.item()
        
        # Step size statistics (if using custom integration)
        if 'step_sizes' in model_info:
            step_sizes = model_info['step_sizes']
            metrics['mean_step_size'] = step_sizes.mean().item()
            metrics['step_size_variance'] = step_sizes.var().item()
            metrics['total_integration_steps'] = len(step_sizes)
        
        return metrics
    
    def expert_specialization(
        self,
        routing_weights: torch.Tensor,
        ground_truth_experts: torch.Tensor,
    ) -> float:
        """Mutual information between learned and ground truth expert assignments."""
        # Convert routing weights to expert assignments
        pred_experts = routing_weights.argmax(dim=-1)
        
        # Flatten for MI computation
        pred_flat = pred_experts.cpu().numpy().flatten()
        true_flat = ground_truth_experts.cpu().numpy().flatten()
        
        # Compute mutual information
        mi = mutual_info_score(true_flat, pred_flat)
        
        # Normalize by entropy of ground truth
        true_entropy = -np.sum([
            p * np.log(p + 1e-8) 
            for p in np.bincount(true_flat) / len(true_flat)
            if p > 0
        ])
        
        return mi / (true_entropy + 1e-8)
    
    def long_term_stability(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
        times: torch.Tensor,
    ) -> float:
        """Estimate error in Lyapunov exponents."""
        # Compute trajectory divergence rate
        T = pred.shape[0]
        if T < 10:
            return 0.0
        
        # Sample pairs of nearby initial conditions
        n_pairs = min(10, pred.shape[1] // 2)
        
        lyapunov_errors = []
        for i in range(n_pairs):
            idx1, idx2 = i * 2, i * 2 + 1
            
            # Initial separation
            initial_sep = torch.norm(true[0, idx1] - true[0, idx2])
            
            if initial_sep < 1e-6:
                continue
            
            # Final separation
            true_final_sep = torch.norm(true[-1, idx1] - true[-1, idx2])
            pred_final_sep = torch.norm(pred[-1, idx1] - pred[-1, idx2])
            
            # Lyapunov exponent estimates
            # Handle both 1D and 2D times tensors
            if times.dim() > 1:
                dt_total = times[0, -1] - times[0, 0]  # Use first batch
            else:
                dt_total = times[-1] - times[0]
            
            true_lyapunov = torch.log(true_final_sep / initial_sep) / dt_total
            pred_lyapunov = torch.log(pred_final_sep / initial_sep) / dt_total
            
            lyapunov_errors.append(
                torch.abs(pred_lyapunov - true_lyapunov).item()
            )
        
        return np.mean(lyapunov_errors) if lyapunov_errors else 0.0
    
    def phase_space_geometry(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> float:
        """Hausdorff distance between predicted and true attractors."""
        # Convert to numpy for scipy
        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()
        
        # Reshape to (n_points, state_dim)
        pred_flat = pred_np.reshape(-1, pred_np.shape[-1])
        true_flat = true_np.reshape(-1, true_np.shape[-1])
        
        # Subsample if too many points
        max_points = 1000
        if len(pred_flat) > max_points:
            indices = np.random.choice(len(pred_flat), max_points, replace=False)
            pred_flat = pred_flat[indices]
        if len(true_flat) > max_points:
            indices = np.random.choice(len(true_flat), max_points, replace=False)
            true_flat = true_flat[indices]
        
        # Compute Hausdorff distance
        d1 = directed_hausdorff(pred_flat, true_flat)[0]
        d2 = directed_hausdorff(true_flat, pred_flat)[0]
        
        return max(d1, d2)
    
    def energy_conservation(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ) -> float:
        """Relative energy drift (placeholder - requires system-specific energy function)."""
        # This is a placeholder - actual implementation would need
        # system-specific Hamiltonian/energy function
        
        # For now, use kinetic energy proxy (velocity magnitude)
        if pred.shape[-1] >= 2:
            # Assume alternating position/velocity dimensions
            pred_vel = pred[..., 1::2]
            true_vel = true[..., 1::2]
            
            pred_energy = 0.5 * torch.sum(pred_vel ** 2, dim=-1)
            true_energy = 0.5 * torch.sum(true_vel ** 2, dim=-1)
            
            # Relative drift
            E0_true = true_energy[0].mean()
            drift_pred = torch.abs(pred_energy - pred_energy[0].unsqueeze(0))
            drift_true = torch.abs(true_energy - true_energy[0].unsqueeze(0))
            
            relative_error = (drift_pred - drift_true).abs() / (E0_true + 1e-8)
            return relative_error.mean().item()
        else:
            return 0.0
    
    def routing_stability(
        self,
        routing_weights: torch.Tensor,
    ) -> float:
        """Temporal consistency of expert routing."""
        # Compute changes in routing weights
        weight_changes = torch.abs(routing_weights[1:] - routing_weights[:-1])
        
        # Average change rate (entropy rate proxy)
        entropy_rate = weight_changes.mean().item()
        
        return entropy_rate


class PerformanceProfiler:
    """Profile computational performance of AME-ODE."""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
    
    def profile_forward_pass(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        n_runs: int = 10,
    ) -> Dict[str, float]:
        """Profile forward pass performance."""
        import time
        
        # Warmup
        for _ in range(3):
            _ = model(x0, t_span)
        
        # Time forward passes
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_runs):
            _ = model(x0, t_span)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        total_time = time.time() - start_time
        
        return {
            'mean_forward_time': total_time / n_runs,
            'throughput': n_runs * x0.shape[0] / total_time,
        }
    
    def profile_memory_usage(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> Dict[str, float]:
        """Profile memory usage."""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0.0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        _ = model(x0, t_span)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return {
            'peak_memory_mb': peak_memory,
            'model_params': sum(p.numel() for p in model.parameters()),
        }


def compute_expert_specialization_matrix(
    model: torch.nn.Module,
    test_states: torch.Tensor,
    n_experts: int,
) -> np.ndarray:
    """Compute expert specialization patterns across state space."""
    device = next(model.parameters()).device
    test_states = test_states.to(device)
    
    # Get expert assignments for test states
    t = torch.zeros(test_states.shape[0]).to(device)
    
    with torch.no_grad():
        # Get initial dynamics estimate
        uniform_weights = torch.ones(test_states.shape[0], n_experts) / n_experts
        uniform_weights = uniform_weights.to(device)
        dx_dt = model.experts(t, test_states, uniform_weights)
        
        # Get routing weights
        weights, _ = model.gating(test_states, dx_dt, t, update_history=False)
    
    return weights.cpu().numpy()


def analyze_expert_diversity(
    model: torch.nn.Module,
    n_samples: int = 1000,
) -> Dict[str, float]:
    """Analyze diversity among learned experts."""
    device = next(model.parameters()).device
    state_dim = model.state_dim
    
    # Sample random states
    test_states = torch.randn(n_samples, state_dim).to(device)
    t = torch.zeros(n_samples).to(device)
    
    with torch.no_grad():
        # Get expert outputs
        expert_outputs = model.experts.get_individual_dynamics(t, test_states)
    
    # Compute pairwise distances
    n_experts = expert_outputs.shape[1]
    distances = np.zeros((n_experts, n_experts))
    
    for i in range(n_experts):
        for j in range(i+1, n_experts):
            diff = expert_outputs[:, i] - expert_outputs[:, j]
            distances[i, j] = torch.norm(diff, dim=-1).mean().item()
            distances[j, i] = distances[i, j]
    
    # Compute diversity metrics
    mean_distance = distances[np.triu_indices_from(distances, k=1)].mean()
    min_distance = distances[distances > 0].min()
    
    return {
        'mean_expert_distance': mean_distance,
        'min_expert_distance': min_distance,
        'distance_matrix': distances,
    }