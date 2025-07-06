import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class AMEODELoss(nn.Module):
    """Complete loss function for AME-ODE training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Loss weights from config
        reg_config = config['training']['regularization']
        self.lambda_route = float(reg_config['route_weight'])
        self.lambda_expert = float(reg_config['expert_weight'])
        self.lambda_div = float(reg_config['diversity_weight'])
        self.lambda_smooth = float(reg_config['smoothness_weight'])
        self.lambda_balance = float(reg_config['balance_weight'])
        
        # Base reconstruction loss
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        pred_trajectory: torch.Tensor,
        true_trajectory: torch.Tensor,
        model_info: Dict[str, Any],
        model: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components.
        
        Args:
            pred_trajectory: Predicted trajectory (T, batch_size, state_dim)
            true_trajectory: True trajectory (T, batch_size, state_dim)
            model_info: Dictionary with routing weights, expert usage, etc.
            model: AME-ODE model instance
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = self.mse_loss(pred_trajectory, true_trajectory)
        
        # Routing efficiency loss (entropy)
        if 'routing_weights' in model_info:
            losses['routing'] = self._routing_efficiency_loss(model_info['routing_weights'])
        else:
            losses['routing'] = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Expert regularization
        losses['expert_reg'] = self._expert_regularization(model)
        
        # Diversity loss
        losses['diversity'] = self._diversity_loss(model, pred_trajectory)
        
        # Routing smoothness loss
        if 'routing_weights' in model_info:
            losses['smoothness'] = self._routing_smoothness_loss(model_info['routing_weights'])
        else:
            losses['smoothness'] = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Load balancing loss
        if 'expert_usage' in model_info:
            losses['balance'] = self._load_balancing_loss(model_info['expert_usage'])
        else:
            losses['balance'] = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Total loss
        losses['total'] = losses['reconstruction']
        
        # Add regularization terms
        if self.lambda_route > 0:
            losses['total'] = losses['total'] + self.lambda_route * losses['routing']
        if self.lambda_expert > 0:
            losses['total'] = losses['total'] + self.lambda_expert * losses['expert_reg']
        if self.lambda_div > 0:
            losses['total'] = losses['total'] + self.lambda_div * losses['diversity']
        if self.lambda_smooth > 0:
            losses['total'] = losses['total'] + self.lambda_smooth * losses['smoothness']
        if self.lambda_balance > 0:
            losses['total'] = losses['total'] + self.lambda_balance * losses['balance']
        
        return losses
    
    def _routing_efficiency_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Encourage sparse expert utilization through entropy minimization.
        
        Args:
            routing_weights: (T, batch_size, n_experts)
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        weights = routing_weights + eps
        
        # Compute entropy at each time step
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        
        # Average over time and batch
        return entropy.mean()
    
    def _expert_regularization(self, model: nn.Module) -> torch.Tensor:
        """L2 regularization on expert parameters."""
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Regularize expert networks
        for expert in model.experts.experts:
            for param in expert.parameters():
                reg_loss = reg_loss + torch.sum(param ** 2)
        
        # Normalize by number of experts
        reg_loss = reg_loss / model.n_experts
        
        return reg_loss
    
    def _diversity_loss(
        self,
        model: nn.Module,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage diversity among experts by maximizing pairwise distances.
        
        Args:
            model: AME-ODE model
            trajectory: Sample trajectory for evaluation
        """
        # Sample random points from trajectory
        T, B, D = trajectory.shape
        n_samples = min(100, T * B)
        
        # Flatten and sample
        flat_traj = trajectory.reshape(-1, D)
        indices = torch.randperm(flat_traj.shape[0])[:n_samples]
        sample_states = flat_traj[indices]
        
        # Compute expert outputs at sample points
        t_sample = torch.zeros(n_samples, device=trajectory.device)
        expert_outputs = model.experts.get_individual_dynamics(t_sample, sample_states)
        
        # Compute pairwise distances between expert outputs
        n_experts = expert_outputs.shape[1]
        
        # Collect all pairwise distances
        distances = []
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                # L2 distance between expert outputs
                diff = expert_outputs[:, i] - expert_outputs[:, j]
                distance = torch.mean(torch.norm(diff, dim=-1))  # Use L2 norm, not squared
                distances.append(distance)
        
        if distances:
            # Average distance between experts
            avg_distance = torch.stack(distances).mean()
            
            # We want to encourage diversity, but with bounded loss
            # When distance is small (experts similar), loss should be high
            # When distance is large (experts diverse), loss should be low
            # Use log(1 + 1/distance) which is high when distance is small
            diversity_loss = torch.log(1.0 + 1.0 / (avg_distance + 1e-6))
        else:
            diversity_loss = torch.tensor(0.0, device=trajectory.device)
        
        return diversity_loss
    
    def _routing_smoothness_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Encourage temporal consistency in routing decisions.
        
        Args:
            routing_weights: (T, batch_size, n_experts)
        """
        # Compute L2 distance between consecutive routing decisions
        diff = routing_weights[1:] - routing_weights[:-1]
        smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=-1))
        
        return smoothness_loss
    
    def _load_balancing_loss(self, expert_usage: torch.Tensor) -> torch.Tensor:
        """Encourage balanced usage of experts across the batch.
        
        Args:
            expert_usage: Average expert usage (batch_size, n_experts)
        """
        # Compute mean usage across batch
        mean_usage = expert_usage.mean(dim=0)
        
        # Variance of expert usage (want low variance = balanced)
        variance = torch.var(mean_usage)
        
        return variance


class StabilityAwareLoss(AMEODELoss):
    """Extended loss with stability considerations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional stability weights
        self.lambda_lyapunov = config['training'].get('lyapunov_weight', 0.1)
        self.lambda_energy = config['training'].get('energy_weight', 0.01)
    
    def forward(
        self,
        pred_trajectory: torch.Tensor,
        true_trajectory: torch.Tensor,
        model_info: Dict[str, Any],
        model: nn.Module,
        hamiltonian: Optional[callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with stability terms."""
        # Get base losses
        losses = super().forward(pred_trajectory, true_trajectory, model_info, model)
        
        # Add Lyapunov stability loss
        if self.lambda_lyapunov > 0:
            losses['lyapunov'] = self._lyapunov_stability_loss(
                pred_trajectory, model, model_info
            )
            losses['total'] += self.lambda_lyapunov * losses['lyapunov']
        
        # Add energy conservation loss (for Hamiltonian systems)
        if self.lambda_energy > 0 and hamiltonian is not None:
            losses['energy'] = self._energy_conservation_loss(
                pred_trajectory, hamiltonian
            )
            losses['total'] += self.lambda_energy * losses['energy']
        
        return losses
    
    def _lyapunov_stability_loss(
        self,
        trajectory: torch.Tensor,
        model: nn.Module,
        model_info: Dict[str, Any],
    ) -> torch.Tensor:
        """Encourage negative Lyapunov exponents for stability."""
        # Estimate local Lyapunov exponents through trajectory divergence
        T, B, D = trajectory.shape
        
        if T < 2:
            return torch.tensor(0.0, device=trajectory.device)
        
        # Compute trajectory divergence rate
        lyapunov_loss = 0.0
        
        for t in range(T - 1):
            x_t = trajectory[t]
            x_next = trajectory[t + 1]
            
            # Add small perturbations
            eps = 1e-4
            perturbations = torch.randn_like(x_t) * eps
            x_perturbed = x_t + perturbations
            
            # Integrate perturbed trajectory one step
            with torch.no_grad():
                dt = torch.tensor(0.01, device=x_t.device)  # Small time step
                dx_perturbed = model.ode_func(torch.tensor(t * dt), x_perturbed)
                x_next_perturbed = x_perturbed + dt * dx_perturbed
            
            # Measure divergence
            initial_sep = torch.norm(perturbations, dim=-1)
            final_sep = torch.norm(x_next_perturbed - x_next, dim=-1)
            
            # Local Lyapunov exponent estimate
            local_lyapunov = torch.log(final_sep / initial_sep) / dt
            
            # Penalize positive Lyapunov exponents
            lyapunov_loss += torch.mean(torch.relu(local_lyapunov))
        
        return lyapunov_loss / (T - 1)
    
    def _energy_conservation_loss(
        self,
        trajectory: torch.Tensor,
        hamiltonian: callable,
    ) -> torch.Tensor:
        """Penalize energy drift in Hamiltonian systems."""
        T = trajectory.shape[0]
        
        # Compute energy at each time step
        energies = []
        for t in range(T):
            H = hamiltonian(trajectory[t])
            energies.append(H)
        
        energies = torch.stack(energies, dim=0)
        
        # Compute relative energy drift
        E0 = energies[0]
        relative_drift = torch.abs(energies - E0.unsqueeze(0)) / (torch.abs(E0) + 1e-8)
        
        return relative_drift.mean()