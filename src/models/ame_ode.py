import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np

from .expert_ode import ExpertODEEnsemble
from .gating import AdaptiveGatingModule


class ODEFunc(nn.Module):
    """ODE function wrapper for torchdiffeq integration."""
    
    def __init__(self, experts, gating, n_experts, expert_threshold):
        super().__init__()
        self.experts = experts
        self.gating = gating
        self.n_experts = n_experts
        self.expert_threshold = expert_threshold
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute dx/dt using AME-ODE dynamics."""
        # For inference, use a simpler approximation
        if not self.training:
            # Use zero dynamics for gating (much faster, minimal accuracy loss)
            dx_dt_init = torch.zeros_like(x)
            weights, _ = self.gating(x, dx_dt_init, t, update_history=False)
            
            # Only compute active experts
            active_mask = weights > self.expert_threshold
            if active_mask.any():
                dx_dt = torch.zeros_like(x)
                for i in range(self.n_experts):
                    if active_mask[:, i].any():
                        expert_out = self.experts.experts[i](t, x)
                        dx_dt += weights[:, i:i+1] * expert_out
            else:
                # Fallback to uniform if no experts active
                uniform_weights = torch.ones(x.shape[0], self.n_experts, device=x.device) / self.n_experts
                dx_dt = self.experts(t, x, uniform_weights)
        else:
            # Training: use full computation for better gradients
            uniform_weights = torch.ones(x.shape[0], self.n_experts, device=x.device) / self.n_experts
            dx_dt_init = self.experts(t, x, uniform_weights)
            weights, _ = self.gating(x, dx_dt_init, t, update_history=True)
            dx_dt = self.experts(t, x, weights)
        
        return dx_dt


class AMEODE(nn.Module):
    """Adaptive Mixture of Expert ODEs with torchdiffeq integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configurations
        self.model_config = config['model']
        self.integration_config = config['integration']
        
        # Model dimensions
        self.state_dim = self.model_config.get('state_dim', 4)  # Will be set dynamically
        self.n_experts = self.model_config['n_experts']
        
        # Expert ensemble
        self.experts = ExpertODEEnsemble(
            n_experts=self.n_experts,
            state_dim=self.state_dim,
            expert_config=self.model_config['expert_architecture'],
            initialization='diverse',
        )
        
        # Gating module
        self.gating = AdaptiveGatingModule(
            state_dim=self.state_dim,
            n_experts=self.n_experts,
            history_config=self.model_config['history_embedding'],
            gating_config=self.model_config['gating_architecture'],
        )
        
        # Integration settings
        self.rtol = self.integration_config['rtol']
        self.atol = self.integration_config['atol']
        self.method = self.integration_config['method']
        self.adjoint = self.integration_config.get('adjoint', True)
        
        # Adaptive step size parameters
        self.max_step_size = self.integration_config['max_step_size']
        self.min_step_size = self.integration_config['min_step_size']
        self.adaptive_step = self.integration_config['adaptive_step']
        self.routing_aware_step = self.integration_config['routing_aware_step']
        
        # Expert threshold for computation
        self.expert_threshold = self.model_config['expert_threshold']
        
        # Temperature for gating (prefer gating_architecture setting, fallback to model setting)
        gating_config = self.model_config.get('gating_architecture', {})
        self.temperature = gating_config.get('temperature', self.model_config.get('temperature', 1.0))
    
    def compute_dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamics dx/dt = Σ g_i(x,h,t) * f_i(x,t)."""
        # Initial estimate using uniform weights
        uniform_weights = torch.ones(x.shape[0], self.n_experts, device=x.device) / self.n_experts
        dx_dt_init = self.experts(t, x, uniform_weights)
        
        # Compute gating weights
        weights, _ = self.gating(x, dx_dt_init, t, update_history=True)
        
        # Compute weighted mixture of expert dynamics
        dx_dt = self.experts(t, x, weights)
        
        return dx_dt
    
    def forward(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward integration of AME-ODE.
        
        Args:
            x0: Initial state (batch_size, state_dim)
            t_span: Time points to evaluate at
            return_all: Whether to return all time points or just final
            
        Returns:
            Integrated trajectory and additional info
        """
        # Reset gating history
        self.gating.reset_history(batch_size=x0.shape[0])
        
        # Ensure t_span is 1D (torchdiffeq requirement)
        if t_span.dim() > 1:
            # For batched times, use the first one (they should be identical for ODE integration)
            # Small numerical differences are okay
            t_span = t_span[0]
        
        # Create ODE function wrapper
        ode_func = ODEFunc(self.experts, self.gating, self.n_experts, self.expert_threshold)
        
        # Select integration method
        # Note: Using regular odeint for now due to adjoint method compatibility issues
        odeint_func = odeint
        
        # Create info collector to gather statistics during integration
        info_collector = {
            'routing_weights': [],
            'expert_usage': torch.zeros(x0.shape[0], self.n_experts, device=x0.device),
        }
        
        # Modify ODE function to collect info
        class ODEFuncWithInfo(nn.Module):
            def __init__(self, ode_func, info_collector, n_timepoints):
                super().__init__()
                self.ode_func = ode_func
                self.info_collector = info_collector
                self.n_timepoints = n_timepoints
                self.call_count = 0
                
            def forward(self, t, x):
                dx_dt = self.ode_func(t, x)
                
                # Collect routing info periodically (not every call)
                if self.call_count % 10 == 0:  # Reduce overhead
                    with torch.no_grad():
                        # Get current weights from gating
                        uniform_weights = torch.ones(x.shape[0], self.ode_func.n_experts, device=x.device) / self.ode_func.n_experts
                        dx_dt_est = self.ode_func.experts(t, x, uniform_weights)
                        weights, _ = self.ode_func.gating(x, dx_dt_est, t, update_history=False)
                        
                        self.info_collector['routing_weights'].append(weights)
                        self.info_collector['expert_usage'] += weights / self.n_timepoints
                
                self.call_count += 1
                return dx_dt
        
        ode_func_with_info = ODEFuncWithInfo(ode_func, info_collector, len(t_span))
        
        # Integrate ODE
        if self.adaptive_step:
            # Use adaptive step size from torchdiffeq
            options = {
                'max_num_steps': 10000,
            }
            
            if self.method == 'dopri5':
                options['first_step'] = self.min_step_size
                options['max_step'] = self.max_step_size
            
            trajectory = odeint_func(
                ode_func_with_info,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
                options=options,
            )
        else:
            # Fixed step integration
            trajectory = odeint_func(
                ode_func_with_info,
                x0,
                t_span,
                method='rk4',
                options={'step_size': 0.01},
            )
        
        # Process collected info
        info = {
            'expert_usage': info_collector['expert_usage'],
            'routing_weights': torch.stack(info_collector['routing_weights'], dim=0) if info_collector['routing_weights'] else torch.zeros(0, x0.shape[0], self.n_experts),
            'routing_entropy': -torch.sum(
                info_collector['expert_usage'] * torch.log(info_collector['expert_usage'] + 1e-8),
                dim=-1
            ).mean()
        }
        
        # Transpose to (batch, time, state) to match dataloader format
        trajectory = trajectory.transpose(0, 1)
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[:, -1], info
    
    
    def compute_adaptive_step_size(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dx_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adaptive step size based on dynamics and routing stability."""
        # Dynamic-based step size
        dynamics_norm = torch.norm(dx_dt, dim=-1) + 1e-6
        dt_dynamics = self.max_step_size * torch.ones_like(dynamics_norm)
        dt_dynamics = torch.minimum(
            dt_dynamics,
            self.rtol / dynamics_norm
        )
        
        if self.routing_aware_step:
            # Routing-based step size
            routing_grads = self.gating.gating_network.get_routing_gradients(
                x, self.gating.history_encoder(x, dx_dt, None)[0], t
            )
            max_routing_grad = torch.max(torch.abs(routing_grads), dim=-1)[0]
            dt_routing = self.temperature / (max_routing_grad + 1e-6)
            dt_routing = torch.clamp(dt_routing, self.min_step_size, self.max_step_size)
            
            # Take minimum of both
            dt = torch.minimum(dt_dynamics, dt_routing)
        else:
            dt = dt_dynamics
        
        # Apply bounds
        dt = torch.clamp(dt, self.min_step_size, self.max_step_size)
        
        return dt
    
    def get_expert_dynamics(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get individual expert dynamics and weights.
        
        Returns:
            Expert dynamics (batch_size, n_experts, state_dim)
            Gating weights (batch_size, n_experts)
        """
        # Get dynamics from each expert
        expert_dynamics = self.experts.get_individual_dynamics(t, x)
        
        # Get current gating weights with zero dynamics for initial history
        dx_dt = torch.zeros_like(x)
        weights, _ = self.gating(x, dx_dt, t, update_history=False)
        
        return expert_dynamics, weights
    
    def compute_dynamics_for_viz(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute dynamics for visualization (no gradient tracking)."""
        with torch.no_grad():
            # Reset gating history for visualization batch size
            original_hidden = self.gating.hidden_states
            self.gating.reset_history(batch_size=x.shape[0])
            
            # Initial estimate using uniform weights
            uniform_weights = torch.ones(x.shape[0], self.n_experts) / self.n_experts
            uniform_weights = uniform_weights.to(x.device)
            dx_dt_init = self.experts(t, x, uniform_weights)
            
            # Compute gating weights
            weights, _ = self.gating(x, dx_dt_init, t, update_history=False)
            
            # Compute mixture dynamics
            dx_dt = self.experts(t, x, weights)
            
            # Restore original hidden states
            self.gating.hidden_states = original_hidden
            
            return dx_dt
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Alias for forward method to match baseline API."""
        return self.forward(x0, t_span, return_all)
    
    @torch.no_grad()
    def fast_inference(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Fast inference mode without info collection."""
        self.eval()
        
        # Reset gating history
        self.gating.reset_history(batch_size=x0.shape[0])
        
        # Ensure t_span is 1D
        if t_span.dim() > 1:
            t_span = t_span[0]
        
        # Create lightweight ODE function
        ode_func = ODEFunc(self.experts, self.gating, self.n_experts, self.expert_threshold)
        
        # Simple integration without info collection
        trajectory = odeint(
            ode_func,
            x0,
            t_span,
            method='rk4' if not self.adaptive_step else self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        
        # Return in batch-first format
        return trajectory.transpose(0, 1)


