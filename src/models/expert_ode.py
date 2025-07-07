import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import numpy as np


class ExpertODE(nn.Module):
    """Single expert ODE function for AME-ODE.
    
    Each expert is a neural network that learns dynamics:
    dx/dt = f_θi(x(t), t)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
        residual: bool = True,
        expert_id: int = 0,
        timescale_init: Optional[float] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.expert_id = expert_id
        
        # Input dimension: state + time + sin(ωt) + cos(ωt)
        input_dim = state_dim + 3
        
        # Expert-specific frequency for temporal encoding
        # Initialize as float then convert to parameter to avoid device issues
        omega_value = 1.0 * (expert_id + 1)
        self.omega = nn.Parameter(torch.tensor(omega_value, dtype=torch.float32), requires_grad=True)
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize with timescale-specific scaling
        if timescale_init is not None:
            with torch.no_grad():
                # Scale final layer weights based on characteristic timescale
                self.net[-1].weight.data *= timescale_init
                if self.net[-1].bias is not None:
                    self.net[-1].bias.data *= timescale_init
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing dx/dt.
        
        Args:
            t: Time tensor of shape (1,) or scalar
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            dx/dt tensor of shape (batch_size, state_dim)
        """
        # Ensure t is properly shaped and on the same device as x
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        t = t.to(x.device)
        
        # Temporal encoding - ensure omega is on the same device
        omega = self.omega.to(x.device)
        sin_wt = torch.sin(omega * t).unsqueeze(-1)
        cos_wt = torch.cos(omega * t).unsqueeze(-1)
        
        # Concatenate inputs
        inputs = torch.cat([x, t.unsqueeze(-1), sin_wt, cos_wt], dim=-1)
        
        # Forward through network
        dx_dt = self.net(inputs)
        
        # Note: For ODEs, dx_dt is the derivative, not x + derivative
        # The residual connection in the network architecture is different
        # from adding x to the output (which would be wrong for ODEs)
        
        return dx_dt
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
        }
        return activation_map.get(activation.lower(), nn.ReLU())


class ExpertODEEnsemble(nn.Module):
    """Ensemble of expert ODEs for AME-ODE."""
    
    def __init__(
        self,
        n_experts: int,
        state_dim: int,
        expert_config: Dict[str, Any],
        initialization: str = "diverse",
    ):
        super().__init__()
        
        self.n_experts = n_experts
        self.state_dim = state_dim
        
        # Create experts with different initializations
        self.experts = nn.ModuleList()
        
        for i in range(n_experts):
            # Determine initialization based on strategy
            if initialization == "diverse":
                # Diverse initialization with different timescales
                timescale = self._get_diverse_timescale(i, n_experts)
                activation = self._get_diverse_activation(i, n_experts)
                depth = self._get_diverse_depth(i, n_experts, expert_config['depth'])
                
                expert = ExpertODE(
                    state_dim=state_dim,
                    hidden_dim=expert_config['width'],
                    num_layers=depth,
                    activation=activation,
                    dropout=expert_config.get('dropout', 0.0),
                    residual=expert_config.get('residual', True),
                    expert_id=i,
                    timescale_init=timescale,
                )
            else:
                # Uniform initialization
                expert = ExpertODE(
                    state_dim=state_dim,
                    hidden_dim=expert_config['width'],
                    num_layers=expert_config['depth'],
                    activation=expert_config['activation'],
                    dropout=expert_config.get('dropout', 0.0),
                    residual=expert_config.get('residual', True),
                    expert_id=i,
                )
            
            self.experts.append(expert)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted mixture of expert dynamics.
        
        Args:
            t: Time tensor
            x: State tensor of shape (batch_size, state_dim)
            expert_weights: Gating weights of shape (batch_size, n_experts)
            
        Returns:
            Weighted mixture dx/dt of shape (batch_size, state_dim)
        """
        # Check if we should use sparse computation
        threshold = 0.01 if not self.training else 1e-6
        active_experts = (expert_weights > threshold).any(dim=0)
        n_active = active_experts.sum().item()
        
        # If most experts are active, use vectorized computation
        if n_active > self.n_experts * 0.7:
            # Compute all experts at once (more efficient for many active experts)
            all_dynamics = self.get_individual_dynamics(t, x)
            dx_dt = torch.sum(expert_weights.unsqueeze(-1) * all_dynamics, dim=1)
        else:
            # Sparse computation for few active experts
            dx_dt = torch.zeros_like(x)
            for i in range(self.n_experts):
                if active_experts[i]:
                    # Check if any sample in batch needs this expert
                    batch_mask = expert_weights[:, i] > threshold
                    if batch_mask.any():
                        expert_dx = self.experts[i](t, x[batch_mask])
                        dx_dt[batch_mask] += expert_weights[batch_mask, i:i+1] * expert_dx
        
        return dx_dt
    
    def get_individual_dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get dynamics from each expert separately.
        
        Returns:
            Tensor of shape (batch_size, n_experts, state_dim)
        """
        dynamics = []
        for expert in self.experts:
            dynamics.append(expert(t, x))
        return torch.stack(dynamics, dim=1)
    
    def _get_diverse_timescale(self, expert_id: int, n_experts: int) -> float:
        """Get diverse timescale initialization."""
        # Logarithmically spaced timescales from 0.1 to 10
        timescales = np.logspace(-1, 1, n_experts)
        return float(timescales[expert_id])
    
    def _get_diverse_activation(self, expert_id: int, n_experts: int) -> str:
        """Get diverse activation functions."""
        activations = ["relu", "tanh", "swish", "gelu"]
        return activations[expert_id % len(activations)]
    
    def _get_diverse_depth(self, expert_id: int, n_experts: int, base_depth: int) -> int:
        """Get diverse network depths."""
        # Vary depth from base_depth-1 to base_depth+1
        depth_offset = (expert_id % 3) - 1
        return max(3, base_depth + depth_offset)