import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from typing import Dict, Any, Tuple, Optional


class SingleNeuralODE(nn.Module):
    """Standard Neural ODE baseline with matching parameter count to AME-ODE."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract dimensions
        self.state_dim = config['model']['state_dim']
        
        # Calculate total parameters in AME-ODE for fair comparison
        expert_params = (
            config['model']['n_experts'] * 
            config['model']['expert_architecture']['width'] * 
            config['model']['expert_architecture']['depth']
        )
        gating_params = (
            config['model']['gating_architecture']['width'] * 
            config['model']['gating_architecture']['depth']
        )
        total_params = expert_params + gating_params
        
        # Design network to match parameter count
        self.num_layers = 4
        self.hidden_dim = int(torch.sqrt(torch.tensor(total_params / self.num_layers)).item())
        
        # Build network
        self.net = self._build_network()
        
        # Integration settings
        self.rtol = float(config['integration']['rtol'])
        self.atol = float(config['integration']['atol'])
        self.method = config['integration']['method']
        self.adjoint = config['integration'].get('adjoint', True)
        self.max_norm = config['integration'].get('dynamics_max_norm', 0.0)  # 0 = no limit
    
    def _build_network(self) -> nn.Module:
        """Build the neural network for dynamics."""
        layers = []
        
        # Input: [x, t, sin(t), cos(t)]
        input_dim = self.state_dim + 3
        
        # Input layer
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.state_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE function: dx/dt = f(x, t)."""
        # Ensure proper shapes
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Temporal encoding
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        
        # Concatenate inputs
        inputs = torch.cat([x, t, sin_t, cos_t], dim=-1)
        
        # Forward through network
        dx_dt = self.net(inputs)
        
        # Optional dynamics bounding for stability
        if hasattr(self, 'max_norm') and self.max_norm > 0:
            dx_dt_norm = torch.norm(dx_dt, dim=-1, keepdim=True)
            scaling_factor = torch.minimum(
                torch.ones_like(dx_dt_norm),
                self.max_norm / (dx_dt_norm + 1e-6)
            )
            dx_dt = dx_dt * scaling_factor
        
        return dx_dt
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate the ODE."""
        # Select integration method
        odeint_func = odeint_adjoint if self.adjoint and self.training else odeint
        
        # Integrate
        if self.method == 'rk4':
            # RK4 is a fixed-step method, doesn't accept rtol/atol
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                method=self.method,
            )
        else:
            # Adaptive methods
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
        
        # Return info for compatibility
        info = {
            'method': 'single_neural_ode',
            'n_function_evals': len(t_span),  # Approximate
        }
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[-1], info


class MultiScaleNeuralODE(nn.Module):
    """Neural ODE with explicit fast/slow timescale separation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.state_dim = config['model']['state_dim']
        self.hidden_dim = config['model']['expert_architecture']['width']
        
        # Separate networks for fast and slow dynamics
        self.fast_net = self._build_network(timescale='fast')
        self.slow_net = self._build_network(timescale='slow')
        
        # Mixing network
        self.mixer = nn.Sequential(
            nn.Linear(self.state_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
        # Integration settings
        self.rtol = float(config['integration']['rtol'])
        self.atol = float(config['integration']['atol'])
        self.method = config['integration']['method']
        self.adjoint = config['integration'].get('adjoint', True)
        self.max_norm = config['integration'].get('dynamics_max_norm', 0.0)  # 0 = no limit
    
    def _build_network(self, timescale: str) -> nn.Module:
        """Build network for specific timescale."""
        layers = []
        input_dim = self.state_dim + 3
        
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.Tanh() if timescale == 'fast' else nn.ReLU())
        
        for _ in range(2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh() if timescale == 'fast' else nn.ReLU())
        
        layers.append(nn.Linear(self.hidden_dim, self.state_dim))
        
        # Initialize with different scales
        net = nn.Sequential(*layers)
        if timescale == 'fast':
            net[-1].weight.data *= 10.0
        else:
            net[-1].weight.data *= 0.1
            
        return net
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale ODE function."""
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Temporal encoding
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        inputs = torch.cat([x, t, sin_t, cos_t], dim=-1)
        
        # Compute fast and slow dynamics
        fast_dx = self.fast_net(inputs)
        slow_dx = self.slow_net(inputs)
        
        # Compute mixing weights
        mix_input = torch.cat([x, t], dim=-1)
        weights = self.mixer(mix_input)
        
        # Weighted combination
        return weights[:, 0:1] * fast_dx + weights[:, 1:2] * slow_dx
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate the ODE."""
        odeint_func = odeint_adjoint if self.adjoint and self.training else odeint
        
        if self.method == 'rk4':
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                method=self.method,
            )
        else:
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
        
        info = {'method': 'multi_scale_neural_ode'}
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[-1], info


class AugmentedNeuralODE(nn.Module):
    """Neural ODE with additional latent dimensions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.state_dim = config['model']['state_dim']
        self.augment_dim = self.state_dim * 2  # Double the state dimension
        self.total_dim = self.state_dim + self.augment_dim
        
        # Network for augmented system
        hidden_dim = config['model']['expert_architecture']['width']
        self.net = nn.Sequential(
            nn.Linear(self.total_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        # Initialize with small weights to prevent instability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Integration settings
        self.rtol = float(config['integration']['rtol'])
        self.atol = float(config['integration']['atol'])
        self.method = config['integration']['method']
        self.adjoint = config['integration'].get('adjoint', True)
        self.max_norm = config['integration'].get('dynamics_max_norm', 0.0)  # 0 = no limit
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Augmented ODE function with stability control."""
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        inputs = torch.cat([z, t], dim=-1)
        dz_dt = self.net(inputs)
        
        # Add stability control: bound the dynamics
        # This prevents explosive growth that causes NaN
        if self.max_norm > 0:  # Only apply if max_norm is set
            dz_dt_norm = torch.norm(dz_dt, dim=-1, keepdim=True)
            scaling_factor = torch.minimum(
                torch.ones_like(dz_dt_norm),
                self.max_norm / (dz_dt_norm + 1e-6)
            )
            dz_dt = dz_dt * scaling_factor
        
        return dz_dt
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate augmented ODE and project back."""
        # Augment initial conditions with zeros
        z0 = torch.cat([x0, torch.zeros_like(x0).repeat(1, 2)], dim=-1)
        
        # Integrate augmented system
        odeint_func = odeint_adjoint if self.adjoint and self.training else odeint
        
        if self.method == 'rk4':
            z_trajectory = odeint_func(
                self,
                z0,
                t_span,
                method=self.method,
            )
        else:
            z_trajectory = odeint_func(
                self,
                z0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
        
        # Project back to original space
        trajectory = z_trajectory[..., :self.state_dim]
        
        info = {
            'method': 'augmented_neural_ode',
            'augment_dim': self.augment_dim,
        }
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[-1], info


class EnsembleNeuralODE(nn.Module):
    """Ensemble of independent Neural ODEs with prediction averaging."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.n_models = config['model']['n_experts']  # Use same number as AME-ODE
        self.state_dim = config['model']['state_dim']
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            self._build_single_model(config, i) 
            for i in range(self.n_models)
        ])
        
        # Integration settings
        self.rtol = float(config['integration']['rtol'])
        self.atol = float(config['integration']['atol'])
        self.method = config['integration']['method']
        self.adjoint = config['integration'].get('adjoint', True)
        self.max_norm = config['integration'].get('dynamics_max_norm', 0.0)  # 0 = no limit
    
    def _build_single_model(self, config: Dict[str, Any], model_id: int) -> nn.Module:
        """Build a single model in the ensemble with diversity."""
        hidden_dim = config['model']['expert_architecture']['width']
        depth = config['model']['expert_architecture']['depth']
        
        layers = []
        input_dim = self.state_dim + 3
        
        # Vary activation functions for diversity
        if model_id % 3 == 0:
            activation = nn.Tanh()
        elif model_id % 3 == 1:
            activation = nn.ReLU()
        else:
            activation = nn.ELU()
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_dim, self.state_dim))
        
        # Different initialization scales for diversity
        net = nn.Sequential(*layers)
        
        # More conservative initialization to prevent instability
        for layer in net:
            if isinstance(layer, nn.Linear):
                # Use xavier initialization with reduced gain
                nn.init.xavier_normal_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Vary final layer scaling more conservatively
        if model_id % 2 == 0:
            net[-1].weight.data *= 0.8
        else:
            net[-1].weight.data *= 1.2
            
        return net
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Ensemble average of dynamics with stability control."""
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        inputs = torch.cat([x, t, sin_t, cos_t], dim=-1)
        
        # Average predictions from all models
        predictions = []
        for model in self.models:
            pred = model(inputs)
            
            # Bound individual model predictions to prevent outliers
            if self.max_norm > 0:
                pred_norm = torch.norm(pred, dim=-1, keepdim=True)
                scaling_factor = torch.minimum(
                    torch.ones_like(pred_norm),
                    self.max_norm / (pred_norm + 1e-6)
                )
                pred = pred * scaling_factor
            
            predictions.append(pred)
        
        # Average the bounded predictions
        dx_dt = torch.stack(predictions).mean(dim=0)
        
        # Additional stability check on the final average
        if self.max_norm > 0:
            dx_dt_norm = torch.norm(dx_dt, dim=-1, keepdim=True)
            final_scaling = torch.minimum(
                torch.ones_like(dx_dt_norm),
                self.max_norm / (dx_dt_norm + 1e-6)
            )
            return dx_dt * final_scaling
        
        return dx_dt
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate ensemble ODE."""
        odeint_func = odeint_adjoint if self.adjoint and self.training else odeint
        
        if self.method == 'rk4':
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                method=self.method,
            )
        else:
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
        
        info = {
            'method': 'ensemble_neural_ode',
            'n_models': self.n_models,
        }
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[-1], info


class TraditionalMoE(nn.Module):
    """Traditional Mixture of Experts with proper ODE integration for fair comparison."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.n_experts = config['model']['n_experts']
        self.state_dim = config['model']['state_dim']
        hidden_dim = config['model']['expert_architecture']['width']
        
        # Expert networks (now with temporal input)
        input_dim = self.state_dim + 3  # state + t + sin(t) + cos(t)
        self.experts = nn.ModuleList()
        
        for i in range(self.n_experts):
            expert = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.state_dim)
            )
            
            # Initialize with small weights for stability
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            
            self.experts.append(expert)
        
        # Gating network with temporal information
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_experts),
            nn.Softmax(dim=-1)
        )
        
        # Integration settings (for fair comparison)
        self.rtol = float(config['integration']['rtol'])
        self.atol = float(config['integration']['atol'])
        self.method = config['integration']['method']
        self.adjoint = config['integration'].get('adjoint', True)
        self.max_norm = float(config['integration'].get('dynamics_max_norm', 0.0))  # 0 = no limit
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """MoE forward with temporal encoding for ODE integration."""
        # Ensure proper shapes
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Temporal encoding
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        
        # Concatenate inputs
        inputs = torch.cat([x, t, sin_t, cos_t], dim=-1)
        
        # Get gating weights
        weights = self.gate(inputs)
        
        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(inputs)
            
            # Bound expert outputs for stability
            if self.max_norm > 0:
                expert_norm = torch.norm(expert_out, dim=-1, keepdim=True)
                scaling_factor = torch.minimum(
                    torch.ones_like(expert_norm),
                    self.max_norm / (expert_norm + 1e-6)
                )
                expert_out = expert_out * scaling_factor
            
            output += weights[:, i:i+1] * expert_out
        
        return output
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        return_all: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate using proper ODE solver for fair comparison."""
        # Use same integration method as other baselines
        odeint_func = odeint_adjoint if self.adjoint and self.training else odeint
        
        if self.method == 'rk4':
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                method=self.method,
            )
        else:
            trajectory = odeint_func(
                self,
                x0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
        
        info = {
            'method': 'traditional_moe',
            'integration': self.method,
            'n_experts': self.n_experts,
        }
        
        if return_all:
            return trajectory, info
        else:
            return trajectory[-1], info