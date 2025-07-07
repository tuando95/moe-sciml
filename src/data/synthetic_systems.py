import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from abc import ABC, abstractmethod
from torchdiffeq import odeint
from tqdm import tqdm
import os
import hashlib
import json
from pathlib import Path


class SyntheticSystem(ABC):
    """Base class for synthetic dynamical systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_dim = self._get_state_dim()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def _get_state_dim(self) -> int:
        """Return the state dimension of the system."""
        pass
    
    @abstractmethod
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute dx/dt for the system."""
        pass
    
    @abstractmethod
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample initial conditions for trajectories."""
        pass
    
    def generate_trajectories(
        self,
        n_trajectories: int,
        t_span: torch.Tensor,
        noise_std: float = 0.0,
        process_noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectories from the system.
        
        Args:
            n_trajectories: Number of trajectories to generate
            t_span: Time points
            noise_std: Observation noise standard deviation
            process_noise_std: Process noise standard deviation for SDE
        
        Returns:
            Initial conditions and trajectories
        """
        # Sample initial conditions
        x0 = self.sample_initial_conditions(n_trajectories)
        
        if process_noise_std > 0:
            # Use Euler-Maruyama for SDE integration with process noise
            trajectories = self._integrate_sde(x0, t_span, process_noise_std)
        else:
            # Use standard ODE integration
            # For large batches, split into chunks to show progress
            if n_trajectories > 100:
                chunk_size = min(500, n_trajectories // 10)  # Process in chunks
                trajectories_list = []
                
                for i in tqdm(range(0, n_trajectories, chunk_size), 
                             desc="Integrating trajectories", 
                             leave=False):
                    end_idx = min(i + chunk_size, n_trajectories)
                    chunk_x0 = x0[i:end_idx]
                    
                    chunk_traj = odeint(
                        self.dynamics,
                        chunk_x0,
                        t_span,
                        method='dopri5',
                        rtol=1e-6,
                        atol=1e-8,
                    )
                    trajectories_list.append(chunk_traj)
                
                # Concatenate all chunks
                trajectories = torch.cat(trajectories_list, dim=1)
            else:
                # Small batch - process all at once
                trajectories = odeint(
                    self.dynamics,
                    x0,
                    t_span,
                    method='dopri5',
                    rtol=1e-6,
                    atol=1e-8,
                )
        
        # Add observation noise if specified
        if noise_std > 0:
            noise = torch.randn_like(trajectories) * noise_std
            trajectories = trajectories + noise
        
        return x0, trajectories
    
    def _integrate_sde(
        self,
        x0: torch.Tensor,
        t_span: torch.Tensor,
        process_noise_std: float,
    ) -> torch.Tensor:
        """Integrate SDE using Euler-Maruyama method.
        
        dx = f(x,t)dt + σ dW
        where dW is Brownian motion
        """
        n_steps = len(t_span)
        n_traj = x0.shape[0]
        state_dim = x0.shape[1]
        
        # Initialize trajectory storage on same device as x0
        trajectories = torch.zeros(n_steps, n_traj, state_dim, device=x0.device)
        trajectories[0] = x0
        
        # Euler-Maruyama integration with stability checks
        for i in tqdm(range(1, n_steps), desc="SDE integration", leave=False):
            dt = (t_span[i] - t_span[i-1]).item()  # Convert to Python scalar
            t = t_span[i-1]
            x = trajectories[i-1]
            
            # Check for NaN/Inf before dynamics computation
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected at step {i-1}, clamping values")
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                trajectories[i-1] = x
            
            # Deterministic part
            dx_det = self.dynamics(t, x) * dt
            
            # Clamp deterministic dynamics to prevent explosion
            dx_det_norm = torch.norm(dx_det, dim=-1, keepdim=True)
            max_step = 10.0 * dt  # Maximum step size relative to dt
            dx_det = torch.where(
                dx_det_norm > max_step,
                dx_det * max_step / (dx_det_norm + 1e-8),
                dx_det
            )
            
            # Stochastic part (Brownian motion)
            dt_tensor = torch.tensor(dt, device=x.device)  # Create tensor on correct device
            dW = torch.randn_like(x) * torch.sqrt(dt_tensor)
            dx_stoch = process_noise_std * dW
            
            # Update state with clamping
            trajectories[i] = x + dx_det + dx_stoch
            
            # Clamp final values to prevent explosion
            trajectories[i] = torch.clamp(trajectories[i], min=-100.0, max=100.0)
        
        return trajectories
    
    def get_ground_truth_expert_assignment(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Get ground truth expert assignment for validation.
        
        Returns:
            Expert indices (batch_size,)
        """
        # Default: no ground truth available
        return torch.zeros(x.shape[0], dtype=torch.long)


class MultiScaleOscillators(SyntheticSystem):
    """Coupled oscillators with multiple timescales."""
    
    def _get_state_dim(self) -> int:
        return 4  # 2D fast + 2D slow
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # System parameters
        params = config.get('params', {})
        self.omega_fast = params.get('freq_fast', 10.0)
        self.omega_slow = params.get('freq_slow', 0.1)
        self.coupling_range = params.get('coupling_strength', [0.01, 0.1])
        
        # Sample fixed parameters
        self.theta_fast = torch.rand(1) * 2 * np.pi
        self.theta_slow = torch.rand(1) * 2 * np.pi
        self.epsilon = np.random.uniform(*self.coupling_range)
        
        # Rotation matrices (move to GPU)
        self.A_fast = self._rotation_matrix(self.theta_fast).to(self.device)
        self.A_slow = self._rotation_matrix(self.theta_slow).to(self.device)
        
        # Coupling matrix (move to GPU)
        self.C = (torch.randn(2, 2) * 0.1).to(self.device)
    
    def _rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """Create 2D rotation matrix."""
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return torch.tensor([
            [0, -1],
            [1, 0]
        ], dtype=torch.float32) * self.omega_fast
    
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Coupled oscillator dynamics."""
        # Split state
        x_fast = x[..., :2]
        x_slow = x[..., 2:]
        
        # Fast dynamics
        dx_fast = self.omega_fast * (x_fast @ self.A_fast.T)
        dx_fast += self.epsilon * (x_slow @ self.C.T)
        
        # Slow dynamics
        dx_slow = self.omega_slow * (x_slow @ self.A_slow.T)
        dx_slow += self.epsilon * (x_fast @ self.C)
        
        return torch.cat([dx_fast, dx_slow], dim=-1)
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from unit Gaussian."""
        return torch.randn(n_samples, self.state_dim, device=self.device)
    
    def get_ground_truth_expert_assignment(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Assign based on dominant dynamics."""
        # Compute speed of fast and slow components
        speed_fast = torch.norm(dx_dt[..., :2], dim=-1)
        speed_slow = torch.norm(dx_dt[..., 2:], dim=-1)
        
        # Expert 0: fast dynamics dominant
        # Expert 1: slow dynamics dominant
        # Expert 2: balanced dynamics
        ratio = speed_fast / (speed_slow + 1e-6)
        
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long)
        expert_assignment[ratio > 5.0] = 0  # Fast dominant
        expert_assignment[ratio < 0.2] = 1  # Slow dominant
        expert_assignment[(ratio >= 0.2) & (ratio <= 5.0)] = 2  # Balanced
        
        return expert_assignment


class PiecewiseLorenz(SyntheticSystem):
    """Lorenz system with piecewise linear/chaotic regions."""
    
    def _get_state_dim(self) -> int:
        return 3
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Lorenz parameters
        params = config.get('params', {})
        self.sigma = np.random.uniform(*params.get('sigma', [8, 12]))
        self.rho = np.random.uniform(*params.get('rho', [24, 32]))
        self.beta = np.random.uniform(*params.get('beta', [2, 3]))
        self.R = np.random.uniform(*params.get('switching_radius', [5, 15]))
    
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Piecewise Lorenz dynamics."""
        # Compute radius
        radius = torch.norm(x, dim=-1, keepdim=True)
        
        # Linear dynamics (inside sphere)
        dx_linear = -0.5 * x
        
        # Lorenz dynamics (outside sphere)
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dx_lorenz = torch.stack([
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2,
            x1 * x2 - self.beta * x3
        ], dim=-1)
        
        # Smooth transition
        alpha = torch.sigmoid(5 * (radius - self.R))
        dx = (1 - alpha) * dx_linear + alpha * dx_lorenz
        
        return dx
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from regions of interest."""
        # Mix of initial conditions inside and outside switching radius
        n_inside = n_samples // 2
        n_outside = n_samples - n_inside
        
        # Inside: small radius
        x_inside = torch.randn(n_inside, 3, device=self.device) * (self.R * 0.5)
        
        # Outside: larger radius
        x_outside = torch.randn(n_outside, 3, device=self.device)
        x_outside = x_outside / torch.norm(x_outside, dim=-1, keepdim=True)
        x_outside = x_outside * (self.R * 1.5 + torch.rand(n_outside, 1, device=self.device))
        
        return torch.cat([x_inside, x_outside], dim=0)
    
    def get_ground_truth_expert_assignment(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Assign based on linear vs chaotic regime."""
        radius = torch.norm(x, dim=-1)
        
        # Expert 0: linear regime (inside)
        # Expert 1: chaotic regime (outside)
        # Expert 2: transition region
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long)
        expert_assignment[radius < self.R * 0.8] = 0
        expert_assignment[radius > self.R * 1.2] = 1
        expert_assignment[(radius >= self.R * 0.8) & (radius <= self.R * 1.2)] = 2
        
        return expert_assignment


class VanDerPolNetwork(SyntheticSystem):
    """Network of coupled Van der Pol oscillators."""
    
    def __init__(self, config: Dict[str, Any]):
        # Network parameters - set before calling super().__init__()
        params = config.get('params', {})
        self.n_oscillators = np.random.randint(*params.get('n_oscillators', [4, 8]))
        self.mu_range = params.get('mu_range', [0.1, 3.0])
        
        # Now call parent init which will call _get_state_dim()
        super().__init__(config)
        
        # Sample parameters
        self.mu = torch.FloatTensor(self.n_oscillators).uniform_(*self.mu_range).to(self.device)
        
        # Coupling matrix (random sparse coupling)
        coupling_prob = 0.3
        self.coupling = torch.randn(self.n_oscillators, self.n_oscillators, device=self.device) * 0.1
        mask = torch.rand(self.n_oscillators, self.n_oscillators, device=self.device) < coupling_prob
        self.coupling = self.coupling * mask
        self.coupling = (self.coupling + self.coupling.T) / 2  # Symmetric
        self.coupling.fill_diagonal_(0)  # No self-coupling
    
    def _get_state_dim(self) -> int:
        return self.n_oscillators * 2  # Position and velocity for each
    
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Van der Pol network dynamics."""
        # Reshape to (batch, n_oscillators, 2)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.n_oscillators, 2)
        
        # Extract positions and velocities
        pos = x_reshaped[..., 0]
        vel = x_reshaped[..., 1]
        
        # Van der Pol dynamics for each oscillator
        dpos = vel
        dvel = self.mu.unsqueeze(0) * (1 - pos**2) * vel - pos
        
        # Add coupling through positions
        coupling_force = (pos.unsqueeze(1) - pos.unsqueeze(2)) @ self.coupling.T
        dvel += coupling_force.squeeze(1)
        
        # Stack derivatives
        dx = torch.stack([dpos, dvel], dim=-1)
        return dx.view(batch_size, -1)
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from limit cycle region."""
        # Random positions and velocities near limit cycle
        pos = torch.randn(n_samples, self.n_oscillators, device=self.device) * 2
        vel = torch.randn(n_samples, self.n_oscillators, device=self.device) * 2
        x0 = torch.stack([pos, vel], dim=-1)
        return x0.view(n_samples, -1)
    
    def get_ground_truth_expert_assignment(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Assign based on oscillation regime."""
        # Compute average mu for current state
        x_reshaped = x.view(x.shape[0], self.n_oscillators, 2)
        pos = x_reshaped[..., 0]
        
        # Classify based on amplitude
        avg_amplitude = torch.mean(torch.abs(pos), dim=-1)
        
        # Expert 0: small amplitude (nearly linear)
        # Expert 1: medium amplitude (weakly nonlinear)
        # Expert 2: large amplitude (strongly nonlinear)
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long)
        expert_assignment[avg_amplitude < 0.5] = 0
        expert_assignment[(avg_amplitude >= 0.5) & (avg_amplitude < 2.0)] = 1
        expert_assignment[avg_amplitude >= 2.0] = 2
        
        return expert_assignment


class FitzHughNagumo(SyntheticSystem):
    """FitzHugh-Nagumo model of neural excitability with spatial coupling."""
    
    def _get_state_dim(self) -> int:
        return self.n_neurons * 2  # voltage + recovery for each neuron
    
    def __init__(self, config: Dict[str, Any]):
        # Set parameters needed by _get_state_dim() before calling super().__init__()
        params = config.get('params', {})
        self.n_neurons = params.get('n_neurons', 10)
        self.a = params.get('a', 0.7)  # Recovery timescale
        self.b = params.get('b', 0.8)  # Recovery coupling
        self.I_ext_range = params.get('I_ext', [0.0, 0.5])  # External current
        
        # Now call parent init
        super().__init__(config)
        
        # Heterogeneous parameters for each neuron
        self.I_ext = torch.linspace(*self.I_ext_range, self.n_neurons).to(self.device)
        
        # Coupling matrix (nearest neighbor + some long-range)
        self.coupling = self._create_coupling_matrix()
        
    def _create_coupling_matrix(self):
        """Create heterogeneous coupling matrix."""
        W = torch.zeros(self.n_neurons, self.n_neurons)
        # Nearest neighbor coupling
        for i in range(self.n_neurons - 1):
            W[i, i+1] = W[i+1, i] = 0.1
        # Add some long-range connections
        W[0, -1] = W[-1, 0] = 0.05  # Ring topology
        return W.to(self.device)
    
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """FitzHugh-Nagumo dynamics with 3 regimes: resting, spiking, bursting."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.n_neurons, 2)
        
        v = x_reshaped[..., 0]  # Voltage
        w = x_reshaped[..., 1]  # Recovery variable
        
        # Voltage dynamics: dv/dt = v - v³/3 - w + I_ext + coupling
        dv = v - v**3 / 3 - w + self.I_ext.unsqueeze(0)
        
        # Add diffusive coupling
        coupling_term = (v.unsqueeze(1) - v.unsqueeze(2)) @ self.coupling.T
        dv += coupling_term.squeeze(1)
        
        # Recovery dynamics: dw/dt = (v + a - b*w) / τ
        # τ varies with activity level creating multiple timescales
        tau = 12.5 + 5.0 * torch.sigmoid(v)  # Activity-dependent timescale
        dw = (v + self.a - self.b * w) / tau
        
        dx = torch.stack([dv, dw], dim=-1)
        return dx.view(batch_size, -1)
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from different activity states."""
        # Mix of resting, spiking, and intermediate states
        v = torch.randn(n_samples, self.n_neurons, device=self.device) * 0.5
        w = torch.randn(n_samples, self.n_neurons, device=self.device) * 0.1
        
        # Some neurons start in excited state
        excited_neurons = torch.rand(n_samples, self.n_neurons, device=self.device) < 0.2
        v[excited_neurons] = 1.5
        
        x0 = torch.stack([v, w], dim=-1)
        return x0.view(n_samples, -1)
    
    def get_ground_truth_expert_assignment(self, x: torch.Tensor, dx_dt: torch.Tensor) -> torch.Tensor:
        """Assign based on neural activity regime."""
        x_reshaped = x.view(x.shape[0], self.n_neurons, 2)
        v = x_reshaped[..., 0]
        
        # Average activity level
        avg_v = torch.mean(v, dim=-1)
        max_v = torch.max(torch.abs(v), dim=-1)[0]
        
        # Expert 0: Resting state (low activity)
        # Expert 1: Spiking (high activity, regular)
        # Expert 2: Bursting/chaotic (very high activity)
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        expert_assignment[max_v < 0.5] = 0  # Resting
        expert_assignment[(max_v >= 0.5) & (max_v < 1.5)] = 1  # Spiking
        expert_assignment[max_v >= 1.5] = 2  # Bursting
        
        return expert_assignment


class PredatorPreyMigration(SyntheticSystem):
    """Spatial predator-prey dynamics with seasonal migration."""
    
    def _get_state_dim(self) -> int:
        return self.n_patches * 2  # prey + predator in each patch
    
    def __init__(self, config: Dict[str, Any]):
        # Set parameters needed by _get_state_dim() before calling super().__init__()
        params = config.get('params', {})
        self.n_patches = params.get('n_patches', 5)
        
        # Now call parent init
        super().__init__(config)
        
        # Heterogeneous carrying capacities (environmental gradient)
        self.K = torch.linspace(0.5, 2.0, self.n_patches).to(self.device)
        
        # Growth and interaction parameters
        self.r = params.get('growth_rate', 1.0)  # Prey growth
        self.a = params.get('predation_rate', 1.2)  # Predation
        self.e = params.get('conversion_efficiency', 0.6)
        self.d = params.get('predator_death', 0.5)
        
        # Migration rates (seasonal)
        self.m_prey = params.get('prey_migration', 0.1)
        self.m_pred = params.get('predator_migration', 0.05)
        
        # Migration connectivity (1D chain with periodic boundary)
        self.migration_matrix = self._create_migration_matrix()
        
    def _create_migration_matrix(self):
        """Create migration connectivity matrix."""
        M = torch.zeros(self.n_patches, self.n_patches)
        for i in range(self.n_patches):
            M[i, (i-1) % self.n_patches] = 1.0
            M[i, (i+1) % self.n_patches] = 1.0
            M[i, i] = -2.0
        return M.to(self.device)
    
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Predator-prey dynamics with migration."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.n_patches, 2)
        
        N = x_reshaped[..., 0]  # Prey density
        P = x_reshaped[..., 1]  # Predator density
        
        # Ensure t is properly shaped for broadcasting
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)
        
        # Seasonal forcing affects carrying capacity
        season = 1.0 + 0.3 * torch.sin(2 * np.pi * t.view(-1, 1) / 10.0)
        K_seasonal = self.K.unsqueeze(0) * season
        
        # Local dynamics (Rosenzweig-MacArthur)
        # Prey: dN/dt = rN(1-N/K) - aNP/(1+N)
        dN_local = self.r * N * (1 - N / K_seasonal) - self.a * N * P / (1 + N)
        
        # Predator: dP/dt = eaNP/(1+N) - dP
        dP_local = self.e * self.a * N * P / (1 + N) - self.d * P
        
        # Migration based on resource gradient
        # Prey migrate towards high carrying capacity
        # Predators follow prey density
        migration_N = self.m_prey * (N @ self.migration_matrix.T)
        migration_P = self.m_pred * (P @ self.migration_matrix.T)
        
        # Total dynamics
        dN = dN_local + migration_N
        dP = dP_local + migration_P
        
        # Ensure non-negative (important for ecological realism)
        dN = torch.where(N <= 0, torch.maximum(dN, torch.zeros_like(dN)), dN)
        dP = torch.where(P <= 0, torch.maximum(dP, torch.zeros_like(dP)), dP)
        
        dx = torch.stack([dN, dP], dim=-1)
        return dx.view(batch_size, -1)
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample ecologically realistic initial conditions."""
        # Prey near carrying capacity with variation
        N0 = self.K.unsqueeze(0) * (0.5 + 0.5 * torch.rand(n_samples, self.n_patches, device=self.device))
        
        # Predators at lower density
        P0 = 0.2 * torch.rand(n_samples, self.n_patches, device=self.device) + 0.1
        
        # Add spatial heterogeneity - some patches empty
        empty_patches = torch.rand(n_samples, self.n_patches, device=self.device) < 0.2
        N0[empty_patches] = 0.1
        P0[empty_patches] = 0.01
        
        x0 = torch.stack([N0, P0], dim=-1)
        return x0.view(n_samples, -1)
    
    def get_ground_truth_expert_assignment(self, x: torch.Tensor, dx_dt: torch.Tensor) -> torch.Tensor:
        """Assign based on ecological regime."""
        x_reshaped = x.view(x.shape[0], self.n_patches, 2)
        N = x_reshaped[..., 0]
        P = x_reshaped[..., 1]
        
        # Total biomass and predator-prey ratio
        total_N = torch.sum(N, dim=-1)
        total_P = torch.sum(P, dim=-1)
        ratio = total_P / (total_N + 0.1)
        
        # Expert 0: Prey-dominated (low predator ratio)
        # Expert 1: Balanced coexistence
        # Expert 2: Predator outbreak or crash dynamics
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        expert_assignment[ratio < 0.1] = 0  # Prey dominated
        expert_assignment[(ratio >= 0.1) & (ratio < 0.5)] = 1  # Balanced
        expert_assignment[ratio >= 0.5] = 2  # Predator dominated/crash
        
        return expert_assignment


class TMDD(SyntheticSystem):
    """Target-Mediated Drug Disposition model with multiple binding states."""
    
    def _get_state_dim(self) -> int:
        return 4  # Drug, Target, Complex, Internalized
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        params = config.get('params', {})
        
        # PK parameters
        self.k_el = params.get('k_elimination', 0.1)  # Drug elimination
        self.k_syn = params.get('k_synthesis', 1.0)  # Target synthesis
        self.k_deg = params.get('k_degradation', 0.2)  # Target degradation
        
        # Binding parameters
        self.k_on = params.get('k_on', 0.5)  # Binding rate
        self.k_off = params.get('k_off', 0.05)  # Dissociation rate
        self.k_int = params.get('k_internalization', 0.1)  # Complex internalization
        
        # Nonlinear feedback
        self.IC50_feedback = params.get('IC50_feedback', 10.0)
        self.n_hill = params.get('n_hill', 2.0)
        
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """TMDD dynamics with nonlinear feedback."""
        # States: [Drug (D), Target (T), Complex (DT), Internalized (I)]
        D = x[..., 0]
        T = x[..., 1]
        DT = x[..., 2]
        I = x[..., 3]
        
        # Feedback: Target synthesis increases when drug is present
        feedback = 1 + 3 * (D**self.n_hill) / (self.IC50_feedback**self.n_hill + D**self.n_hill)
        
        # Drug dynamics
        dD = -self.k_el * D - self.k_on * D * T + self.k_off * DT
        
        # Target dynamics with feedback
        dT = self.k_syn * feedback - self.k_deg * T - self.k_on * D * T + self.k_off * DT
        
        # Complex dynamics
        dDT = self.k_on * D * T - self.k_off * DT - self.k_int * DT
        
        # Internalized complex
        dI = self.k_int * DT - 0.5 * self.k_deg * I
        
        return torch.stack([dD, dT, dDT, dI], dim=-1)
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from different dosing scenarios."""
        # Different initial drug concentrations (doses)
        D0 = torch.exp(torch.randn(n_samples, device=self.device) * 2) * 10  # Log-normal around 10
        
        # Target at steady state
        T0 = torch.ones(n_samples, device=self.device) * (self.k_syn / self.k_deg)
        
        # No complex or internalized initially
        DT0 = torch.zeros(n_samples, device=self.device)
        I0 = torch.zeros(n_samples, device=self.device)
        
        return torch.stack([D0, T0, DT0, I0], dim=-1)
    
    def get_ground_truth_expert_assignment(self, x: torch.Tensor, dx_dt: torch.Tensor) -> torch.Tensor:
        """Assign based on binding regime."""
        D = x[..., 0]
        T = x[..., 1]
        DT = x[..., 2]
        
        # Binding saturation
        saturation = DT / (T + DT + 1e-6)
        
        # Expert 0: Linear phase (low drug, unsaturated)
        # Expert 1: Saturation phase (high binding)
        # Expert 2: Depletion phase (target depleted)
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        expert_assignment[(D < 1.0) & (saturation < 0.2)] = 0  # Linear
        expert_assignment[(saturation >= 0.2) & (saturation < 0.8)] = 1  # Saturating
        expert_assignment[(saturation >= 0.8) | (T < 0.1)] = 2  # Depleted
        
        return expert_assignment


class TumorImmune(SyntheticSystem):
    """Tumor-immune dynamics with PD-1/PD-L1 checkpoint inhibition."""
    
    def _get_state_dim(self) -> int:
        return 5  # Tumor, Effector T cells, Regulatory T cells, PD-1, PD-L1
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        params = config.get('params', {})
        
        # Tumor parameters
        self.r_tumor = params.get('tumor_growth', 0.5)
        self.K_tumor = params.get('tumor_capacity', 100.0)
        
        # Immune parameters
        self.k_kill = params.get('kill_rate', 1.0)
        self.k_stim = params.get('stimulation', 0.2)
        self.d_eff = params.get('effector_death', 0.1)
        self.d_reg = params.get('regulatory_death', 0.05)
        
        # Checkpoint parameters
        self.k_pd1 = params.get('pd1_expression', 0.1)
        self.k_pdl1 = params.get('pdl1_expression', 0.2)
        self.k_bind = params.get('checkpoint_binding', 0.5)
        self.k_inhibit = params.get('inhibition_strength', 0.8)
        
        # Treatment (anti-PD-1 antibody concentration)
        self.treatment_schedule = params.get('treatment_schedule', 'none')
        
    def dynamics(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Tumor-immune dynamics with checkpoint interactions."""
        T = x[..., 0]   # Tumor cells
        E = x[..., 1]   # Effector T cells
        R = x[..., 2]   # Regulatory T cells
        P1 = x[..., 3]  # PD-1 expression
        PL = x[..., 4]  # PD-L1 expression
        
        # Ensure t is properly shaped
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        
        # Treatment effect (periodic dosing)
        if self.treatment_schedule == 'periodic':
            drug = 10.0 * (torch.sin(0.5 * t) > 0.8).float()
        else:
            drug = torch.zeros_like(t)
        
        # Ensure drug is properly shaped for broadcasting
        if drug.dim() == 1:
            drug = drug.view(-1, 1)
        
        # Checkpoint inhibition reduces immune suppression
        checkpoint_bound = self.k_bind * P1 * PL / (1 + P1 * PL)
        inhibition = self.k_inhibit * checkpoint_bound / (1 + drug.squeeze(-1))
        
        # Tumor dynamics (logistic growth - immune killing)
        kill_rate = self.k_kill * (1 - inhibition)  # Inhibition reduces killing
        dT = self.r_tumor * T * (1 - T / self.K_tumor) - kill_rate * E * T / (1 + R)
        
        # Effector T cell dynamics (stimulation by tumor - suppression)
        dE = self.k_stim * T * E / (10 + T) - self.d_eff * E - inhibition * E
        
        # Regulatory T cell dynamics (recruited by tumor)
        dR = 0.1 * T * R / (50 + T) - self.d_reg * R
        
        # PD-1 expression (upregulated by activation)
        dP1 = self.k_pd1 * E / (1 + E) - 0.1 * P1 - drug.squeeze(-1) * P1
        
        # PD-L1 expression (upregulated by tumor)
        dPL = self.k_pdl1 * T / (10 + T) - 0.1 * PL
        
        # Ensure non-negative
        dx = torch.stack([dT, dE, dR, dP1, dPL], dim=-1)
        return torch.where(x > 0, dx, torch.maximum(dx, torch.zeros_like(dx)))
    
    def sample_initial_conditions(self, n_samples: int) -> torch.Tensor:
        """Sample from different disease stages."""
        # Early stage: small tumor, active immune
        early = n_samples // 3
        T0_early = torch.rand(early, device=self.device) * 10 + 1
        E0_early = torch.rand(early, device=self.device) * 5 + 5
        
        # Mid stage: growing tumor, struggling immune
        mid = n_samples // 3
        T0_mid = torch.rand(mid, device=self.device) * 30 + 20
        E0_mid = torch.rand(mid, device=self.device) * 3 + 2
        
        # Late stage: large tumor, exhausted immune
        late = n_samples - early - mid
        T0_late = torch.rand(late, device=self.device) * 30 + 60
        E0_late = torch.rand(late, device=self.device) * 2 + 0.5
        
        # Combine
        T0 = torch.cat([T0_early, T0_mid, T0_late])
        E0 = torch.cat([E0_early, E0_mid, E0_late])
        R0 = torch.rand(n_samples, device=self.device) * 2
        P1_0 = torch.rand(n_samples, device=self.device) * 0.5
        PL_0 = T0 / 100  # PD-L1 proportional to tumor
        
        return torch.stack([T0, E0, R0, P1_0, PL_0], dim=-1)
    
    def get_ground_truth_expert_assignment(self, x: torch.Tensor, dx_dt: torch.Tensor) -> torch.Tensor:
        """Assign based on disease/treatment phase."""
        T = x[..., 0]
        E = x[..., 1]
        
        # Immune-tumor ratio
        ratio = E / (T + 1)
        
        # Expert 0: Immune control (high ratio)
        # Expert 1: Dynamic balance (medium ratio)
        # Expert 2: Tumor escape (low ratio)
        expert_assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        expert_assignment[ratio > 0.5] = 0  # Immune control
        expert_assignment[(ratio >= 0.1) & (ratio <= 0.5)] = 1  # Balance
        expert_assignment[ratio < 0.1] = 2  # Tumor escape
        
        return expert_assignment


class SyntheticDataGenerator:
    """Generate synthetic data for AME-ODE experiments with caching support."""
    
    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str] = None):
        self.config = config
        self.systems = self._initialize_systems()
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = config.get('cache_dir', 'data/cache/synthetic')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_systems(self) -> Dict[str, SyntheticSystem]:
        """Initialize all enabled synthetic systems."""
        systems = {}
        
        for system_config in self.config['data']['synthetic_systems']:
            if system_config['enabled']:
                name = system_config['name']
                
                if name == 'multi_scale_oscillators':
                    systems[name] = MultiScaleOscillators(system_config)
                elif name == 'piecewise_lorenz':
                    systems[name] = PiecewiseLorenz(system_config)
                elif name == 'van_der_pol_network':
                    systems[name] = VanDerPolNetwork(system_config)
                elif name == 'fitzhugh_nagumo':
                    systems[name] = FitzHughNagumo(system_config)
                elif name == 'predator_prey_migration':
                    systems[name] = PredatorPreyMigration(system_config)
                elif name == 'tmdd':
                    systems[name] = TMDD(system_config)
                elif name == 'tumor_immune':
                    systems[name] = TumorImmune(system_config)
                else:
                    print(f"Unknown system: {name}")
        
        return systems
    
    def _get_cache_key(self, system_name: str, split: str, system_config: Dict) -> str:
        """Generate a unique cache key based on system configuration."""
        # Include noise configuration in the cache key
        cache_config = {
            'system': system_config,
            'noise': self.config['data']['noise'],
            'augmentation': self.config['data'].get('augmentation', {})
        }
        
        # Create a deterministic hash of the configuration
        config_str = json.dumps(cache_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{system_name}_{split}_{config_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.pt"
    
    def generate_dataset(
        self,
        system_name: str,
        split: str = 'train',
        force_regenerate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate dataset for a specific system and split with caching.
        
        Args:
            system_name: Name of the synthetic system
            split: Dataset split ('train', 'val', or 'test')
            force_regenerate: If True, regenerate even if cached version exists
        """
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not found or not enabled")
        
        # Get system configuration
        system_config = next(
            cfg for cfg in self.config['data']['synthetic_systems']
            if cfg['name'] == system_name
        )
        
        # Check cache
        cache_key = self._get_cache_key(system_name, split, system_config)
        cache_path = self._get_cache_path(cache_key)
        
        if not force_regenerate and cache_path.exists():
            print(f"Loading cached {system_name} dataset ({split} split) from {cache_path}")
            try:
                dataset = torch.load(cache_path)
                print(f"  Loaded {dataset['trajectories'].shape[1]} trajectories")
                return dataset
            except Exception as e:
                print(f"  Failed to load cache: {e}")
                print("  Regenerating dataset...")
        
        # Generate dataset
        system = self.systems[system_name]
        
        # Determine number of trajectories for split
        n_total = system_config['n_trajectories']
        splits = self.config['data']['train_val_test_split']
        
        if split == 'train':
            n_traj = int(n_total * splits[0])
        elif split == 'val':
            n_traj = int(n_total * splits[1])
        elif split == 'test':
            n_traj = int(n_total * splits[2])
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Generate time span on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_span = torch.linspace(
            0,
            system_config['trajectory_length'],
            int(system_config['trajectory_length'] / system_config['sampling_dt']) + 1,
            device=device
        )
        
        # Generate trajectories
        print(f"Generating {n_traj} trajectories for {system_name} ({split} split)...")
        print(f"  Using device: {device}")
        noise_std = self.config['data']['noise']['observation_noise']
        process_noise_std = self.config['data']['noise'].get('process_noise', 0.0)
        
        # Show if using SDE integration
        if process_noise_std > 0:
            print(f"  Using SDE integration with process noise std={process_noise_std}")
        
        # Generate trajectories with progress tracking
        x0, trajectories = system.generate_trajectories(
            n_traj, t_span, noise_std, process_noise_std
        )
        print(f"  Generated trajectories shape: {trajectories.shape}")
        
        # Compute ground truth expert assignments (vectorized)
        print("  Computing expert assignments...")
        # Compute all derivatives at once
        dx_dt = (trajectories[1:] - trajectories[:-1]) / system_config['sampling_dt']
        
        # Get expert assignments for all timesteps at once
        # Reshape to (n_timesteps * n_trajectories, state_dim) for batch processing
        n_timesteps = len(t_span) - 1
        x_flat = trajectories[:-1].reshape(-1, system.state_dim)
        dx_dt_flat = dx_dt.reshape(-1, system.state_dim)
        
        # Get all expert assignments at once
        expert_assignments_flat = system.get_ground_truth_expert_assignment(x_flat, dx_dt_flat)
        
        # Reshape back to (n_timesteps, n_trajectories)
        gt_experts = expert_assignments_flat.reshape(n_timesteps, n_traj)
        print(f"  Expert assignments shape: {gt_experts.shape}")
        
        # Move to CPU for caching (to save GPU memory and for compatibility)
        dataset = {
            'initial_conditions': x0.cpu(),
            'trajectories': trajectories.cpu(),
            'times': t_span.cpu(),
            'ground_truth_experts': gt_experts.cpu(),
            'system_name': system_name,
            'state_dim': system.state_dim,
            'cache_key': cache_key,
        }
        
        # Save to cache
        print(f"  Saving dataset to cache: {cache_path}")
        torch.save(dataset, cache_path)
        
        return dataset