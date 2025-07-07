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
        
        # Initialize trajectory storage
        trajectories = torch.zeros(n_steps, n_traj, state_dim)
        trajectories[0] = x0
        
        # Euler-Maruyama integration
        for i in range(1, n_steps):
            dt = t_span[i] - t_span[i-1]
            t = t_span[i-1]
            x = trajectories[i-1]
            
            # Deterministic part
            dx_det = self.dynamics(t, x) * dt
            
            # Stochastic part (Brownian motion)
            dW = torch.randn_like(x) * torch.sqrt(dt)
            dx_stoch = process_noise_std * dW
            
            # Update state
            trajectories[i] = x + dx_det + dx_stoch
        
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
        
        # Rotation matrices
        self.A_fast = self._rotation_matrix(self.theta_fast)
        self.A_slow = self._rotation_matrix(self.theta_slow)
        
        # Coupling matrix
        self.C = torch.randn(2, 2) * 0.1
    
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
        return torch.randn(n_samples, self.state_dim)
    
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
        x_inside = torch.randn(n_inside, 3) * (self.R * 0.5)
        
        # Outside: larger radius
        x_outside = torch.randn(n_outside, 3)
        x_outside = x_outside / torch.norm(x_outside, dim=-1, keepdim=True)
        x_outside = x_outside * (self.R * 1.5 + torch.rand(n_outside, 1))
        
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
        super().__init__(config)
        
        # Network parameters
        params = config.get('params', {})
        self.n_oscillators = np.random.randint(*params.get('n_oscillators', [4, 8]))
        self.mu_range = params.get('mu_range', [0.1, 3.0])
        
        # Sample parameters
        self.mu = torch.FloatTensor(self.n_oscillators).uniform_(*self.mu_range)
        
        # Coupling matrix (random sparse coupling)
        coupling_prob = 0.3
        self.coupling = torch.randn(self.n_oscillators, self.n_oscillators) * 0.1
        mask = torch.rand(self.n_oscillators, self.n_oscillators) < coupling_prob
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
        pos = torch.randn(n_samples, self.n_oscillators) * 2
        vel = torch.randn(n_samples, self.n_oscillators) * 2
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
        
        # Generate time span
        t_span = torch.linspace(
            0,
            system_config['trajectory_length'],
            int(system_config['trajectory_length'] / system_config['sampling_dt']) + 1
        )
        
        # Generate trajectories
        print(f"Generating {n_traj} trajectories for {system_name} ({split} split)...")
        noise_std = self.config['data']['noise']['observation_noise']
        process_noise_std = self.config['data']['noise'].get('process_noise', 0.0)
        
        # Show if using SDE integration
        if process_noise_std > 0:
            print(f"  Using SDE integration with process noise std={process_noise_std}")
        
        with tqdm(total=1, desc="  Integrating ODEs", bar_format='{desc}: {bar}') as pbar:
            x0, trajectories = system.generate_trajectories(
                n_traj, t_span, noise_std, process_noise_std
            )
            pbar.update(1)
        
        # Compute ground truth expert assignments (vectorized)
        with tqdm(total=1, desc="  Computing expert assignments", bar_format='{desc}: {bar}') as pbar:
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
            pbar.update(1)
        
        dataset = {
            'initial_conditions': x0,
            'trajectories': trajectories,
            'times': t_span,
            'ground_truth_experts': gt_experts,
            'system_name': system_name,
            'state_dim': system.state_dim,
            'cache_key': cache_key,
        }
        
        # Save to cache
        print(f"  Saving dataset to cache: {cache_path}")
        torch.save(dataset, cache_path)
        
        return dataset