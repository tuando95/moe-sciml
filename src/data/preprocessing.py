import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader
import warnings


class DataPreprocessor:
    """Preprocessing pipeline for dynamical systems data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalization_stats = {}
        self.outlier_threshold = 5.0  # Standard deviations
        
    def fit(self, trajectories: torch.Tensor):
        """Fit preprocessing parameters on training data."""
        # Compute normalization statistics
        flat_data = trajectories.reshape(-1, trajectories.shape[-1])
        
        self.normalization_stats['mean'] = flat_data.mean(dim=0)
        self.normalization_stats['std'] = flat_data.std(dim=0)
        
        # Prevent division by zero
        self.normalization_stats['std'] = torch.maximum(
            self.normalization_stats['std'],
            torch.tensor(1e-6)
        )
        
        return self
    
    def transform(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing transformations."""
        # Normalize
        normalized = self.normalize(trajectories)
        
        # Remove outliers
        cleaned = self.remove_outliers(normalized)
        
        return cleaned
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize to zero mean and unit variance."""
        if not self.normalization_stats:
            warnings.warn("Normalization stats not fitted. Returning original data.")
            return data
        
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        
        # Handle different tensor shapes
        if data.dim() == 2:
            return (data - mean) / std
        elif data.dim() == 3:
            return (data - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported data dimension: {data.dim()}")
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if not self.normalization_stats:
            return data
        
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        
        if data.dim() == 2:
            return data * std + mean
        elif data.dim() == 3:
            return data * std.unsqueeze(0).unsqueeze(0) + mean.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported data dimension: {data.dim()}")
    
    def remove_outliers(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Remove trajectories with outliers."""
        # Compute trajectory-wise max norm
        if trajectories.dim() == 3:  # (time, batch, state)
            max_norms = torch.max(
                torch.norm(trajectories, dim=-1),
                dim=0
            )[0]
            
            # Keep trajectories within threshold
            mask = max_norms < self.outlier_threshold
            return trajectories[:, mask]
        else:
            return trajectories
    
    def temporal_subsample(
        self,
        trajectories: torch.Tensor,
        times: torch.Tensor,
        sampling_mode: str = 'uniform',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample trajectories at irregular intervals."""
        if sampling_mode == 'uniform':
            return trajectories, times
        
        elif sampling_mode == 'random':
            # Random subsampling with variable density
            n_points = len(times)
            n_keep = max(n_points // 2, 10)
            
            # Always keep first and last
            indices = [0, n_points - 1]
            
            # Random selection for middle points
            middle_indices = torch.randperm(n_points - 2)[:n_keep-2] + 1
            indices.extend(middle_indices.tolist())
            indices = sorted(indices)
            
            return trajectories[indices], times[indices]
        
        elif sampling_mode == 'adaptive':
            # Keep more points where dynamics change rapidly
            if trajectories.shape[0] < 3:
                return trajectories, times
            
            # Compute local dynamics magnitude
            dx_dt = torch.diff(trajectories, dim=0) / torch.diff(times).unsqueeze(-1).unsqueeze(-1)
            dynamics_norm = torch.norm(dx_dt, dim=-1).mean(dim=-1)
            
            # Adaptive sampling based on dynamics
            threshold = dynamics_norm.quantile(0.5)
            keep_mask = torch.zeros(len(times), dtype=torch.bool)
            keep_mask[0] = True  # Keep first
            keep_mask[-1] = True  # Keep last
            keep_mask[1:-1] = dynamics_norm > threshold
            
            # Ensure minimum number of points
            if keep_mask.sum() < 10:
                indices = torch.argsort(dynamics_norm, descending=True)[:10]
                keep_mask[indices] = True
            
            return trajectories[keep_mask], times[keep_mask]
        
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")


class DataAugmentation:
    """Data augmentation for dynamical systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augmentation_config = config['data'].get('augmentation', {})
    
    def augment_trajectory(
        self,
        trajectory: torch.Tensor,
        mode: str = 'all'
    ) -> torch.Tensor:
        """Apply augmentation to trajectory."""
        if mode == 'none':
            return trajectory
        
        augmented = trajectory.clone()
        
        if self.augmentation_config.get('random_rotation', False) and mode in ['all', 'rotation']:
            augmented = self.random_rotation(augmented)
        
        if self.augmentation_config.get('random_scaling', False) and mode in ['all', 'scaling']:
            augmented = self.random_scaling(augmented)
        
        if self.augmentation_config.get('noise_injection', False) and mode in ['all', 'noise']:
            augmented = self.add_noise(augmented)
        
        return augmented
    
    def random_rotation(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply random rotation in state space."""
        state_dim = trajectory.shape[-1]
        
        if state_dim < 2:
            return trajectory
        
        # Generate random rotation matrix (2D subspace)
        theta = torch.rand(1) * 2 * np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Apply to first two dimensions
        rotated = trajectory.clone()
        x, y = trajectory[..., 0], trajectory[..., 1]
        rotated[..., 0] = cos_theta * x - sin_theta * y
        rotated[..., 1] = sin_theta * x + cos_theta * y
        
        return rotated
    
    def random_scaling(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply random scaling while preserving dynamics structure."""
        # Scale factor between 0.8 and 1.2
        scale = 0.8 + 0.4 * torch.rand(1)
        return trajectory * scale
    
    def add_noise(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Add small noise to trajectory."""
        noise_level = self.augmentation_config.get('noise_level', 0.01)
        noise = torch.randn_like(trajectory) * noise_level
        return trajectory + noise


class ExperimentalDataset(Dataset):
    """Dataset for experimental evaluation with preprocessing."""
    
    def __init__(
        self,
        data_dict: Dict[str, torch.Tensor],
        preprocessor: Optional[DataPreprocessor] = None,
        augmentation: Optional[DataAugmentation] = None,
        sequence_length: Optional[int] = None,
        temporal_sampling: str = 'uniform',
    ):
        self.trajectories = data_dict['trajectories'].transpose(0, 1)  # (batch, time, state)
        self.times = data_dict['times']
        self.initial_conditions = data_dict['initial_conditions']
        self.state_dim = data_dict['state_dim']
        
        # Optional ground truth expert assignments
        if 'ground_truth_experts' in data_dict:
            self.gt_experts = data_dict['ground_truth_experts'].transpose(0, 1)
        else:
            self.gt_experts = None
        
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.sequence_length = sequence_length
        self.temporal_sampling = temporal_sampling
        
        # Apply preprocessing if provided
        if self.preprocessor is not None:
            self.trajectories = self.preprocessor.transform(
                self.trajectories.transpose(0, 1)
            ).transpose(0, 1)
    
    def __len__(self):
        return self.trajectories.shape[0]
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        times = self.times.clone()
        
        # Apply temporal subsampling
        if self.temporal_sampling != 'uniform':
            if self.preprocessor:
                trajectory_for_sampling = trajectory.unsqueeze(1)  # Add batch dim
                times_for_sampling = times
                
                trajectory_sampled, times_sampled = self.preprocessor.temporal_subsample(
                    trajectory_for_sampling.transpose(0, 1),
                    times_for_sampling,
                    self.temporal_sampling
                )
                trajectory = trajectory_sampled.transpose(0, 1).squeeze(1)
                times = times_sampled
        
        # Apply sequence length truncation
        if self.sequence_length is not None and len(times) > self.sequence_length:
            start_idx = np.random.randint(0, len(times) - self.sequence_length)
            end_idx = start_idx + self.sequence_length
            trajectory = trajectory[start_idx:end_idx]
            times = times[start_idx:end_idx]
            
            if self.gt_experts is not None:
                gt_experts = self.gt_experts[idx, start_idx:end_idx-1]
            else:
                gt_experts = None
        else:
            if self.gt_experts is not None:
                gt_experts = self.gt_experts[idx]
            else:
                gt_experts = None
        
        # Apply augmentation
        if self.augmentation is not None and np.random.rand() < 0.5:
            trajectory = self.augmentation.augment_trajectory(trajectory)
        
        sample = {
            'trajectory': trajectory,
            'times': times,
            'initial_condition': trajectory[0],
        }
        
        if gt_experts is not None:
            sample['ground_truth_experts'] = gt_experts
        
        return sample


def create_experimental_dataloaders(
    config: Dict[str, Any],
    system_name: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataPreprocessor]:
    """Create data loaders with full preprocessing pipeline."""
    from ..data.synthetic_systems import SyntheticDataGenerator
    
    # Generate raw data
    data_gen = SyntheticDataGenerator(config)
    train_data = data_gen.generate_dataset(system_name, 'train')
    val_data = data_gen.generate_dataset(system_name, 'val')
    test_data = data_gen.generate_dataset(system_name, 'test')
    
    # Update config with state dimension
    config['model']['state_dim'] = train_data['state_dim']
    
    # Create preprocessor and fit on training data
    preprocessor = DataPreprocessor(config)
    preprocessor.fit(train_data['trajectories'])
    
    # Create augmentation
    augmentation = DataAugmentation(config) if config['data'].get('augmentation', {}) else None
    
    # Determine temporal sampling
    temporal_sampling = config['data'].get('temporal_sampling', 'uniform')
    sequence_length = config['training'].get('sequence_length', None)
    
    # Create datasets
    train_dataset = ExperimentalDataset(
        train_data,
        preprocessor=preprocessor,
        augmentation=augmentation,
        sequence_length=sequence_length,
        temporal_sampling=temporal_sampling if temporal_sampling != 'uniform' else 'uniform',
    )
    
    val_dataset = ExperimentalDataset(
        val_data,
        preprocessor=preprocessor,
        augmentation=None,  # No augmentation for validation
        sequence_length=None,  # Full sequences for validation
        temporal_sampling='uniform',
    )
    
    test_dataset = ExperimentalDataset(
        test_data,
        preprocessor=preprocessor,
        augmentation=None,
        sequence_length=None,
        temporal_sampling='uniform',
    )
    
    # Create data loaders
    if batch_size is None:
        batch_size = config['training']['batch_size']
    if num_workers is None:
        num_workers = config['compute']['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, preprocessor