import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..models.ame_ode import AMEODE
from ..models.losses import AMEODELoss, StabilityAwareLoss
from ..data.synthetic_systems import SyntheticDataGenerator
from ..utils.config import Config


class TrajectoryDataset(Dataset):
    """Dataset for trajectory data."""
    
    def __init__(self, data_dict: Dict[str, torch.Tensor], sequence_length: Optional[int] = None):
        self.trajectories = data_dict['trajectories']
        self.times = data_dict['times']
        self.initial_conditions = data_dict['initial_conditions']
        self.state_dim = data_dict['state_dim']
        self.sequence_length = sequence_length
        
        # Transpose to (batch, time, state)
        self.trajectories = self.trajectories.transpose(0, 1)
        
    def __len__(self):
        return self.trajectories.shape[0]
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        # Optionally truncate to sequence length
        if self.sequence_length is not None and len(self.times) > self.sequence_length:
            start_idx = np.random.randint(0, len(self.times) - self.sequence_length)
            end_idx = start_idx + self.sequence_length
            trajectory = trajectory[start_idx:end_idx]
            times = self.times[start_idx:end_idx]
        else:
            times = self.times
        
        return {
            'trajectory': trajectory,
            'times': times,
            'initial_condition': trajectory[0],
        }


class AMEODETrainer:
    """Trainer for AME-ODE model."""
    
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = AMEODE(config.to_dict()).to(self.device)
        
        # Apply improved initialization if requested
        if config.model.get('use_improved_init', True):
            from src.models.initialization import initialize_ame_ode
            initialize_ame_ode(self.model, config.model)
        
        # Initialize loss function
        if config.training.get('use_stability_loss', False):
            self.loss_fn = StabilityAwareLoss(config.to_dict())
        else:
            self.loss_fn = AMEODELoss(config.to_dict())
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._build_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'expert_usage': [],
            'routing_entropy': [],
        }
        
        # Create directories
        self.checkpoint_dir = Path(config.logging['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.logging['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Adaptive regularization state
        self.adaptive_regularization = config.training.get('adaptive_regularization', True)
        self.reg_adjustment_history = []
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_name = self.config.training['optimizer'].lower()
        lr = float(self.config.training['learning_rate'])
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        elif opt_name == 'adamw':
            weight_decay = float(self.config.training.get('weight_decay', 1e-5))
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        scheduler_config = self.config.training.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training['num_epochs'],
                eta_min=float(scheduler_config.get('min_lr', 1e-6)),
            )
        elif scheduler_type == 'cosine_with_restarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_config.get('T_0', 50)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('min_lr', 1e-6)),
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 50)),
                gamma=float(scheduler_config.get('gamma', 0.1)),
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'reconstruction': 0.0,
            'routing': 0.0,
            'expert_reg': 0.0,
            'diversity': 0.0,
            'smoothness': 0.0,
            'balance': 0.0,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            trajectory = batch['trajectory'].to(self.device)
            times = batch['times'].to(self.device)
            x0 = batch['initial_condition'].to(self.device)
            
            # Forward pass
            pred_trajectory, model_info = self.model(x0, times)
            
            # Compute loss
            losses = self.loss_fn(pred_trajectory, trajectory, model_info, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training['gradient_clip_norm']
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_metrics:
                if key in losses:
                    epoch_metrics[key] += losses[key].item()
                elif key == 'loss' and 'total' in losses:
                    epoch_metrics[key] += losses['total'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
            })
        
        # Average metrics
        n_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'reconstruction': 0.0,
            'expert_usage_variance': 0.0,
            'routing_entropy': 0.0,
        }
        
        expert_usage_all = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Forward pass
                pred_trajectory, model_info = self.model(x0, times)
                
                # Compute loss
                losses = self.loss_fn(pred_trajectory, trajectory, model_info, self.model)
                
                # Update metrics
                val_metrics['loss'] += losses['total'].item()
                val_metrics['reconstruction'] += losses['reconstruction'].item()
                
                # Collect expert usage
                if 'expert_usage' in model_info:
                    expert_usage_all.append(model_info['expert_usage'])
                
                if 'routing_entropy' in model_info:
                    val_metrics['routing_entropy'] += model_info['routing_entropy'].item()
        
        # Average metrics
        n_batches = len(val_loader)
        for key in val_metrics:
            if key != 'expert_usage_variance':
                val_metrics[key] /= n_batches
        
        # Compute expert usage statistics
        if expert_usage_all:
            expert_usage_all = torch.cat(expert_usage_all, dim=0)
            mean_usage = expert_usage_all.mean(dim=0)
            val_metrics['expert_usage_variance'] = mean_usage.var().item()
        
        return val_metrics
    
    def adaptive_regularization_update(self, metrics: Dict[str, float]):
        """Update regularization weights based on current performance."""
        if not self.adaptive_regularization:
            return
        
        # Check expert similarity (diversity)
        expert_similarity = self._compute_expert_similarity()
        
        # Adjust diversity weight if experts are collapsing
        if expert_similarity > 0.95:
            old_weight = self.loss_fn.lambda_div
            self.loss_fn.lambda_div = min(old_weight * 1.1, 10.0)
            self.reg_adjustment_history.append({
                'epoch': self.current_epoch,
                'type': 'diversity',
                'old': old_weight,
                'new': self.loss_fn.lambda_div,
                'reason': f'expert_similarity={expert_similarity:.3f}',
            })
        
        # Check routing stability
        if 'routing_entropy' in metrics and metrics['routing_entropy'] > 2.0:
            old_weight = self.loss_fn.lambda_smooth
            self.loss_fn.lambda_smooth = min(old_weight * 1.2, 1.0)
            self.reg_adjustment_history.append({
                'epoch': self.current_epoch,
                'type': 'smoothness',
                'old': old_weight,
                'new': self.loss_fn.lambda_smooth,
                'reason': f'routing_entropy={metrics["routing_entropy"]:.3f}',
            })
        
        # Check load balancing
        if 'expert_usage_variance' in metrics and metrics['expert_usage_variance'] > 0.1:
            old_weight = self.loss_fn.lambda_balance
            self.loss_fn.lambda_balance = min(old_weight * 1.1, 2.0)
            self.reg_adjustment_history.append({
                'epoch': self.current_epoch,
                'type': 'balance',
                'old': old_weight,
                'new': self.loss_fn.lambda_balance,
                'reason': f'usage_variance={metrics["expert_usage_variance"]:.3f}',
            })
    
    def _compute_expert_similarity(self) -> float:
        """Compute pairwise similarity between experts."""
        # Sample random states
        x_sample = torch.randn(100, self.model.state_dim).to(self.device)
        t_sample = torch.zeros(100).to(self.device)
        
        # Get expert outputs
        with torch.no_grad():
            expert_outputs = self.model.experts.get_individual_dynamics(t_sample, x_sample)
        
        # Compute pairwise cosine similarities
        n_experts = expert_outputs.shape[1]
        similarities = []
        
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                out_i = expert_outputs[:, i].flatten()
                out_j = expert_outputs[:, j].flatten()
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    out_i.unsqueeze(0),
                    out_j.unsqueeze(0)
                ).item()
                similarities.append(abs(cos_sim))
        
        return max(similarities) if similarities else 0.0
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'reg_adjustment_history': self.reg_adjustment_history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.metrics_history = checkpoint['metrics_history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.reg_adjustment_history = checkpoint.get('reg_adjustment_history', [])
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config.training['num_epochs']
        
        patience = self.config.training['early_stopping_patience']
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Adaptive regularization
            self.adaptive_regularization_update(val_metrics)
            
            # Record metrics
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['routing_entropy'].append(val_metrics['routing_entropy'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Recon: {val_metrics['reconstruction']:.4f}")
            print(f"Routing Entropy: {val_metrics['routing_entropy']:.4f}")
            print(f"Expert Usage Var: {val_metrics['expert_usage_variance']:.4f}")
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.logging['save_frequency'] == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final metrics
        self._save_training_summary()
    
    def _save_training_summary(self):
        """Save training summary and metrics."""
        summary = {
            'final_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history,
            'reg_adjustment_history': self.reg_adjustment_history,
            'config': self.config.to_dict(),
        }
        
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")


def create_data_loaders(
    config: Config,
    system_name: str,
    force_regenerate: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    # Initialize data generator with cache directory
    cache_dir = config.get('cache_dir', 'data/cache/synthetic')
    data_gen = SyntheticDataGenerator(config.to_dict(), cache_dir=cache_dir)
    
    # Generate datasets (will use cache if available)
    train_data = data_gen.generate_dataset(system_name, 'train', force_regenerate=force_regenerate)
    val_data = data_gen.generate_dataset(system_name, 'val', force_regenerate=force_regenerate)
    test_data = data_gen.generate_dataset(system_name, 'test', force_regenerate=force_regenerate)
    
    # Update model config with state dimension
    config._config['model']['state_dim'] = train_data['state_dim']
    
    # Create datasets
    sequence_length = config.training.get('sequence_length', None)
    train_dataset = TrajectoryDataset(train_data, sequence_length)
    val_dataset = TrajectoryDataset(val_data, sequence_length)
    test_dataset = TrajectoryDataset(test_data, sequence_length)
    
    # Create data loaders
    batch_size = config.training['batch_size']
    num_workers = config.compute['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
    
    return train_loader, val_loader, test_loader