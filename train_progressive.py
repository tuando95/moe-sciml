#!/usr/bin/env python3
"""Train AME-ODE with progressive regularization schedule."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path

from src.utils.config import Config
from src.training.trainer import AMEODETrainer, create_data_loaders


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProgressiveAMEODETrainer(AMEODETrainer):
    """AME-ODE trainer with progressive regularization."""
    
    def __init__(self, config, device='cuda'):
        super().__init__(config, device)
        
        # Store initial regularization weights - ensure they are floats
        self.initial_reg_weights = {
            'route': 0.0,
            'expert': float(config.training['regularization']['expert_weight']),
            'diversity': 0.0,
            'smoothness': float(config.training['regularization']['smoothness_weight']),
            'balance': 0.0
        }
        
        # Target regularization weights (reached at end of training)
        self.target_reg_weights = {
            'route': 0.0001,
            'expert': 1e-5,
            'diversity': 0.00001,  # Very light diversity
            'smoothness': 0.0001,
            'balance': 0.00001    # Very light balance
        }
        
        # Phase transitions
        self.warmup_epochs = 20  # Pure reconstruction
        self.rampup_epochs = 50  # Gradually introduce regularization
        
    def update_regularization_weights(self, epoch):
        """Progressively increase regularization weights."""
        if epoch < self.warmup_epochs:
            # Pure reconstruction phase
            progress = 0.0
        elif epoch < self.warmup_epochs + self.rampup_epochs:
            # Ramp up phase
            progress = (epoch - self.warmup_epochs) / self.rampup_epochs
        else:
            # Full regularization
            progress = 1.0
        
        # Update weights in the loss function
        for key in self.initial_reg_weights:
            current_weight = (
                self.initial_reg_weights[key] * (1 - progress) +
                self.target_reg_weights[key] * progress
            )
            # Update the loss function's weights directly
            if key == 'route':
                self.loss_fn.lambda_route = current_weight
            elif key == 'expert':
                self.loss_fn.lambda_expert = current_weight
            elif key == 'diversity':
                self.loss_fn.lambda_div = current_weight
            elif key == 'smoothness':
                self.loss_fn.lambda_smooth = current_weight
            elif key == 'balance':
                self.loss_fn.lambda_balance = current_weight
    
    def train_epoch(self, dataloader):
        """Train for one epoch with progressive regularization."""
        # Update regularization weights based on current epoch
        self.update_regularization_weights(self.current_epoch)
        
        # Call parent train_epoch
        metrics = super().train_epoch(dataloader)
        
        # Log current regularization weights every 10 epochs
        if self.current_epoch % 10 == 0:
            print(f"\nRegularization weights at epoch {self.current_epoch}:")
            print(f"  route_weight: {self.loss_fn.lambda_route:.2e}")
            print(f"  expert_weight: {self.loss_fn.lambda_expert:.2e}")
            print(f"  diversity_weight: {self.loss_fn.lambda_div:.2e}")
            print(f"  smoothness_weight: {self.loss_fn.lambda_smooth:.2e}")
            print(f"  balance_weight: {self.loss_fn.lambda_balance:.2e}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train AME-ODE with progressive regularization')
    parser.add_argument('--config', type=str, default='configs/minimal_reg_ame.yml',
                        help='Path to configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(Path(args.config))
    
    # Add improved initialization settings
    if 'use_improved_init' not in config.model:
        config._config['model']['use_improved_init'] = True
    if 'expert_init_strategy' not in config.model:
        config._config['model']['expert_init_strategy'] = 'spectral'
    if 'gating_init_strategy' not in config.model:
        config._config['model']['gating_init_strategy'] = 'sparse'
    
    # Set random seed
    set_seed(args.seed)
    
    # Create data loaders
    print(f"Loading {args.system} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.system, force_regenerate=False
    )
    
    # Initialize trainer with progressive regularization
    print("\nInitializing AME-ODE with progressive regularization...")
    trainer = ProgressiveAMEODETrainer(config, device=args.device)
    
    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nModel initialized with {total_params:,} parameters")
    print(f"Number of experts: {trainer.model.n_experts}")
    print(f"Gating temperature: {trainer.model.temperature}")
    print(f"\nProgressive training schedule:")
    print(f"  Warmup epochs (pure reconstruction): {trainer.warmup_epochs}")
    print(f"  Rampup epochs (gradual regularization): {trainer.rampup_epochs}")
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    trainer.model.eval()
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            trajectory = batch['trajectory'].to(trainer.device)
            times = batch['times'].to(trainer.device)
            x0 = batch['initial_condition'].to(trainer.device)
            
            # Ensure times is 1D
            if times.dim() > 1:
                times = times[0]
            
            # Forward pass
            pred_traj, info = trainer.model(x0, times)
            
            # Compute loss
            losses = trainer.loss_fn(pred_traj, trajectory, info, trainer.model)
            test_losses.append(losses['reconstruction'].item())
    
    # Print results
    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    
    print(f"\nTest Reconstruction Loss: {avg_test_loss:.6f} ± {std_test_loss:.6f}")
    
    # Compare with baselines
    baseline_results = {
        'Single Neural ODE': 0.529915,
        'Multi-Scale Neural ODE': 0.484677,
        'Augmented Neural ODE': 0.471452,
        'Ensemble Neural ODE': 0.494241,
        'Traditional MoE': 0.481780
    }
    
    print("\nComparison with baselines:")
    for name, mse in baseline_results.items():
        improvement = (mse - avg_test_loss) / mse * 100
        symbol = "✓" if improvement > 0 else "✗"
        print(f"  {symbol} {name}: {mse:.6f} (AME-ODE is {abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'})")
    
    # Save final checkpoint
    checkpoint_path = Path(config.logging['checkpoint_dir']) / 'final_model.pt'
    print(f"\nSaving final model to {checkpoint_path}")
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config.to_dict(),
        'test_mse': avg_test_loss,
        'epoch': trainer.current_epoch,
    }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()