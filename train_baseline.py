#!/usr/bin/env python3
"""Train baseline models for comparison with AME-ODE."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import json
from datetime import datetime

from src.utils.config import Config
from src.baselines.single_neural_ode import (
    SingleNeuralODE, 
    MultiScaleNeuralODE,
    AugmentedNeuralODE,
    EnsembleNeuralODE,
    TraditionalMoE
)
from src.training.trainer import create_data_loaders
from src.evaluation.metrics import AMEODEMetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaselineTrainer:
    """Trainer for baseline models."""
    
    def __init__(self, model, config, device='cuda', baseline='single'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.baseline = baseline
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.training['learning_rate'])
        )
        
        # Learning rate scheduler - reduce on plateau for stability
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Loss
        self.loss_fn = nn.MSELoss()
        
        # Metrics
        self.best_val_loss = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Early stopping
        self.early_stopping_patience = config.training.get('early_stopping_patience', 20)
        self.early_stopping_counter = 0
        self.early_stopped = False
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            trajectory = batch['trajectory'].to(self.device)
            times = batch['times'].to(self.device)
            x0 = batch['initial_condition'].to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'integrate'):
                # Ensure times is 1D for torchdiffeq
                if times.dim() > 1:
                    times_1d = times[0]  # All trajectories should have same time points
                else:
                    times_1d = times
                    
                pred_trajectory, _ = self.model.integrate(x0, times_1d)
                # Transpose to match batch format
                pred_trajectory = pred_trajectory.transpose(0, 1)
            else:
                # For models without integrate method
                pred_trajectory = self.model(times, x0)
            
            # Compute loss
            loss = self.loss_fn(pred_trajectory, trajectory)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - use stricter clipping for augmented models
            if self.baseline == 'augmented':
                clip_norm = min(0.5, self.config.training['gradient_clip_norm'])
            else:
                clip_norm = self.config.training['gradient_clip_norm']
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                clip_norm
            )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'integrate'):
                    # Ensure times is 1D for torchdiffeq
                    if times.dim() > 1:
                        times_1d = times[0]  # All trajectories should have same time points
                    else:
                        times_1d = times
                        
                    pred_trajectory, _ = self.model.integrate(x0, times_1d)
                    pred_trajectory = pred_trajectory.transpose(0, 1)
                else:
                    pred_trajectory = self.model(times, x0)
                
                # Compute loss
                loss = self.loss_fn(pred_trajectory, trajectory)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def evaluate_full_metrics(self, test_loader):
        """Compute comprehensive metrics on test set."""
        self.model.eval()
        
        # Initialize metrics calculator
        metrics_calc = AMEODEMetrics(self.config.to_dict())
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating metrics"):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                gt_experts = batch.get('ground_truth_experts', None)
                
                if gt_experts is not None:
                    gt_experts = gt_experts.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'integrate'):
                    # Ensure times is 1D for torchdiffeq
                    if times.dim() > 1:
                        times_1d = times[0]
                    else:
                        times_1d = times
                        
                    pred_trajectory, info = self.model.integrate(x0, times_1d)
                    pred_trajectory = pred_trajectory.transpose(0, 1)
                else:
                    pred_trajectory = self.model(times, x0)
                    info = {}
                
                # Compute comprehensive metrics
                metrics = metrics_calc.compute_all_metrics(
                    pred_trajectory, trajectory, info, times, gt_experts
                )
                all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            if values:  # Only aggregate if values exist
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        return aggregated_metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop."""
        print(f"\nTraining {self.model.__class__.__name__}")
        print("="*50)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.metrics_history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.metrics_history['val_loss'].append(val_loss)
            
            # Step the learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model and check early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                
            # Check early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                self.early_stopped = True
                break
        
        if self.early_stopped:
            print(f"\nTraining stopped early at epoch {epoch+1}")
        else:
            print(f"\nCompleted all {num_epochs} epochs")
            
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        return self.metrics_history
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.logging['checkpoint_dir']) / self.model.__class__.__name__
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, checkpoint_dir / 'best_model.pt')


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--config', type=str, default='configs/quick_test.yml',
                        help='Path to configuration file')
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['single', 'multiscale', 'augmented', 'ensemble', 'moe'],
                        help='Which baseline to train')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to train on')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load config
    config = Config(Path(args.config))
    
    # Create data loaders
    print(f"Loading {args.system} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(config, args.system)
    
    # Create baseline model
    baseline_classes = {
        'single': SingleNeuralODE,
        'multiscale': MultiScaleNeuralODE,
        'augmented': AugmentedNeuralODE,
        'ensemble': EnsembleNeuralODE,
        'moe': TraditionalMoE,
    }
    
    model_class = baseline_classes[args.baseline]
    model = model_class(config.to_dict())
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer and train
    trainer = BaselineTrainer(model, config, device=args.device, baseline=args.baseline)
    metrics = trainer.train(train_loader, val_loader, config.training['num_epochs'])
    
    # Save final results
    results = {
        'model': model.__class__.__name__,
        'config': args.config,
        'system': args.system,
        'best_val_loss': trainer.best_val_loss,
        'metrics_history': metrics,
        'total_params': total_params,
        'timestamp': datetime.now().isoformat(),
    }
    
    results_dir = Path('results') / 'baselines'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{args.baseline}_{args.system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Test evaluation with comprehensive metrics
    print("\nEvaluating on test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    
    # Compute full metrics on test set
    print("\nComputing comprehensive metrics on test set...")
    test_metrics = trainer.evaluate_full_metrics(test_loader)
    
    # Update results with test metrics
    results['test_loss'] = test_loss
    results['test_metrics'] = test_metrics
    
    # Print key metrics
    print("\nKey Test Metrics:")
    print(f"  Trajectory MSE: {test_metrics['trajectory_mse']['mean']:.6f} ± {test_metrics['trajectory_mse']['std']:.6f}")
    if 'long_term_mse' in test_metrics:
        print(f"  Long-term MSE: {test_metrics['long_term_mse']['mean']:.6f} ± {test_metrics['long_term_mse']['std']:.6f}")
    if 'phase_space_accuracy' in test_metrics:
        print(f"  Phase Space Accuracy: {test_metrics['phase_space_accuracy']['mean']:.6f}")
    
    # Re-save results with comprehensive metrics
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()