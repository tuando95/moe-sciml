#!/usr/bin/env python3
"""Comparative analysis experiments for AME-ODE vs baselines."""

import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import pandas as pd
from scipy import stats

from src.models.ame_ode import AMEODE
from src.baselines.single_neural_ode import (
    SingleNeuralODE, MultiScaleNeuralODE, AugmentedNeuralODE,
    EnsembleNeuralODE, TraditionalMoE
)
from src.data.preprocessing import create_experimental_dataloaders
from src.evaluation.metrics import AMEODEMetrics, PerformanceProfiler
from src.evaluation.visualization import AMEODEVisualizer
from src.utils.config import Config
from src.models.losses import AMEODELoss


class ComparativeExperiment:
    """Run comparative analysis between AME-ODE and baselines."""
    
    def __init__(self, config_path: Path, output_dir: Path, use_fast_inference: bool = True):
        self.config = Config(config_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_fast_inference = use_fast_inference
        
        # Results storage
        self.results = {
            'models': {},
            'metrics': {},
            'timing': {},
            'memory': {},
        }
        
        # Set device
        self.device = torch.device(
            self.config.compute['device'] 
            if torch.cuda.is_available() 
            else 'cpu'
        )
        
        # Initialize metrics calculator
        self.metrics_calc = AMEODEMetrics(self.config.to_dict())
        self.profiler = PerformanceProfiler()
        
    def create_model(self, model_type: str) -> torch.nn.Module:
        """Create model based on type."""
        config_dict = self.config.to_dict()
        
        if model_type == 'ame_ode':
            return AMEODE(config_dict).to(self.device)
        elif model_type == 'single_neural_ode':
            return SingleNeuralODE(config_dict).to(self.device)
        elif model_type == 'multi_scale_neural_ode':
            return MultiScaleNeuralODE(config_dict).to(self.device)
        elif model_type == 'augmented_neural_ode':
            return AugmentedNeuralODE(config_dict).to(self.device)
        elif model_type == 'ensemble_neural_ode':
            return EnsembleNeuralODE(config_dict).to(self.device)
        elif model_type == 'traditional_moe':
            return TraditionalMoE(config_dict).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(
        self,
        model: torch.nn.Module,
        model_type: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """Train a model and return training history."""
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training['learning_rate']
        )
        
        # Setup loss function
        if model_type == 'ame_ode':
            loss_fn = AMEODELoss(self.config.to_dict())
        else:
            loss_fn = torch.nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_time': [],
        }
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            model.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Forward pass
                # Ensure times is 1D for torchdiffeq
                if times.dim() > 1:
                    times_1d = times[0]
                else:
                    times_1d = times
                
                # During training, always use regular forward pass
                pred_traj, info = model.integrate(x0, times_1d)
                    
                # For AME-ODE, transpose to match batch format
                if model_type == 'ame_ode':
                    pred_traj = pred_traj  # Already in correct format
                else:
                    pred_traj = pred_traj.transpose(0, 1)  # From (time, batch, state) to (batch, time, state)
                
                # Compute loss
                if model_type == 'ame_ode':
                    losses = loss_fn(pred_traj, trajectory, info, model)
                    loss = losses['total']
                else:
                    loss = loss_fn(pred_traj, trajectory)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    trajectory = batch['trajectory'].to(self.device)
                    times = batch['times'].to(self.device)
                    x0 = batch['initial_condition'].to(self.device)
                    
                    # Ensure times is 1D for torchdiffeq
                    if times.dim() > 1:
                        times_1d = times[0]
                    else:
                        times_1d = times
                    
                    pred_traj, info = model.integrate(x0, times_1d)
                    
                    # For AME-ODE, transpose to match batch format
                    if model_type == 'ame_ode':
                        pred_traj = pred_traj  # Already in correct format
                    else:
                        pred_traj = pred_traj.transpose(0, 1)  # From (time, batch, state) to (batch, time, state)
                    
                    if model_type == 'ame_ode':
                        losses = loss_fn(pred_traj, trajectory, info, model)
                        loss = losses['total']
                    else:
                        loss = loss_fn(pred_traj, trajectory)
                    
                    val_losses.append(loss.item())
            
            # Record history
            epoch_time = time.time() - start_time
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            history['epoch_time'].append(epoch_time)
            
            print(f"Epoch {epoch+1}: Train Loss = {history['train_loss'][-1]:.4f}, "
                  f"Val Loss = {history['val_loss'][-1]:.4f}, Time = {epoch_time:.2f}s")
        
        return history
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        model_type: str,
        test_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        model.eval()
        
        all_metrics = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_type}"):
                trajectory = batch['trajectory'].to(self.device)
                times = batch['times'].to(self.device)
                x0 = batch['initial_condition'].to(self.device)
                
                # Time inference
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                # Forward pass
                # Ensure times is 1D for torchdiffeq
                if times.dim() > 1:
                    times_1d = times[0]
                else:
                    times_1d = times
                
                # Use fast inference for AME-ODE during evaluation if enabled
                if self.use_fast_inference and model_type == 'ame_ode' and hasattr(model, 'fast_inference'):
                    pred_traj = model.fast_inference(x0, times_1d)
                    # Create minimal info for metrics
                    info = {
                        'expert_usage': torch.zeros(x0.shape[0], model.n_experts, device=self.device),
                        'routing_weights': torch.zeros(len(times_1d), x0.shape[0], model.n_experts, device=self.device),
                        'routing_entropy': torch.tensor(0.0, device=self.device)
                    }
                else:
                    pred_traj, info = model.integrate(x0, times_1d)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # For AME-ODE, transpose to match batch format
                if model_type == 'ame_ode':
                    pred_traj = pred_traj  # Already in correct format
                else:
                    pred_traj = pred_traj.transpose(0, 1)  # From (time, batch, state) to (batch, time, state)
                
                # Compute metrics
                gt_experts = batch.get('ground_truth_experts', None)
                if gt_experts is not None:
                    gt_experts = gt_experts.to(self.device)
                
                metrics = self.metrics_calc.compute_all_metrics(
                    pred_traj, trajectory, info, times, gt_experts
                )
                all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            aggregated_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
        
        # Add timing statistics
        aggregated_metrics['inference_time'] = {
            'mean': np.mean(inference_times),
            'std': np.std(inference_times),
            'total': np.sum(inference_times),
        }
        
        # Profile performance
        sample_batch = next(iter(test_loader))
        sample_x0 = sample_batch['initial_condition'][:10].to(self.device)
        sample_times = sample_batch['times'].to(self.device)
        
        # Ensure times is 1D for profiling
        if sample_times.dim() > 1:
            sample_times_1d = sample_times[0]
        else:
            sample_times_1d = sample_times
        
        perf_metrics = self.profiler.profile_forward_pass(
            model, sample_x0, sample_times_1d, n_runs=10
        )
        memory_metrics = self.profiler.profile_memory_usage(
            model, sample_x0, sample_times_1d
        )
        
        aggregated_metrics['performance'] = perf_metrics
        aggregated_metrics['memory'] = memory_metrics
        
        return aggregated_metrics
    
    def run_comparison(
        self,
        system_name: str,
        model_types: List[str],
        num_epochs: int = 50,
        num_seeds: int = 3,
    ):
        """Run full comparison across models and seeds."""
        # Store results for each model and seed
        all_results = {model_type: [] for model_type in model_types}
        
        for seed in range(num_seeds):
            print(f"\n{'='*60}")
            print(f"Running experiments with seed {seed}")
            print('='*60)
            
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create data loaders
            train_loader, val_loader, test_loader, preprocessor = \
                create_experimental_dataloaders(self.config.to_dict(), system_name)
            
            # Train and evaluate each model
            for model_type in model_types:
                print(f"\nTraining {model_type}...")
                
                # Create model
                model = self.create_model(model_type)
                
                # Train
                history = self.train_model(
                    model, model_type, train_loader, val_loader, num_epochs
                )
                
                # Evaluate
                metrics = self.evaluate_model(model, model_type, test_loader)
                
                # Store results
                result = {
                    'seed': seed,
                    'history': history,
                    'metrics': metrics,
                    'model_params': sum(p.numel() for p in model.parameters()),
                }
                all_results[model_type].append(result)
                
                # Save checkpoint
                checkpoint_path = self.output_dir / f"{model_type}_seed{seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config.to_dict(),
                    'metrics': metrics,
                    'history': history,
                }, checkpoint_path)
        
        # Aggregate results across seeds
        self.aggregate_and_save_results(all_results, system_name)
        
        # Create comparison visualizations
        self.create_comparison_plots(all_results, system_name)
        
        # Statistical significance testing
        self.statistical_analysis(all_results)
    
    def aggregate_and_save_results(
        self,
        all_results: Dict[str, List[Dict]],
        system_name: str,
    ):
        """Aggregate results across seeds and save."""
        aggregated = {}
        
        for model_type, results in all_results.items():
            # Extract key metrics across seeds
            trajectory_mse = [r['metrics']['trajectory_mse']['mean'] for r in results]
            inference_time = [r['metrics']['inference_time']['mean'] for r in results]
            
            aggregated[model_type] = {
                'trajectory_mse': {
                    'mean': np.mean(trajectory_mse),
                    'std': np.std(trajectory_mse),
                    'values': trajectory_mse,
                },
                'inference_time': {
                    'mean': np.mean(inference_time),
                    'std': np.std(inference_time),
                    'values': inference_time,
                },
                'model_params': results[0]['model_params'],
            }
            
            # Add model-specific metrics
            if model_type == 'ame_ode':
                if 'mean_active_experts' in results[0]['metrics']:
                    active_experts = [r['metrics']['mean_active_experts']['mean'] for r in results]
                    aggregated[model_type]['mean_active_experts'] = {
                        'mean': np.mean(active_experts),
                        'std': np.std(active_experts),
                    }
        
        # Save aggregated results
        results_path = self.output_dir / f"{system_name}_comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        # Create results table
        self.create_results_table(aggregated, system_name)
    
    def create_results_table(self, aggregated: Dict[str, Any], system_name: str):
        """Create a formatted results table."""
        # Create DataFrame
        rows = []
        for model_type, metrics in aggregated.items():
            row = {
                'Model': model_type,
                'MSE (mean ± std)': f"{metrics['trajectory_mse']['mean']:.4f} ± {metrics['trajectory_mse']['std']:.4f}",
                'Inference Time (ms)': f"{metrics['inference_time']['mean']*1000:.2f} ± {metrics['inference_time']['std']*1000:.2f}",
                'Parameters': f"{metrics['model_params']:,}",
            }
            
            if 'mean_active_experts' in metrics:
                row['Active Experts'] = f"{metrics['mean_active_experts']['mean']:.2f} ± {metrics['mean_active_experts']['std']:.2f}"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save as CSV
        df.to_csv(self.output_dir / f"{system_name}_results_table.csv", index=False)
        
        # Print table
        print("\n" + "="*80)
        print(f"RESULTS SUMMARY: {system_name}")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    
    def create_comparison_plots(
        self,
        all_results: Dict[str, List[Dict]],
        system_name: str,
    ):
        """Create visualization comparing models."""
        import matplotlib.pyplot as plt
        
        # Training curves comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for model_type, results in all_results.items():
            # Average training curves across seeds
            train_losses = np.array([r['history']['train_loss'] for r in results])
            val_losses = np.array([r['history']['val_loss'] for r in results])
            
            mean_train = train_losses.mean(axis=0)
            std_train = train_losses.std(axis=0)
            mean_val = val_losses.mean(axis=0)
            std_val = val_losses.std(axis=0)
            
            epochs = range(len(mean_train))
            
            # Plot with confidence intervals
            ax1.plot(epochs, mean_train, label=model_type)
            ax1.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2)
            
            ax2.plot(epochs, mean_val, label=model_type)
            ax2.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{system_name}_training_curves.png", dpi=150)
        plt.close()
        
        # Performance vs efficiency plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for model_type, results in all_results.items():
            mse_values = [r['metrics']['trajectory_mse']['mean'] for r in results]
            time_values = [r['metrics']['inference_time']['mean'] * 1000 for r in results]  # ms
            
            ax.scatter(time_values, mse_values, s=100, label=model_type, alpha=0.7)
            ax.errorbar(
                np.mean(time_values), np.mean(mse_values),
                xerr=np.std(time_values), yerr=np.std(mse_values),
                fmt='o', markersize=10, capsize=5
            )
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Trajectory MSE')
        ax.set_title('Accuracy vs Computational Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{system_name}_efficiency_plot.png", dpi=150)
        plt.close()
    
    def statistical_analysis(self, all_results: Dict[str, List[Dict]]):
        """Perform statistical significance testing."""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        # Extract AME-ODE results as baseline
        if 'ame_ode' not in all_results:
            print("AME-ODE not found in results. Skipping statistical analysis.")
            return
        
        ame_mse = [r['metrics']['trajectory_mse']['mean'] for r in all_results['ame_ode']]
        
        # Compare each baseline to AME-ODE
        for model_type, results in all_results.items():
            if model_type == 'ame_ode':
                continue
            
            baseline_mse = [r['metrics']['trajectory_mse']['mean'] for r in results]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(ame_mse, baseline_mse)
            
            # Effect size (Cohen's d)
            diff = np.array(ame_mse) - np.array(baseline_mse)
            cohens_d = np.mean(diff) / np.std(diff)
            
            print(f"\n{model_type} vs AME-ODE:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comparative analysis experiments')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to use')
    parser.add_argument('--models', nargs='+', 
                        default=['ame_ode', 'single_neural_ode', 'multi_scale_neural_ode'],
                        help='Models to compare')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Output directory for results')
    parser.add_argument('--fast-inference', action='store_true',
                        help='Use fast inference mode for AME-ODE evaluation')
    
    args = parser.parse_args()
    
    # Run comparison
    experiment = ComparativeExperiment(
        Path(args.config),
        Path(args.output_dir),
        use_fast_inference=args.fast_inference
    )
    
    experiment.run_comparison(
        args.system,
        args.models,
        num_epochs=args.epochs,
        num_seeds=args.seeds
    )


if __name__ == '__main__':
    main()