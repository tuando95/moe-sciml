#!/usr/bin/env python3
"""Compare all baseline models with AME-ODE."""

import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def train_all_baselines(config='configs/quick_test.yml', system='multi_scale_oscillators', 
                       parallel=False, n_gpus=1):
    """Train all baseline models."""
    baselines = ['single', 'multiscale', 'augmented', 'ensemble', 'moe']
    results = {}
    
    if parallel and n_gpus > 0:
        # Parallel training using multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        
        print(f"\nTraining {len(baselines)} baselines in parallel using {n_gpus} GPU(s)")
        print('='*60)
        
        def train_single_baseline(baseline, gpu_id):
            """Train a single baseline on specified GPU."""
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            cmd = [
                'python', 'train_baseline.py',
                '--config', config,
                '--baseline', baseline,
                '--system', system
            ]
            
            print(f"Starting {baseline} on GPU {gpu_id}...")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Training failed: {result.stderr}")
            
            # Find and load results
            results_dir = Path('results') / 'baselines'
            baseline_files = list(results_dir.glob(f"{baseline}_{system}_*.json"))
            if baseline_files:
                latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
                with open(latest_file) as f:
                    return baseline, json.load(f)
            return baseline, None
        
        # Distribute baselines across GPUs
        max_workers = min(len(baselines), n_gpus)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, baseline in enumerate(baselines):
                gpu_id = i % n_gpus
                future = executor.submit(train_single_baseline, baseline, gpu_id)
                futures[future] = baseline
            
            # Collect results as they complete
            for future in as_completed(futures):
                baseline = futures[future]
                try:
                    baseline_name, result = future.result()
                    if result:
                        results[baseline_name] = result
                        print(f"✓ Completed: {baseline_name}")
                    else:
                        print(f"✗ No results for: {baseline_name}")
                except Exception as e:
                    print(f"✗ Error training {baseline}: {e}")
    
    else:
        # Sequential training
        for baseline in baselines:
            print(f"\n{'='*60}")
            print(f"Training {baseline} baseline")
            print('='*60)
            
            cmd = [
                'python', 'train_baseline.py',
                '--config', config,
                '--baseline', baseline,
                '--system', system
            ]
            
            try:
                subprocess.run(cmd, check=True)
                
                # Find the latest results file
                results_dir = Path('results') / 'baselines'
                baseline_files = list(results_dir.glob(f"{baseline}_{system}_*.json"))
                if baseline_files:
                    latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file) as f:
                        results[baseline] = json.load(f)
            except subprocess.CalledProcessError as e:
                print(f"Error training {baseline}: {e}")
    
    return results


def plot_baseline_comparison(results, save_path='results/baseline_comparison.png'):
    """Plot comparison of all baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Test MSE comparison
    models = list(results.keys())
    test_mses = []
    test_mse_stds = []
    param_counts = []
    
    for m in models:
        if 'test_metrics' in results[m] and 'trajectory_mse' in results[m]['test_metrics']:
            test_mses.append(results[m]['test_metrics']['trajectory_mse']['mean'])
            test_mse_stds.append(results[m]['test_metrics']['trajectory_mse']['std'])
        else:
            # Fallback to test_loss if metrics not available
            test_mses.append(results[m].get('test_loss', results[m].get('best_val_loss', 0)))
            test_mse_stds.append(0)
        param_counts.append(results[m]['total_params'])
    
    x = np.arange(len(models))
    ax1.bar(x, test_mses, yerr=test_mse_stds, capsize=5, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel('Test Trajectory MSE')
    ax1.set_title('Model Performance Comparison (Test Set)')
    ax1.set_yscale('log')
    
    # Add parameter counts as text
    for i, (model, params, mse) in enumerate(zip(models, param_counts, test_mses)):
        ax1.text(i, mse, f'{params:,}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Training curves (if available)
    has_training_history = any('metrics_history' in data for data in results.values())
    
    if has_training_history:
        for model, data in results.items():
            if 'metrics_history' in data:
                epochs = range(1, len(data['metrics_history']['val_loss']) + 1)
                ax2.plot(epochs, data['metrics_history']['val_loss'], label=model, marker='o')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Training Progress')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        # Alternative plot: Parameter efficiency
        test_mses_for_plot = []
        param_counts_for_plot = []
        model_names = []
        
        for m in models:
            if 'test_metrics' in results[m] and 'trajectory_mse' in results[m]['test_metrics']:
                test_mses_for_plot.append(results[m]['test_metrics']['trajectory_mse']['mean'])
                param_counts_for_plot.append(results[m]['total_params'])
                model_names.append(m)
        
        if test_mses_for_plot:
            scatter = ax2.scatter(param_counts_for_plot, test_mses_for_plot, s=100, alpha=0.7)
            
            # Add model labels
            for i, name in enumerate(model_names):
                ax2.annotate(name, (param_counts_for_plot[i], test_mses_for_plot[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Number of Parameters')
            ax2.set_ylabel('Test MSE')
            ax2.set_title('Parameter Efficiency')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nComparison plot saved to: {save_path}")


def generate_comparison_report(results, ame_ode_results=None):
    """Generate a detailed comparison report."""
    report = []
    report.append("# Baseline Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table with test metrics
    report.append("## Summary Results (Test Set Performance)\n")
    report.append("| Model | Parameters | Test MSE | Long-term MSE | Phase Accuracy | Relative to Single |")
    report.append("|-------|------------|----------|---------------|----------------|-------------------|")
    
    # Get reference MSE from single model
    single_mse = 1.0
    if 'single' in results and 'test_metrics' in results['single']:
        single_mse = results['single']['test_metrics']['trajectory_mse']['mean']
    
    for model, data in sorted(results.items()):
        params = data['total_params']
        
        # Extract test metrics
        if 'test_metrics' in data:
            test_mse = data['test_metrics']['trajectory_mse']['mean']
            test_mse_str = f"{test_mse:.6f}"
            
            long_term_mse = data['test_metrics'].get('long_term_mse', {}).get('mean', '-')
            if long_term_mse != '-':
                long_term_mse = f"{long_term_mse:.6f}"
            
            phase_acc = data['test_metrics'].get('phase_space_accuracy', {}).get('mean', '-')
            if phase_acc != '-':
                phase_acc = f"{phase_acc:.4f}"
        else:
            test_mse = data.get('test_loss', data.get('best_val_loss', '-'))
            test_mse_str = f"{test_mse:.6f}" if test_mse != '-' else '-'
            long_term_mse = '-'
            phase_acc = '-'
        
        relative = (test_mse / single_mse - 1) * 100 if isinstance(test_mse, (int, float)) else 0
        report.append(f"| {model.capitalize()} | {params:,} | {test_mse_str} | {long_term_mse} | {phase_acc} | {relative:+.1f}% |")
    
    if ame_ode_results:
        ame_test_mse = ame_ode_results.get('test_mse', ame_ode_results.get('val_loss', '-'))
        if isinstance(ame_test_mse, (int, float)):
            ame_relative = (ame_test_mse / single_mse - 1) * 100
            report.append(f"| **AME-ODE** | {ame_ode_results['params']:,} | "
                         f"{ame_test_mse:.6f} | - | - | {ame_relative:+.1f}% |")
        else:
            report.append(f"| **AME-ODE** | {ame_ode_results['params']:,} | {ame_test_mse} | - | - | - |")
    
    # Detailed analysis
    report.append("\n## Detailed Analysis\n")
    
    # Find best model based on test MSE
    models_with_metrics = [(m, d) for m, d in results.items() 
                          if 'test_metrics' in d and 'trajectory_mse' in d['test_metrics']]
    if models_with_metrics:
        best_model = min(models_with_metrics, 
                        key=lambda x: x[1]['test_metrics']['trajectory_mse']['mean'])
        report.append(f"**Best Baseline**: {best_model[0].capitalize()} "
                     f"(Test MSE: {best_model[1]['test_metrics']['trajectory_mse']['mean']:.6f})")
    else:
        # Fallback to validation loss
        best_model = min(results.items(), key=lambda x: x[1].get('test_loss', x[1].get('best_val_loss', float('inf'))))
        metric_value = best_model[1].get('test_loss', best_model[1].get('best_val_loss', 'N/A'))
        report.append(f"**Best Baseline**: {best_model[0].capitalize()} "
                     f"(Loss: {metric_value:.4f})")
    
    # Model-specific insights
    report.append("\n### Model-Specific Insights\n")
    
    insights = {
        'single': "Standard Neural ODE baseline with matched parameters",
        'multiscale': "Explicitly models fast/slow dynamics with separate networks",
        'augmented': "Uses additional latent dimensions for increased expressiveness",
        'ensemble': "Averages predictions from multiple diverse models",
        'moe': "Traditional MoE with static routing (no temporal adaptation)"
    }
    
    for model, desc in insights.items():
        if model in results:
            report.append(f"- **{model.capitalize()}**: {desc}")
            if 'test_metrics' in results[model]:
                test_mse = results[model]['test_metrics']['trajectory_mse']['mean']
                test_mse_std = results[model]['test_metrics']['trajectory_mse']['std']
                report.append(f"  - Test MSE: {test_mse:.6f} ± {test_mse_std:.6f}")
            else:
                report.append(f"  - Test Loss: {results[model].get('test_loss', 'N/A')}")
            report.append(f"  - Parameters: {results[model]['total_params']:,}")
    
    # Save report
    report_path = Path('results') / 'baseline_comparison_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nDetailed report saved to: {report_path}")
    return '\n'.join(report)


def evaluate_from_checkpoints(config_path='configs/quick_test.yml', 
                             checkpoint_dir='checkpoints_test',
                             system='multi_scale_oscillators',
                             use_fast_inference=False):
    """Evaluate models from saved checkpoints."""
    import torch
    from src.utils.config import Config
    from src.training.trainer import create_data_loaders
    from src.evaluation.metrics import AMEODEMetrics
    from src.baselines.single_neural_ode import (
        SingleNeuralODE, MultiScaleNeuralODE, AugmentedNeuralODE,
        EnsembleNeuralODE, TraditionalMoE
    )
    from tqdm import tqdm
    
    config = Config(Path(config_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    print(f"Loading {system} dataset...")
    _, _, test_loader = create_data_loaders(config, system)
    
    # Model mapping
    model_classes = {
        'SingleNeuralODE': SingleNeuralODE,
        'MultiScaleNeuralODE': MultiScaleNeuralODE,
        'AugmentedNeuralODE': AugmentedNeuralODE,
        'EnsembleNeuralODE': EnsembleNeuralODE,
        'TraditionalMoE': TraditionalMoE,
    }
    
    results = {}
    checkpoint_path = Path(checkpoint_dir)
    
    for model_name, model_class in model_classes.items():
        model_checkpoint = checkpoint_path / model_name / 'best_model.pt'
        
        if not model_checkpoint.exists():
            print(f"Checkpoint not found for {model_name}, skipping...")
            continue
            
        print(f"\nEvaluating {model_name} from checkpoint...")
        
        # Load model
        model = model_class(config.to_dict()).to(device)
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        # Get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Evaluate on test set
        metrics_calc = AMEODEMetrics(config.to_dict())
        all_metrics = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                trajectory = batch['trajectory'].to(device)
                times = batch['times'].to(device)
                x0 = batch['initial_condition'].to(device)
                gt_experts = batch.get('ground_truth_experts', None)
                
                if gt_experts is not None:
                    gt_experts = gt_experts.to(device)
                
                # Ensure times is 1D for torchdiffeq
                if times.dim() > 1:
                    times_1d = times[0]
                else:
                    times_1d = times
                
                # Forward pass
                pred_trajectory, info = model.integrate(x0, times_1d)
                pred_trajectory = pred_trajectory.transpose(0, 1)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(pred_trajectory, trajectory)
                test_loss += loss.item()
                
                # Compute comprehensive metrics
                metrics = metrics_calc.compute_all_metrics(
                    pred_trajectory, trajectory, info, times, gt_experts
                )
                all_metrics.append(metrics)
        
        # Aggregate metrics
        test_loss /= len(test_loader)
        aggregated_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        # Store results
        results[model_name.replace('NeuralODE', '').lower()] = {
            'model': model_name,
            'total_params': total_params,
            'test_loss': test_loss,
            'test_metrics': aggregated_metrics,
            'checkpoint_info': checkpoint.get('epoch', 'N/A'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"{model_name} - Test MSE: {aggregated_metrics['trajectory_mse']['mean']:.6f}")
    
    # Also evaluate AME-ODE if checkpoint exists
    ame_checkpoint = checkpoint_path / 'best_model.pt'
    if ame_checkpoint.exists():
        print(f"\nEvaluating AME-ODE from checkpoint...")
        
        from src.models.ame_ode import AMEODE
        
        # Load model
        model = AMEODE(config.to_dict()).to(device)
        checkpoint = torch.load(ame_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Evaluate on test set
        all_metrics = []
        test_loss = 0.0
        
        from src.models.losses import AMEODELoss
        loss_fn = AMEODELoss(config.to_dict())
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating AME-ODE"):
                trajectory = batch['trajectory'].to(device)
                times = batch['times'].to(device)
                x0 = batch['initial_condition'].to(device)
                gt_experts = batch.get('ground_truth_experts', None)
                
                if gt_experts is not None:
                    gt_experts = gt_experts.to(device)
                
                # Ensure times is 1D for torchdiffeq
                if times.dim() > 1:
                    times_1d = times[0]
                else:
                    times_1d = times
                
                # Forward pass
                if use_fast_inference:
                    # Use fast inference for speed
                    pred_trajectory = model.fast_inference(x0, times_1d)
                    # Create minimal info for metrics
                    info = {
                        'expert_usage': torch.zeros(x0.shape[0], model.n_experts, device=device),
                        'routing_weights': torch.zeros(len(times_1d), x0.shape[0], model.n_experts, device=device),
                        'routing_entropy': torch.tensor(0.0, device=device)
                    }
                else:
                    pred_trajectory, info = model.integrate(x0, times_1d)
                # AME-ODE already outputs in correct format
                
                # Compute losses
                losses = loss_fn(pred_trajectory, trajectory, info, model)
                test_loss += losses['reconstruction'].item()
                
                # Compute comprehensive metrics
                metrics = metrics_calc.compute_all_metrics(
                    pred_trajectory, trajectory, info, times, gt_experts
                )
                all_metrics.append(metrics)
        
        # Aggregate metrics
        test_loss /= len(test_loader)
        aggregated_metrics = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        # Store results
        results['ame_ode'] = {
            'model': 'AME-ODE',
            'total_params': total_params,
            'test_loss': test_loss,
            'test_metrics': aggregated_metrics,
            'checkpoint_info': checkpoint.get('epoch', 'N/A'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"AME-ODE - Test MSE: {aggregated_metrics['trajectory_mse']['mean']:.6f}")
    
    return results


def main():
    """Run full baseline comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline models')
    parser.add_argument('--config', type=str, default='configs/quick_test.yml',
                        help='Configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system')
    parser.add_argument('--train', action='store_true',
                        help='Train all baselines (otherwise use existing results)')
    parser.add_argument('--use-checkpoints', action='store_true',
                        help='Evaluate from saved checkpoints instead of results files')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_test',
                        help='Directory containing model checkpoints')
    parser.add_argument('--fast-inference', action='store_true',
                        help='Use fast inference mode for AME-ODE evaluation')
    parser.add_argument('--parallel', action='store_true',
                        help='Train models in parallel using multiple processes')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='Number of GPUs to use for parallel training')
    
    args = parser.parse_args()
    
    if args.train:
        # Train all baselines
        results = train_all_baselines(args.config, args.system, 
                                    parallel=args.parallel, n_gpus=args.n_gpus)
    elif args.use_checkpoints:
        # Evaluate from saved checkpoints
        results = evaluate_from_checkpoints(args.config, args.checkpoint_dir, args.system, 
                                          use_fast_inference=args.fast_inference)
        
        # Save results for future reference
        results_dir = Path('results') / 'baselines' 
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"checkpoint_evaluation_{args.system}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {results_file}")
    else:
        # Load existing results
        results = {}
        results_dir = Path('results') / 'baselines'
        
        for baseline in ['single', 'multiscale', 'augmented', 'ensemble', 'moe']:
            files = list(results_dir.glob(f"{baseline}_{args.system}_*.json"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                with open(latest) as f:
                    results[baseline] = json.load(f)
    
    if results:
        # Generate comparison plot
        plot_baseline_comparison(results)
        
        # Generate report
        report = generate_comparison_report(results)
        print("\n" + report)
    else:
        print("No results found. Run with --train flag to train baselines or --use-checkpoints to evaluate from checkpoints.")


if __name__ == '__main__':
    main()