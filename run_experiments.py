#!/usr/bin/env python3
"""Comprehensive experiment runner for AME-ODE research."""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import yaml
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.utils.config import Config


class ExperimentRunner:
    """Main experiment runner coordinating all experiments."""
    
    def __init__(self, base_dir: Path = Path('.')):
        self.base_dir = base_dir
        self.results_dir = base_dir / 'experiment_results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.results_dir / f'experiment_{timestamp}'
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Log file
        self.log_file = self.experiment_dir / 'experiment_log.txt'
        
    def log(self, message: str):
        """Log message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    def run_command(self, cmd: List[str], name: str) -> bool:
        """Run a command and log output."""
        self.log(f"\nRunning {name}...")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            # Run command and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            output_file = self.experiment_dir / f"{name}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            self.log(f"{name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"ERROR in {name}: {e}")
            error_file = self.experiment_dir / f"{name}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Return code: {e.returncode}\n")
                f.write(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
            return False
    
    def train_ame_ode(
        self,
        config_path: str,
        system: str,
        epochs: int = 100,
        device: str = 'cuda',
        force_regenerate: bool = False,
    ) -> Optional[Path]:
        """Train AME-ODE model."""
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, 'train.py',
            '--config', config_path,
            '--system', system,
            '--device', device,
            '--seed', '42',
        ]
        
        if force_regenerate:
            cmd.append('--force-regenerate')
        
        # Modify config to use our checkpoint directory
        config = Config(Path(config_path))
        config._config['logging']['checkpoint_dir'] = str(checkpoint_dir)
        config._config['training']['num_epochs'] = epochs
        
        # Save modified config
        modified_config = self.experiment_dir / 'config_modified.yml'
        with open(modified_config, 'w') as f:
            yaml.dump(config.to_dict(), f)
        
        cmd[2] = str(modified_config)  # Use modified config
        
        success = self.run_command(cmd, 'ame_ode_training')
        
        if success:
            # Return path to best model
            best_model = checkpoint_dir / 'best_model.pt'
            if best_model.exists():
                return best_model
            else:
                # Try latest model
                latest_model = checkpoint_dir / 'latest_model.pt'
                if latest_model.exists():
                    return latest_model
        
        return None
    
    def run_ablation_studies(
        self,
        config_path: str,
        system: str,
        ablation_types: List[str],
    ):
        """Run ablation studies."""
        ablation_dir = self.experiment_dir / 'ablations'
        ablation_dir.mkdir(exist_ok=True)
        
        for ablation_type in ablation_types:
            cmd = [
                sys.executable, 'experiments/run_ablations.py',
                '--config', config_path,
                '--ablation', ablation_type,
                '--system', system,
                '--output-dir', str(ablation_dir),
            ]
            
            self.run_command(cmd, f'ablation_{ablation_type}')
    
    def run_comparative_analysis(
        self,
        config_path: str,
        system: str,
        models: List[str],
        epochs: int = 50,
        seeds: int = 3,
        use_fast_inference: bool = True,
    ):
        """Run comparative analysis against baselines."""
        comparison_dir = self.experiment_dir / 'comparisons'
        comparison_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, 'experiments/comparative_analysis.py',
            '--config', config_path,
            '--system', system,
            '--models', *models,
            '--epochs', str(epochs),
            '--seeds', str(seeds),
            '--output-dir', str(comparison_dir),
        ]
        
        if use_fast_inference:
            cmd.append('--fast-inference')
        
        self.run_command(cmd, 'comparative_analysis')
    
    def run_error_analysis(
        self,
        config_path: str,
        model_checkpoint: Path,
        system: str,
        use_fast_inference: bool = True,
    ):
        """Run error analysis experiments."""
        error_dir = self.experiment_dir / 'error_analysis'
        error_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, 'experiments/error_analysis.py',
            '--config', config_path,
            '--checkpoint', str(model_checkpoint),
            '--system', system,
            '--output-dir', str(error_dir),
        ]
        
        if use_fast_inference:
            cmd.append('--fast-inference')
        
        self.run_command(cmd, 'error_analysis')
    
    def run_visualization_experiments(
        self,
        config_path: str,
        model_checkpoint: Path,
        system: str,
    ):
        """Run visualization experiments."""
        viz_dir = self.experiment_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        checkpoint_dir = model_checkpoint.parent
        
        cmd = [
            sys.executable, 'experiments/visualization_experiments.py',
            '--config', config_path,
            '--checkpoint', str(model_checkpoint),
            '--system', system,
            '--checkpoint-dir', str(checkpoint_dir),
            '--output-dir', str(viz_dir),
        ]
        
        self.run_command(cmd, 'visualization_experiments')
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        self.log("\nGenerating experiment report...")
        
        report = {
            'experiment_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat(),
            'experiments_run': [],
        }
        
        # Check which experiments were run
        if (self.experiment_dir / 'checkpoints').exists():
            report['experiments_run'].append('training')
        
        if (self.experiment_dir / 'ablations').exists():
            report['experiments_run'].append('ablations')
            
        if (self.experiment_dir / 'comparisons').exists():
            report['experiments_run'].append('comparative_analysis')
            
        if (self.experiment_dir / 'error_analysis').exists():
            report['experiments_run'].append('error_analysis')
            
        if (self.experiment_dir / 'visualizations').exists():
            report['experiments_run'].append('visualizations')
        
        # Collect key results
        results_summary = {}
        
        # Training results
        training_summary_path = self.experiment_dir / 'checkpoints' / 'logs' / 'training_summary.json'
        if training_summary_path.exists():
            with open(training_summary_path) as f:
                training_summary = json.load(f)
                results_summary['training'] = {
                    'final_epoch': training_summary.get('final_epoch'),
                    'best_val_loss': training_summary.get('best_val_loss'),
                }
        
        # Comparison results
        comparison_files = list((self.experiment_dir / 'comparisons').glob('*_comparison_results.json'))
        if comparison_files:
            with open(comparison_files[0]) as f:
                comparison_results = json.load(f)
                results_summary['comparison'] = comparison_results
        
        report['results_summary'] = results_summary
        
        # Save report
        report_path = self.experiment_dir / 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Report saved to: {report_path}")
        
        # Print summary
        self.log("\n" + "="*60)
        self.log("EXPERIMENT SUMMARY")
        self.log("="*60)
        self.log(f"Experiment directory: {self.experiment_dir}")
        self.log(f"Experiments run: {', '.join(report['experiments_run'])}")
        
        if 'training' in results_summary:
            self.log(f"\nTraining Results:")
            self.log(f"  Final epoch: {results_summary['training']['final_epoch']}")
            self.log(f"  Best val loss: {results_summary['training']['best_val_loss']:.4f}")
        
        self.log("="*60)
    
    def run_full_experiment_suite(
        self,
        config_path: str,
        system: str,
        skip_training: bool = False,
        model_checkpoint: Optional[str] = None,
        experiments: Optional[List[str]] = None,
        force_regenerate: bool = False,
        use_fast_inference: bool = True,
    ):
        """Run the full suite of experiments."""
        self.log("="*60)
        self.log("AME-ODE FULL EXPERIMENT SUITE")
        self.log("="*60)
        self.log(f"Config: {config_path}")
        self.log(f"System: {system}")
        self.log(f"Output directory: {self.experiment_dir}")
        
        # Default to all experiments
        if experiments is None:
            experiments = ['training', 'ablations', 'comparison', 'error', 'visualization']
        
        # Train model (or use provided checkpoint)
        if not skip_training and 'training' in experiments:
            model_checkpoint_path = self.train_ame_ode(
                config_path, system, epochs=100, force_regenerate=force_regenerate
            )
            if model_checkpoint_path is None:
                self.log("ERROR: Training failed. Exiting.")
                return
        else:
            if model_checkpoint:
                model_checkpoint_path = Path(model_checkpoint)
            else:
                self.log("ERROR: No model checkpoint provided and training skipped.")
                return
        
        # Run experiments
        if 'ablations' in experiments:
            self.run_ablation_studies(
                config_path, system,
                ['routing_mechanism', 'expert_initialization', 'n_experts', 'regularization']
            )
        
        if 'comparison' in experiments:
            self.run_comparative_analysis(
                config_path, system,
                ['ame_ode', 'single_neural_ode', 'multi_scale_neural_ode', 
                 'augmented_neural_ode', 'ensemble_neural_ode'],
                epochs=50, seeds=3, use_fast_inference=use_fast_inference
            )
        
        if 'error' in experiments:
            self.run_error_analysis(
                config_path, model_checkpoint_path, system,
                use_fast_inference=use_fast_inference
            )
        
        if 'visualization' in experiments:
            self.run_visualization_experiments(
                config_path, model_checkpoint_path, system
            )
        
        # Generate final report
        self.generate_report()
        
        self.log("\nAll experiments completed!")


def main():
    parser = argparse.ArgumentParser(description='Run AME-ODE experiments')
    
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--system', type=str, default='multi_scale_oscillators',
                        help='Synthetic system to use')
    parser.add_argument('--experiments', nargs='+',
                        choices=['training', 'ablations', 'comparison', 'error', 'visualization'],
                        help='Which experiments to run (default: all)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to existing model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration of synthetic datasets (ignore cache)')
    parser.add_argument('--fast-inference', action='store_true',
                        help='Use fast inference mode for AME-ODE evaluation')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run experiments
    runner = ExperimentRunner()
    runner.run_full_experiment_suite(
        args.config,
        args.system,
        skip_training=args.skip_training,
        model_checkpoint=args.checkpoint,
        experiments=args.experiments,
        force_regenerate=args.force_regenerate,
        use_fast_inference=args.fast_inference,
    )


if __name__ == '__main__':
    main()