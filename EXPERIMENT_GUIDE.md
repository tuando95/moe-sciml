# AME-ODE Experiment Guide: Step-by-Step Instructions

This guide provides detailed instructions for running all experiments in the AME-ODE project.

## Prerequisites

1. **Install Dependencies**
   ```bash
   cd /mnt/d/biomed-research/moe-sciml
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python test_implementation.py
   ```
   This should show "✅ ALL TESTS PASSED!"

## Quick Start (Minimal Test)

Run a quick test to ensure everything is working:

```bash
# Quick test with minimal configuration
python train.py --config configs/quick_test.yml --system multi_scale_oscillators --device cuda
```

## Full Experimental Pipeline

### Option 1: Run All Experiments Automatically

The easiest way to run all experiments is using the comprehensive experiment runner:

```bash
# Run complete evaluation on all systems
python run_experiments.py \
    --config config.yml \
    --systems multi_scale_oscillators piecewise_lorenz van_der_pol_network \
    --experiments all \
    --results-dir experiment_results_full
```

This will automatically:
1. Train AME-ODE models for each system
2. Run comparative analysis against all baselines
3. Perform ablation studies
4. Conduct error analysis
5. Generate visualizations
6. Create a comprehensive report

### Option 2: Run Experiments Step-by-Step

If you prefer more control, run each experiment individually:

#### Step 1: Train AME-ODE Model

```bash
# Train on multi-scale oscillators
python train.py \
    --config config.yml \
    --system multi_scale_oscillators \
    --device cuda

# Train on Lorenz system with specialized config
python train.py \
    --config configs/experiment_lorenz.yml \
    --system piecewise_lorenz \
    --device cuda

# Train on Van der Pol network (stiff dynamics)
python train.py \
    --config configs/experiment_stiff.yml \
    --system van_der_pol_network \
    --device cuda
```

After training, checkpoints will be saved in `checkpoints/best_model.pt`.

#### Step 2: Run Comparative Analysis

Compare AME-ODE against baseline methods:

```bash
# Compare on multi-scale oscillators
python experiments/comparative_analysis.py \
    --config config.yml \
    --system multi_scale_oscillators \
    --models ame_ode single_neural_ode multi_scale_neural_ode augmented_neural_ode ensemble_neural_ode \
    --epochs 50 \
    --seeds 3 \
    --output-dir comparison_results
```

This generates:
- `comparison_results/multi_scale_oscillators_results_table.csv` - Performance summary
- `comparison_results/multi_scale_oscillators_training_curves.png` - Training curves
- `comparison_results/multi_scale_oscillators_efficiency_plot.png` - Accuracy vs speed

#### Step 3: Run Ablation Studies

Test the importance of each component:

```bash
# Routing mechanism ablation
python experiments/run_ablations.py \
    --config config.yml \
    --ablation routing_mechanism \
    --system multi_scale_oscillators \
    --output-dir ablation_results

# Expert initialization ablation
python experiments/run_ablations.py \
    --config config.yml \
    --ablation expert_initialization \
    --system multi_scale_oscillators \
    --output-dir ablation_results

# Number of experts ablation
python experiments/run_ablations.py \
    --config config.yml \
    --ablation n_experts \
    --system multi_scale_oscillators \
    --output-dir ablation_results

# Regularization ablation
python experiments/run_ablations.py \
    --config config.yml \
    --ablation regularization \
    --system multi_scale_oscillators \
    --output-dir ablation_results
```

#### Step 4: Error Analysis

Analyze failure cases and computational efficiency:

```bash
# Run error analysis on trained model
python experiments/error_analysis.py \
    --config config.yml \
    --checkpoint checkpoints/best_model.pt \
    --system multi_scale_oscillators \
    --output-dir error_analysis_results
```

This generates:
- `error_analysis_results/failure_analysis.json` - Failure case statistics
- `error_analysis_results/worst_failure_cases.png` - Visualization of failures
- `error_analysis_results/expert_distance_matrix.png` - Expert similarity analysis
- `error_analysis_results/routing_patterns.png` - Routing behavior analysis
- `error_analysis_results/throughput_scaling.png` - Computational efficiency

#### Step 5: Visualization Experiments

Generate comprehensive visualizations:

```bash
# Run visualization experiments
python experiments/visualization_experiments.py \
    --config config.yml \
    --checkpoint checkpoints/best_model.pt \
    --system multi_scale_oscillators \
    --checkpoint-dir checkpoints \
    --output-dir visualization_results
```

This creates:
- Expert vector fields at different time points
- Expert specialization regions (2D projections)
- Learning dynamics evolution
- Interactive 3D trajectories (HTML files)
- Expert activation patterns (t-SNE)

## Experiment Configurations

### For Different Systems

1. **Chaotic Systems (Lorenz)**
   ```bash
   python train.py --config configs/experiment_lorenz.yml --system piecewise_lorenz
   ```

2. **Large-Scale Systems (Kuramoto)**
   ```bash
   python train.py --config configs/experiment_large_scale.yml --system kuramoto_model
   ```

3. **Stiff Systems (Van der Pol)**
   ```bash
   python train.py --config configs/experiment_stiff.yml --system van_der_pol_network
   ```

### For Different Scenarios

1. **Quick Development Test**
   ```bash
   python train.py --config configs/quick_test.yml --system multi_scale_oscillators
   ```

2. **Memory-Constrained Systems**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Long Training Runs**
   - Increase num_epochs in config
   - Use larger patience for early stopping
   - Enable wandb logging for monitoring

## Analyzing Results

### 1. Training Metrics

Check the training logs:
```bash
# View training progress
tensorboard --logdir logs

# Check training summary
cat logs/training_summary.json
```

### 2. Comparison Results

```bash
# View comparison table
cat comparison_results/multi_scale_oscillators_results_table.csv

# Check statistical significance
grep -A 20 "STATISTICAL ANALYSIS" comparison_results/experiment_log.txt
```

### 3. Ablation Results

```bash
# View ablation summary
cat ablation_results/routing_mechanism/summary.json

# Check specific ablation impact
python -c "import json; print(json.load(open('ablation_results/routing_mechanism/summary.json'))['baseline']['trajectory_mse'])"
```

## Common Issues and Solutions

### 1. CUDA Out of Memory

**Solution**: Reduce batch size or sequence length in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
  sequence_length: 50  # Reduce from 100
```

### 2. Training Instability

**Solution**: Use stiff system configuration:
```bash
python train.py --config configs/experiment_stiff.yml --system your_system
```

### 3. Slow Training

**Solution**: Enable mixed precision and use larger batch:
```yaml
compute:
  mixed_precision: true
training:
  batch_size: 64
```

### 4. Poor Expert Diversity

**Solution**: Increase diversity regularization:
```yaml
training:
  regularization:
    diversity_weight: 2.0  # Increase from 1.0
```

## Reproducing Paper Results

To reproduce the main results from the paper:

```bash
# 1. Run main experiments
python run_experiments.py \
    --config config.yml \
    --systems multi_scale_oscillators piecewise_lorenz van_der_pol_network kuramoto_model \
    --experiments all \
    --results-dir paper_results

# 2. Generate final report
cd paper_results
python -c "
import json
with open('experiment_summary.json') as f:
    summary = json.load(f)
print('Experiments completed:', len(summary['experiments']))
print('Check final_report.md for comprehensive results')
"
```

## Custom Experiments

### Adding a New System

1. Create system class in `src/data/synthetic_systems.py`
2. Add to `SyntheticDataGenerator._initialize_systems()`
3. Create config in `configs/experiment_yoursystem.yml`
4. Run: `python train.py --config configs/experiment_yoursystem.yml --system your_system`

### Adding a New Baseline

1. Implement model in `src/baselines/`
2. Add to `create_model()` in `experiments/comparative_analysis.py`
3. Include in comparison: `--models ame_ode your_baseline`

## Summary Checklist

- [ ] Install dependencies and verify with `test_implementation.py`
- [ ] Run quick test to ensure setup works
- [ ] Train AME-ODE models for each system
- [ ] Run comparative analysis against baselines
- [ ] Perform ablation studies
- [ ] Conduct error analysis
- [ ] Generate visualizations
- [ ] Review final report and results

## Contact and Issues

If you encounter any issues:
1. Check the experiment logs in the results directory
2. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure all dependencies are installed: `pip install -r requirements.txt --upgrade`