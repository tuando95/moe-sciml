# Adaptive Mixture of Expert ODEs (AME-ODE)

Implementation of Adaptive Mixture of Expert ODEs for modeling heterogeneous dynamical systems using PyTorch and torchdiffeq.

## Overview

AME-ODE is a novel approach that combines mixture of experts with neural ODEs to capture heterogeneous dynamics in complex systems. The model adaptively routes different regions of state space and time to specialized expert networks, enabling efficient and accurate modeling of multi-scale, multi-regime dynamical systems.

## Key Features

- **Adaptive Expert Routing**: Dynamic selection of experts based on current state and dynamics history
- **Heterogeneous Dynamics**: Specialized experts for different timescales, stability regimes, and nonlinearity levels
- **Efficient Integration**: Adaptive time-stepping with routing-aware step size control
- **Comprehensive Training**: Multi-component loss function encouraging specialization and efficiency

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/moe-sciml.git
cd moe-sciml

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
# Train on multi-scale oscillators
python train.py --system multi_scale_oscillators --device cuda

# Train on piecewise Lorenz system
python train.py --system piecewise_lorenz --config config.yml
```

### Evaluating a Trained Model

```bash
# Evaluate a checkpoint
python train.py --evaluate --resume checkpoints/best_model.pt
```

### Running Ablation Studies

```bash
# Run routing mechanism ablation
python experiments/run_ablations.py --ablation routing_mechanism

# Run expert initialization ablation
python experiments/run_ablations.py --ablation expert_initialization
```

## Project Structure

```
moe-sciml/
├── config.yml              # Main configuration file
├── requirements.txt        # Python dependencies
├── train.py               # Main training script
├── src/
│   ├── models/
│   │   ├── expert_ode.py      # Expert ODE networks
│   │   ├── gating.py          # Gating and routing mechanisms
│   │   ├── ame_ode.py         # Main AME-ODE model
│   │   └── losses.py          # Loss functions
│   ├── data/
│   │   └── synthetic_systems.py  # Synthetic dynamical systems
│   ├── training/
│   │   └── trainer.py         # Training loop and utilities
│   ├── evaluation/
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualization.py   # Visualization tools
│   └── utils/
│       └── config.py          # Configuration management
├── experiments/
│   └── run_ablations.py      # Ablation study scripts
└── tests/                     # Unit tests
```

## Configuration

The model is configured through `config.yml`. Key parameters include:

- `model.n_experts`: Number of expert ODEs
- `model.expert_architecture`: Expert network architecture
- `model.gating_architecture`: Gating network architecture
- `training.num_epochs`: Number of training epochs
- `integration.method`: ODE integration method (e.g., 'dopri5')

## Synthetic Systems

The framework includes several synthetic dynamical systems for evaluation:

1. **Multi-Scale Oscillators**: Coupled oscillators with fast/slow timescales
2. **Piecewise Lorenz**: Lorenz system with linear/chaotic regions
3. **Van der Pol Network**: Network of coupled Van der Pol oscillators
4. **Kuramoto Model**: Phase oscillator synchronization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ame-ode2024,
  title={Adaptive Mixture of Expert ODEs for Heterogeneous Dynamics},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.