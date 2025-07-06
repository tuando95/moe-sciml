"""Improved initialization strategies for AME-ODE."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


def initialize_expert_ode(
    expert: nn.Module,
    expert_id: int,
    n_experts: int,
    state_dim: int,
    init_strategy: str = "spectral"
):
    """Initialize expert ODE with specialized strategies.
    
    Args:
        expert: Expert ODE module
        expert_id: Index of this expert
        n_experts: Total number of experts
        state_dim: Dimension of state space
        init_strategy: Initialization strategy ('spectral', 'stability', 'mixed')
    """
    
    if init_strategy == "spectral":
        # Initialize experts to capture different frequency bands
        freq_scale = np.logspace(-1, 1, n_experts)[expert_id]
        
        # Initialize network weights
        for name, param in expert.named_parameters():
            if 'weight' in name:
                if 'net.-1' in name or name.endswith('net.6.weight'):  # Output layer
                    # Scale output to match frequency
                    nn.init.xavier_normal_(param)
                    param.data *= freq_scale * 0.1
                else:
                    # Hidden layers - use spectral normalization
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Set omega parameter for temporal encoding
        if hasattr(expert, 'omega'):
            expert.omega.data = torch.tensor(freq_scale)
            
    elif init_strategy == "stability":
        # Initialize for different stability regimes
        stability_factor = np.linspace(-0.5, 0.5, n_experts)[expert_id]
        
        for name, param in expert.named_parameters():
            if 'weight' in name:
                if 'net.-1' in name or name.endswith('net.6.weight'):  # Output layer
                    # Initialize to be slightly contractive/expansive
                    nn.init.xavier_normal_(param)
                    param.data *= (1.0 + stability_factor) * 0.1
                else:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                if 'net.-1' in name or name.endswith('net.6.bias'):  # Output layer
                    # Bias towards stability
                    nn.init.constant_(param, -0.1 * stability_factor)
                else:
                    nn.init.zeros_(param)
                    
    elif init_strategy == "mixed":
        # Mixed strategy - combine spectral and stability
        freq_scale = np.logspace(-1, 1, n_experts)[expert_id]
        stability_factor = np.sin(2 * np.pi * expert_id / n_experts) * 0.3
        
        for name, param in expert.named_parameters():
            if 'weight' in name:
                if 'net.-1' in name or name.endswith('net.6.weight'):  # Output layer
                    # Combine frequency and stability scaling
                    nn.init.xavier_normal_(param)
                    param.data *= freq_scale * (1.0 + stability_factor) * 0.1
                else:
                    # Use different init for different experts
                    if expert_id % 2 == 0:
                        nn.init.orthogonal_(param, gain=np.sqrt(2))
                    else:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        # Set omega parameter
        if hasattr(expert, 'omega'):
            expert.omega.data = torch.tensor(freq_scale)


def initialize_gating_network(
    gating: nn.Module,
    n_experts: int,
    temperature: float = 1.0,
    init_strategy: str = "uniform"
):
    """Initialize gating network for good routing.
    
    Args:
        gating: Gating network module
        n_experts: Number of experts
        temperature: Gating temperature
        init_strategy: Initialization strategy ('uniform', 'sparse', 'learned')
    """
    
    if init_strategy == "uniform":
        # Initialize to give roughly uniform routing initially
        for name, param in gating.named_parameters():
            if 'weight' in name:
                if 'output' in name or name.endswith('2.weight'):  # Output layer
                    # Small random init for logits
                    nn.init.normal_(param, mean=0.0, std=0.01 / temperature)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                if 'output' in name or name.endswith('2.bias'):  # Output layer
                    # Initialize biases to make routing roughly uniform
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.zeros_(param)
                    
    elif init_strategy == "sparse":
        # Initialize for sparse routing (one-hot like)
        for name, param in gating.named_parameters():
            if 'weight' in name:
                if 'output' in name or name.endswith('2.weight'):  # Output layer
                    # Larger variance for more decisive routing
                    nn.init.normal_(param, mean=0.0, std=1.0 / temperature)
                else:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    elif init_strategy == "learned":
        # Initialize with small random weights, let it learn from scratch
        for name, param in gating.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)


def initialize_ame_ode(model, config: dict):
    """Initialize complete AME-ODE model with improved strategies.
    
    Args:
        model: AME-ODE model
        config: Model configuration dict
    """
    n_experts = model.n_experts
    state_dim = model.state_dim
    
    # Initialize experts with diverse strategies
    expert_init_strategy = config.get('expert_init_strategy', 'mixed')
    for i, expert in enumerate(model.experts.experts):
        initialize_expert_ode(
            expert, i, n_experts, state_dim, expert_init_strategy
        )
    
    # Initialize gating network
    gating_init_strategy = config.get('gating_init_strategy', 'uniform')
    temperature = config.get('temperature', 1.0)
    
    if hasattr(model.gating, 'gating_network'):
        for gating_net in model.gating.gating_network.gating_networks:
            initialize_gating_network(
                gating_net, n_experts, temperature, gating_init_strategy
            )
    
    # Initialize LSTM in history encoder
    if hasattr(model.gating, 'history_encoder') and hasattr(model.gating.history_encoder, 'lstm'):
        lstm = model.gating.history_encoder.lstm
        # Use orthogonal initialization for LSTM
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    print(f"Initialized AME-ODE with {n_experts} experts using {expert_init_strategy} strategy")
    
    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for p in model.experts.parameters())
    gating_params = sum(p.numel() for p in model.gating.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Expert parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)")
    print(f"Gating parameters: {gating_params:,} ({gating_params/total_params*100:.1f}%)")