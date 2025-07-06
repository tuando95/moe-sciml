import torch
import torch.nn as nn
from typing import Optional, Tuple


class DynamicsHistoryEncoder(nn.Module):
    """LSTM-based encoder for dynamics history."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: [x(t), dx/dt]
        input_dim = state_dim * 2
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Initialize hidden states
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Update history embedding.
        
        Args:
            x: Current state (batch_size, state_dim)
            dx_dt: Current dynamics (batch_size, state_dim)
            hidden: Previous hidden state tuple (h, c)
            
        Returns:
            History embedding and updated hidden state
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = self.h0.expand(-1, batch_size, -1).contiguous()
            c = self.c0.expand(-1, batch_size, -1).contiguous()
            hidden = (h, c)
        
        # Concatenate state and dynamics
        lstm_input = torch.cat([x, dx_dt], dim=-1)
        lstm_input = lstm_input.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Extract history embedding (last hidden state)
        history_embedding = hidden[0][-1]  # Shape: (batch_size, hidden_dim)
        
        return history_embedding, hidden


class GatingNetwork(nn.Module):
    """Gating network for expert selection."""
    
    def __init__(
        self,
        state_dim: int,
        history_dim: int,
        n_experts: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.history_dim = history_dim
        self.n_experts = n_experts
        self.temperature = temperature
        
        # Input: [x, h, t, ||x||, ||h||]
        input_dim = state_dim + history_dim + 3
        
        # Build gating network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (logits for each expert)
        layers.append(nn.Linear(hidden_dim, n_experts))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        history: torch.Tensor,
        t: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Compute gating weights.
        
        Args:
            x: State tensor (batch_size, state_dim)
            history: History embedding (batch_size, history_dim)
            t: Time tensor (batch_size, 1) or (batch_size,)
            return_logits: Whether to return logits before softmax
            
        Returns:
            Gating weights (batch_size, n_experts)
        """
        # Ensure proper shapes
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Compute norms
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        h_norm = torch.norm(history, dim=-1, keepdim=True)
        
        # Concatenate inputs
        inputs = torch.cat([x, history, t, x_norm, h_norm], dim=-1)
        
        # Forward through network
        logits = self.net(inputs)
        
        if return_logits:
            return logits
        
        # Apply temperature and softmax
        weights = torch.softmax(logits / self.temperature, dim=-1)
        
        return weights
    
    def get_routing_gradients(
        self,
        x: torch.Tensor,
        history: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradients of routing weights w.r.t. time.
        
        Used for adaptive time stepping.
        """
        # Don't compute expensive gradients during training
        if self.training:
            return torch.zeros(x.shape[0], self.n_experts, device=x.device)
        
        # Clone to avoid modifying input
        t_grad = t.clone().detach().requires_grad_(True)
        
        # Forward pass with gradient-enabled time
        weights = self.forward(x, history, t_grad)
        
        # Compute gradient of each weight w.r.t. time
        grads = []
        for i in range(self.n_experts):
            if i < self.n_experts - 1:
                grad = torch.autograd.grad(
                    weights[:, i].sum(),
                    t_grad,
                    create_graph=False,  # Don't create graph
                    retain_graph=True,   # Keep for next iteration
                )[0]
            else:
                # Last iteration, don't retain
                grad = torch.autograd.grad(
                    weights[:, i].sum(),
                    t_grad,
                    create_graph=False,
                    retain_graph=False,
                )[0]
            grads.append(grad)
        
        return torch.stack(grads, dim=-1)


class AdaptiveGatingModule(nn.Module):
    """Complete gating module with history encoding."""
    
    def __init__(
        self,
        state_dim: int,
        n_experts: int,
        history_config: dict,
        gating_config: dict,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_experts = n_experts
        
        # History encoder
        self.history_encoder = DynamicsHistoryEncoder(
            state_dim=state_dim,
            hidden_dim=history_config['hidden_dim'],
            num_layers=history_config.get('num_layers', 1),
        )
        
        # Gating network
        self.gating_network = GatingNetwork(
            state_dim=state_dim,
            history_dim=history_config['hidden_dim'],
            n_experts=n_experts,
            hidden_dim=gating_config['width'],
            num_layers=gating_config['depth'],
            temperature=gating_config.get('temperature', 1.0),
        )
        
        # Cache for history states
        self.hidden_states = None
    
    def reset_history(self, batch_size: Optional[int] = None):
        """Reset history states."""
        self.hidden_states = None
    
    def forward(
        self,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
        t: torch.Tensor,
        update_history: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating weights with history update.
        
        Args:
            x: Current state
            dx_dt: Current dynamics
            t: Current time
            update_history: Whether to update history state
            
        Returns:
            Gating weights and history embedding
        """
        # Update history
        history, new_hidden = self.history_encoder(x, dx_dt, self.hidden_states)
        
        if update_history:
            self.hidden_states = new_hidden
        
        # Compute gating weights
        weights = self.gating_network(x, history, t)
        
        return weights, history
    
    def compute_routing_smoothness(
        self,
        trajectory: torch.Tensor,
        dynamics: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Compute routing smoothness over a trajectory.
        
        Used for regularization during training.
        """
        self.reset_history()
        
        smoothness = 0.0
        prev_weights = None
        
        for t in range(trajectory.shape[1]):
            x_t = trajectory[:, t]
            dx_t = dynamics[:, t]
            t_val = times[t] if times.dim() > 0 else times
            
            weights, _ = self.forward(x_t, dx_t, t_val)
            
            if prev_weights is not None:
                # L2 distance between consecutive weights
                smoothness += torch.mean((weights - prev_weights) ** 2)
            
            prev_weights = weights
        
        return smoothness / (trajectory.shape[1] - 1)