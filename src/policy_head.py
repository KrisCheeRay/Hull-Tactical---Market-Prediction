import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PolicyHead(nn.Module):
    """
    Policy Head: Outputs position [0, 2] based on SFT predictions and market state.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            input_dim: Input feature dimension (1 + n_stats)
                       1: y_hat (SFT prediction)
                       n_stats: Historical window statistics (e.g., vol, trend)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # Output alpha, beta for Beta distribution
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha, beta: Parameters for Beta distribution, guaranteed > 0
        """
        out = self.net(x)
        # Softplus ensures positivity, +1e-6 prevents division by zero
        alpha = F.softplus(out[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 1]) + 1.0 + 1e-6
        return alpha, beta
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get position [0, 2]
        """
        alpha, beta = self.forward(x)
        
        if deterministic:
            # Inference/Evaluation: Use mean
            # Mean of Beta = alpha / (alpha + beta)
            # Map to [0, 2] -> 2 * Mean
            action = 2.0 * (alpha / (alpha + beta))
        else:
            # Training: Sample from distribution
            dist = torch.distributions.Beta(alpha, beta)
            sample = dist.sample()
            action = 2.0 * sample
            
        return action.clamp(0.0, 2.0)

    def get_log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probability of the action (for Policy Gradient)
        Args:
            x: state
            action: actual action taken [0, 2]
        """
        alpha, beta = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)
        
        # action is [0, 2], need to normalize back to [0, 1] for Beta distribution probability
        action_norm = action / 2.0
        action_norm = action_norm.clamp(1e-6, 1.0 - 1e-6) # Prevent boundary numerical instability
        
        log_prob = dist.log_prob(action_norm)
        return log_prob
