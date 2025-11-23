import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyHead(nn.Module):
    """
    Policy Head: 根据 SFT 预测值与市场状态，输出仓位 [0, 2]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            input_dim: 输入特征维度 (1 + n_stats)
                       1: y_hat (SFT 预测值)
                       n_stats: 历史窗口统计量 (如 vol, trend 等)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # 输出 alpha, beta 用于 Beta 分布
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha, beta: Beta 分布的参数，保证 > 0
        """
        out = self.net(x)
        # Softplus 保证正数，+1e-6 防止除零
        alpha = F.softplus(out[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 1]) + 1.0 + 1e-6
        return alpha, beta
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        获取仓位 [0, 2]
        """
        alpha, beta = self.forward(x)
        
        if deterministic:
            # 推理/评估：使用均值
            # Mean of Beta = alpha / (alpha + beta)
            # 映射到 [0, 2] -> 2 * Mean
            action = 2.0 * (alpha / (alpha + beta))
        else:
            # 训练：从分布采样
            dist = torch.distributions.Beta(alpha, beta)
            sample = dist.sample()
            action = 2.0 * sample
            
        return action.clamp(0.0, 2.0)

    def get_log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算动作的 log probability (用于 Policy Gradient)
        Args:
            x: state
            action: 实际采取的动作 [0, 2]
        """
        alpha, beta = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)
        
        # action 是 [0, 2]，需归一化回 [0, 1] 计算 Beta 分布概率
        action_norm = action / 2.0
        action_norm = action_norm.clamp(1e-6, 1.0 - 1e-6) # 防止边界数值不稳定
        
        log_prob = dist.log_prob(action_norm)
        return log_prob

