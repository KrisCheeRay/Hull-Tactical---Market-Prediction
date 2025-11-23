import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path

from src.policy_head import PolicyHead
from src.configs import ArtifactPaths

# -----------------------------------------------------------------------------
# 1. 奖励函数 (由量化同学提供，此处为示例)
# -----------------------------------------------------------------------------
def compute_reward(
    position: torch.Tensor,
    returns: torch.Tensor,
    volatility: torch.Tensor,
    lambda_risk: float = 0.1
) -> torch.Tensor:
    """
    R = Position * Return - lambda * Risk
    """
    pnl = position * returns
    # 简单风险惩罚：波动率越高，持有仓位的惩罚越大
    # 或者：(Position * Volatility)^2
    risk_penalty = lambda_risk * (position * volatility)**2
    return pnl - risk_penalty

# -----------------------------------------------------------------------------
# 2. 数据集定义
# -----------------------------------------------------------------------------
class GRPODataset(Dataset):
    def __init__(self, y_hat_path: str, raw_csv_path: str, window_size: int = 20):
        # 加载 OOS 信号
        self.signals = pl.read_parquet(y_hat_path)
        
        # 加载原始数据用于计算统计量和 reward
        # 假设 raw_csv 包含: ds, y (真实收益), 以及原始价格用于算 vol
        # 这里简化：假设 train.csv 里 y 就是收益
        self.raw = pl.read_csv(raw_csv_path).rename({"market_forward_excess_returns": "y"})
        self.raw = self.raw.with_columns(pl.col("date_id").cast(pl.Int64).alias("ds"))
        
        # Join 信号与原始数据
        self.df = self.raw.join(self.signals, on="ds", how="inner").sort("ds")
        
        self.window_size = window_size
        
        # 预计算滚动统计量 (作为 Policy 输入的一部分)
        # Volatility: 过去 N 天 y 的标准差
        self.df = self.df.with_columns([
            pl.col("y").rolling_std(window_size).alias("rolling_vol").fill_null(0.0),
            pl.col("y").rolling_mean(window_size).alias("rolling_trend").fill_null(0.0)
        ])
        
        # 转换为 numpy 以便快速索引
        self.data = self.df.select([
            "y_hat",       # SFT 预测
            "rolling_vol", # 市场状态 1
            "rolling_trend", # 市场状态 2
            "y"            # 真实收益 (用于计算 Reward)
        ]).to_numpy()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Input State: [y_hat, vol, trend]
        state = self.data[idx, :3].astype(np.float32)
        # Target info for reward: [y_true, vol]
        # 注意：Reward 取决于当天的真实收益 y
        target_info = self.data[idx, [3, 1]].astype(np.float32) 
        
        return state, target_info

# -----------------------------------------------------------------------------
# 3. 训练脚本
# -----------------------------------------------------------------------------
def train_grpo():
    artifacts = ArtifactPaths()
    signals_path = Path(artifacts.models_dir) / "sft_y_hat_oos.parquet"
    raw_data_path = Path("/kaggle/input/hull-tactical-market-prediction/train.csv")
    
    if not signals_path.exists():
        print(f"Signals file not found: {signals_path}")
        print("Please run kaggle/work/gen_sft_signals.py first.")
        return

    # 配置
    EPOCHS = 10
    BATCH_SIZE = 256
    LR = 1e-3
    
    # 准备数据
    dataset = GRPODataset(str(signals_path), str(raw_data_path))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    # Input dim = 3 (y_hat, vol, trend)
    policy = PolicyHead(input_dim=3, hidden_dim=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    print("Starting GRPO Training...")
    
    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        
        for states, targets in dataloader:
            # states: [B, 3]
            # targets: [B, 2] -> y_true, vol
            
            # 1. 采样动作
            actions = policy.get_action(states, deterministic=False) # [B]
            
            # 2. 计算奖励
            y_true = targets[:, 0]
            vol = targets[:, 1]
            rewards = compute_reward(actions, y_true, vol) # [B]
            
            # 3. 计算 Loss (Policy Gradient / REINFORCE)
            # Loss = - log_prob(a) * (R - baseline)
            # 简单起见用 batch 均值做 baseline
            baseline = rewards.mean()
            adv = rewards - baseline
            
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 4. 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
        avg_rew = total_reward / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Reward: {avg_rew:.4f} | Loss: {total_loss:.4f}")
        
    # 保存
    save_path = Path(artifacts.models_dir) / "policy_head.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Policy Head saved to {save_path}")

if __name__ == "__main__":
    train_grpo()

