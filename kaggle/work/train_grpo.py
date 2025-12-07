import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.policy_head import PolicyHead
from src.configs import ArtifactPaths, DataSchema

# -----------------------------------------------------------------------------
# 1. Reward Function (Comprehensive Version)
# -----------------------------------------------------------------------------
def compute_reward(
    position: torch.Tensor,           # Current position [B]
    prev_position: torch.Tensor,      # Previous position [B] (for transaction cost)
    returns: torch.Tensor,            # Actual returns [B] (forward_returns[T])
    volatility: torch.Tensor,         # Volatility [B]
    y_hat: torch.Tensor,              # Prediction [B] (for direction judgment)
    lambda_risk: float = 0.5,         # Risk penalty coefficient
    lambda_turnover: float = 0.05,    # Transaction cost coefficient
    lambda_direction: float = 0.05,   # Direction bonus coefficient
    transaction_cost: float = 0.001   # Single transaction cost (e.g., 0.1%)
) -> torch.Tensor:
    """
    Comprehensive Reward Function:
    R = PnL - Risk_Penalty - Turnover_Penalty + Direction_Bonus
    
    Args:
        position: Current position [0, 2]
        prev_position: Previous position [0, 2]
        returns: Actual returns (forward_returns)
        volatility: Volatility
        y_hat: Prediction (used for direction judgment)
        lambda_risk: Risk penalty coefficient
        lambda_turnover: Transaction cost coefficient
        lambda_direction: Direction bonus coefficient
        transaction_cost: Single transaction cost
    """
    # 1. PnL (Core Profit)
    # Amplify PnL signal because returns are small (approx 0.1%), need amplification for model learning
    pnl = position * returns * 100.0  # Amplify by 100x to make signal more obvious
    
    # 2. Risk Penalty (Higher penalty for higher volatility)
    risk_penalty = lambda_risk * (position * volatility) ** 2
    
    # 3. Transaction Cost (Reduce frequent trading)
    position_change = torch.abs(position - prev_position)
    turnover_penalty = lambda_turnover * position_change * transaction_cost
    
    # 4. Direction Bonus (Reward if predicted direction matches actual)
    # Assume position > 1.0 means Long, < 1.0 means Short
    predicted_direction = torch.sign(y_hat)  # +1 Long, -1 Short, 0 Neutral
    actual_direction = torch.sign(returns)   # +1 Up, -1 Down, 0 Unchanged
    
    # Check if direction matches
    # Long & Up, OR Short & Down, OR Neutral & Unchanged
    direction_match = (
        ((predicted_direction > 0) & (actual_direction > 0)) |  # Long & Up
        ((predicted_direction < 0) & (actual_direction < 0)) |  # Short & Down
        ((predicted_direction == 0) & (actual_direction == 0))   # Neutral & Unchanged
    ).float()
    direction_bonus = lambda_direction * direction_match
    
    return pnl - risk_penalty - turnover_penalty + direction_bonus

# -----------------------------------------------------------------------------
# 2. Dataset Definition
# -----------------------------------------------------------------------------
class GRPODataset(Dataset):
    def __init__(self, y_hat_path: str, raw_csv_path: str, feature_cols: list, window_size: int = 20):
        """
        Args:
            y_hat_path: OOS prediction signal file path (sft_y_hat_oos.parquet)
            raw_csv_path: Raw training data path (train_feature_selected.csv)
            feature_cols: Important factor column names (E2, M13, P8, P5, V9, S2, M12, S5)
            window_size: Rolling window size for calculating vol and trend
        """
        # Load OOS signals (use nhits_y_hat only, not patchtst_y_hat)
        self.signals = pl.read_parquet(y_hat_path)
        
        # [CRITICAL ALIGNMENT FIX]
        # Current status (gen_sft_signals.py did shift(1)):
        #   - Input: y[ds=T] = forward_returns[T-1]
        #   - Prediction: y_hat[ds=T] corresponds to y[ds=T], i.e., forward_returns[T-1]
        #   So in signals: y_hat[ds=T] predicts forward_returns[T-1]
        #
        # Target Alignment:
        #   - y_hat[ds=T] should correspond to forward_returns[T] (predicting tomorrow today)
        #
        # Solution:
        #   - signals[ds=T] predicts forward_returns[T-1]
        #   - To make y_hat[ds=T] correspond to forward_returns[T], we need:
        #     signals[ds=T+1] (predicting forward_returns[T]) to align with raw[ds=T]'s forward_returns[T]
        #   - Therefore: shift signals' ds forward by 1 (ds = ds + 1)
        self.signals = self.signals.with_columns(
            (pl.col("ds") + 1).alias("ds")  # shift forward by 1 to align: y_hat[ds=T] predicts forward_returns[T]
        )
        
        # Load raw data (contains factor columns and actual returns)
        self.raw = pl.read_csv(raw_csv_path)
        if "date_id" in self.raw.columns:
            self.raw = self.raw.with_columns(pl.col("date_id").cast(pl.Int64).alias("ds"))
        if "forward_returns" in self.raw.columns:
            self.raw = self.raw.rename({"forward_returns": "y"})
        
        # [Data Cleaning: Consistent with SFT Training]
        # Apply same filtering logic: null_count <= 4
        original_len = len(self.raw)
        null_counts = self.raw.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        self.raw = self.raw.with_columns(null_counts)
        
        # Filter: Keep rows with <= 4 nulls AND valid y
        self.raw = self.raw.filter(
            (pl.col("null_count") <= 4) & (pl.col("y").is_not_null())
        )
        self.raw = self.raw.drop("null_count")  # cleanup
        
        print(f"Data Cleaning: Dropped {original_len - len(self.raw)} rows. Remaining: {len(self.raw)}")
        
        # Join signals with raw data
        # Now: y_hat[ds=T] corresponds to forward_returns[ds=T] (predicting tomorrow today)
        self.df = self.raw.join(self.signals, on="ds", how="inner").sort("ds")
        
        # [Critical: Clean again after Join]
        # Join might result in NaNs in some feature columns (if signals missing for some rows)
        # Re-apply null_count <= 4 filtering logic
        join_original_len = len(self.df)
        null_counts_after_join = self.df.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        self.df = self.df.with_columns(null_counts_after_join)
        
        # Filter: Keep rows with <= 4 nulls in features AND valid y AND valid nhits_y_hat
        self.df = self.df.filter(
            (pl.col("null_count") <= 4) & 
            (pl.col("y").is_not_null()) & 
            (pl.col("nhits_y_hat").is_not_null())
        )
        self.df = self.df.drop("null_count")  # cleanup
        
        print(f"After join cleaning: Dropped {join_original_len - len(self.df)} rows. Remaining: {len(self.df)}")
        
        self.window_size = window_size
        self.feature_cols = feature_cols
        
        # Pre-compute rolling statistics (as part of Policy Input)
        # Volatility: rolling std of y over N days
        # Trend: rolling mean of y over N days
        self.df = self.df.with_columns([
            pl.col("y").rolling_std(window_size).alias("rolling_vol").fill_null(0.0),
            pl.col("y").rolling_mean(window_size).alias("rolling_trend").fill_null(0.0)
        ])
        
        # Ensure all required columns exist
        required_cols = ["nhits_y_hat", "rolling_vol", "rolling_trend", "y"] + feature_cols
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numpy for fast indexing
        # State: [nhits_y_hat, rolling_vol, rolling_trend, E2, M13, P8, P5, V9, S2, M12, S5]
        # Target: [y_true, vol, prev_position, y_hat] (for Reward calculation)
        state_cols = ["nhits_y_hat", "rolling_vol", "rolling_trend"] + feature_cols
        
        # Clean NaNs: Check and fill (before converting to numpy)
        # Note: We already filtered rows with null_count <= 4, so at most 4 NaNs per row
        # Filling remaining NaNs with 0.0 is reasonable
        state_df = self.df.select(state_cols)
        
        # Fill all NaNs: Fill with 0.0 directly in Polars
        fill_exprs = [pl.col(col).fill_null(0.0) for col in state_cols]
        state_df = state_df.with_columns(fill_exprs)
        
        # Convert to numpy, ensure float type
        self.state_data = state_df.to_numpy().astype(np.float32)
        
        # Final cleanup: Ensure no NaNs or Inf
        self.state_data = np.nan_to_num(self.state_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate previous position (for transaction cost calculation)
        # Note: In training loop, prev_position should come from previous action
        # But in Dataset, we cannot access previous action
        # Solution: Use previous y_hat sign as approximation (simple but not accurate enough)
        # Initialize as neutral position 1.0, update dynamically in training loop
        prev_positions = np.ones(len(self.df), dtype=np.float32)  # Initialize as neutral 1.0
        if len(self.df) > 1:
            # Approximate prev_position using previous y_hat sign
            # position = 1.0 + sign(y_hat) * 0.5 (Long 1.5, Short 0.5, Neutral 1.0)
            nhits_y_hat = self.df["nhits_y_hat"].to_numpy()
            prev_positions[1:] = 1.0 + np.sign(nhits_y_hat[:-1]) * 0.5
            prev_positions = prev_positions.astype(np.float32)
        
        # Clean NaNs in target data
        y_true = self.df["y"].to_numpy()
        vol = self.df["rolling_vol"].to_numpy()
        y_hat = self.df["nhits_y_hat"].to_numpy()
        
        # Fill NaNs
        y_true = np.nan_to_num(y_true, nan=0.0)
        vol = np.nan_to_num(vol, nan=0.0)
        y_hat = np.nan_to_num(y_hat, nan=0.0)
        
        self.target_data = np.column_stack([
            y_true,           # y_true (forward_returns)
            vol,              # vol
            prev_positions,   # prev_position
            y_hat             # y_hat (for direction judgment)
        ])
        
    def __len__(self):
        return len(self.state_data)
    
    def __getitem__(self, idx):
        # Input State: [nhits_y_hat, vol, trend, E2, M13, P8, P5, V9, S2, M12, S5] (11 dimensions)
        state = self.state_data[idx].astype(np.float32)
        # Target info for reward: [y_true, vol, prev_position, y_hat]
        # y_true: Actual returns (forward_returns[T])
        # vol: Volatility
        # prev_position: Previous position (for transaction cost)
        # y_hat: Prediction (for direction judgment)
        target_info = self.target_data[idx].astype(np.float32)
        
        # Final check: Ensure no NaN or Inf
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        target_info = np.nan_to_num(target_info, nan=0.0, posinf=0.0, neginf=0.0)
        
        return state, target_info

# -----------------------------------------------------------------------------
# 3. Training Script
# -----------------------------------------------------------------------------
def train_grpo():
    artifacts = ArtifactPaths()
    schema = DataSchema()
    
    signals_path = Path(artifacts.models_dir) / "sft_y_hat_oos.parquet"
    # Use train_feature_selected.csv because it contains the 8 factor columns we need
    raw_data_path = Path("/kaggle/input/hull-tactical-market-prediction/train_feature_selected.csv")
    
    if not signals_path.exists():
        print(f"Signals file not found: {signals_path}")
        print("Please run kaggle/work/gen_sft_signals.py first.")
        return
    
    if not raw_data_path.exists():
        # Try local path
        raw_data_path = Path("kaggle/input/hull-tactical-market-prediction/train_feature_selected.csv")
        if not raw_data_path.exists():
            print(f"Raw data file not found: {raw_data_path}")
            return

    # Configuration
    EPOCHS = 50  # Increase epochs for more learning
    BATCH_SIZE = 128  # Decrease batch size, increase updates per epoch (7953/128 approx 62 batches/epoch)
    LR = 1e-3  # Decrease learning rate for stability
    GRAD_CLIP = 1.0  # Gradient clipping
    
    # Prepare Data
    # State dimension: 1 (nhits_y_hat) + 1 (rolling_vol) + 1 (rolling_trend) + 8 (features) = 11
    dataset = GRPODataset(
        str(signals_path), 
        str(raw_data_path),
        feature_cols=schema.feature_cols,
        window_size=20
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    # Input dim = 11 (nhits_y_hat, vol, trend, E2, M13, P8, P5, V9, S2, M12, S5)
    input_dim = 1 + 1 + 1 + len(schema.feature_cols)  # nhits_y_hat + vol + trend + 8 features
    policy = PolicyHead(input_dim=input_dim, hidden_dim=128)  # Increase hidden_dim for more inputs
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    # LR Scheduler: Reduce LR every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Save best model (Based on Loss, lower is better)
    best_loss = float('inf')
    best_model_path = Path(artifacts.models_dir) / "policy_head_best.pt"
    
    print(f"Starting GRPO Training...")
    print(f"State dimension: {input_dim} (nhits_y_hat + vol + trend + {len(schema.feature_cols)} features)")
    print(f"Features: {schema.feature_cols}")
    
    # Track previous positions (for transaction cost)
    # Note: Since DataLoader shuffles, reset at start of epoch
    prev_positions_batch = None
    
    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        prev_positions_batch = None  # Reset per epoch
        
        # Debug: collect stats
        all_actions = []
        all_rewards = []
        
        for batch_idx, (states, targets) in enumerate(dataloader):
            # states: [B, 11] -> [nhits_y_hat, vol, trend, E2, M13, P8, P5, V9, S2, M12, S5]
            # targets: [B, 4] -> [y_true, vol, prev_position_approx, y_hat]
            
            # 1. Sample Actions
            actions = policy.get_action(states, deterministic=False) # [B]
            
            # 2. Get prev_position
            # Prioritize using actions from previous batch (more accurate), else use approx from Dataset
            # Note: Ensure batch size matches
            if prev_positions_batch is not None and len(prev_positions_batch) == len(actions):
                prev_position = prev_positions_batch
            else:
                # If batch size mismatch (e.g. last batch), use approx from Dataset
                prev_position = targets[:, 2]  # Approx from Dataset
            
            # 3. Compute Reward
            y_true = targets[:, 0]        # forward_returns[T]
            vol = targets[:, 1]           # volatility
            y_hat = targets[:, 3]         # prediction (for direction)
            
            rewards = compute_reward(
                position=actions,
                prev_position=prev_position,
                returns=y_true,
                volatility=vol,
                y_hat=y_hat,
                lambda_risk=0.05,  # Reduce risk penalty to encourage risk-taking
                lambda_turnover=0.005,  # Reduce turnover penalty for flexible trading
                lambda_direction=0.005,  # Restore small Direction Bonus as auxiliary signal
                transaction_cost=0.001
            ) # [B]
            
            # 4. Save current actions as next batch's prev_position
            prev_positions_batch = actions.detach().clone()
            
            # Debug: Collect stats (first batch of first epoch)
            if epoch == 0 and batch_idx == 0:
                all_actions.append(actions.detach().cpu().numpy())
                all_rewards.append(rewards.detach().cpu().numpy())
                
                # Detailed breakdown
                pnl_sample = (actions * y_true * 10.0).detach().cpu().numpy()  # Consistent with training (10x amp)
                risk_penalty_sample = (0.05 * (actions * vol) ** 2).detach().cpu().numpy()
                turnover_penalty_sample = (0.005 * torch.abs(actions - prev_position) * 0.001).detach().cpu().numpy()
                direction_bonus_sample = (0.005 * (
                    ((torch.sign(y_hat) > 0) & (torch.sign(y_true) > 0)) |
                    ((torch.sign(y_hat) < 0) & (torch.sign(y_true) < 0)) |
                    ((torch.sign(y_hat) == 0) & (torch.sign(y_true) == 0))
                ).float()).detach().cpu().numpy()
                
                print(f"\n  Detailed Reward Breakdown (first batch):")
                print(f"    PnL: mean={pnl_sample.mean():.6f}, std={pnl_sample.std():.6f}, range=[{pnl_sample.min():.6f}, {pnl_sample.max():.6f}]")
                print(f"    Risk Penalty: mean={risk_penalty_sample.mean():.6f}, std={risk_penalty_sample.std():.6f}")
                print(f"    Turnover Penalty: mean={turnover_penalty_sample.mean():.6f}, std={turnover_penalty_sample.std():.6f}")
                print(f"    Direction Bonus: mean={direction_bonus_sample.mean():.6f}, std={direction_bonus_sample.std():.6f}")
                print(f"    Returns (y_true): mean={y_true.mean():.6f}, std={y_true.std():.6f}, range=[{y_true.min():.6f}, {y_true.max():.6f}]")
                print(f"    Volatility: mean={vol.mean():.6f}, std={vol.std():.6f}, range=[{vol.min():.6f}, {vol.max():.6f}]")
                print(f"    y_hat: mean={y_hat.mean():.6f}, std={y_hat.std():.6f}, range=[{y_hat.min():.6f}, {y_hat.max():.6f}]")
            
            # 3. Compute Loss (Policy Gradient / REINFORCE)
            # Loss = - log_prob(a) * (R - baseline)
            # Use batch mean as baseline for simplicity
            baseline = rewards.mean()
            adv = rewards - baseline
            
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 4. Update
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
        avg_rew = total_reward / len(dataloader)
        avg_loss_train = total_loss / len(dataloader)  # Average training loss
        current_lr = optimizer.param_groups[0]['lr']
        
        # [Evaluation Mode]: Recalculate loss on all data after each epoch (without update)
        # This provides a more accurate metric for evaluation
        policy.eval()
        eval_total_loss = 0.0
        eval_total_reward = 0.0
        with torch.no_grad():
            for states, targets in dataloader:
                # Sample actions (Deterministic for evaluation)
                actions = policy.get_action(states, deterministic=True)
                
                # Get prev_position
                prev_position = targets[:, 2]
                
                # Compute Reward
                y_true = targets[:, 0]
                vol = targets[:, 1]
                y_hat = targets[:, 3]
                
                rewards = compute_reward(
                    position=actions,
                    prev_position=prev_position,
                    returns=y_true,
                    volatility=vol,
                    y_hat=y_hat,
                    lambda_risk=0.05,
                    lambda_turnover=0.005,
                    lambda_direction=0.005,
                    transaction_cost=0.001
                )
                
                # Compute Loss (for evaluation)
                baseline = rewards.mean()
                adv = rewards - baseline
                log_probs = policy.get_log_prob(states, actions)
                loss = -(log_probs * adv).mean()
                
                eval_total_loss += loss.item()
                eval_total_reward += rewards.mean().item()
        
        policy.train()  # Restore training mode
        
        # Evaluation metrics
        eval_avg_loss = eval_total_loss / len(dataloader)
        eval_avg_reward = eval_total_reward / len(dataloader)
        
        # Save Best Model (Based on Eval Loss, lower is better)
        if eval_avg_loss < best_loss:
            best_loss = eval_avg_loss
            torch.save(policy.state_dict(), best_model_path)
            is_best = "★"
        else:
            is_best = ""
        
        # Debug Info (First epoch)
        if epoch == 0 and len(all_actions) > 0:
            actions_sample = np.concatenate(all_actions)
            rewards_sample = np.concatenate(all_rewards)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Reward: {avg_rew:.4f} | Train Loss: {avg_loss_train:.4f} | Eval Reward: {eval_avg_reward:.4f} | Eval Loss: {eval_avg_loss:.4f} | LR: {current_lr:.6f} {is_best}")
            print(f"  Debug - Actions: mean={actions_sample.mean():.4f}, std={actions_sample.std():.4f}, range=[{actions_sample.min():.4f}, {actions_sample.max():.4f}]")
            print(f"  Debug - Rewards: mean={rewards_sample.mean():.4f}, std={rewards_sample.std():.4f}, range=[{rewards_sample.min():.4f}, {rewards_sample.max():.4f}]")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Reward: {avg_rew:.4f} | Train Loss: {avg_loss_train:.4f} | Eval Reward: {eval_avg_reward:.4f} | Eval Loss: {eval_avg_loss:.4f} | LR: {current_lr:.6f} {is_best}")
        
        # Update LR
        scheduler.step()
        
    # Save Final Model
    save_path = Path(artifacts.models_dir) / "policy_head.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"\nFinal Policy Head saved to {save_path}")
    print(f"Best Policy Head (Loss: {best_loss:.4f}) saved to {best_model_path}")

if __name__ == "__main__":
    train_grpo()
