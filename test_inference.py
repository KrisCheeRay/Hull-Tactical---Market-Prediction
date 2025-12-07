import sys
from pathlib import Path
import polars as pl
import pandas as pd
import torch
import numpy as np
from neuralforecast import NeuralForecast

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore

def run_test_inference():
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        print("Data root not found.")
        return

    # Load Configs
    schema = DataSchema()
    cfg = SFTConfig()
    artifacts = ArtifactPaths()

    # 1. Prepare Context (Last 60 days from Train)
    # This serves as the 'History' for the model
    print("Loading Train Context...")
    train_df = pl.read_csv(data_root / "train_feature_selected.csv")
    
    # Shift y logic for Train context
    if "forward_returns" in train_df.columns:
        train_df = train_df.with_columns(
            pl.col("forward_returns").shift(1).alias("y")
        )
    train_df = train_df.rename({"date_id": "ds"})
    
    # Drop nulls and fill features
    feature_cols = schema.feature_cols
    train_df = train_df.with_columns([
        pl.col(c).fill_null(0.0) for c in feature_cols
    ])
    train_df = train_df.filter(pl.col("y").is_not_null())
    
    # Context Window
    context_df = train_df.tail(cfg.input_size)
    context_df = context_df.with_columns(pl.lit("series_0").alias("unique_id"))
    
    print(f"Context shape: {context_df.shape}")

    # 2. Prepare Test Data (Future)
    print("Loading Test Data...")
    test_df = pl.read_csv(data_root / "test.csv")
    
    # Rename lagged_forward_returns to y (This is our Ground Truth for validation)
    if "lagged_forward_returns" in test_df.columns:
        test_df = test_df.rename({"lagged_forward_returns": "y"})
    
    test_df = test_df.rename({"date_id": "ds"})
    
    # Feature cleaning for Test
    test_df = test_df.with_columns([
        pl.col(c).fill_null(0.0) for c in feature_cols
    ])
    # We keep 'y' in test_df for MSE calculation later, but predict() might ignore it or use it as futr_exog?
    # NeuralForecast predict(futr_df) expects futr_df to have unique_id, ds, and Future Exog variables.
    # It does NOT expect 'y'.
    # So we separate Y for metric calculation.
    test_y_gt = test_df["y"].to_numpy()
    
    test_df = test_df.with_columns(pl.lit("series_0").alias("unique_id"))
    print(f"Test shape: {test_df.shape}")

    # 3. Feature Store Transform
    print("Applying Feature Store...")
    fs = FeatureStore(schema=schema)
    fs.load(artifacts) # Load saved scaler
    
    # Transform Context
    context_pd = fs.transform(context_df).to_pandas()
    # Transform Test (Future Input)
    test_pd = fs.transform(test_df).to_pandas()
    
    # Ensure types are correct (Schema Alignment)
    context_pd['ds'] = context_pd['ds'].astype(int)
    test_pd['ds'] = test_pd['ds'].astype(int)
    context_pd['unique_id'] = context_pd['unique_id'].astype(str)
    test_pd['unique_id'] = test_pd['unique_id'].astype(str)

    # CRITICAL FIX: NeuralForecast expects futr_df to immediately follow df in time.
    # Since test.csv might have arbitrary date_ids (e.g. 8980 overlapping with train),
    # we must re-index test_pd['ds'] to ensure continuity.
    last_train_ds = context_pd['ds'].max()
    new_test_ds = np.arange(last_train_ds + 1, last_train_ds + 1 + len(test_pd))
    print(f"Re-indexing test ds from {test_pd['ds'].min()}... to {new_test_ds[0]}...")
    test_pd['ds'] = new_test_ds

    # 4. Load Models (Zero-Shot) & Manual Rolling Loop
    print("Loading PatchTST from checkpoint...")
    nf_patch = NeuralForecast.load(path=artifacts.patchtst_save_dir)
    
    print("Loading N-HiTS from checkpoint...")
    nf_nhits = NeuralForecast.load(path=artifacts.nhits_save_dir)

    # ---------------------------------------------------------
    # Manual Rolling Forecast Loop
    # 模拟线上环境：一步步预测，一步步更新 Context
    # ---------------------------------------------------------
    
    # 修复: Align Columns (Schema Alignment)
    # train_df 可能包含 test_df 没有的列 (如 forward_returns), 反之亦然
    # concat 之前必须取交集，否则会出现 NaN 导致报错
    common_cols = context_pd.columns.intersection(test_pd.columns)
    print(f"Aligning columns... Kept {len(common_cols)} common columns.")
    
    # 确保关键列在 common_cols 里
    required_cols = ['unique_id', 'ds', 'y']
    for col in required_cols:
        if col not in common_cols:
            # 如果 y 不在 test_pd 里 (前面逻辑处理过，应该在)，这里会有问题
            # 我们假设前面的逻辑已经保证了 test_pd 有 'y'
            print(f"Warning: {col} not in common columns!")
            
    current_context = context_pd[common_cols].copy()
    test_pd_aligned = test_pd[common_cols].copy()
    
    # 确保 context 里有 y (因为 predict 需要完整的历史)
    if 'y' not in current_context.columns:
        current_context['y'] = context_df['y'].to_numpy()
        
    rolling_results = []
    
    print(f"\nStarting Manual Rolling Forecast for {len(test_pd_aligned)} steps...")
    
    for i in range(len(test_pd_aligned)):
        # 取出当前这一步的 "未来" 特征 (即测试集的第 i 行)
        # 注意：predict(futr_df) 需要 DataFrame 格式
        next_step_row = test_pd_aligned.iloc[[i]].copy()
        target_ds = next_step_row['ds'].values[0]
        
        # --- 1. 预测 ---
        # 使用当前 context 预测下一步
        # PatchTST
        fcst_p = nf_patch.predict(df=current_context, futr_df=next_step_row)
        y_hat_p = fcst_p['PatchTST'].values[0]
        
        # N-HiTS
        fcst_n = nf_nhits.predict(df=current_context, futr_df=next_step_row)
        y_hat_n = fcst_n['NHITS'].values[0]
        
        # --- 2. 记录结果 ---
        y_true = next_step_row['y'].values[0] if 'y' in next_step_row.columns else np.nan
        
        rolling_results.append({
            'ds': target_ds,
            'y_true': y_true,
            'PatchTST': y_hat_p,
            'NHITS': y_hat_n,
            'Ensemble': (y_hat_p + y_hat_n) / 2
        })
        
        # --- 3. 更新 Context (Rolling) ---
        # 把这一步的真实数据加入 context，作为下一步的历史
        # 必须包含 y (作为 autoregressive 输入) 和 exog
        
        # 构造要追加的行
        # 必须确保列名和 context 一致
        update_row = next_step_row.copy()
        # 确保 y 存在
        if 'y' not in update_row.columns:
             update_row['y'] = y_true
             
        # 追加到末尾
        current_context = pd.concat([current_context, update_row], ignore_index=True)
        
        # 保持窗口大小不变 (可选，但这能防止 context 无限膨胀变慢)
        # PatchTST input_size=60, N-HiTS input_size=60
        # 我们保持比如最近 100 天，足够覆盖 input_size
        if len(current_context) > 200:
            current_context = current_context.iloc[-200:].reset_index(drop=True)
            
        print(f"Step {i+1}/{len(test_pd_aligned)}: ds={target_ds}, True={y_true:.4f}, P={y_hat_p:.4f}, N={y_hat_n:.4f}")

    # ---------------------------------------------------------
    # 5. 评估
    # ---------------------------------------------------------
    results = pd.DataFrame(rolling_results)
    
    print("\n--- Rolling Forecast Results (Head) ---")
    print(results.head())
    
    # Calculate MSE
    mse_patch = ((results['y_true'] - results['PatchTST'])**2).mean()
    mse_nhits = ((results['y_true'] - results['NHITS'])**2).mean()
    mse_ens = ((results['y_true'] - results['Ensemble'])**2).mean()
    
    # Calculate IC (Information Coefficient - Pearson Correlation)
    # 如果样本太少 (如10个)，IC 波动极大，仅供参考
    ic_patch = results['y_true'].corr(results['PatchTST'])
    ic_nhits = results['y_true'].corr(results['NHITS'])
    ic_ens = results['y_true'].corr(results['Ensemble'])
    
    # Calculate Hit Rate (Directional Accuracy)
    # sign(0) = 0, 需要注意
    hit_patch = (np.sign(results['PatchTST']) == np.sign(results['y_true'])).mean()
    hit_nhits = (np.sign(results['NHITS']) == np.sign(results['y_true'])).mean()
    hit_ens = (np.sign(results['Ensemble']) == np.sign(results['y_true'])).mean()
    
    print(f"\n--- Metrics over {len(results)} Test Samples ---")
    print(f"{'Model':<10} | {'MSE':<12} | {'IC (Corr)':<10} | {'Hit Rate':<10}")
    print("-" * 50)
    print(f"{'PatchTST':<10} | {mse_patch:.8f}   | {ic_patch:.4f}     | {hit_patch:.2%}")
    print(f"{'NHITS':<10} | {mse_nhits:.8f}   | {ic_nhits:.4f}     | {hit_nhits:.2%}")
    print(f"{'Ensemble':<10} | {mse_ens:.8f}   | {ic_ens:.4f}     | {hit_ens:.2%}")
    
    # Save detailed results
    results.to_csv('test_inference_rolling_results.csv', index=False)
    print("\nDetailed results saved to 'test_inference_rolling_results.csv'")

if __name__ == "__main__":
    run_test_inference()
