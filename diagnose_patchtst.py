
import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import torch
from neuralforecast import NeuralForecast

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore

def diagnose_models():
    print("=== Starting Deep Diagnosis for PatchTST vs N-HiTS ===")
    
    # 1. Load Data & Feature Store
    schema = DataSchema()
    artifacts = ArtifactPaths()
    
    print("Loading Train Context (Last 200 days)...")
    train_df = pl.read_csv("kaggle/input/hull-tactical-market-prediction/train_feature_selected.csv")
    
    # Shift y logic
    if "forward_returns" in train_df.columns:
        train_df = train_df.with_columns(
            pl.col("forward_returns").shift(1).alias("y")
        )
    train_df = train_df.rename({"date_id": "ds"})
    feature_cols = schema.feature_cols
    train_df = train_df.with_columns([pl.col(c).fill_null(0.0) for c in feature_cols])
    train_df = train_df.filter(pl.col("y").is_not_null())
    
    # Take a longer context to see history
    context_df = train_df.tail(200).with_columns(pl.lit("series_0").alias("unique_id"))
    
    # Feature Store Transform
    fs = FeatureStore(schema=schema)
    fs.load(artifacts)
    context_pd = fs.transform(context_df).to_pandas()
    context_pd['ds'] = np.arange(len(context_pd)) # Reset ds for simplicity
    
    print(f"Context loaded. Y mean: {context_pd['y'].mean():.6f}, Y std: {context_pd['y'].std():.6f}")

    # 2. Load Models
    print("\nLoading Models...")
    nf_patch = NeuralForecast.load(path=artifacts.patchtst_save_dir)
    nf_nhits = NeuralForecast.load(path=artifacts.nhits_save_dir)
    
    # 3. Sensitivity Analysis (Perturbation Test)
    # 这里的目的是看模型对输入的反应是否合理
    print("\n--- Sensitivity Analysis (Predicting Next Step) ---")
    
    # Case A: Normal Input
    # 构造一个 dummy future (只需要1行)
    last_ds = context_pd['ds'].max()
    futr_df = pd.DataFrame({
        'unique_id': ['series_0'],
        'ds': [last_ds + 1]
    })
    # Fill features with 0 (mean value after scaling roughly)
    for col in feature_cols:
        futr_df[col] = 0.0
        
    pred_p_normal = nf_patch.predict(df=context_pd, futr_df=futr_df)['PatchTST'].values[0]
    pred_n_normal = nf_nhits.predict(df=context_pd, futr_df=futr_df)['NHITS'].values[0]
    
    print(f"Normal Prediction     | PatchTST: {pred_p_normal:.6f} | N-HiTS: {pred_n_normal:.6f}")

    # Case B: Zero History (All features = 0, y = 0)
    # 如果模型学到了 Bias，即使全是0它也会预测一个非零值
    zero_context = context_pd.copy()
    for col in feature_cols + ['y']:
        zero_context[col] = 0.0
        
    pred_p_zero = nf_patch.predict(df=zero_context, futr_df=futr_df)['PatchTST'].values[0]
    pred_n_zero = nf_nhits.predict(df=zero_context, futr_df=futr_df)['NHITS'].values[0]
    
    print(f"Zero Input Prediction | PatchTST: {pred_p_zero:.6f} | N-HiTS: {pred_n_zero:.6f}")
    
    # Case C: Positive Trend Injection
    # 强行把最后 10 天的 y 变成正的大值，看模型是否跟随
    pos_context = context_pd.copy()
    pos_context.loc[pos_context.index[-10:], 'y'] = 2.0 # 2 sigma
    
    pred_p_pos = nf_patch.predict(df=pos_context, futr_df=futr_df)['PatchTST'].values[0]
    pred_n_pos = nf_nhits.predict(df=pos_context, futr_df=futr_df)['NHITS'].values[0]
    
    print(f"Positive Trend Input  | PatchTST: {pred_p_pos:.6f} | N-HiTS: {pred_n_pos:.6f}")
    
    # Case D: Negative Trend Injection
    neg_context = context_pd.copy()
    neg_context.loc[neg_context.index[-10:], 'y'] = -2.0
    
    pred_p_neg = nf_patch.predict(df=neg_context, futr_df=futr_df)['PatchTST'].values[0]
    pred_n_neg = nf_nhits.predict(df=neg_context, futr_df=futr_df)['NHITS'].values[0]
    
    print(f"Negative Trend Input  | PatchTST: {pred_p_neg:.6f} | N-HiTS: {pred_n_neg:.6f}")
    
    print("\n--- Analysis ---")
    print(f"PatchTST Reactivity (Pos - Normal): {pred_p_pos - pred_p_normal:.6f}")
    print(f"N-HiTS Reactivity   (Pos - Normal): {pred_n_pos - pred_n_normal:.6f}")
    
    # 4. Distribution Check (on Train Context)
    # 我们看看 PatchTST 在训练集上的拟合情况 (In-Sample Check)
    # 虽然是 loaded model，我们可以用它预测过去的一段来看看
    print("\n--- In-Sample Fit Check (Last 10 steps of Context) ---")
    # 我们手动滚动预测 context 的最后 10 步
    # 注意：这其实是 test on train data，但能看出模型到底有没有学会哪怕一点点 pattern
    
    insample_res = []
    # 从倒数第10个开始
    start_idx = len(context_pd) - 10
    
    # 这里的逻辑是：用 t-input_size 到 t-1 预测 t
    # 简便起见，我们只做一步演示
    # 取 context 的一段切片
    slice_df = context_pd.iloc[start_idx-60:start_idx].reset_index(drop=True) # 假设 input_size=60
    slice_df['unique_id'] = 'series_0'
    slice_df['ds'] = np.arange(len(slice_df))
    
    target_row = context_pd.iloc[[start_idx]].copy()
    target_row['ds'] = len(slice_df) # Next step
    target_y = target_row['y'].values[0]
    
    p_is = nf_patch.predict(df=slice_df, futr_df=target_row)['PatchTST'].values[0]
    n_is = nf_nhits.predict(df=slice_df, futr_df=target_row)['NHITS'].values[0]
    
    print(f"Sample In-Sample Prediction:")
    print(f"True Y: {target_y:.6f}")
    print(f"PatchTST: {p_is:.6f} (Diff: {p_is - target_y:.6f})")
    print(f"N-HiTS  : {n_is:.6f} (Diff: {n_is - target_y:.6f})")

if __name__ == "__main__":
    diagnose_models()

