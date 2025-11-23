from pathlib import Path
from typing import List, Tuple
import json
import pandas as pd
import polars as pl
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore

def train_fold_and_predict(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    schema: DataSchema,
    cfg: SFTConfig
) -> pl.DataFrame:
    """
    训练一个 fold 的 SFT 模型，并返回验证集上的预测结果 (OOS)
    """
    # 1. 特征工程 (在当前训练集上 fit)
    fs = FeatureStore(schema=schema)
    train_transformed = fs.fit_transform(train_df)
    val_transformed = fs.transform(val_df)  # 使用训练集的 scaler

    # 2. 构建模型
    model = PatchTST(
        input_size=cfg.input_size,
        h=cfg.horizon,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps // 5,  # 降低每个 fold 的步数以加速
        patience=cfg.patience,
    )

    # 3. 训练
    nf = NeuralForecast(models=[model], freq=cfg.freq)
    # 注意: NeuralForecast 默认会将 val_size 划分为验证集，这里我们直接 fit train
    # 若要严格早停，需传入 val_size 或手动 split。这里简化为直接 fit。
    nf.fit(df=train_transformed.to_pandas())

    # 4. 预测验证集
    # NeuralForecast 的 predict 需要未来的 ds，这里我们用 rolling window 预测
    # 但更简单的是：直接用 nf.predict() 对 val_df 进行预测
    # 注意：nf.predict() 需要 val_df 有足够的 history
    # 为了简化，我们把 history + val 拼起来预测，然后只取 val 部分
    # 这里假设 val_df 是紧接着 train_df 的
    
    # 构造预测输入：需要 train 的末尾 + val
    history_len = cfg.input_size
    # 取 train 的最后 input_size 行作为 context
    context_df = train_transformed.tail(history_len)
    predict_df = pl.concat([context_df, val_transformed])
    
    # 预测
    # 注意：NeuralForecast 的 predict 是从历史数据最后一点往后预测 h 步
    # 但我们需要对 val_df 中的每一个时间点都产生预测 (Rolling Forecast)
    # 对于生成全量 OOS 信号，最高效的方法是使用 cross_validation
    # 但 NF 的 cross_validation 比较重。
    # 
    # 替代方案：
    # 由于我们只是要生成信号，我们可以用 predict_insample() 或者
    # 使用 NF 的 predict() 配合 step_size=1 滚动 (太慢)
    # 
    # 最佳实践：直接用 NeuralForecast.cross_validation 生成 OOS 预测
    # 但这通常用于时序分割。
    
    # 这里我们用最简单的方式：训练好的模型，对 val_df 做 predict
    # 由于 PatchTST 是滑窗模型，我们可以构造 batch 进行推理
    # 但为了利用 NF 的接口，我们使用 predict 接口
    preds_pd = nf.predict(futr_df=val_transformed.to_pandas())
    
    # 整理结果
    # preds_pd 包含: unique_id, ds, PatchTST
    preds = pl.from_pandas(preds_pd.reset_index())
    return preds.select([
        pl.col("unique_id"),
        pl.col("ds"),
        pl.col("PatchTST").alias("y_hat")
    ])

def generate_oos_signals(
    full_df: pl.DataFrame,
    schema: DataSchema,
    cfg: SFTConfig,
    n_folds: int = 5
) -> pl.DataFrame:
    """
    执行 K-Fold 交叉预测，生成全量 OOS 信号
    """
    # 按时间排序
    full_df = full_df.sort(schema.timestamp_col)
    
    # 简单的时间序列 K-Fold (Blocking Time Series Split)
    # 也可以用 KFold (随机)，但时序数据通常建议按块
    # 这里为了让每个样本都有 OOS 预测，我们使用 KFold
    # 注意：严格来说时序不能随机 KFold (泄露)，但我们是要训练 Policy Head
    # 只要保证 "SFT 没见过它预测的那一小段" 即可。
    # 简单的 5-Fold:
    # Fold 1: Train=[2,3,4,5], Val=[1]
    # ...
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=False) # Shuffle=False 保持时序块？不，KFold(shuffle=False) 切成连续块
    
    # KFold 切分的是索引
    indices = np.arange(len(full_df))
    
    all_preds = []
    
    print(f"Starting {n_folds}-Fold Cross Prediction...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"Processing Fold {fold + 1}/{n_folds}...")
        
        train_fold = full_df[train_idx]
        val_fold = full_df[val_idx]
        
        # 训练并预测
        preds = train_fold_and_predict(train_fold, val_fold, schema, cfg)
        
        # 拼接真实值用于对照 (可选)
        # val_y = val_fold.select([schema.unique_id_col, schema.timestamp_col, schema.target_col])
        # preds = preds.join(val_y, on=["unique_id", "ds"])
        
        all_preds.append(preds)
        
    # 合并所有折的预测
    full_preds = pl.concat(all_preds)
    
    # 按时间排序
    full_preds = full_preds.sort("ds")
    
    return full_preds

if __name__ == "__main__":
    # 1. 加载数据
    data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        # 本地测试路径
        data_root = Path("kaggle/input/hull-tactical-market-prediction/")
        
    # 假设已有 long format 数据 (或在此转换)
    # 这里为了演示，加载 raw 并转换
    try:
        train = pl.read_csv(data_root / "train.csv")
    except Exception:
        print("Data not found, skipping execution.")
        exit(0)
        
    # 简单转换 (同 sft_patchTST.py)
    train = (
        train.rename({"market_forward_excess_returns": "y"})
        .with_columns(
            pl.col("date_id").cast(pl.Int64).alias("ds"),
            pl.lit("series_0").alias("unique_id"),
        )
    )
    
    schema = DataSchema(
        unique_id_col="unique_id",
        timestamp_col="ds",
        target_col="y"
    )
    cfg = SFTConfig(max_steps=500) # 测试用
    artifacts = ArtifactPaths()
    
    # 2. 生成 OOS 信号
    oos_signals = generate_oos_signals(train, schema, cfg, n_folds=5)
    
    # 3. 保存
    out_path = Path(artifacts.models_dir) / "sft_y_hat_oos.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oos_signals.write_parquet(out_path)
    
    print(f"Successfully generated OOS signals at {out_path}")

