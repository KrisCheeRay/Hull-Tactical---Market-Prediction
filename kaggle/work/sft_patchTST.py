import json
from pathlib import Path
from typing import Optional

import polars as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore


def to_long_format(df_wide: pl.DataFrame, schema: DataSchema) -> pl.DataFrame:
    # Expect df_wide already includes columns: unique_id, ds, y, and feature columns
    # If your raw csv是宽表，请在数据工程侧转换后再调用本函数
    required = {schema.unique_id_col, schema.timestamp_col, schema.target_col}
    missing = required - set(df_wide.columns)
    if missing:
        raise ValueError(f"Missing columns for long format: {missing}")
    return df_wide


def build_model(cfg: SFTConfig) -> PatchTST:
    return PatchTST(
        input_size=cfg.input_size,
        h=cfg.horizon,
        patch_len=cfg.patch_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        revin=cfg.revin,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps,
        patience=cfg.patience,
    )


def train_sft(
    train_df: pl.DataFrame,
    schema: DataSchema,
    cfg: SFTConfig,
    artifacts: ArtifactPaths,
) -> None:
    fs = FeatureStore(schema=schema)
    train_df = fs.fit_transform(train_df)

    model = build_model(cfg)
    nf = NeuralForecast(models=[model], freq=cfg.freq)
    nf.fit(df=train_df.to_pandas())

    Path(artifacts.models_dir).mkdir(parents=True, exist_ok=True)
    # 保存权重
    import torch
    torch.save(nf.models[0].state_dict(), artifacts.patchtst_weights)
    # 保存配置与特征
    with open(artifacts.patchtst_config, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
    fs.save(artifacts)


if __name__ == "__main__":
    # 示例：从 Kaggle 官方数据读取并重命名为 long format
    data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
    train = pl.read_csv(data_root / "train.csv")
    # 将官方列名映射为 schema 要求；这里仅示例，真实映射请按数据工程成果替换
    # 假设 'date_id' 可映射成时间索引，这里演示生成 'ds'（需在数据工程侧提供真实日期）
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
        target_col="y",
        feature_cols=None,  # 自动推断除保留列外的所有列为特征
    )
    cfg = SFTConfig()
    artifacts = ArtifactPaths()

    train_long = to_long_format(train, schema)
    train_sft(train_long, schema, cfg, artifacts)


