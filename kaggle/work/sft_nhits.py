import json
from pathlib import Path
import polars as pl
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore
from kaggle.work.sft_patchTST import to_long_format


def build_model(cfg: SFTConfig) -> NHITS:
    # NHITS 特有参数保持硬编码或扩展 Config，通用参数从 cfg 读取
    return NHITS(
        input_size=cfg.input_size,
        h=cfg.horizon,
        # NHITS specific defaults
        n_blocks=1,
        mlp_units=[256, 256],
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        patience=cfg.patience,
    )


def train_nhits(
    train_df: pl.DataFrame, 
    schema: DataSchema, 
    cfg: SFTConfig,
    artifacts: ArtifactPaths
) -> None:
    fs = FeatureStore(schema=schema)
    train_df = fs.fit_transform(train_df)

    model = build_model(cfg)
    nf = NeuralForecast(models=[model], freq=cfg.freq)
    nf.fit(df=train_df.to_pandas())

    Path(artifacts.models_dir).mkdir(parents=True, exist_ok=True)
    torch.save(nf.models[0].state_dict(), artifacts.nhits_weights)
    
    # 保存 Config 以便推理加载
    # 注意：这里应该保存构建 NHITS 所需的所有参数
    nhits_cfg = {
        "input_size": cfg.input_size,
        "h": cfg.horizon,
        "n_blocks": 1,
        "mlp_units": [256, 256],
        # ... other static params used in build_model
    }
    with open(artifacts.nhits_config, "w", encoding="utf-8") as f:
        json.dump(nhits_cfg, f, ensure_ascii=False, indent=2)
        
    fs.save(artifacts)


if __name__ == "__main__":
    data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
    train = pl.read_csv(data_root / "train.csv")
    
    # 示例预处理，实际应由数据工程完成
    if "date_id" in train.columns:
        train = (
            train.rename({"market_forward_excess_returns": "y"})
            .with_columns(
                pl.col("date_id").cast(pl.Int64).alias("ds"),
                pl.lit("series_0").alias("unique_id"),
            )
        )
    
    schema = DataSchema(unique_id_col="unique_id", timestamp_col="ds", target_col="y")
    cfg = SFTConfig()
    artifacts = ArtifactPaths()
    
    train_long = to_long_format(train, schema)
    train_nhits(train_long, schema, cfg, artifacts)


