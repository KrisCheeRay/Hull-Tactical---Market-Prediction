import json
from pathlib import Path
import polars as pl

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from src.configs import DataSchema, ArtifactPaths
from src.feature_store import FeatureStore


def build_model() -> NHITS:
    # 轻量稳定版 NHITS（可按需在此调整）
    return NHITS(
        input_size=60,
        h=1,
        n_blocks=1,
        mlp_units=[256, 256],
        batch_size=128,
        learning_rate=1e-3,
        max_steps=8000,
        patience=10,
    )


def train_nhits(train_df: pl.DataFrame, schema: DataSchema, artifacts: ArtifactPaths) -> None:
    fs = FeatureStore(schema=schema)
    train_df = fs.fit_transform(train_df)

    model = build_model()
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df.to_pandas())

    Path(artifacts.models_dir).mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(nf.models[0].state_dict(), artifacts.nhits_weights)
    with open(artifacts.nhits_config, "w", encoding="utf-8") as f:
        json.dump({"input_size": 60, "h": 1}, f, ensure_ascii=False, indent=2)
    fs.save(artifacts)


if __name__ == "__main__":
    data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
    train = pl.read_csv(data_root / "train.csv")
    train = (
        train.rename({"market_forward_excess_returns": "y"})
        .with_columns(
            pl.col("date_id").cast(pl.Int64).alias("ds"),
            pl.lit("series_0").alias("unique_id"),
        )
    )
    schema = DataSchema(unique_id_col="unique_id", timestamp_col="ds", target_col="y")
    artifacts = ArtifactPaths()
    train_nhits(train, schema, artifacts)


