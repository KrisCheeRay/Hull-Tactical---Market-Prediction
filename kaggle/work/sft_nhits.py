import json
import sys
import logging
from pathlib import Path
import polars as pl
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore
from kaggle.work.sft_patchTST import to_long_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_model(cfg: SFTConfig, schema: DataSchema) -> NHITS:
    # NHITS 特有参数保持硬编码或扩展 Config，通用参数从 cfg 读取
    return NHITS(
        input_size=cfg.input_size,
        h=cfg.horizon,
        hist_exog_list=schema.feature_cols, # CRITICAL: Use features as historical exogenous variables
        # NHITS specific defaults
        stack_types=['identity', 'identity', 'identity'],
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        early_stop_patience_steps=cfg.patience,
    )


def train_nhits(
    train_df: pl.DataFrame, 
    schema: DataSchema, 
    cfg: SFTConfig, 
    artifacts: ArtifactPaths
) -> None:
    
    fs = FeatureStore(schema=schema)
    train_df = fs.fit_transform(train_df)

    model = build_model(cfg, schema)
    nf = NeuralForecast(models=[model], freq=cfg.freq)
    nf.fit(df=train_df.to_pandas())

    # 4. Save Weights & Config (Using NeuralForecast native save)
    logger.info(f"Saving model artifacts to {artifacts.nhits_save_dir}...")
    # nf.save() saves the entire NeuralForecast object including model weights and config
    nf.save(path=artifacts.nhits_save_dir, overwrite=True)
        
    logger.info("SFT Training Completed.")


if __name__ == "__main__":
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
        
    train_path = data_root / "train_feature_selected.csv"
    
    if train_path.exists():
        print(f"Loading data from {train_path}...")
        train = pl.read_csv(train_path)
        
        # CRITICAL FIX: Shift forward_returns by 1
        if "forward_returns" in train.columns:
            train = train.with_columns(
                pl.col("forward_returns").shift(1).alias("y")
            )
        if "date_id" in train.columns:
            train = train.rename({"date_id": "ds"})
            
        # DROP NULLS
        feature_cols = ["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]
        original_len = len(train)
        
        # Data Cleaning Strategy:
        # 1. Count nulls in feature columns
        null_counts = train.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        train = train.with_columns(null_counts)
        
        # 2. Filter: Keep rows with <= 4 nulls AND valid y
        train = train.filter(
            (pl.col("null_count") <= 4) & (pl.col("y").is_not_null())
        )
        
        # 3. Fill remaining nulls with 0.0
        train = train.with_columns([
            pl.col(c).fill_null(0.0) for c in feature_cols
        ])
        train = train.drop("null_count") # cleanup
        
        print(f"Data Cleaning: Dropped {original_len - len(train)} rows. Remaining: {len(train)}")

        train = train.with_columns(
            pl.lit("series_0").alias("unique_id"),
            pl.col("ds").cast(pl.Int64)
        )
    
        schema = DataSchema()
        cfg = SFTConfig()
        artifacts = ArtifactPaths()
        
        train_long = to_long_format(train, schema)
        train_nhits(train_long, schema, cfg, artifacts)
