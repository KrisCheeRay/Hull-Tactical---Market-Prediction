import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import polars as pl
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_long_format(df: pl.DataFrame, schema: DataSchema) -> pl.DataFrame:
    """
    Convert wide format to long format expected by NeuralForecast.
    Actually, NeuralForecast expects: [unique_id, ds, y, feature_1, feature_2...]
    This function ensures columns are correct.
    """
    required_cols = [schema.unique_id_col, schema.timestamp_col, "y"]
    # Check if all feature cols exist
    missing_features = [c for c in schema.feature_cols if c not in df.columns]
    if missing_features:
        logger.warning(f"Missing feature columns: {missing_features}")
    
    # Ensure required columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df

def build_model(cfg: SFTConfig, schema: DataSchema) -> PatchTST:
    # PatchTST in NeuralForecast currently strictly follows Channel Independence 
    # and does not support 'hist_exog_list' (explicit channel mixing).
    # We use it in its native CI mode: weight sharing across variables, but independent inference.
    return PatchTST(
        h=cfg.horizon,
        input_size=cfg.input_size,
        # hist_exog_list=schema.feature_cols, # REMOVED: Not supported by PatchTST implementation
        patch_len=cfg.patch_len,
        stride=cfg.patch_len,
        hidden_size=cfg.d_model,

        n_heads=cfg.n_heads,
        encoder_layers=cfg.n_layers,
        dropout=cfg.dropout,
        revin=cfg.revin,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        random_seed=42,
        early_stop_patience_steps=cfg.patience,
    )

def train_sft(
    train_df: pl.DataFrame,
    schema: DataSchema,
    cfg: SFTConfig,
    artifacts: ArtifactPaths,
    pretrained_weights_path: Optional[str] = None
) -> None:
    """
    Train the SFT model (PatchTST) and save artifacts.
    
    Args:
        pretrained_weights_path: Path to pretrained encoder weights from MAE pretraining.
                                 If provided, loads encoder weights before fine-tuning.
    """
    logger.info("Starting SFT Training (PatchTST)...")
    
    # 1. Feature Store (Fit & Transform)
    logger.info("Fitting Feature Store...")
    fs = FeatureStore(schema=schema)
    train_scaled = fs.fit_transform(train_df)
    fs.save(artifacts)

    # 2. Initialize NeuralForecast Model
    logger.info("Initializing PatchTST...")
    model = build_model(cfg, schema)
    
    # 3. Load pretrained weights if provided
    if pretrained_weights_path and Path(pretrained_weights_path).exists():
        logger.info(f"Loading pretrained encoder weights from {pretrained_weights_path}...")
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        
        # Access backbone and encoder
        backbone = model.model
        encoder = backbone.backbone
        
        # Load encoder weights
        encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        logger.info("Pretrained encoder weights loaded successfully")
    elif pretrained_weights_path:
        logger.warning(f"Pretrained weights path provided but file not found: {pretrained_weights_path}")
    
    # 4. Train
    nf = NeuralForecast(models=[model], freq=cfg.freq)
    
    # NeuralForecast expects pandas DataFrame
    logger.info("Fitting model...")
    nf.fit(df=train_scaled.to_pandas())
    
    # 5. Save Weights & Config (Using NeuralForecast native save)
    logger.info(f"Saving model artifacts to {artifacts.patchtst_save_dir}...")
    # nf.save() saves the entire NeuralForecast object including model weights and config
    nf.save(path=artifacts.patchtst_save_dir, overwrite=True)
        
    logger.info("SFT Training Completed.")

if __name__ == "__main__":
    # Example usage with local csv
    data_root = Path("kaggle/input/hull-tactical-market-prediction/")
    if not data_root.exists():
        data_root = Path("/kaggle/input/hull-tactical-market-prediction/")
        
    train_path = data_root / "train_feature_selected.csv"
    
    if train_path.exists():
        print(f"Loading data from {train_path}...")
        train = pl.read_csv(train_path)
        
        # Rename cols for NeuralForecast
        # CRITICAL FIX: Shift forward_returns by 1 to align with inference logic.
        # Train: y_t = forward_returns_{t-1}
        # Inference: y_t = lagged_forward_returns (which is forward_returns_{t-1})
        # This ensures we predict Day T's return using information available up to Day T.
        if "forward_returns" in train.columns:
            train = train.with_columns(
                pl.col("forward_returns").shift(1).alias("y")
            )
        
        if "date_id" in train.columns:
            train = train.rename({"date_id": "ds"})
            
        # DROP NULLS for features AND the newly shifted y
        # Based on user input, first ~1000 rows have empty features.
        feature_cols = ["E2", "M13", "P8", "P5", "V9", "S2", "M12", "S5"]
        original_len = len(train)
        
        # Data Cleaning Strategy:
        # 1. Count nulls in feature columns
        null_counts = train.select(
            pl.sum_horizontal([pl.col(c).is_null() for c in feature_cols]).alias("null_count")
        )
        train = train.with_columns(null_counts)
        
        # 2. Filter: Keep rows with <= 4 nulls (drops first ~1000 sparse rows)
        # Also drop rows where 'y' is null (due to shift or original missing)
        train = train.filter(
            (pl.col("null_count") <= 4) & (pl.col("y").is_not_null())
        )
        
        # 3. Fill remaining nulls with 0.0
        train = train.with_columns([
            pl.col(c).fill_null(0.0) for c in feature_cols
        ])
        train = train.drop("null_count") # cleanup
        
        print(f"Data Cleaning: Dropped {original_len - len(train)} rows. Remaining: {len(train)}")

        # Add unique_id
        train = train.with_columns(
            pl.lit("series_0").alias("unique_id"),
            pl.col("ds").cast(pl.Int64)
        )
        
        # Configs
        schema = DataSchema()
        cfg = SFTConfig()
        artifacts = ArtifactPaths()
        
        train_long = to_long_format(train, schema)
        train_sft(train_long, schema, cfg, artifacts)
    else:
        print("Train file not found, skipping local test.")
