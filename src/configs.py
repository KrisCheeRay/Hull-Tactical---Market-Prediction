from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataSchema:
    unique_id_col: str = "unique_id"
    timestamp_col: str = "ds"
    target_col: str = "y"
    feature_cols: Optional[List[str]] = None


@dataclass
class SFTConfig:
    # windowing
    input_size: int = 60
    horizon: int = 1
    # PatchTST model
    patch_len: int = 16
    d_model: int = 128  # Reduced from 256 to avoid overfitting on 9k samples
    n_heads: int = 4    # Reduced from 8
    n_layers: int = 3
    dropout: float = 0.1
    revin: bool = True  # Critical for handling 30-year distribution shifts
    # training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int = 10_000
    patience: int = 10
    freq: str = "D"  # daily by default


@dataclass
class ArtifactPaths:
    models_dir: str = "models"
    scaler_path: str = "models/scaler.pkl"
    features_path: str = "models/features.json"
    patchtst_weights: str = "models/patchtst_v1.pt"
    patchtst_config: str = "models/patchtst_config.json"
    nhits_weights: str = "models/nhits_v1.pt"
    nhits_config: str = "models/nhits_config.json"
    ensemble_config: str = "models/ensemble.json"


