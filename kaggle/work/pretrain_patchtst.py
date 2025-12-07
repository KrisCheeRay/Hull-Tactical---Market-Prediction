import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.configs import SFTConfig,DataSchema,ArtifactPaths
from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_long_format(df_wide: pl.DataFrame, schema: DataSchema) -> pl.DataFrame:
    # Expect df_wide already includes columns: unique_id, ds, y, and feature columns
    # If your raw csv是宽表，请在数据工程侧转换后再调用本函数
    required = {schema.unique_id_col, schema.timestamp_col, schema.target_col}
    missing = required - set(df_wide.columns)
    if missing:
        raise ValueError(f"Missing columns for long format: {missing}")
    return df_wide


# ---------------------------------------------------------------------
# ① 构建 PatchTST，并提取 patcher + encoder + dims
# ---------------------------------------------------------------------
def load_encoder_patcher(cfg: SFTConfig):
    """
    创建 PatchTST，然后提取：
    - patcher
    - encoder(backbone)
    - d_model
    - patch_dim = input_channels × patch_len
    """
    nf = NeuralForecast(models=[
        PatchTST(
            input_size=cfg.input_size,
            h=cfg.horizon,                  # 虽然 MAE 不用，但 NF 需要此参数
            patch_len=cfg.patch_len,
            stride=cfg.patch_len,
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
    ], freq='D')

    model = nf.models[0].model

    # patch_dim = C * patch_len
    patch_dim = cfg.input_channels * cfg.patch_len

    return (
        model.patcher,
        model.backbone,
        model.d_model,
        patch_dim,
    )


# ---------------------------------------------------------------------
# ② MAE 模型：patcher + encoder + reconstruction head
# ---------------------------------------------------------------------
class PatchTST_MAE(nn.Module):
    def __init__(self, patcher, encoder, d_model, patch_dim):
        super().__init__()
        self.patcher = patcher
        self.encoder = encoder

        # Linear(d_model → patch_dim)
        self.reconstruct = nn.Linear(d_model, patch_dim)

    def forward(self, x, mask):
        """
        x: (B, T, C)
        mask: (num_patches,) boolean
        """
        patches = self.patcher(x)                   # (B, num_patches, patch_dim)
        patches_masked = patches.clone()
        patches_masked[:, mask] = 0.0

        latent = self.encoder(patches_masked)       # (B, num_patches, d_model)

        # reconstruct only masked positions
        recon = self.reconstruct(latent[:, mask])   # (B, num_mask, patch_dim)

        return recon, patches[:, mask]


# ---------------------------------------------------------------------
# ③ mask 生成
# ---------------------------------------------------------------------
def random_mask(num_patches, ratio=0.30):
    num_mask = int(num_patches * ratio)
    mask_idx = np.random.choice(num_patches, num_mask, replace=False)

    mask = torch.zeros(num_patches, dtype=torch.bool)
    mask[mask_idx] = True
    return mask


# ---------------------------------------------------------------------
# ④ MAE 训练循环（支持多 mask per sample）
# ---------------------------------------------------------------------
def train_mae(mae_model, dataloader, epochs=10, lr=1e-4,
              mask_ratio=0.3, num_mask_versions=2):

    optimizer = torch.optim.Adam(mae_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    mae_model.train()

    for e in range(epochs):
        for x in dataloader:  # (B, T, C)
            x = x.to(device)

            # 发现patch数量
            with torch.no_grad():
                num_patches = mae_model.patcher(x).shape[1]

            # 多 mask（增强）
            loss_total = 0
            for _ in range(num_mask_versions):
                mask = random_mask(num_patches, mask_ratio).to(device)
                recon, target = mae_model(x, mask)
                loss_total += loss_fn(recon, target)

            loss = loss_total / num_mask_versions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {e}] MAE Loss = {loss.item():.6f}")




class PatchTST_MAE_Dataset(Dataset):
    """
    从长格式 df 构造 (T,C) 滑动窗口
    """

    def __init__(self, df: pl.DataFrame, schema: DataSchema, window: int):
        df = df.sort(schema.timestamp_col)

        # 自动选出特征列
        feature_cols = [
            c for c in df.columns
            if c not in (schema.unique_id_col, schema.timestamp_col, schema.target_col)
        ]
        self.feature_cols = feature_cols

        # (N, C)
        arr = df.select(feature_cols).to_numpy().astype(np.float32)
        self.arr = arr

        self.window = window

    def __len__(self):
        return len(self.arr) - self.window

    def __getitem__(self, idx):
        x = self.arr[idx : idx + self.window]  # (T, C)
        return torch.tensor(x, dtype=torch.float32)




# ---------------------------------------------------------------------
# ⑤ 主流程：加载 encoder → MAE → 保存权重
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = SFTConfig()

    patcher, encoder, d_model, patch_dim = load_encoder_patcher(cfg)

    mae = PatchTST_MAE(
        patcher=patcher,
        encoder=encoder,
        d_model=d_model,
        patch_dim=patch_dim
    ).to(device)

    data_root = Path("/kaggle/input/hull-tactical-market-prediction/")    
    train = pl.read_csv(data_root / "train_feature_selected.csv")
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
        feature_cols=['E2','M13','P8','P5','V9','S2','M12','S5','forward_returns'],  # 自动推断除保留列外的所有列为特征
    )
    cfg = SFTConfig()
    artifacts = ArtifactPaths()
    train_long = to_long_format(train, schema)
    dataset = PatchTST_MAE_Dataset(
        df=train_long,
        schema=schema,
        window=cfg.input_size  # 如 256
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)   

    train_mae(mae, loader, epochs=100)

    Path("models").mkdir(exist_ok=True)
    torch.save(mae.encoder.state_dict(), "models/patchtst_encoder_pretrained.pt")
    print("Saved pretrained encoder → models/patchtst_encoder_pretrained.pt")
