import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit

from src.configs import DataSchema, SFTConfig, ArtifactPaths
from src.feature_store import FeatureStore
from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
用于训练时微调
model.backbone.load_state_dict(torch.load("patchtst_encoder_pretrained.pt"))
nf.fit(train_df)
'''
# ---------------------------------------------------------------------
# ① 构建 PatchTST，并提取 patcher + encoder（加 patch_dim / d_model）
# ---------------------------------------------------------------------
def load_encoder_patcher(cfg: SFTConfig):
    nf = NeuralForecast(models=[
        PatchTST(
            input_size=cfg.input_size,
            h=cfg.horizon,
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

    return (
        model.patcher,
        model.backbone,
        model.d_model,
        model.patch_dim,
        model.patch_len
    )


# ---------------------------------------------------------------------
# ② MAE 模型
# ---------------------------------------------------------------------
class PatchTST_MAE(nn.Module):
    def __init__(self, patcher, encoder, d_model, patch_dim):
        super().__init__()
        self.patcher = patcher
        self.encoder = encoder
        self.reconstruct = nn.Linear(d_model, patch_dim)

    def forward(self, x, mask):
        patches = self.patcher(x)              # (B, num_patches, patch_dim)
        patches_masked = patches.clone()
        patches_masked[:, mask] = 0.0

        latent = self.encoder(patches_masked)  # (B, num_patches, d_model)
        recon = self.reconstruct(latent[:, mask])  # (B, num_mask, patch_dim)

        return recon, patches[:, mask]


# ---------------------------------------------------------------------
# ③ mask 生成
# ---------------------------------------------------------------------
def random_mask(num_patches, mask_ratio=0.30):
    num_mask = int(num_patches * mask_ratio)
    mask_idx = np.random.choice(num_patches, num_mask, replace=False)

    mask = torch.zeros(num_patches, dtype=torch.bool)
    mask[mask_idx] = True
    return mask


# ---------------------------------------------------------------------
# ④ MAE 训练循环
# ---------------------------------------------------------------------
def train_mae(mae_model, dataloader, epochs=10, lr=1e-4, mask_ratio=0.3):
    optimizer = torch.optim.Adam(mae_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    mae_model.train()

    for e in range(epochs):
        for x in dataloader:   # x: (B, T, C)
            x = x.to(device)

            num_patches = mae_model.patcher(x).shape[1]
            mask = random_mask(num_patches, mask_ratio).to(device)

            recon, target = mae_model(x, mask)
            loss = loss_fn(recon, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {e}] Loss = {loss.item():.6f}")


# ---------------------------------------------------------------------
# ⑤ 示例入口：加载模型 → 预训练 → 保存权重
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Load config
    cfg = SFTConfig()  # 默认值

    patcher, encoder, d_model, patch_dim, patch_len = load_encoder_patcher(cfg)

    mae = PatchTST_MAE(
        patcher=patcher,
        encoder=encoder,
        d_model=d_model,
        patch_dim=patch_dim
    ).to(device)

    #换成真实dataloader
    fake_data = torch.randn(32, cfg.input_size, cfg.input_channels)
    loader = DataLoader(fake_data, batch_size=cfg.batch_size, shuffle=True)

    train_mae(mae, loader, epochs=10)

    # 保存 MAE encoder 权重
    torch.save(mae.encoder.state_dict(), "models/patchtst_encoder_pretrained.pt")
    print("Saved pretrained encoder to models/patchtst_encoder_pretrained.pt")



