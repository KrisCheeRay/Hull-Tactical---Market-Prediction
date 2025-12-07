from collections import deque
import json
from pathlib import Path
from typing import Deque, Optional, Dict

import polars as pl
import torch
from neuralforecast.models import PatchTST
from neuralforecast.models import NHITS

from .configs import DataSchema, SFTConfig, ArtifactPaths
from .feature_store import FeatureStore


class SFTPredictor:
    def __init__(self, schema: DataSchema, cfg: SFTConfig, artifacts: ArtifactPaths) -> None:
        self.schema = schema
        self.cfg = cfg
        self.artifacts = artifacts
        self._history: Deque[pl.DataFrame] = deque(maxlen=cfg.input_size)
        self._fs = FeatureStore(schema=schema)
        self._patch_model: Optional[PatchTST] = None
        self._nhits_model: Optional[NHITS] = None
        self._ready: bool = False
        self._ensemble: Dict = {}

    def load(self) -> None:
        self._fs.load(self.artifacts)
        with open(self.artifacts.patchtst_config, "r", encoding="utf-8") as f:
            patch_cfg = json.load(f)
        self._patch_model = PatchTST(**patch_cfg)
        patch_state = torch.load(self.artifacts.patchtst_weights, map_location="cpu")
        self._patch_model.load_state_dict(patch_state)
        self._patch_model.eval()
        # NHITS 可选加载（若存在）
        if Path(self.artifacts.nhits_weights).exists():
            try:
                with open(self.artifacts.nhits_config, "r", encoding="utf-8") as f:
                    nhits_cfg = json.load(f)
                self._nhits_model = NHITS(**nhits_cfg)
                nhits_state = torch.load(self.artifacts.nhits_weights, map_location="cpu")
                self._nhits_model.load_state_dict(nhits_state)
                self._nhits_model.eval()
            except Exception:
                self._nhits_model = None
        # ensemble
        if Path(self.artifacts.ensemble_config).exists():
            with open(self.artifacts.ensemble_config, "r", encoding="utf-8") as f:
                self._ensemble = json.load(f)
        self._ready = True

    def predict_next(self, batch_df: pl.DataFrame) -> float:
        if not self._ready:
            self.load()
        # 轻量处理与标准化
        batch_df = self._fs.transform(batch_df)   # 标准化这个对应的
        preds_patch = self._patch_model(batch_df)
        preds_nhits = self._nhits_model(batch_df)
        y_hat = 0.5*preds_nhits + 0.5*preds_patch
        return y_hat
        # 维护窗口（此处留作集成：在外部构建 [B,T,C]，或由库内部滑窗）
        # 简化：直接调用库期望的 DataFrame 输入预测接口（如需要）
        # 由于 online 预测通常需自定义 reshape，本类主要负责加载与标准化，具体拼接由调用方实现
        #raise NotImplementedError("Integrate with your gateway batching to produce [B,T,C] for model forward.")

    def combine(self, pred_patch: float, pred_nhits: Optional[float], recent_vol: Optional[float] = None) -> float:
        # 固定权重
        if not self._ensemble or self._ensemble.get("type", "fixed") == "fixed":
            w = self._ensemble.get("weights", {"patchtst": 1.0, "nhits": 0.0})
            total = w.get("patchtst", 1.0) * pred_patch + w.get("nhits", 0.0) * (pred_nhits or 0.0)
            return float(total)
        # 制度切换（无标签的阈值选择）
        regime = self._ensemble.get("regime_vol", {})
        if regime.get("enabled", False) and recent_vol is not None:
            t = regime.get("thresholds", [])[0:2]
            if len(t) != 2:
                return float(pred_patch)
            if recent_vol < t[0]:
                w = regime["weights"]["low"]
            elif recent_vol < t[1]:
                w = regime["weights"]["mid"]
            else:
                w = regime["weights"]["high"]
            total = w.get("patchtst", 1.0) * pred_patch + w.get("nhits", 0.0) * (pred_nhits or 0.0)
            return float(total)
        return float(pred_patch)


