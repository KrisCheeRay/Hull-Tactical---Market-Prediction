# ... (existing content) ...

## 端到端流程与文件职责（SFT → Ensemble → 推理 → 预留 GRPO）

下面按执行顺序、不重不漏地说明各模块的职责与产物，便于团队协作与复现。

### 0) 环境与依赖
- 建议使用我们已提供的 `environment.yml` 创建 conda 环境；或参考 `requirements.txt`。

### 1) 数据格式与特征一致性（FeatureStore）
- 目标：把原始数据整理为 NeuralForecast 所需的 long format，并固化“列顺序/标准化器/缺失处理”。
- 相关文件：
  - `src/configs.py`
    - `DataSchema`：数据列名规范（`unique_id`, `ds`, `y` 与特征列）。
    - `SFTConfig`：SFT 超参（window、patchTST 超参、训练参数等）。
    - `ArtifactPaths`：模型与配置产物路径（统一在 `models/`）。
  - `src/feature_store.py`
    - `FeatureStore.fit_transform(df)`：训练期在训练集上 `fit` 标准化器并变换；保存 `scaler.pkl`、`features.json`。
    - `FeatureStore.transform(df)`：推理期复用同一套标准化器与列顺序，保证训练/推理一致。
  - 产物（自动写入 `models/`）：
    - `models/scaler.pkl`：`StandardScaler`。
    - `models/features.json`：列顺序与 schema 记录。

数据工程交付：把原始 CSV 整理成包含至少三列的 long format：
- `unique_id`：序列 ID（单序列可固定为 `"series_0"`）。
- `ds`：时间索引（日期或有序整数，训练/推理保持一致）。
- `y`：目标（SFT 的 gt，建议为“下一步超额收益”）。
- 其他列为特征（可时变/静态）。

### 2) SFT 训练与信号生成（"烂苹果"生成）
- **目标**：
  1. 训练 PatchTST/NHITS 模型（用于最终推理）。
  2. **生成全量 OOS (Out-of-Sample) 预测信号**：通过 K-Fold 交叉预测，为 GRPO 阶段提供“带真实噪声”的输入信号（y_hat）。
- **脚本**：
  - `kaggle/work/sft_patchTST.py`：全量训练 PatchTST。
  - `kaggle/work/sft_nhits.py`：全量训练 NHITS。
  - **新增 `kaggle/work/gen_sft_signals.py`**：执行 K-Fold，拼接生成 `sft_y_hat_oos.parquet`。
- **产物**：
  - 模型权重：`models/patchtst_v1.pt` 等。
  - **GRPO 训练数据**：`models/sft_y_hat_oos.parquet` (包含 unique_id, ds, y_hat, y_gt)。

### 3) GRPO 策略训练（"吃烂苹果"）
- **目标**：训练一个轻量 Policy Head (MLP)，输入 SFT 信号 + 历史统计量，输出仓位 [0, 2]。
- **核心逻辑**：使用 OOS 预测值 (y_hat) 训练，迫使策略学会适应预测误差（Train as you serve）。
- **脚本**：
  - `src/policy_head.py`：定义 MLP 结构。
  - `kaggle/work/train_grpo.py`：加载 `sft_y_hat_oos.parquet`，构造 State，训练 MLP。
- **产物**：
  - `models/policy_head.pt`

### 4) 推理侧装载与组合（满足 Kaggle 网关）
- 目标：一次性加载所有产物，批处理内快速返回预测（首批 ≤15min，之后每批 ≤5min）。
- 文件：`src/predict_runtime.py`
  - `SFTPredictor.load()`：加载所有权重。
  - `SFTPredictor.predict_next(batch_df)`：
    1. SFT 前向 -> 得到 y_hat。
    2. 实时计算历史统计量 (Vol, Trend)。
    3. 拼接 State -> Policy Head -> 仓位。
- 线上流程（与 `demo.py` 配合）：
  1. 首次调用前或第一次 `predict` 内部：完成一次性加载（≤15min）。
  2. 每批数据到达：轻量特征处理 → 维护长度 `input_size` 的历史窗口 → 两模型前向 → `combine()` 加权 → 输出预测。
  3. 推荐把最终分数单调映射到 [0,2] 仓位（例如 `2*sigmoid(k*y_hat+b)` 或线性+截断）。

### 5) 打包为 Kaggle Dataset（推理使用）
- 目标：将训练产物以 **Private** Dataset 的形式上传，供提交 Notebook 使用（若仅用官方数据训练，可保持私有）。
- 打包内容（建议）：
  - `models/patchtst_v1.pt`、`patchtst_config.json`
  - `models/nhits_v1.pt`、`nhits_config.json`（若使用）
  - `models/policy_head.pt`
  - `models/scaler.pkl`、`features.json`
  - `models/ensemble.json`
- 提交 Notebook 顶部添加该 Dataset，并复用本仓库的推理代码。

---
