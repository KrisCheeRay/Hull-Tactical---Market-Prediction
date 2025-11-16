## 项目概览

本仓库用于参加 Kaggle 比赛 `Hull-Tactical Market Prediction`。Notebook 在评测时会通过 `DefaultInferenceServer` 调用我们自定义的 `predict` 函数，对时间序列数据进行滚动推理。当前核心脚本是 `demo.py`，后续模型、特征与部署说明都会在此文档中同步。

---

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

### 2) SFT 训练（PatchTST 与 NHITS）
- 目标：用监督学习在训练集上拟合时间序列模型，输出可复现实验的权重与配置。
- 脚本：
  - `kaggle/work/sft_patchTST.py`：训练 PatchTST。
  - `kaggle/work/sft_nhits.py`：训练 NHITS（轻量、快速、与 PatchTST 互补）。
- 两个脚本都会：
  1. 调用 `FeatureStore.fit_transform()` 进行标准化。
  2. 使用 NeuralForecast `.fit()` 自动做滑窗/分 batch/早停。
  3. 保存权重与配置到 `models/`。
- 产物：
  - PatchTST：`models/patchtst_v1.pt`、`models/patchtst_config.json`
  - NHITS：`models/nhits_v1.pt`、`models/nhits_config.json`
  - 公共：`models/scaler.pkl`、`models/features.json`

推荐 PatchTST 默认超参（可在脚本内调整）：
- `input_size=60`, `h=1`, `patch_len=16`, `d_model=256`, `n_heads=8`, `n_layers=3`, `dropout=0.1`
- `batch_size=64`, `learning_rate=1e-3`, `weight_decay=1e-4`, `max_steps=10000`, `patience=10`

命令示例（本地或 Kaggle Notebook 中运行）：
```bash
python kaggle/work/sft_patchTST.py
python kaggle/work/sft_nhits.py
```

### 3) 加权集成（Ensemble）
- 目标：在验证集上为 PatchTST 与 NHITS 搜索一组稳健的加权，或定义制度切换（无标签）权重。
- 配置文件：`models/ensemble.json`
  - 固定加权（默认）：
    ```json
    { "type": "fixed", "weights": { "patchtst": 0.6, "nhits": 0.4 } }
    ```
  - 制度切换（可选）：基于近窗波动的阈值切换到不同权重（无需标签）。
    - `regime_vol.enabled=true`，设置 `vol_window` 与 `thresholds`，并提供 `low/mid/high` 三组权重。
- 后续可提供独立验证脚本，对验证切片计算 IC/Rank-IC 并网格搜索权重/阈值，然后更新该 JSON。

### 4) 推理侧装载与组合（满足 Kaggle 网关）
- 目标：一次性加载所有产物，批处理内快速返回预测（首批 ≤15min，之后每批 ≤5min）。
- 文件：`src/predict_runtime.py`
  - `SFTPredictor.load()`：加载 `scaler.pkl`、PatchTST/NHITS 权重与配置、`ensemble.json`。
  - `SFTPredictor.combine(p1, p2, recent_vol)`：按固定或制度切换权重合成最终预测。
  - `SFTPredictor.predict_next(batch_df)`：留口与网关对接（你在此拼接窗口并调用模型前向；保持与训练一致的列与标准化）。
- 线上流程（与 `demo.py` 配合）：
  1. 首次调用前或第一次 `predict` 内部：完成一次性加载（≤15min）。
  2. 每批数据到达：轻量特征处理 → 维护长度 `input_size` 的历史窗口 → 两模型前向 → `combine()` 加权 → 输出预测。
  3. 推荐把最终分数单调映射到 [0,2] 仓位（例如 `2*sigmoid(k*y_hat+b)` 或线性+截断）。

### 5) 打包为 Kaggle Dataset（推理使用）
- 目标：将训练产物以 **Private** Dataset 的形式上传，供提交 Notebook 使用（若仅用官方数据训练，可保持私有）。
- 打包内容（建议）：
  - `models/patchtst_v1.pt`、`patchtst_config.json`
  - `models/nhits_v1.pt`、`nhits_config.json`（若使用）
  - `models/scaler.pkl`、`features.json`
  - `models/ensemble.json`
- 提交 Notebook 顶部添加该 Dataset，并复用本仓库的推理代码。

### 6) 预留 GRPO 接口（第二阶段）
- 复用 `FeatureStore` 与 `SFTPredictor.load()` 的加载路径；把 SFT 的输出分数/embedding 与必要市场特征拼接后，接一个小 `policy/value` MLP：
  - policy 输出到 [0,2]（`2*sigmoid` 或 Beta 分布采样/期望×2）。
  - 先冻结 PatchTST/NHITS（backbone），只训 policy/value 头；稳定后小学习率解冻尾层微调。
  - 奖励示例：`reward = pnl − λ1*max_drawdown − λ2*turnover − λ3*volatility`（做标准化/裁剪）。

---

## 文件结构

- `demo.py`：Kaggle 官方提供的推理入口，需要补全 `predict` 函数。
- `Hull-Tactical---Market-Prediction/`：比赛原始数据与官方工具，其中：
  - `hull-tactical-market-prediction/train.csv`、`test.csv`：训练与评测数据。
  - `hull-tactical-market-prediction/kaggle_evaluation/`：`DefaultGateway`、`DefaultInferenceServer` 等评测相关代码。
  - `README.md`、`tecnic.md`：比赛说明与前期资料。
  
新增/重要目录：
- `src/configs.py`：数据 schema、SFT 超参、产物路径。
- `src/feature_store.py`：特征一致性与标准化导出/加载。
- `src/predict_runtime.py`：线上装载、加权逻辑与对接口。
- `kaggle/work/sft_patchTST.py`：PatchTST 训练入口。
- `kaggle/work/sft_nhits.py`：NHITS 训练入口。
- `models/`：统一存放权重与配置（`*.pt`、`*_config.json`、`scaler.pkl`、`features.json`、`ensemble.json`）。

## 快速开始

1. 安装依赖：
   - 必备：`polars`、`pandas`、`numpy`、`kaggle-evaluation`（Kaggle 环境自带）。
2. 本地调试：
   - 运行 `demo.py` 会启动 `inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))`，模拟官方评测流程。
3. 在线评测：
   - Kaggle 平台会设置环境变量 `KAGGLE_IS_COMPETITION_RERUN=1`，此时脚本会调用 `inference_server.serve()` 与官方评测网关通信。

## `predict` 函数要求

- 输入：`polars.DataFrame`，含当前时间步的特征。
- 输出：`float` 或 `DataFrame`（推荐返回 `polars.DataFrame`，列名遵循官方规范）。
- 时限：
  - 首个 batch 提供 15 分钟加载模型、构造状态。
  - 后续每个 batch 需在 5 分钟内返回，建议预加载模型和参数。
- 状态管理：如需跨时间步缓存，可使用模块级变量（例如 `global` 模型对象、滑动窗口）。
- 容错：使用 `try/except` 捕获异常并记录日志，以便定位问题。

## 推荐开发流程

1. **EDA（Exploratory Data Analysis）**：理解特征含义、缺失值、目标定义。
2. **特征工程**：滑动窗口、技术指标、宏观因子等，尽量保持计算高效。
3. **SFT 模型训练**：先训 PatchTST，再训 NHITS；导出权重与配置；验证集做加权搜索，更新 `models/ensemble.json`。
4. **推理脚本整合**：用 `SFTPredictor` 一次加载全部产物；`combine()` 完成加权；确保批次耗时达标。
5. **本地模拟**：使用 `DefaultGateway` 自测滚动推理。
6. **提交与回测**：冷启动 ≤15min，批次 ≤5min，总时长 ≤9h；离线回测关注 Sharpe/DD/turnover。

## 监控与调试建议

- 记录 `predict` 内关键耗时，确保满足 5 分钟限制。
- 输出预测分布统计（mean、std）以监控漂移。
- 若本地网关出现卡顿，检查是否有阻塞 I/O 或重复加载模型。

## 反思与后续改进

- 短期：补全 `predict`，实现最小可用模型，并添加日志记录。
- 中期：构建多窗口输入、模型集成，完善策略回测（Sharpe、Drawdown 等）。
- 长期：引入在线监控与自动化测试，确保 Notebook 版本可重复运行。
- 文档化：新增 `docs/kaggle_evaluation_notes.md` 归纳评测框架，有助于后续新人快速理解 Gateway/InferenceServer 交互。

---

## FAQ（与你的关心点对齐）
- SFT 的 gt 是什么？通常取“下一步超额收益”。也可以在有“老师仓位”时做位置监督，但默认用收益更稳。
- 为什么做小型 Ensemble？不同模型误差结构互补，固定/制度权重能显著提高稳健性且在线成本低。
- 与 GRPO 如何衔接？复用同一套加载与特征，给 SFT 输出接一个小 policy/value 头，输出到 [0,2]，先冻干线只训头，稳定后再小学习率微调。

---

## 复现与环境（Conda）

1) 导出当前环境（建议由维护者执行）
```bash
# 导出精确版本（无构建号），覆盖 environment.yml
conda env export --no-builds > environment.yml
```

2) 其他成员创建环境
```bash
conda env create -f environment.yml
conda activate hull_tactical
```

3) 验证关键依赖
```bash
python -c "import polars, neuralforecast, torch; print(polars.__version__, getattr(neuralforecast,'__version__','NA'), torch.__version__)"
```

> 说明：模型权重、scaler 等大文件不纳入 Git，由 Kaggle Private Dataset 提供；若需版本化大文件，使用 Git LFS。

---

## 把本地仓库连接到远程并推送

在项目根目录执行（将 `YOUR_REMOTE_URL.git` 替换为你的远程地址）：
```bash
git init
git add .
git commit -m "Init: SFT + Ensemble skeleton, docs and env"
git branch -M main
git remote add origin YOUR_REMOTE_URL.git
git push -u origin main
```

若远端已有历史、你想强制以本地为准：
```bash
git push -u origin main --force
```

常见问题：
- 权限报错：检查你是否有该远程仓库的写权限（HTTPS 推荐使用 Token，SSH 确保已配公钥）。
- 大文件被拒：把模型权重/缓存加入 `.gitignore`（见下），或启用 Git LFS。

---

## .gitignore 建议

```
# Python
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
*.pyc

# Env / IDE
.env
.venv/
.DS_Store
.vscode/
.idea/

# Models & Artifacts (改由 Kaggle Dataset/LFS 管理)
models/*.pt
models/*.pth
models/*.ckpt
models/*.npz
models/*.npy
models/*.onnx
models/*.bin
models/*.safetensors
models/*.h5
models/*.joblib
models/*.pkl
lightning_logs/
artifacts/

# Data cache
data/
*.parquet
*.feather
*.csv
```


