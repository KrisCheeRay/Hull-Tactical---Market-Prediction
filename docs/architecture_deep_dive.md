# End-to-End Architecture Deep Dive (架构深度剖析)

本文档旨在从代码实现层面，手把手剖析本项目的 **SFT (Supervised Fine-Tuning) + GRPO (Policy Optimization)** 双阶段架构。

---

## 0. 全局数据流 (The Big Picture)

整个流程分为三个阶段：

1.  **Phase 0: 数据准备 (Data Prep)**
    *   **输入**: 原始 CSV (Wide Format)。
    *   **动作**: 清洗、特征工程、标准化 (`FeatureStore`)。
    *   **产物**: 符合 `DataSchema` 的 Long Format 数据。
2.  **Phase 1: 信号生成 (Signal Generation)**
    *   **输入**: 处理后的训练数据。
    *   **动作**: `NeuralForecast` 交叉验证 (Rolling CV) + 集成 (Ensemble)。
    *   **产物**: `sft_y_hat_oos.parquet` (烂苹果数据)。
3.  **Phase 2: 策略训练 (Policy Training)**
    *   **输入**: `sft_y_hat_oos.parquet`。
    *   **动作**: 训练 Policy Head (MLP) 使用 GRPO。
    *   **产物**: `policy_head.pt`。
4.  **Phase 3: 在线推理 (Online Inference)**
    *   **输入**: 实时市场数据。
    *   **动作**: 特征变换 -> SFT 推理 -> Policy 决策。

---

## 0.5 数据接口与 Feature Store (`src/feature_store.py`)

在进入复杂的模型训练前，必须统一数据“语言”。

### 接口契约 (`src/configs.py`)
所有模型都认准以下列名：
*   `unique_id`: 序列标识 (Kaggle 比赛通常只有一个序列，如 'series_0')。
*   `ds`: 时间戳 (Date/Time Step)。
*   `y`: 预测目标 (Forward Returns)。
*   `features`: 其他所有列。

### FeatureStore 的作用
我们不能让训练和推理的数据分布不一致。`FeatureStore` 保证了这一点。

```python
# 训练时 (Fit & Transform)
fs = FeatureStore(schema)
train_scaled = fs.fit_transform(train_df) 
# -> 计算 mean/std，保存到 scaler.pkl，记录列顺序到 features.json

# 推理/验证时 (Transform Only)
fs.load(artifacts)
test_scaled = fs.transform(test_df)
# -> 使用之前保存的 mean/std 进行转换，并强制对齐列顺序
```

---

## 1. Phase 1: 烂苹果工厂 (`kaggle/work/gen_sft_signals.py`)

这个脚本的核心任务是**“诚实地模拟误差”**。我们使用 `NeuralForecast` 的 `cross_validation` 功能来实现高效的滚动预测。

### 核心代码剖析

```python
def generate_oos_signals(df, ...):
    # 1. 全局特征缩放
    # 注意：PatchTST 对缩放不敏感(RevIN)，但为了统一流程，我们先做一次 transform。
    fs = FeatureStore(schema)
    df_scaled = fs.fit_transform(df)
    
    # 2. 交叉验证 (Rolling Cross Validation)
    # 我们不进行重训 (refit=False)，让模型只学一次，然后一直预测未来。
    # 这样生成的预测值随着时间推移误差会变大（变得更“烂”），
    # 强迫 Policy Head 学会处理失效的信号。
    cv_df = nf.cross_validation(
        df=df_scaled.to_pandas(),
        n_windows=total_len - input_size, # 覆盖几乎所有历史
        step_size=1,                      # 逐日滚动
        refit=False                       # 关键点：不作弊，不重训
    )
    
    # 3. 动态集成 (Dynamic Ensemble)
    # 模拟推理时的行为：同时获取 PatchTST 和 NHITS 的预测，并根据波动率加权。
    # 这一步至关重要，确保 Policy Head 看到的输入分布与上线时一致。
    vol = compute_recent_vol(cv_df) # 计算环境波动率
    final_y_hat = combine_predictions(
        pred_patch=cv_df["PatchTST"],
        pred_nhits=cv_df.get("NHITS"),
        vol=vol,
        config=ensemble_cfg
    )
    
    # 4. 保存烂苹果
    # 只有这些带有真实误差的 y_hat，才是 GRPO 训练的合格饲料。
    result_df.write_parquet("models/sft_y_hat_oos.parquet")
```

**关键点：Ensemble Consistency**
*   **原则**: Train as you serve.
*   如果在 `predict_runtime.py` 里你要用 `0.6*A + 0.4*B`，那么在这里你生成训练数据时，也必须用同样的逻辑生成 `0.6*A + 0.4*B`。
*   绝不能把纯 PatchTST 的结果喂给 GRPO，上线却给它吃混合模型的结果。

---

## 2. Phase 2: 决策大脑 (`src/policy_head.py`)

这个文件定义了我们的决策智能体。它是一个简单的多层感知机 (MLP)，但输出层有讲究。

### 核心代码剖析

```python
class PolicyHead(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        # 输入层：为什么是 3？
        # 因为我们要喂给它：[SFT集成预测值, 历史波动率, 历史趋势]
        # 这三个数构成了它做决策的全部依据。
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # ... 中间层省略 ...
            nn.Linear(hidden_dim // 2, 2)  # 输出层：输出 2 个数
        )
```

#### 关键点：如何输出连续动作 [0, 2]？

我们不能直接输出一个标量，因为我们需要**概率分布**来支持强化学习的探索。我们选择 **Beta 分布**。

```python
    def forward(self, x):
        out = self.net(x)
        # Softplus 保证参数大于 0
        # +1.0 让 Beta 分布倾向于单峰，训练更稳定
        alpha = F.softplus(out[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 1]) + 1.0 + 1e-6
        return alpha, beta
```

#### 动作采样 (Training vs Inference)

```python
    def get_action(self, x, deterministic=False):
        alpha, beta = self.forward(x)
        
        if deterministic:
            # 推理模式 (上线时)：取均值
            # Beta 均值 = alpha / (alpha + beta) -> [0, 1]
            action = 2.0 * (alpha / (alpha + beta))
        else:
            # 训练模式 (GRPO 时)：采样
            # 即使均值是 1.0，也可能抽到 1.2 或 0.8 进行尝试
            dist = torch.distributions.Beta(alpha, beta)
            action = 2.0 * dist.sample()
            
        return action
```

---

## 3. Phase 3: 训练道场 (`kaggle/work/train_grpo.py`)

这是最神奇的地方：模型是如何在没有“正确答案”的情况下学会赚钱的。

### 数据集构建 (`GRPODataset`)

```python
    def __getitem__(self, idx):
        # State: 此时此刻模型能看到的信息 (y_hat, vol)
        state = self.data[idx, :state_dim]
        # Target: 此时此刻模型看不到，但稍后用来判卷的信息 (y_true)
        target_info = self.data[idx, target_idx]
        return state, target_info
```

### 训练循环 (The Loop)

```python
        for states, targets in dataloader:
            # 1. 决策 (Actor)
            # Policy Head 看着 state，心里没底地给出了 actions
            actions = policy.get_action(states, deterministic=False)
            
            # 2. 判卷 (Critic/Environment)
            # 市场根据 actions 和 y_true 计算 PnL (盈亏)
            rewards = compute_reward(actions, y_true, vol)
            
            # 3. 计算优势 (Advantage)
            # 比平均水平好多少？
            adv = rewards - rewards.mean()
            
            # 4. 策略梯度 (Policy Gradient)
            # Loss = - log_prob(action) * advantage
            # 赚了就增大这个动作的概率，亏了就减小
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 5. 修正大脑
            optimizer.step()
```

---

## 4. 总结：Sim-to-Real 的闭环

1.  **Phase 0**: 保证了训练和推理特征的一致性。
2.  **Phase 1**: 制造了带有真实误差和集成特性的信号，防止了“过拟合完美预测”。
3.  **Phase 2/3**: 直接优化 PnL，让模型学会了在预测不准时如何控制仓位（通常表现为低波动时敢上杠杆，预测模糊时降仓）。

这套架构在 Kaggle 金融赛题中具有很强的鲁棒性。
