# End-to-End Architecture Deep Dive (架构深度剖析)

本文档旨在从代码实现层面，手把手剖析本项目的 **SFT (Supervised Fine-Tuning) + GRPO (Policy Optimization)** 双阶段架构。我们将不跳跃地解释每一行核心代码背后的原理与工程考量。

---

## 0. 全局数据流 (The Big Picture)

我们的目标是训练一个能根据“不完美的预测”做出“最优仓位决策”的智能体。整个流程分为三个阶段：

1.  **Phase 1: 信号生成 (Signal Generation)**
    *   **输入**: 原始训练数据 (Train CSV)。
    *   **动作**: 使用 PatchTST 模型进行 K-Fold 交叉预测。
    *   **产物**: `sft_y_hat_oos.parquet` (包含全量数据的“模拟实盘预测值”，即我们所谓的“烂苹果”)。
    *   **目的**: 制造出真实的预测误差，供策略头学习如何应对。

2.  **Phase 2: 策略训练 (Policy Training)**
    *   **输入**: `sft_y_hat_oos.parquet` + 原始价格数据。
    *   **动作**: 训练 Policy Head (MLP) 使用 Policy Gradient (GRPO)。
    *   **产物**: `policy_head.pt` (学会了何时该信 SFT，何时该怂的策略网络)。

3.  **Phase 3: 在线推理 (Online Inference)**
    *   **输入**: 实时市场数据。
    *   **动作**: SFT 预测 -> 拼接特征 -> Policy Head 决策。
    *   **产物**: 最终仓位 [0, 2]。

---

## 1. Phase 1: 烂苹果工厂 (`kaggle/work/gen_sft_signals.py`)

这个脚本的核心任务是**“诚实地模拟误差”**。

### 核心代码剖析

```python
def generate_oos_signals(full_df, ...):
    # 1. 准备 K-Fold 切分器
    # KFold(shuffle=False) 意味着我们按顺序切分数据，不打乱时间顺序。
    # 虽然这在时序上看似有点问题（Fold 1 用后来的数据训），但在“特征提取”的意义上，
    # 只要保证“模型没见过它预测的那一段”，就能模拟出 Out-of-Sample 的效果。
    kf = KFold(n_splits=5, shuffle=False) 
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # train_idx: [2, 3, 4, 5] (假设 fold=0)
        # val_idx:   [1]
        train_fold = full_df[train_idx]
        val_fold = full_df[val_idx]
        
        # 2. 训练 SFT 模型 (train_fold_and_predict)
        # 在这一步，我们用 80% 的数据训练一个新的 PatchTST 模型。
        # 这个模型完全不知道 val_fold 长什么样。
        preds = train_fold_and_predict(train_fold, val_fold, ...)
        
        # 3. 预测 Val Fold
        # 我们得到的 preds，就是模型在“瞎猜”情况下的预测值。
        # 这完全模拟了上线后，模型面对未知未来的状态。
        all_preds.append(preds)
        
    # 4. 拼接
    # 最后把 5 个 fold 的预测拼起来，我们就拥有了覆盖全时段的“真实误差信号”。
    full_preds = pl.concat(all_preds).sort("ds")
```

**初学者常见疑问**：
*   **问**：为什么不直接用全量数据训一个模型，然后预测全量数据？
*   **答**：那样模型“见过答案”了。它对第 60 天的预测会非常准（过拟合）。如果用这个超准的信号去训 Policy Head，Policy Head 会学会“无脑信预测”。等上线实盘，预测变准了（因为没见过未来），Policy Head 就会被坑死。

---

## 2. Phase 2: 决策大脑 (`src/policy_head.py`)

这个文件定义了我们的决策智能体。它是一个简单的多层感知机 (MLP)，但输出层有讲究。

### 核心代码剖析

```python
class PolicyHead(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        # 输入层：为什么是 3？
        # 因为我们要喂给它：[SFT预测值, 历史波动率, 历史趋势]
        # 这三个数构成了它做决策的全部依据。
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # ... 中间层省略 ...
            nn.Linear(hidden_dim // 2, 2)  # 输出层：输出 2 个数
        )
```

#### 关键点：如何输出连续动作 [0, 2]？

我们不能直接输出一个标量（比如用 Tanh），因为我们需要**概率分布**来支持强化学习的探索。我们选择 **Beta 分布**。

```python
    def forward(self, x):
        out = self.net(x)
        # Softplus(x) = log(1 + exp(x))
        # 作用：把网络输出的任意实数 (-inf, +inf) 映射到 (0, +inf)。
        # 为什么要 +1.0？为了让 Beta 分布倾向于“单峰”而不是两端极化，训练更稳定。
        alpha = F.softplus(out[:, 0]) + 1.0 + 1e-6
        beta = F.softplus(out[:, 1]) + 1.0 + 1e-6
        return alpha, beta
```

#### 动作采样 (Training vs Inference)

```python
    def get_action(self, x, deterministic=False):
        alpha, beta = self.forward(x)
        
        if deterministic:
            # 推理模式 (上线时)：
            # 我们不希望决策忽高忽低，所以直接取分布的“平均值”。
            # Beta 分布均值公式 = alpha / (alpha + beta) -> 结果在 [0, 1]
            mean = alpha / (alpha + beta)
            action = 2.0 * mean  # 放大到 [0, 2]
        else:
            # 训练模式 (GRPO 时)：
            # 我们需要“试错”。
            # 从分布里随机抽一个值。有时候均值是 1.0，但我可能抽到 1.2 试试看。
            # 如果 1.2 赚了，我就通过梯度告诉网络：“下次把均值往 1.2 挪一挪”。
            dist = torch.distributions.Beta(alpha, beta)
            sample = dist.sample()
            action = 2.0 * sample
            
        return action
```

---

## 3. Phase 3: 训练道场 (`kaggle/work/train_grpo.py`)

这是最神奇的地方：模型是如何在没有“正确答案”的情况下学会赚钱的。

### 数据集构建 (`GRPODataset`)

```python
    def __getitem__(self, idx):
        # 我们把两类信息拼在一起：
        # 1. State (决策依据): [y_hat (SFT预测), vol (波动率), trend (趋势)]
        # 2. Target (客观事实): [y_true (真实收益), vol (用于计算风险)]
        state = self.data[idx, :3]
        target_info = self.data[idx, [3, 1]]
        return state, target_info
```

### 训练循环 (The Loop)

```python
        for states, targets in dataloader:
            # 1. 决策 (Actor)
            # Policy Head 看着 state，心里没底地给出了 actions (比如 [1.2, 0.5, ...])
            actions = policy.get_action(states, deterministic=False)
            
            # 2. 判卷 (Critic/Environment)
            # 市场拿着 actions 和当天的真实收益 y_true，计算你赚了多少。
            # 赚了就是正分，亏了就是负分。
            # 还可以扣除风险分 (lambda * risk)。
            rewards = compute_reward(actions, y_true, vol)
            
            # 3. 计算优势 (Advantage)
            # baseline 是全班平均分。
            # 如果你这笔交易得了 5 分，平均分是 2 分，那你的优势(adv)就是 +3 分。
            baseline = rewards.mean()
            adv = rewards - baseline
            
            # 4. 策略梯度 (Policy Gradient)
            # 核心公式：Loss = - log_prob(action) * advantage
            # log_prob(action): 网络认为“出这个动作”的概率是对数。
            # 
            # 逻辑演绎：
            # - 如果 adv > 0 (表现好): Loss 变负 -> 梯度下降 -> log_prob 增大。
            #   (网络学会了：下次遇到这种情况，还这么干！)
            # - 如果 adv < 0 (表现差): Loss 变正 -> 梯度下降 -> log_prob 减小。
            #   (网络学会了：下次别这么干了！)
            log_probs = policy.get_log_prob(states, actions)
            loss = -(log_probs * adv).mean()
            
            # 5. 修正大脑
            optimizer.step()
```

---

## 4. 总结：为什么这套能行？

1.  **不依赖上帝**：我们从头到尾没有假设“SFT 预测是准的”。我们是通过 K-Fold 强行让 SFT 暴露出它的不准。
2.  **拥抱不确定性**：Policy Head 通过 Beta 分布采样，探索了不同仓位在不同信号下的收益，而不是死记硬背。
3.  **端到端优化**：最终优化目标直接是 Reward (赚钱)，而不是 MSE (预测准)。这直接对齐了比赛目标。

这套架构虽然代码量不大，但逻辑非常严密，是金融时间序列中**Sim-to-Real (模拟到实盘)** 的最佳实践之一。

