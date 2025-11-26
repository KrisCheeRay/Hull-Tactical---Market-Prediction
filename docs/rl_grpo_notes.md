## RL & GRPO 学习笔记（按你刚才的思考路径整理）

本笔记不是正式论文推导，而是**从直觉到公式再到代码**，完全按照你刚才的提问顺序，把概念串成一条线，方便你之后复习。

---

## 1. 从我们的项目出发：我们到底在干什么？

先从你已经完全想通的那句开始：

- **我们当前的交易 RL 设定其实是「单步轨迹」**：
  - 每天视为一条轨迹：  
    `state_t`（今天市场状态 + SFT 预测因子）  
    → `action_t`（今天仓位 0–2）  
    → `reward_t`（今天真实 PnL + 风险惩罚）
  - 所以一条 trajectory 只有 1 个时间步 t。
  - 这就是为什么代码里的 loss 里看不到对时间 t 的显式求和。

损失函数在代码里长这样（概念版）：

```python
loss = -(log_probs * advantages).mean()
```

- `log_probs`: 每个样本（每天）的 `log πθ(a_t | s_t)`
- `advantages`: 每个样本的优势 `A_t`
- `.mean()`: 对整个 batch 做平均 = 用这批样本的平均值来近似论文里的期望 `E[...]`

你已经意识到：

- **如果轨迹只有 1 步**：  
  数学上的 `Σ_t` 退化为 1 项；  
  只剩下对「不同轨迹」(不同天) 的平均。

---

## 2. MLP 输出的两个数：alpha / beta 到底是什么？

你一开始的迷惑：

- 「我们不是就俩 action 吗，alpha 和 beta？」

澄清：

- **alpha / beta 不是 action，是「分布的形状参数」**。
- 我们的 PolicyHead 做的是：
  1. 看 state，输出两个实数 `raw_alpha`, `raw_beta`
  2. 经过 Softplus 转成正数，再 +1 得到 `alpha`, `beta`
  3. 用它们定义一个 **Beta 分布**：`dist = Beta(alpha, beta)`
  4. **训练时**从中 sample 出一个 `sample ~ Beta(...)`，并映射到 `[0, 2]`：
     ```python
     sample = dist.sample()      # 在 [0,1] 上的随机数
     action = 2.0 * sample       # 缩放到 [0,2]
     ```

类比：

- **alpha / beta**：骰子的形状（公平骰子？偏向某一面？）
- **action**：这一次扔骰子扔出来的点数

我们之所以想要「分布」而不是「一个定值」，是为了：

- **训练时**有随机性（exploration），可以尝试不同仓位；
- **推理时**用分布的均值（expectation）作为一个稳定的 deterministic 决策。

---

## 3. 轨迹、时间步、action：Σ 到底在对什么求和？

你的原始理解（非常接近真相）：

- 一条轨迹里有很多个 action；
- loss 里应该是「每个 action 的 log_prob × 它的 advantage」，然后加起来；
- 最后对轨迹做平均。

标准 RL 的数学写法正是这样：

```text
J(θ) ≈ (1/N) Σ_{i=1..N} Σ_{t=1..T_i} [ log πθ(a_{i,t} | s_{i,t}) * A_{i,t} ]
```

- 外层 Σ_i：对 **不同轨迹 τ_i** 求平均（N 条轨迹）
- 内层 Σ_t：对同一条轨迹内部的 **不同时刻 t 的 action** 做贡献求和

而在我们的项目里：

- 我们**人为简化成「单步轨迹」**：  
  每天一条轨迹，轨迹长度 T=1。
- 所以 `Σ_t` 只剩下 1 项，代码里看起来就只剩下：

```python
loss = -(log_probs * advantages).mean()
```

这里的 `.mean()`：

- 是对所有「轨迹 i」（每天样本）做平均；
- 数学上对应 `(1/N) Σ_i`，内部的 Σ_t 因为 T=1 被省略了。

你总结得很好：

> 蒙特卡洛实际上就是 sample 多个轨迹，然后求和是对一条轨迹上多个 action 求优势加权，最后再对所有轨迹做 uniform 平均。

在我们这里：**一条轨迹只有 1 个 action**，所以只剩最后那一步平均。

> 额外提示：在 **LLM 场景** 下，一个 prompt (state_0) 通常会 sample 多条完整轨迹（不同回答），每条轨迹内部又包含多次 action（token 级决策）。  
> 所以一个 batch 可能有 B 个 prompt、每个 prompt 采样 N 条轨迹，总共有 B×N 条轨迹，每条轨迹再按 token 维度求和。  
> 我们当前的 trading 项目是单步轨迹，所以“每个 state 只 sample 一个 action”已经足以构建 Monte Carlo 估计；不需要像 LLM 那样对同一个 prompt 反复采样多条轨迹。区别只是任务结构不同，抽象的数学框架是一样的。

---

## 4. 期望 & 蒙特卡洛：论文里的 E[...] 和代码里的 .mean()

论文里经常写：

```text
J(θ) = E_{τ ~ πθ}[ R(τ) ]
∇J(θ) = E_{τ ~ πθ}[ Σ_t log πθ(a_t | s_t) * A_t ]
```

你抓住了这点：

- 外层的 `E[...]` 是「在所有可能轨迹上的期望」
- 实际上我们不可能枚举所有轨迹，只能：
  - sample 出 N 条轨迹（或 N 个样本）
  - 用样本平均 **近似期望**。

这就是「蒙特卡洛估计」在 RL 里的具体含义：

```python
loss = -(log_probs * advantages).mean()
```

- `log_probs * advantages`：对应单个样本/时间步的 `log π * A`
- `.mean()`：对整个 batch（样本集合）做平均，近似 `E[...]`

你后面精准地总结出了本质：

> 蒙特卡洛实际上就是 sample 多个轨迹，然后对一条轨迹内做 Σ，再对所有轨迹做平均。

---

## 5. baseline / Advantage：为什么要减去「平均奖励」？

原始 REINFORCE：

```text
Loss_pg = - E[ log πθ(a|s) * R ]
```

问题：

- 如果所有 R 都是正的（比如都是 +10），那所有动作的概率都会一起被拉大；
- 方差很大，训练不稳定。

解决办法：引入 **baseline**，构造 Advantage：

```text
A = R - baseline
Loss = - E[ log π(a|s) * A ]
```

最简单的 baseline：

- 就是当前 batch 的平均奖励 `baseline = rewards.mean()`；
- 这样：
  - 比平均好的（A>0）：log_prob 会被「鼓励」变大；
  - 比平均差的（A<0）：log_prob 会被「惩罚」变小。

你抓到了重点：

- GRPO 在最简实现里，**可以只用 batch 奖励均值作为 baseline**，
- 不引入独立的 Value 网络（Critic），先保持 Actor-only，降低复杂度。

这对我们现在的项目是非常合适的。

---

## 6. PPO / GRPO 里的 ratio & clip / KL 惩罚到底在干嘛？

你问到：

- 为啥很多地方还有 KL 惩罚？
- 这和 clip 有啥区别？

本质都是同一件事：**防止更新太猛（“学过头”）**。

### 6.1 ratio + clip（PPO / GRPO）

PPO 风格的目标：

```text
r(θ) = π_new(a|s) / π_old(a|s)
L_clip(θ) = E[ min( r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A ) ]
```

解释：

- 如果新策略 π_new 相比旧策略 π_old 改变不大（r 在 [1-ε,1+ε] 之间），就正常用 `r * A`；
- 如果改太多（r 超出区间），就强制截断到 `1±ε`；
- 防止某一次梯度太大，把策略整形直接打崩。

### 6.2 KL 惩罚

另一种常见做法是加 KL penalty：

```text
loss = loss_pg + β * KL(π_new || π_old)
```

- KL 越大，说明新旧策略差得越多；
- 乘上一个权重 β，当偏离太多时提供惩罚，逼着更新「慢一点」。

**在我们当前的小 MLP 上：**

- 可以先不加 KL / clip，  
  用小学习率 + baseline + 适当的 early stopping；
- 如果之后发现训练不稳定，再考虑加一个简单的 KL penalty，是最容易落地的加强版。

---

## 7. 训练时的“随机 sample” vs 推理时的“确定动作”

你一开始困惑：

- “既然我们要用 RL，为什么还要随机 sample？那线上推理的时候怎么办？”

关键区别：

### 7.1 训练阶段（Exploration 模式）

我们希望策略多试试不同动作：

```python
alpha, beta = policy_head(state)
dist = Beta(alpha, beta)
sample = dist.sample()      # 在 [0,1] 内抽一个
action = 2.0 * sample       # 仓位 [0,2]
```

- 每次训练，同一个 state 可能抽出不同的 action；
- 不同的 action 会导致不同的 reward；
- 这样我们才能比较：**“action A 比 action B 好多少？”**，并通过 advantage 把这个信息写回网络参数里。

### 7.2 推理阶段（Exploitation 模式）

上线时，我们只需要一个稳定、可重复的决策：

```python
alpha, beta = policy_head(state)
mean = alpha / (alpha + beta)   # Beta 的均值 in [0,1]
action = 2.0 * mean             # 映射到 [0,2]
```

- 不再 sample；
- 行为上就退化成「输入向量 → 一个实数仓位」的普通回归模型。

结论：

- **多余的随机性只出现在训练**，是为了探索（exploration）；
- **推理时完全可以 deterministic**。

---

## 8. 类比到 Transformer / 生成式模型：一整句给奖励还是每个 token 给奖励？

你的问题：

- 「LLM 生成的时候是给整段话一个奖励，还是每个 token 一个奖励？」
- 「如果是整段话，那怎么把奖励用到每个 token 上？」

标准 RLHF / GRPO-LLM 做法：

1. **生成整句 response**：
   - 给一个 prompt，模型生成一个完整序列 `y_1,...,y_T`。
2. **对整句打一个分 `R`**（来自 reward model 或打分规则）。
3. **把这个 R 分配到每个 token 的 log_prob 上**：

   ```text
   Loss = - Σ_{t=1..T} [ log πθ(y_t | y_<t, x) * R ]
   ```

   - 虽然 R 是对整句的评分，但每个 token 的选择都对这句好坏有贡献；
   - 所以每个时间步 t 都乘上同一个 R，告诉模型：
     > “这整句都挺好，你在这些步骤里的决策都该被推一推。”

你之前的想法：

- 「总不能对每个词都枚举排列组合生成一句话吧？」

的确不会，做法是：

- 用当前策略 πθ **sample 若干条句子**（轨迹 τ）；
- 对每条轨迹评 Reward；
- 用这些 sample 的统计来近似理论上的 `E_{τ ~ πθ}[·]`；
- 完全不需要枚举所有可能句子。

这和我们交易版的 RL 是一模一样的逻辑，只是：

- 交易：一条轨迹是「多天的仓位序列」；
- 文本：一条轨迹是「多 token 的输出序列」。

---

## 9. 总结：把你现在的理解提炼成几句「金句」方便复习

1. **蒙特卡洛**：就是「从当前策略里 sample 多条轨迹，用它们的平均近似理论上的期望 E[...]」。
2. **轨迹 vs action**：一条轨迹由多步 (s_t, a_t, r_t) 组成，内层 Σ_t 是对一条轨迹内所有时间步的贡献求和，外层 Σ_i / mean 是对多条轨迹的平均。
3. **alpha / beta**：不是 action，是「动作分布的性格」；真正的 action 是从 Beta(α,β) 里 sample 出来的仓位。
4. **Advantage**：`A = R - baseline`，最简单 baseline 就是 batch reward 均值；A>0 的动作会被鼓励，A<0 的动作会被压低概率。
5. **我们当前项目**：每条轨迹只有 1 步（当天 state → 当天 action → 当天 reward），所以时间维上的 Σ_t 退化，只剩对 batch 的平均。
6. **训练 vs 推理**：训练用随机 sample（exploration），推理用均值（exploitation），对外看起来就是一个普通的“输入 → 仓位”模型。
7. **PPO/GRPO**：在基本的 `log_prob * A` 上，加了 ratio / clip / KL 等约束，目的是“别一次改太猛”，提高训练稳定性。

如果以后你想把这些再和具体代码一一对照，我们可以在 `kaggle/work/grpo.py` 里写一个「6 行伪环境 + 20 行训练循环」的玩具版，把每个 Σ / E[...] 都标在具体的 tensor 维度上。那样你就能从「论文公式 → 代码实现」完全自由切换了。


