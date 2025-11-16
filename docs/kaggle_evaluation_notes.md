## Kaggle Evaluation 框架快速笔记

> 目标：用简单语言描述 `kaggle_evaluation` 包和 `demo.py` 的核心逻辑，帮助第一次参加 Kaggle 比赛的同学理解线上评测是怎么跑的。

---

### 1. 整体流程（像接力赛）

1. 我们写好的 `predict` 函数被包装进 `InferenceServer`。
2. Kaggle 官方写的 `Gateway` 负责按时间顺序把测试集一段段送给 `predict`。
3. `Gateway` 把每一段预测收集起来，合成最终的 `submission.parquet`。
4. 传输数据靠 `relay`（使用 gRPC），就像一个双向对讲机。

---

### 2. `demo.py` 做了什么

- 顶部提醒：`predict` 必须自己实现，而且要在 5 分钟内回应每个 batch。
- 目前的 `predict` 直接返回 `0.0`，只是占位符。
- `DefaultInferenceServer(predict)` 会把我们的 `predict` 注册成服务端的监听函数。
- 若环境变量 `KAGGLE_IS_COMPETITION_RERUN` 存在，说明在官方评测环境，会执行 `inference_server.serve()` 持续等待请求。
- 本地调试时会调用 `run_local_gateway(...)`，使用公开数据模拟官方测试流程。

---

### 3. `kaggle_evaluation.default_inference_server`

- **类：** `DefaultInferenceServer`
- **功能：** 继承 `templates.InferenceServer`，只重写 `_get_gateway_for_test`，返回 `DefaultGateway`。
- **理解：** 这是官方帮我们准备好的“默认网关 + 服务端”组合，方便我们本地自测。

---

### 4. `kaggle_evaluation.default_gateway`

- **类：** `DefaultGateway`
- 继承 `templates.Gateway`，主要定制了三件事：
  1. `unpack_data_paths`：找出比赛数据所在目录（默认是包内的 `test.csv`）。
  2. `generate_data_batches`：把 `test.csv` 读成 `polars.DataFrame`，按 `batch_id`（第一列）拆成多个 batch，一次发给 `predict`。
  3. `competition_specific_validation`：这里没做额外校验（`pass`），但子类里可以自定义检查逻辑。
- 构造函数里还设置：预测列名 `prediction`、行 ID 列名 `batch_id`、响应时限 5 分钟。

---

### 5. `core.templates`

- **`Gateway` 抽象类**（主角是 `get_all_predictions` 用的三个抽象方法）：
  - `unpack_data_paths()`：绑定数据路径。
  - `generate_data_batches()`：将数据切片并附带 row IDs。
  - `competition_specific_validation()`：写比赛特定的预测检查。
- **`InferenceServer` 抽象类**：
  - 构造时会把传入的函数（比如 `predict`）注册到 gRPC 服务端。
  - `serve()`：在 Kaggle 线上模式会阻塞等待请求。
  - `run_local_gateway()`：本地测试用，超时超过 15 分钟会发出警告。

---

### 6. `core.base_gateway`

`BaseGateway` 是 `DefaultGateway` 的父类，处理所有通用细节：

- **初始化：** 建立 gRPC 客户端 (`relay.Client`)，保存数据路径、目标列名、行 ID 列名。
- **超时设置：** `set_response_timeout_seconds` 控制 `predict` 的响应期限（除首 batch）。
- **主流程 `run()`：**
  1. `unpack_data_paths()` 设置数据目录。
  2. `get_all_predictions()` 循环 `generate_data_batches()`。
  3. 每个 batch 调 `predict`，做通用校验 `competition_agnostic_validation()`，再做自定义校验。
  4. 收集所有预测，写出 `submission.parquet`。
  5. 若在官方环境，还会写 `result.json` 记录是否成功。
- **关键校验：** `competition_agnostic_validation` 确保预测行数和 row IDs 对齐，防止作弊或格式错误。
- **文件共享：** `share_files` 用于把大文件挂载到选手容器（本地用符号链接，线上用 `mount`）。
- **错误处理：** 通过 `GatewayRuntimeErrorType` 枚举提供清晰的用户提示。

---

### 7. `core.relay`

- 承担“数据传输 + 序列化”任务。
- **`Client`：**
  - 负责和 `InferenceServer` 建立 gRPC 连接。
  - 第一次连接允许等待 15 分钟，让服务器完成启动。
  - 后续请求会受到 `endpoint_deadline_seconds` 限制。
- **`define_server`：** 把函数列表（例如 `predict`）注册成 gRPC 服务端。
- **序列化 `_serialize` / `_deserialize`：**
  - 支持 `list`、`tuple`、`dict`、`numpy`、`pandas`、`polars` 等常见数据类型。
  - 传输格式使用 Parquet / Arrow / NPY，保证速度与类型安全。

---

### 8. 开发者要点总结

- `predict` 必须快速返回，并保证输出行数与 `Gateway` 提供的 row IDs 一致。
- 若要跨 batch 记忆状态，可以在模块级别保存变量（`global`），但要注意内存。
- 本地跑 `run_local_gateway` 前要确保数据路径正确（默认指向 `/kaggle/input/...`，本地需映射或修改）。
- 一旦切换到 Kaggle 线上环境，`InferenceServer` 会一直监听，直到评测结束。
- 所有日志和错误会通过 `result.json`、`submission.parquet` 反馈给我们。

---

> 有了这些信息，我们就能从 0 开始实现一个最小可用版的 `predict`，并在本地模拟 Kaggle 的在线测试流程，后续再逐步增强特征和模型。



