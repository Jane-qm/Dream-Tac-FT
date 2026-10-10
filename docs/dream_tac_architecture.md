# Dream-Tac 架构、输入数据格式与输出说明

> 基于仓库 `/media/jushen/zhixuan/Dream-Tac` 源码梳理（非论文转述）。行号以当前代码为准。

---

## 1. 一句话概括

Dream-Tac = **Cosmos Policy（Cosmos-Predict2-2B Video2World DiT 微调成策略）+ 触觉**。
它把「当前观测（本体状态 / 相机图像 / 触觉图像）+ 动作块 + 未来观测」全部排成一条**潜空间视频序列**，用同一个扩散 Transformer 联合去噪：

- 条件帧（当前状态、当前图像、当前触觉）保持干净；
- 需要预测的帧（动作块、未来本体状态、未来图像、未来触觉）加噪后去噪；
- 额外加入 **CASA（Contact-Aware Self-Attention）**：在自注意力 logits 上加一个 rank-1 偏置，让非触觉 token 更关注触觉 token，强度由“接触门控” \(g_t\) 决定。

---

## 2. 代码结构（与训练链路相关的部分）

| 模块 | 路径 | 作用 |
|---|---|---|
| 原始数据预处理 | `cosmos_policy/experiments/robot/franka/preprocess_tactile_franka_data.py` | 原始 HDF5（JPEG 帧 + 位姿）→ 预处理 HDF5 + MP4 |
| T5 文本嵌入 | `cosmos_policy/datasets/save_franka_t5_text_embeddings.py` | 读 `preprocessing_metadata.json` 中的 task_name，生成 `t5_embeddings.pkl` |
| 数据集 | `cosmos_policy/datasets/franka_dataset.py`（继承 `libero_dataset.py`） | 组装 12 槽位潜序列所需的视频帧、动作、本体、触觉门控 |
| 统计/归一化 | `cosmos_policy/datasets/dataset_utils.py`、`dataset_common.py` | min/max 归一化到 [-1, 1]，统计写 `dataset_statistics_franka.json` |
| 触觉门控 | `cosmos_policy/utils/tactile_self_attn_gate.py` | 由相邻两帧触觉图像差分算 \(g_t\) |
| 策略模型 | `cosmos_policy/models/policy_video2world_model.py`、`policy_text2world_model.py` | 条件掩码、向量模态注入、损失计算 |
| DiT 主干 | `cosmos_policy/_src/predict2/networks/minimal_v4_dit.py` | `MiniTrainDIT`，含 CASA 开关 |
| CASA 注意力内核 | `cosmos_policy/_src/predict2/networks/tactile_self_attn_chunked.py` | FlashBias 风格 SDPA / chunked SDPA / eager |
| 实验配置 | `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` | 数据集 + 模型超参 + dataloader + job |
| 训练入口 | `cosmos_policy/scripts/train.py` | `torchrun -m cosmos_policy.scripts.train --config=... -- experiment=...` |
| 推理工具 | `cosmos_policy/experiments/robot/cosmos_utils.py` | `get_action`：构造序列、采样、解码动作/未来图像/未来本体 |
| 开环评估 | `cosmos_policy/experiments/robot/franka/run_franka_openloop.py` | 回放一条 episode，输出动作对比图与未来预测 IoU |

---

## 3. 基础组件

| 组件 | 说明 |
|---|---|
| 主干 | Cosmos-Predict2-2B Video2World（`model-480p-16fps.pt`）上微调，`resolution="224"` |
| Tokenizer | Wan2.1 VAE：时间压缩 4×（`num_duplicates_per_image=4`，每张图复制 4 次占满 1 个潜帧），空间 8×，16 通道。224×224 图像 → 每个潜帧 16×28×28 = **12544 个元素** |
| 文本 | T5-XXL 嵌入（预先离线算好，训练时 `conditioner.text.dropout_rate=0`） |
| 扩散 | EDM 框架，`HybridEDMSDE`（sigma ∈ [0.01, 200]），`conditioning_strategy="frame_replace"`，`sigma_conditional=0` |

---

## 4. 输入数据格式要求

### 4.1 原始数据（预处理脚本期望的格式）

`preprocess_tactile_franka_data.py` 递归查找 `--dataset_path` 下所有 `.hdf5`，每个文件需包含：

| HDF5 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `observations/images/cam_front` | (T,) JPEG bytes | 主视角（primary） |
| `observations/images/cam_high` | (T,) JPEG bytes | 当作腕部视角（wrist） |
| `observations/images/tactile_rectify_left` | (T,) JPEG bytes，可选 | 左指尖触觉图像（Xense） |
| `observations/images/tactile_rectify_right` | (T,) JPEG bytes，可选 | 右指尖触觉图像 |
| `puppet/pose` | (T, 6) | 末端位姿 xyz + rpy |
| `puppet/gripper` | (T,) 或 (T,1)，可选 | 缺省填 1.0 |

任务名 = `--dataset_path` 的目录名去掉下划线（注意代码是 `replace("_", "")`，不是替换为空格）。

### 4.2 预处理后格式（`FrankaDataset` 直接读取）

```
<out_base_dir>/<task_folder>/
├── train/episode_{i}.hdf5, episode_{i}_{cam}.mp4
├── val/...
├── preprocessing_metadata.json
├── t5_embeddings.pkl                       # 由 save_franka_t5_text_embeddings 生成
├── dataset_statistics_franka.json          # 首次训练时自动计算
└── dataset_statistics_post_norm_franka.json
```

每个 `episode_{i}.hdf5`：

| 键 | 形状 | 说明 |
|---|---|---|
| attrs `task_name` | str | 用作 T5 嵌入的 key |
| `observations/qpos` | (T, 6) | **proprio = 上一步动作的位姿**（`qpos[t] = action[t-1, :6]`） |
| `observations/video_paths/{cam_front, cam_high, tactile_rectify_left, tactile_rectify_right}` | bytes | 同目录下 MP4 相对路径（默认 256×256、25 fps） |
| `action` | (T, 7) | 绝对位姿 6 + 夹爪 1 |
| `relative_action` | (T, 7) | 差分动作（训练未使用） |

注意：`FrankaDataset.__init__` 会把**所有 episode 的全部视频帧解码并 resize 到 224 后常驻内存**（`load_video_as_images`），数据量大时会成为内存瓶颈。

### 4.3 单个训练样本（`FrankaDataset.__getitem__` 输出）

以 tactile 版本（`state_t=12`）为例：

| 字段 | 形状 / 值 | 说明 |
|---|---|---|
| `video` | (3, 45, 224, 224) uint8 | 1 帧 blank + 11 个槽位 × 4 帧 |
| `actions` | (chunk_size, 7) | 从 t 开始的动作块，越界重复最后一帧，已归一化到 [-1,1] |
| `proprio` / `future_proprio` | (6,) | t 与 t+chunk_size 时刻的本体，已归一化 |
| `t5_text_embeddings` | (512, 1024) | 由 `command` 查表 |
| `*_latent_idx` | int | 各模态所在潜帧下标（见 4.4） |
| `future_tactile_left/right_latent_idx` | 10 / 11 | 用于触觉预测损失统计 |
| `tactile_self_attn_gate` | 标量 float | 接触门控 \(g_t\) |
| `rollout_data_mask`、`world_model_sample_mask`、`value_function_sample_mask` | 0/1 | Franka 只有 demo，均为 0 |
| `value_function_return` | -100 占位 | Franka 不训练价值函数 |

### 4.4 潜序列槽位布局

**Franka + 触觉（12 个潜帧，`chunk_duration=45`）**

| 潜帧 idx | 内容 | 来源 | 角色 |
|---|---|---|---|
| 0 | blank | 全零图 | 条件 |
| 1 | 当前 proprio | 向量注入 | 条件 |
| 2 | 当前 wrist（cam_high） | 图像 | 条件 |
| 3 | 当前 primary（cam_front） | 图像 | 条件 |
| 4 | 当前触觉左 | 图像 | 条件（CASA 触觉 token） |
| 5 | 当前触觉右 | 图像 | 条件（CASA 触觉 token） |
| 6 | 动作块 | 向量注入 | **预测** |
| 7 | 未来 proprio | 向量注入 | **预测** |
| 8 | 未来 wrist | 图像 | **预测** |
| 9 | 未来 primary | 图像 | **预测** |
| 10 | 未来触觉左 | 图像 | **预测**（CASA 触觉 token） |
| 11 | 未来触觉右 | 图像 | **预测**（CASA 触觉 token） |

对应配置：`state_t=12`、`min/max_num_conditional_frames=6`、`tactile_latent_t_indices=(4,5,10,11)`。
无触觉基线是 LIBERO 的 8 槽布局：blank, proprio, wrist, primary, action, future proprio, future wrist, future primary。
ALOHA（三相机）布局支持第二个腕部视角：`current_wrist_image2_latent_idx` / `future_wrist_image2_latent_idx`，模型侧已经支持这两个 key。

### 4.5 非图像模态如何进入扩散序列

数据集先在动作 / proprio 槽位放全零图，VAE 编码后由模型把该潜帧整体替换为向量：

- `replace_latent_with_action_chunk`：把 (chunk_size, action_dim) 展平后**循环平铺**填满 16×28×28 的潜帧；
- `replace_latent_with_proprio`：同理平铺 proprio；
- 推理时 `extract_action_chunk_from_latent_sequence` 把潜帧切成若干份，取平均还原动作块。

约束：向量元素数必须不超过 12544（例如 chunk 20 × 动作 20 维 = 400，没有问题）。

---

## 5. 训练前向与损失

1. `get_data_and_condition`（`policy_video2world_model.py`）
   - VAE 编码 `video` → `x0` (B, 16, 12, 28, 28)；
   - 构造 `condition_video_input_mask`：前 6 个潜帧为条件，动作 / 未来帧为预测目标；
   - 把 proprio / action 注入 `gt_frames`；
   - 把 `tactile_self_attn_gate` 放进 condition，作为 `tactile_self_attn_gate_B` 传给网络。
2. `compute_loss_with_epsilon_and_sigma`（`policy_text2world_model.py`）
   - `x0` 中注入 action、proprio、future proprio；
   - 加噪 → `denoise` → 对条件帧用 GT 替换（`denoise_replace_gt_frames=True`），因此**损失只来自非条件帧**；
   - 损失为按 sigma 加权的 MSE；另外分模态记录 action / future proprio / future image / future tactile 的 MSE、L1 日志；
   - 可选 `action_loss_multiplier`，以及若干 mask 开关（Franka 配置下均为默认 False）。

---

## 6. CASA（Contact-Aware Self-Attention）

**门控**（`tactile_self_attn_gate.py`，在数据集中计算）：

\[
\text{raw}_t = \max\Big(\tfrac{1}{255}\,\overline{|L_t - L_{t-1}|},\ \tfrac{1}{255}\,\overline{|R_t - R_{t-1}|}\Big),\qquad
g_t = g_{\min} + (g_{\max}-g_{\min})\,\sigma\!\Big(k\,\frac{\text{raw}_t - \text{median}}{\text{MAD}}\Big)
\]

默认 median=0.002、MAD=0.001、k=4，\(g_t\in[0.15, 1.0]\)。这几个常数是针对触觉 RGB 图像差分标定的。

**注意力偏置**（`minimal_v4_dit.py` + `tactile_self_attn_chunked.py`）：

\[
\text{score}_{ij} \mathrel{+}= \alpha\, g_t\,(1-M_i)\,M_j
\]

其中 \(M\) 在 `tactile_latent_t_indices` 对应的潜帧 token 上取 1，\(\alpha\) 为 `tactile_self_attn_alpha`（默认 2.0）。效果是：只有非触觉 query 看向触觉 key 时被加强。

由于偏置是 rank-1 的，`flashbias_sdpa` 后端把 \(\sqrt{\gamma}a_i\)、\(\sqrt{\gamma}b_j\) 拼接到 Q/K 的额外维度上，因此不需要 float mask，可以继续走 Flash/cuDNN 内核。后端由环境变量 `COSMOS_TACTILE_SELF_ATTN_BACKEND` 选择。

**这一机制并不依赖触觉是图像**：只需要 (a) 有若干“接触模态”潜帧下标，(b) 每个样本有一个标量门控。这是后面接入六维力的关键。

---

## 7. 实验配置要点（以 `cosmos_predict2_2b_480p_franka_cut_banana_20260321` 为例）

```python
dataset = L(FrankaDataset)(data_dir=..., t5_text_embeddings_path=..., chunk_size=20,
                           use_image_aug=True, normalize_actions=True, normalize_proprio=True,
                           num_duplicates_per_image=4, demonstration_sampling_prob=1.0,
                           return_value_function_returns=False)
model.config: state_t=12, min/max_num_conditional_frames=6, tokenizer.chunk_duration=45,
              net.use_tactile_self_attn_bias=True, tactile_self_attn_alpha=2.0,
              tactile_latent_t_indices=(4,5,10,11)
scheduler: 20k 步余弦周期 + 2k warmup；lr=1e-4（继承 libero 配置）
dataloader: batch_size=16, num_workers=12, DistributedSampler
checkpoint: 从 Cosmos-Predict2-2B-Video2World 加载，save_iter=50
```

新实验需要定义 `LazyDict` 并加入文件末尾的注册列表（约第 1016 行起）。

---

## 8. 输出信息

### 8.1 训练输出

- Checkpoint：`$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints/iter_XXXXXXXXX`
- 日志（W&B / 控制台）：总损失，以及 action、future proprio、future image / wrist、future tactile left/right 的 MSE / L1，迭代速度
- 数据目录下自动生成 `dataset_statistics_franka.json`（动作 / 本体的 min、max、mean、std），**推理反归一化必须用它**

### 8.2 推理输出（`cosmos_utils.get_action`）

| 键 | 内容 |
|---|---|
| `actions` | (chunk_size, action_dim)，用 dataset statistics 反归一化后的动作块 |
| `future_image_predictions` | 解码后的未来 primary / wrist / tactile_left / tactile_right 图像 |
| future proprio | 从未来 proprio 潜帧还原 |
| `value_prediction` | Franka 配置下无意义 |

### 8.3 开环评估输出（`run_franka_openloop.py`）

- 预测动作与 GT 动作逐维对比曲线；
- `future_prediction_eval/` 下保存每个 chunk 的预测 / GT 图像；
- `future_pred_iou_summary.json`：前景 IoU，以及触觉纹理 IoU（`edge_texture_iou_p88`、`dark_inv_otsu_iou`）。

---

## 9. 端到端流程

```
原始 HDF5 ──preprocess──▶ 预处理 HDF5 + MP4 + metadata
                 └─save_*_t5_text_embeddings─▶ t5_embeddings.pkl
                                 │
           FrankaDataset（全部载入内存，算统计量并归一化）
                                 │  video(3,45,224,224), actions, proprio, idx..., gate
                                 ▼
       VAE 编码 → 向量注入 → 加噪 → MiniTrainDIT(+CASA) → x0_pred → 加权 MSE
                                 │
                       checkpoints + W&B 日志
                                 ▼
             get_action / openloop：动作块 + 未来图像 / 触觉预测
```
