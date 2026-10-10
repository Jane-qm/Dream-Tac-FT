# 交接文档：tienyi 双臂 + 六维力数据接入 Dream-Tac（8×A100）

> 相关文档：`dream_tac_architecture.md`（架构说明）、`tienyi_force_integration_plan.md`（数据分析与整体方案）。
> 本文回答三个问题：**做到哪了、输入输出是什么、按什么步骤在 8×A100 上跑起来。**

---

## 0. 当前进度

| 模块 | 状态 | 位置 |
|---|---|---|
| 原始数据分析（字段、六维力、重力影响、四元数顺序） | ✅ 完成 | `tienyi_force_integration_plan.md` 第 1 节 |
| **数据转换脚本** | ✅ 完成，已在 3 条数据上验证 | `cosmos_policy/experiments/robot/tienyi/preprocess_tienyi_force_data.py` |
| **转换结果校验与可视化脚本** | ✅ 完成 | `cosmos_policy/experiments/robot/tienyi/verify_tienyi_preprocessed.py` |
| 冒烟测试样例输出 | ✅ | `/media/jushen/zhixuan/tienyi_preprocessed/smoke3/`（3 条），可视化在 `smoke3_verify/` |
| 全量数据（160 条，因果力滤波） | ✅ 校验 0 错误（train 152 / val 8，157,715 步） | `/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160` |
| T5 指令嵌入 | ✅ 已生成 `$DATA/t5_embeddings.pkl`（本机用 Cosmos 仓库自带的 T5-11B encoder，见 `tienyi_setup/06_t5_emb.sh`） | `cosmos_policy/datasets/save_franka_t5_text_embeddings.py` |
| 训练用数据集类 `TienyiForceDataset` | ✅ 完成，已在 smoke3 上取样测试（CPU） | `cosmos_policy/datasets/tienyi_dataset.py` |
| 模型侧力槽位注入与 loss 日志 | ✅ 完成，CPU 小张量单测通过；尚未在 GPU 上实跑 | `models/policy_video2world_model.py`、`policy_text2world_model.py`、`_src/predict2/callbacks/iter_speed.py` |
| 实验配置 | ✅ 完成并注册 3 个 | `config/experiment/cosmos_policy_experiment_configs.py` |
| 开环评估 | ✅ 脚本完成 | `cosmos_policy/experiments/robot/tienyi/openloop_eval.py` |
| 本机（2×A100）环境与训练 | 见 `tienyi_single_gpu_plan.md` | `/media/jushen/zhixuan/tienyi_setup/` |

---

## 1. 数据转换：输入与输出

### 1.1 输入（原始数据，只读）

```
/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover/
└── <date_dir>/success_episodes/<episode_id>/data/trajectory.hdf5
```

- 31 个日期目录，2769 条，约 3.5 TB，单条 1–2 GB，平均约 1969 帧，相机约 29 Hz。
- `<date_dir>` 以 `_fail` 结尾的是**失败数据**（21 条），默认排除。
- 使用的键：三路相机 JPEG、从臂 EE 位姿 / 夹爪 / 关节、主臂关节 / 夹爪、左右六维力（TCP 系）。完整列表见脚本开头的 docstring。

### 1.2 输出（预处理数据，即训练配置里的 `data_dir`）

```
<out_dir>/
├── train/
│   ├── episode_00001.hdf5
│   ├── episode_00001_cam_high.mp4          # 头部相机（主视角），224×224，约 14.5 fps
│   ├── episode_00001_cam_left_wrist.mp4    # 左腕
│   └── episode_00001_cam_right_wrist.mp4   # 右腕
├── val/ ...                                # 同上，默认占 5%
├── preprocessing_metadata.json             # 参数、来源映射、被跳过的条目；dataset_info.task_name = 指令
└── force_statistics.json                   # 力的逐维统计 + 门控 median/MAD（只用 train 集统计）
```

每个 `episode_XXXXX.hdf5`（T' ≈ T/2）：

| 键 | 形状 | 含义 |
|---|---|---|
| attrs `task_name` | str | 语言指令，同时是 T5 嵌入的 key |
| attrs `source_path` / `stride` / `fps` / `quat_order` | | 溯源信息 |
| attrs `force_window_s` / `force_baseline_s` | float | 力的因果平均窗口（约 0.067 s）和零偏时长（0.5 s），**部署端必须用同样的值** |
| `observations/qpos` | (T', 20) | **proprio**：每只臂 [x, y, z, rot6d(6), gripper]，左臂在前。rot6d 是旋转矩阵的前两列 |
| `action` | (T', 20) | **动作**：`action[t] = qpos[t+1]`（下一时刻的绝对 EE 位姿 + 夹爪），最后一步重复 |
| `observations/force` | (T', 12) | **六维力**：[左 Fx Fy Fz Tx Ty Tz, 右 …]，TCP 系。取原始约 59 Hz 读数（`*_raw`，带时间戳）在 \((t-\text{一个控制周期}, t]\) 内的平均，t 为该步图像的相机时间戳；再减去 episode 开头 0.5 s 的零偏。**只用过去的数据（因果）**，部署端可用同一函数复现，见第 7 节 |
| `observations/force_abs` | (T', 12) | `*_align` 读数（不滤波、不去零偏），供消融用 |
| `observations/force_baseline` | (12,) | 被减掉的零偏 |
| `observations/force_gate_raw` | (T',) | 接触事件强度：4 步窗口差分，按 std 归一化后左右臂取最大值 |
| `observations/qpos_joint` | (T', 16) | 备选：从臂 7 关节 + 夹爪 ×2 |
| `action_joint` | (T', 16) | 备选：主臂 7 关节 + 主臂夹爪指令 ×2 |
| `observations/timestamp` | (T',) | 原始时间戳 |
| `observations/video_paths/{cam_high, cam_left_wrist, cam_right_wrist}` | bytes | 同目录 MP4 的相对路径（命名与 `ALOHADataset` 一致） |

`force_statistics.json` 的关键字段：
- `force_mean/std/min/max/p01/p99`：12 维，供数据集归一化使用（建议先按 p01/p99 截断，再 min-max 到 [-1, 1]）；
- `gate.raw_median`、`gate.raw_mad`：在数据集中通过 `scalar_gate_from_raw(force_gate_raw[t], median=raw_median, mad=raw_mad)` 得到门控 \(g_t\)，取代原来针对触觉图像标定的 0.002 / 0.001。

### 1.3 实测数据（3 条冒烟测试）

- 转换速度约 **17 秒/条/进程**，瓶颈是网络盘读取和 JPEG 解码；输出约 **12 MB/条**（MP4 约 11 MB + HDF5 约 0.5 MB）。
- 校验结果：3 条、3467 步，**0 个错误**。三路相机颜色和映射正确，力曲线的接触尖峰与门控 raw 值吻合（因果版本见 `smoke3_causal_verify/*.png`）。
- 门控 raw 值（因果版本 `smoke3_causal`）：median 约 0.27，MAD 约 0.14。全量数据要以正式运行时生成的 `force_statistics.json` 为准。
- 因果平均与旧版对称高斯平滑（`smoke3/`，已弃用）相比：相关系数 0.96–0.995，整体滞后约 1 步（69 ms），逐帧抖动中位数从约 0.13 N 增加到约 0.24 N（相对力的标准差约 3 N，影响很小）。
- 原始数据还有一个现象：`*_align` 取的是相机帧之后 8–15 ms（最多约 27 ms）的那个原始读数，也就是本身就略微"偷看未来"。改用按时间戳加窗后，这个问题也一并消除。

---

## 2. 环境

### 2.1 数据转换（CPU 即可，不需要 GPU）

只依赖 `h5py numpy opencv-python(-headless) tqdm`，校验脚本另外需要 `matplotlib`。用训练环境也可以，单独装也可以：

```bash
pip install h5py numpy opencv-python-headless tqdm matplotlib
```

### 2.2 训练（8×A100 80GB）

| 项 | 要求 |
|---|---|
| OS / 驱动 | Linux，CUDA 12.8 驱动 |
| Python | 3.10 |
| 依赖 | `cosmos_policy/pyproject.toml`：PyTorch 2.7（cu128）、flash-attn 2.7.3、transformer-engine 2.2.0、natten 0.21.0、transformers 4.51.3 等 |
| 安装方式 1（推荐） | 在仓库根目录 `docker build -t dream-tac docker`，运行时加 `--gpus all --ipc=host`（见 README 1.1） |
| 安装方式 2 | `cd cosmos_policy && uv sync --extra cu128 --group franka` |
| 基础权重 | Hugging Face `nvidia/Cosmos-Predict2-2B-Video2World`：`model-480p-16fps.pt` 和 `tokenizer/tokenizer.pth`（Wan2.1 VAE） |
| 文本编码器 | Hugging Face `google-t5/t5-11b`（约 45 GB，只在生成 T5 嵌入时用一次） |
| 权重路径 | 代码里写死占位路径 `/path/to/Cosmos-Predict2-2B-Video2World/...`（6 处）。**不改代码**，建软链接：`mkdir -p /path/to && ln -sfn <实际权重目录> /path/to/Cosmos-Predict2-2B-Video2World` |
| 国内网络 | HF 不可达时，权重、VAE、T5 encoder 都可从 ModelScope `nv-community/Cosmos-Predict2-2B-Video2World` 下载（同内容，无需 token），见 `tienyi_setup/04_models.sh` |
| 环境变量 | `PYTHONPATH=<repo>`，`IMAGINAIRE_OUTPUT_ROOT=<checkpoint 输出目录>`，`COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa`，以及 W&B 相关的 `WANDB_API_KEY` 或 `WANDB_MODE=offline` |
| 节点资源 | 内存 ≥ 800 GB（见 4.3），预处理输出约 2 GB，checkpoint 每个约 20–30 GB |

---

## 3. 操作步骤（8×A100）

```bash
export REPO=/media/jushen/zhixuan/Dream-Tac
export RAW=/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover
export DATA=/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160
cd $REPO && export PYTHONPATH=$REPO:$PYTHONPATH
```

**Step 1 数据转换**（随机抽 160 条成功数据；32 进程预计 5–10 分钟，取决于网络盘速度）

```bash
python -m cosmos_policy.experiments.robot.tienyi.preprocess_tienyi_force_data \
  --raw_root $RAW --out_dir $DATA --max_episodes 160 --num_workers 32
```

常用参数：`--stride`（默认 2）、`--img_size`（默认 224）、`--date_filter 20260916,20260917`（只用指定日期）、`--percent_val`（默认 0.05）、`--instruction "..."`（默认是英文的电池安装指令）、`--quat_order`（默认 xyzw）、`--force_window_s`（默认 0，即一个控制周期 = stride / 相机帧率）、`--force_baseline_s`（默认 0.5）。

**Step 2 校验**（必须做：检查 `[BAD]` 行，并人工看一眼 PNG）

```bash
python -m cosmos_policy.experiments.robot.tienyi.verify_tienyi_preprocessed \
  --data_dir $DATA --vis_dir $DATA/../verify_160 --num_vis 3
```

**Step 3 生成 T5 嵌入**（1 张 GPU，几分钟）

```bash
python -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir $DATA
# 生成 $DATA/t5_embeddings.pkl，key = 指令字符串
```

**Step 4 指定数据目录**（代码已完成，见第 5 节）

实验配置通过环境变量读取数据目录，默认 `/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160`：

```bash
export TIENYI_DATA_DIR=$DATA
```

三个实验：

| 实验名 | 槽位 | 说明 |
|---|---|---|
| `cosmos_predict2_2b_480p_tienyi_battery_no_force` | 10 | 基线：3 相机 + proprio，不用力 |
| `cosmos_predict2_2b_480p_tienyi_battery_force` | 12 | 主实验：当前力 + 未来力槽位 + 力门控 CASA |
| `cosmos_predict2_2b_480p_tienyi_battery_force_no_attn_bias` | 12 | 消融：有力槽位，关掉 CASA |

**Step 5 配置检查（dry-run）**

```bash
python -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py --dryrun -- \
  experiment="cosmos_predict2_2b_480p_tienyi_battery_no_force"
```

**Step 6 短训练冒烟**（只加载 1 个文件，确认 loss 下降、checkpoint 能保存）

```bash
DEBUGGING=True torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_tienyi_battery_no_force" trainer.max_iter=20
```

**Step 7 正式训练：8×A100，约 2 小时**（先跑基线，再跑加力版本，或者用两台机器并行）

```bash
export IMAGINAIRE_OUTPUT_ROOT=/media/jushen/zhixuan/checkpoints
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_tienyi_battery_force"
```

开训后 10 分钟内看一下每步的耗时：如果明显慢于 3 s/iter，就把 `max_iter` 调到 7200 / 实测秒数，保证在 2 小时内跑完。

**Step 8 开环评估**（待开发）：回放验证集 episode，输出动作曲线、未来图像和未来力的对比。

---

## 4. 8×A100 训练：输入、输出与推荐配置

### 4.1 训练输入

| 输入 | 来源 |
|---|---|
| 预处理数据 `$DATA/train/*` | Step 1 |
| `$DATA/t5_embeddings.pkl` | Step 3 |
| `$DATA/force_statistics.json` | Step 1（力归一化和门控参数） |
| `model-480p-16fps.pt` + `tokenizer.pth` | Hugging Face 下载 |

**单个样本**（数据集类输出，以加力版本为例）：`video` (3, 45, 224, 224)；`actions` (16, 20)；`proprio` / `future_proprio` (20,)；`force` (12,)；`future_force` (12,)，目标是 \(F_{t+H}-F_t\)；`tactile_self_attn_gate` 标量；各模态的 `*_latent_idx`；`t5_text_embeddings` (512, 1024)。

### 4.2 训练输出

| 输出 | 位置 |
|---|---|
| Checkpoints | `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/checkpoints/iter_XXXXXXXXX`，每 500 步一个 |
| 动作 / 本体 / 未来力归一化统计 | 首次训练时在 `$DATA/` 下自动生成 `dataset_statistics_tienyi.json`（只用 train 集，原子写入，多卡安全），**推理反归一化必须用它**，所以 `$DATA` 需要可写 |
| 日志 | 控制台：总 loss，以及 action、future proprio、future 图像、future force（`demo_sample_future_force_mse/l1_loss`）各自的 MSE / L1，迭代速度。W&B 曲线沿用原有项，没有新增 future force 曲线（保持对原代码的最小改动） |

### 4.3 推荐配置（2 小时预算）

| 参数 | 值 | 说明 |
|---|---|---|
| GPU | 8×A100 80GB，`--nproc_per_node=8` | |
| 每卡 batch | 8（全局 64） | 2 小时约 2.4k–2.8k 步（估算值） |
| `chunk_size` | 16 | 约 1.1 s @14.5 Hz |
| `max_iter` | 2500–3000 | 根据实测速度调整 |
| `scheduler.warm_up_steps` | [150, 0] | 原配置是 2000，2 小时内会一直处于 warmup |
| `scheduler.cycle_lengths` | [3000, 1e14] | |
| `checkpoint.save_iter` | 500 | 原配置 50 太频繁，每个 checkpoint 20–30 GB |
| `dataloader.num_workers` | 8 | 116 核 / 8 个 rank |
| 加力版本布局 | `state_t=12`、`chunk_duration=45`、`min/max_num_conditional_frames=6`、`tactile_latent_t_indices=(5, 11)`、`use_tactile_self_attn_bias=True` | 槽位定义见 `tienyi_force_integration_plan.md` 4.2 |
| 无力基线布局 | `state_t=10`、`chunk_duration=37`、`min/max_num_conditional_frames=5`、`use_tactile_self_attn_bias=False` | blank, proprio, 左腕, 右腕, head, action, 未来 proprio, 未来左腕, 未来右腕, 未来 head |
| 数据量 | 约 160 条 | 默认 `load_videos_to_memory=True`：**每个 rank 都把全部视频解码进内存**，每条约 450 MB，8 个 rank 共约 580 GB。内存不够时改为 `False`（按需从 MP4 解码，几乎不占内存，但取样更慢，需要更多 `num_workers`） |

---

## 5. 训练侧实现说明

### 5.1 六维力怎么进入模型

力和 proprio 一样被当作"向量模态"：像素里放全黑占位帧，VAE 编码后，把该 latent 帧（16×28×28 = 12544 个数）用力向量平铺覆盖（复用原有的 `replace_latent_with_proprio`）。

| 槽位 | 内容 | 角色 | 归一化 |
|---|---|---|---|
| 5 `force` | 当前 12 维力 \(F_t\)（左 Fx..Tz，右 Fx..Tz） | **条件**（不加噪、不算 loss） | 按 train 集 p01/p99 截断后映射到 [-1, 1] |
| 11 `future_force` | \(F_{t+H}-F_t\)，H = chunk_size = 16 步（约 1.1 s） | **预测目标**（加噪、算 loss，与动作、未来图像一起去噪） | 除以 train 集 \(|F_{t+H}-F_t|\) 的逐维 p99，再截断到 [-1, 1] |
| CASA 门控 | \(g_t\) = `scalar_gate_from_raw(force_gate_raw[t], median, mad)` ∈ [0.15, 1] | 每个样本一个标量：接触变化越剧烈，其他 token 对槽位 5、11 的注意力偏置越大 | median / MAD 来自 `force_statistics.json` |

预测未来力**不直接用于控制**，作用是辅助任务：逼模型学会"动作 → 接触力变化"的关系。推理时只需要给出当前力。

### 5.2 `TienyiForceDataset`（`cosmos_policy/datasets/tienyi_dataset.py`，新文件）

- 读取 `action` / `observations/qpos`（可改成 `action_joint` / `observations/qpos_joint`）、`observations/force`、`observations/force_gate_raw`、三路 MP4、attrs `task_name`。
- 统计写入 `$DATA/dataset_statistics_tienyi.json`：actions / proprio 的 min/max/mean/std/median，以及 `future_force_scale_chunk{H}`。
- 输出键与 LIBERO / ALOHA 一致，另外加 `force`、`future_force`、`current_force_latent_idx`、`future_force_latent_idx`、`tactile_self_attn_gate`。`use_force=False` 时不输出这 5 个键。
- 进程内缓存：配置框架会为 DataLoader 和 DistributedSampler 各实例化一次数据集，缓存保证视频只在内存里存一份。
- `unnormalize_future_force(pred, current_raw)`：评估时把预测还原成牛顿 / 牛米。

### 5.3 对原有代码的改动（只有新增，没有删改；batch 里没有 `force` 键时不生效）

| 文件 | 改动 |
|---|---|
| `models/policy_video2world_model.py` | `get_data_and_condition`：把 `force` / `future_force` 注入 `gt_frames`；`get_x0_fn_from_batch`（推理）：把当前力槽位的条件掩码置 1 并注入；batch 里有 `tactile_self_attn_gate` 时把门控写入 condition（与训练一致，原推理调用方不传该键，行为不变） |
| `models/policy_text2world_model.py` | `training_step` 传入 4 个可选参数；`compute_loss_with_epsilon_and_sigma` 把力注入 x0，在两个 loss 掩码分支中加入未来力槽位，新增 `demo_sample_future_force_mse/l1_loss` |
| `_src/predict2/callbacks/iter_speed.py` | 控制台日志里加入上面两个 loss 名 |
| `config/experiment/cosmos_policy_experiment_configs.py` | 新增并注册 3 个实验 |

网络（`minimal_v4_dit.py`）、触觉代码（`FrankaDataset`、`tactile_self_attn_gate.py`）都没有改。

### 5.4 测试

- 模型注入：`/media/jushen/zhixuan/tienyi_tests/test_force_injection.py`。用 AST 从源码抽出真实的 `replace_latent_with_proprio` 和 `compute_loss_with_epsilon_and_sigma`，配合 mock 的 `self` 在 CPU 上运行。验证了三点：力被写入槽位 5 和 11；未来力误差只体现在 future force loss 中；没有力的 batch 得到 NaN，不影响其他实验。
- 数据集：`tienyi_tests/test_tienyi_ds.py`（`D=<数据目录>` 指定），在 `smoke3_causal` 上测试形状、范围、缓存、按需解码模式、DataLoader 多进程取样。
- 力滤波：`tienyi_tests/test_force_filter.py`。在真实 episode 上验证：在线版与离线版逐值一致；改动观测时刻之后的读数不影响结果（因果）；每个窗口约 4 个读数，没有空窗口。
- 推理门控：`tienyi_tests/test_inference_gate.py`。验证 batch 带门控时写入 condition / uncondition，不带时与原行为一致。
- 以上测试在本机 CPU 环境 `/media/jushen/zhixuan/.venv_cpu_test`（CPU 版 torch）中运行，命令前加 `PYTHONPATH=<repo>`。

---

## 6. 已知问题与注意事项

- **四元数顺序**：默认 xyzw。依据是头部位姿按 xyzw 解读为向下俯视约 38°，符合实际；如果采集方说明不同，用 `--quat_order wxyz` 重新转换即可。校验脚本会检查 rot6d 的正交性，但无法区分这两种顺序。
- **力**：TCP 系，没有做完整重力补偿（重力项只能解释约 10–20% 的力变化）。抓着物体时 \(|F|\) 会有约 5 N 的持续偏移，这是正常现象，能被当前力槽位保留下来。
- 右臂 Tz 几乎不变（std 约 0.02），计算门控时已对 std 设了下限，避免放大噪声。
- 图像是 640×480 直接缩放到 224×224（与原 Dream-Tac 的 Franka 做法一致），有轻微的长宽比变形。
- 网络盘读取较慢；转换时 `--num_workers` 不宜超过 32–48，否则会被 IO 卡住。
- 要扩大到全量 2748 条，设置 `load_videos_to_memory=False`（按需解码），并适当增加 `num_workers`。

---

## 7. 部署端：六维力的处理（必须与训练一致）

训练数据里的力由 `cosmos_policy/experiments/robot/tienyi/force_filter.py` 生成；部署端**直接 import 同一个文件**，不要自己重写。

```python
import json
import numpy as np
from cosmos_policy.experiments.robot.tienyi.force_filter import OnlineForceFilter

WINDOW_S = 0.0672    # = 训练数据 hdf5 attrs["force_window_s"]（stride / 相机帧率）
BASELINE_S = 0.5     # = attrs["force_baseline_s"]
filt = {arm: OnlineForceFilter(window_s=WINDOW_S) for arm in ("left", "right")}

# 1) 传感器回调：每来一条原始读数就 push（时间戳与相机用同一个时钟，单位秒）
def on_wrench(arm, ts, wrench6):          # wrench6 = [Fx, Fy, Fz, Tx, Ty, Tz]，TCP 系
    filt[arm].push(ts, wrench6)

# 2) 开始执行前：机械臂静止、夹爪未接触，等待 BASELINE_S 秒后取零偏
t_start = ...                             # 静止段开始时刻
for arm in filt:
    filt[arm].capture_baseline(t_start, BASELINE_S)

# 3) 每个控制步：t_obs 用本步送进模型的那帧图像的相机时间戳
f_rel = np.concatenate([filt["left"].get(t_obs), filt["right"].get(t_obs)])   # (12,)，单位 N / N·m

# 4) 归一化：与 TienyiForceDataset.normalize_force 相同（force_statistics.json 的 p01/p99）
stats = json.load(open(f"{DATA}/force_statistics.json"))
lo, hi = np.array(stats["force_p01"]), np.array(stats["force_p99"])
force_in = np.clip(2 * (np.clip(f_rel, lo, hi) - lo) / (hi - lo) - 1, -1, 1)
# batch["force"] = force_in[None]；batch["current_force_latent_idx"] = 5
```

检查清单：

| 项 | 要求 |
|---|---|
| 时间戳 | 力读数与相机使用同一时钟；`t_obs` 是**图像**时间戳，不是推理开始或结束的时间 |
| 控制频率 | 与训练相同：每步约 69 ms（相机 29 Hz、stride 2）。如果改了频率，要用新的 `--stride` / `--force_window_s` 重新转换数据并重新训练 |
| 读数来源 | 使用录制时 `force_*_end_effector_raw` 对应的同一话题（约 59 Hz）。频率更高也可以，因为区间平均对采样率不敏感 |
| 零偏 | 每次执行前重新采集，条件尽量与录制时 episode 开头相同（静止、未抓取物体） |
| 坐标系 / 工具 | TCP 系；更换夹爪或末端工具会改变偏置，需要重新采集数据 |
| 门控 | 原 Dream-Tac 推理路径不传门控，推理时没有 CASA 偏置。现在 `get_x0_fn_from_batch` 已支持：调用方在 batch 里放 `tactile_self_attn_gate`（形状 (B,)）即生效。部署端用 `force_window_mean` 的结果按 `compute_gate_raw` 的公式在线计算 raw，再用 `force_statistics.json` 里的 median/MAD 经 `scalar_gate_from_raw` 映射成门控。开环评估脚本直接用数据集里的门控 |
