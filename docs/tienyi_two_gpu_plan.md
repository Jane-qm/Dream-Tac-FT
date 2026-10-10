# 本机训练计划：tienyi 双臂 + 六维力 × Dream-Tac（本机 2×A100）

> 文件名沿用 single_gpu，机器已从 1 卡变为 2 卡（2026-10-09）。

## Goal（完成标准）

在本机 2×A100 80GB 上训练 `cosmos_predict2_2b_480p_tienyi_battery_force`（可超过 2 小时），并满足：

1. 训练正常进行：总 loss 和 `demo_sample_action_mse_loss` 明显下降，没有 NaN，`demo_sample_future_force_mse_loss` 有数值且在下降；
2. 至少保存 2 个 checkpoint（含最终一个），能被重新加载；
3. 在 val 集 episode 上做开环评估：预测动作块 / 未来力与真值对比的曲线和误差数字（`metrics.json`）。

不在本次范围：无力基线对比（主实验完成后有余量再跑）、真机部署。

## 本机现状

| 项 | 状态 |
|---|---|
| GPU | 2×A100-SXM4-80GB，驱动 535（CUDA 12.2）|
| 内存 / 磁盘 | 962 GB；GPFS `/media/jushen/zhixuan` 剩余约 20 TB |
| 网络（切换 2 卡后） | 可用：百度 / 清华 pip 镜像、download.pytorch.org、developer.download.nvidia.com、**ModelScope**。不可用：GitHub、conda-forge、pypi.org、huggingface.co、hf-mirror.com |
| 数据 | `tienyi_preprocessed/back_battery_and_cover_160`：160 条（train 152 / val 8），157,715 步，校验 0 错误 |

## 关键风险与对策

| 风险 | 对策 |
|---|---|
| transformer-engine 2.2.0 在镜像上只有源码包，本机没有 nvcc | 从 NVIDIA 官网下载 CUDA 12.8 toolkit 装到 GPFS（不装驱动），cuDNN 用 pip 包，本地编译 TE（只编 sm_80）。flash-attn / natten 为可选依赖，不装 |
| 驱动只有 CUDA 12.2，项目要求 cu128 | CUDA 12.x 小版本兼容；只生成 sm_80 机器码，避免 PTX JIT。不行就退回 torch cu126 |
| HF 不可达 | Cosmos 权重、Wan2.1 VAE、T5-11B encoder 全部从 ModelScope `nv-community/Cosmos-Predict2-2B-Video2World` 下载（与 HF 官方仓库同内容，无需 token），用多连接分段下载脚本 `ms_dl.py` |
| T5-11B 全量 45 GB | 只用 Cosmos 仓库自带的 `text_encoder/`（T5-11B encoder，约 9.7 GB）+ `tokenizer/` 里的 T5 分词器；用项目自带的 `CosmosT5TextEncoder(model_name=本地目录)` 加载，代码不改 |
| 代码里写死的权重占位路径 | 软链接 `/path/to/Cosmos-Predict2-2B-Video2World` → 实际目录，代码不改 |
| 后台任务被机器切换打断 | 所有步骤都是可重入脚本（下载断点续传、已完成步骤跳过），日志在 `tienyi_setup/logs/` |

## 步骤（脚本均在 `/media/jushen/zhixuan/tienyi_setup/`）

| # | 步骤 | 脚本 | 产出 |
|---|---|---|---|
| 1 | conda 环境 `envs/dreamtac`（Python 3.10）+ torch 2.7.0 cu128 + 项目依赖 | `01_env.sh` | `STEP1_DONE` |
| 2 | CUDA 12.8 toolkit → 编译 transformer-engine 2.2.0 | `02a_cuda.sh`、`02b_te.sh` | `import transformer_engine` 成功 |
| 3 | 转换 160 条数据（因果力滤波）并校验 | `03_data.sh` | ✅ 已完成 |
| 4 | ModelScope 下载 `model-480p-16fps.pt`、`tokenizer/tokenizer.pth`、T5 encoder + 分词器，建软链接 | `04_models.sh`（调用 `ms_dl.py`） | `STEP4_DONE` |
| 5 | 生成 `$DATA/t5_embeddings.pkl` | `06_t5_emb.sh` | `STEP4B_DONE` |
| 6 | dry-run + 单卡 20 步冒烟，测每步耗时与显存，试 batch 8 / 16 | — | 确定 batch / max_iter |
| 7 | 双卡正式训练（`nohup torchrun --nproc_per_node=2`，W&B offline） | — | checkpoints + 日志 |
| 8 | 开环评估 | `cosmos_policy/experiments/robot/tienyi/openloop_eval.py` | `metrics.json` + 曲线图 |

## 冒烟实测（2026-10-09）

| 配置 | 每步耗时（稳态） | 峰值显存 | loss |
|---|---|---|---|
| 1 卡，batch 8，不分片 | 约 4.0 s | 47 GB | 20 步正常，checkpoint 19 GB |
| 2 卡，每卡 batch 16（全局 32），`fsdp_shard_size=2` | **6.9 s** | 58 GB（reserved 73 GB） | 60 步内 14.9 → 2.8，future force loss 有值 |

- 数据加载：152 条 train episode 读进内存约 35 s。
- 第 1、5 步各有一次约 50 s 的停顿（首次编译 / 关闭自动 GC 后的手动回收），之后稳定。
- 开环评估脚本在冒烟 checkpoint 上跑通（60 步时动作基本是均值预测，正常）。

## 正式训练（已启动）

- 命令：`tienyi_setup/08_pipeline.sh`（无人值守：训练失败自动续训最多 3 次 → 训练完成后在 2 张卡上并行评估最终和中间 checkpoint）
- 实验 `cosmos_predict2_2b_480p_tienyi_battery_force`，job 名 `force_2gpu_b16_4k`
- 2 卡，每卡 batch 16（全局 32），`max_iter=4000`，warmup 200，LR 周期 4000 步，每 500 步存 checkpoint
- 4000 × 6.9 s ≈ 7.7 小时；样本数约 128k（≈ 0.86 个 epoch），与原配置 3000 步 × 全局 64 ≈ 192k 同量级
- 输出：checkpoint 在 `$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy_tienyi_battery/cosmos_v2_finetune/force_2gpu_b16_4k/checkpoints/`；日志 `tienyi_setup/logs/force_2gpu_b16_4k_*.log`；评估 `/media/jushen/zhixuan/tienyi_eval/force_2gpu_b16_4k/<iter>/metrics.json`

手动启动单个训练（参数都可覆盖）：

```bash
cd /media/jushen/zhixuan/tienyi_setup
NGPU=2 BATCH=16 MAX_ITER=4000 JOB=<名字> ./07_train.sh          # 同名 job 重跑会从最新 checkpoint 续训
EXP=cosmos_predict2_2b_480p_tienyi_battery_no_force NGPU=2 BATCH=16 MAX_ITER=4000 JOB=no_force_2gpu ./07_train.sh   # 无力基线
```

## 环境搭建中遇到的问题（已解决）

| 问题 | 解决 |
|---|---|
| `cosmos_policy/` 不能 `pip install -e`（缺 README） | 训练本来就用 `PYTHONPATH=<repo>`，改为直接装依赖（`requirements_dreamtac.txt`） |
| 容器里没有 g++ | `apt-get install g++`（编译 TE / flash-attn 用，产物在 GPFS 上的 conda 环境里） |
| TE import 时找不到 `libcudnn_adv.so.9` | `env.sh` 把 pip 装的 `nvidia/*/lib` 加进 `LD_LIBRARY_PATH` |
| config 导入链上 `qwen2_5_vl.py` 断言 flash-attn 可用 | 源码编译 flash-attn 2.7.3（只编 sm_80，64 并行约 15 min） |
| 缺 `decord`、`peft` | pip 安装 |
| LIBERO 基础实验的数据集参数被 LazyConfig 合并进 `TienyiForceDataset` | 数据集类显式接受并忽略这 9 个继承参数，其他未知参数仍报错 |
| `WANDB_MODE=offline` 不生效（config 里 `job.wandb_mode=online` 优先） | 命令行加 `job.wandb_mode=offline` |
| 文档里的 `MAGINAIRE_OUTPUT_ROOT` 是笔误 | 代码读的是 `IMAGINAIRE_OUTPUT_ROOT`，已全部改正 |
| `fsdp_shard_size` 默认 8 | 命令行设为 GPU 数 |
| 开环评估输入 dtype 与模型不一致 | 与 `get_action` 一样，浮点输入转 bfloat16 |
| pip 直连 download.pytorch.org 只有 ~200 KB/s | `dl_torch.sh` 多连接下载 wheel 后本地安装 |
