# tienyi 双臂数据（含六维力）接入 Dream-Tac 训练方案

> 数据：`/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515`
> 目标：先跑通 Dream-Tac 训练流程，再以 Dream-Tac 的方式把六维力（Fx, Fy, Fz, Tx, Ty, Tz）作为“接触模态”加入。
> 架构背景见同目录 `dream_tac_architecture.md`。

---

## 1. 数据检查结果

### 1.1 规模

- 只有一个任务：`back_battery_and_cover`（背面小电池安装 + 小电池盖安装），共 31 个日期子目录、**2769 个 HDF5，约 3.5 TB**。
  - 正常目录 21 个，共 2748 条；`*_fail` 目录 10 个，共 21 条，**已确认是失败数据**（子目录名 `success_episodes` 只是沿用了同样的命名）。
- 路径模式：`<task>/<date_dir>/success_episodes/<episode_id>/data/trajectory.hdf5`，单文件约 1–2 GB。
- 抽样 82 条（每个目录 3 条）：**字段结构完全一致**；长度 130–3032 帧，均值约 1969 帧（相机约 29 Hz，平均每条约 68 s）。
- `metadata/language_instruction` 全是 `test01`，**没有可用的语言指令**。
- 抽样文件中 `is_intervene` 全为 False。

### 1.2 字段

| 组 | 键（`*_align` 已与相机时间戳对齐，长度为 T；`*_raw` 长度为 2T，约 58 Hz） | 形状 | 说明 |
|---|---|---|---|
| 相机 | `camera_observations/color_images/{camera_head, camera_left, camera_right}` | (T,) JPEG，640×480，BGR | head = Orbbec Gemini336L（主视角）；left / right = RealSense D405（腕部） |
| 相机 | `camera_observations/depth_images/*` | (T,) 编码字节 | 深度，暂不使用 |
| **力** | `force_observations/force_{left,right}_end_effector_align/data` | **(T, 6) float64** | **左右臂末端六维力** |
| 力 | `force_observations/force_{left,right}_end_effector_raw/data` | (2T, 6) | 高频原始力 |
| 从臂 | `puppet/end_effector_{left,right}_pose_align/data` | (T, 7) | xyz + 四元数（单位范数） |
| 从臂 | `puppet/end_effector_{left,right}_position_align/data` | (T, 1) | 夹爪开度，约 0.012–0.90 |
| 从臂 | `puppet/arm_{left,right}_position_align/data` | (T, 7) | 7 个关节角 |
| 主臂 | `master/arm_{left,right}_position_align/data`、`master/end_effector_*_position_align` | (T, 7)、(T,) | 遥操作指令（关节角、夹爪 0–0.99） |
| 其他 | `puppet/head_*`、`master/waist_*` | | 基本恒定，忽略 |
| 标定 | `camera_extrinsics/*`、`camera_intrinsics/*` | | 暂不使用 |

### 1.3 六维力确认：**有**

在 82 条抽样中全部存在且非零。以 `20260916/3858357` 为例（单位推测为 N 和 N·m）：

| | Fx | Fy | Fz | Tx | Ty | Tz |
|---|---|---|---|---|---|---|
| 左臂 mean | -3.73 | **19.61** | -2.25 | -0.71 | 0.14 | 0.00 |
| 左臂 std | 2.00 | 1.67 | 2.69 | 0.33 | 0.36 | 0.20 |
| 右臂 mean | 8.19 | -4.09 | **-9.28** | 0.95 | -0.14 | 0.26 |
| 右臂 range Fz | | | -29.6 ~ -2.7 | | | |

需要处理的几个特性：

1. **存在静态零偏**：左臂刚开始、没有接触的前 100 帧里 Fy 就约为 20.5 N，而且不同 episode 都在 18–21 N 左右，看起来是传感器零偏。最终的模型输入应该是“相对零偏的变化量”，而不是绝对读数。
2. **噪声大**：逐帧 \(\|\Delta F_{xyz}\|\) 的中位数约 1.1 N（MAD 约 0.33，p99 约 4 N，最大约 15 N）。直接拿逐帧差分做门控基本只能看到噪声，**必须先做低通滤波**，或者在更长的时间窗口上做差分。
3. **四元数有符号翻转**（每条 2–6 次），直接做 min-max 归一化或回归会出问题，需要改用 rot6d 表示，或者先做半球连续化处理。
4. **坐标系已确认是 TCP 系**。传感器安装在法兰（TCP）和夹爪之间，读数里包含夹爪自重；由于坐标系跟着末端一起转，自重分量会随姿态变化（详见 1.4）。

### 1.4 重力影响评估（为什么不需要完整的重力补偿）

用 4 条 episode、左右两臂，按标准负载辨识模型做了最小二乘拟合：

\[
F = R^\top w + b,\qquad T = p \times (R^\top w) + c
\]

其中 \(R\) 是 TCP 在基座系下的姿态（来自 pose 中的四元数），\(w\) 是工具重力，\(b, c\) 是零偏，\(p\) 是质心位置。

| 结果 | 数值 |
|---|---|
| 姿态（重力项）能解释的力方差 R² | 0.04–0.46（大多数在 0.1–0.2） |
| 能解释的力矩方差 R² | 0.02–0.22 |
| 不同 episode 估出的工具质量 | 0.5–1.4 kg，**很不稳定** |
| 四元数顺序 | 力的拟合区分度很弱（wxyz 略好）；**更可靠的依据是头部位姿** `[0.0003, 0.325, 0.0001, 0.9457]`：按 xyzw 解读是向下俯视约 38°，按 wxyz 解读是约 180° 旋转，后者不合理。因此**采用 xyzw** |

结论：

- 这个任务里末端姿态变化不大，**重力随姿态变化的部分只占力变化的一小部分**；主要变化来自接触、加减速和噪声。
- 因此**不需要做完整的负载辨识和重力补偿**。“去掉每条 episode 的零偏”加上“差分”就够用。另外，夹爪抓取电池后负载会改变，这时完整补偿反而容易出错。

### 1.5 差分怎么用

差分会消掉零偏和缓慢变化的重力分量，但同时也会消掉**持续接触的力水平**：比如压电池盖时力很大但几乎不变，这时 ΔF≈0。所以按用途分开处理：

| 用途 | 推荐做法 | 原因 |
|---|---|---|
| **接触门控** \(g_t\) | 先低通滤波，再对窗口做差分：\(\text{raw}_t=\max_{\text{arm}}\|(\bar F_t-\bar F_{t-k})/\sigma\|\)，k 取 3–5 步（约 0.2–0.35 s） | 门控需要的是“接触事件”，差分正好合适；窗口差分能压住约 1 N 的逐帧噪声 |
| **当前力槽位**（条件输入） | \(F_t - F_{\text{baseline}}\)（减去 episode 开头静止段的均值）并滤波；可再拼接窗口差分 \(\Delta_k F_t\) | 保留持续接触的力水平；剩余的重力项很小（见 1.4） |
| **未来力槽位**（预测目标） | \(F_{t+H} - F_t\)，即在一个动作块时间内力的变化 | 与零偏和重力无关，预测目标更干净；语义上就是“这一段动作会带来多大的力变化” |

---

## 2. 与 Dream-Tac 输入要求的差异

| Dream-Tac 现有要求（Franka） | tienyi 数据 | 处理方式 |
|---|---|---|
| 单臂，2 路相机（primary + wrist） | 双臂，3 路相机（head + 左腕 + 右腕） | 采用 ALOHA 式 3 视角布局（模型已支持 `*_wrist_image2_latent_idx`） |
| 触觉 = 两路 Xense 触觉**图像** | **没有触觉图像**，只有两路六维力**向量** | 力作为向量模态注入潜帧，并沿用 CASA（见第 4 节） |
| 动作 7 维（位姿 6 + 夹爪） | 双臂位姿（四元数）+ 夹爪 | 新定义 20 维动作 |
| 原始 HDF5 键 `observations/images/cam_front` 等 | 键名完全不同 | 编写新的预处理脚本 |
| 任务名来自目录名 | 指令是 `test01` | 人工指定英文指令 |
| 数据集把视频全部载入内存 | 约 540 万帧 × 3 路相机 | 第一阶段用子集，后续改为懒加载 |

---

## 3. 数据转换设计

### 3.1 时间轴

- 以相机时间轴（`*_align`，约 29 Hz）作为主时间轴。
- **时间下采样 stride=2，约 14.5 Hz**：一方面相邻帧冗余，另一方面能把内存减半；对应 `chunk_size=16`，约 1.1 s，与原 Franka 配置的 20 步 @25 Hz（0.8 s）在同一量级。stride 和 chunk_size 都写成参数，方便后续调整。
- 力信号可以利用 `*_raw`（约 58 Hz）构建历史窗口（见 4.2 中的可选项）。

### 3.2 各模态定义

| 模态 | 定义 | 维度 |
|---|---|---|
| proprio \(s_t\) | 每只臂 [xyz(3), rot6d(6), gripper(1)]，左右拼接（**用真实从臂状态**，而不是 Franka 脚本里“上一步动作”的近似） | 20 |
| action \(a_t\) | \(s_{t+1}\)，即下一时刻的从臂绝对 EE 位姿 + 夹爪，格式同 proprio | 20 |
| 备选 action | 主臂关节角 7 + 主臂夹爪 1，双臂共 16 维（关节空间，不依赖 IK） | 16 |
| 图像 | head → primary；left → wrist；right → wrist2。JPEG 解码，**BGR→RGB**，resize 到 224（直接存 224，省掉一次 resize） | 3×224×224 |
| 力 \(f_t\) | 左右各 6 维，拼成 12 维（处理流程见 3.3） | 12 |
| 指令 | 例如 `"install the small battery on the back and then install the small battery cover"` | T5 |

### 3.3 力信号预处理

1. 取 `force_{left,right}_end_effector_align/data`，转为 float32。
2. **去零偏**（TCP 系，不做重力补偿，原因见 1.4）：减去每条 episode 开头静止段（例如前 0.5 s）的均值，得到 \(\tilde F\)。原始绝对值另存一份 `force_abs`，供消融使用。
3. **因果滤波（已实现）**：取原始约 59 Hz 读数（`*_raw`，带时间戳），在 \((t-\text{一个控制周期}, t]\) 内取平均，t 为该步图像的相机时间戳。只用过去的数据，部署端用同一函数复现（`experiments/robot/tienyi/force_filter.py`，见交接文档第 7 节）。最初设想的零相位滤波会用到未来数据，部署时无法复现，已放弃。
4. **构造槽位内容**（见 1.5）：当前力槽位用 \(\tilde F^{\text{filt}}_t\)；未来力槽位的目标用 \(\tilde F^{\text{filt}}_{t+H}-\tilde F^{\text{filt}}_t\)。
5. **归一化**：先按 p1/p99 截断，再按 Dream-Tac 统一的 min-max 归一化到 [-1, 1]（统计量写入 `dataset_statistics_*.json`，新增 `force_min/max/mean/std`）。力和力矩量级相差约 10 倍，所以**必须逐维归一化**。
6. **门控统计**：在全数据集上计算 \(\text{raw}_t = \max_{\text{arm}} \|(\tilde F^{\text{filt}}_t-\tilde F^{\text{filt}}_{t-k})/\sigma_F\|\) 的 median 和 MAD，写进统计文件，代替 `scalar_gate_from_raw` 里针对图像标定的常数 0.002 / 0.001。

### 3.4 过滤

- 排除 `*_fail` 目录（Franka 配置只用 demo；以后可以作为 rollout 数据训练价值函数）。
- 去掉长度低于某个阈值的 episode（例如 < 300 帧，最短的只有 130 帧）。
- 去掉含 `is_intervene=True` 的片段（如果存在）。
- 可选：裁掉开头和结尾的静止段（位姿和夹爪都不变）。

### 3.5 预处理输出格式（沿用 Dream-Tac 约定，便于复用 T5、统计、train/val 切分逻辑）

```
<out_base>/back_battery_and_cover/
├── train/episode_{i}.hdf5 + episode_{i}_{cam_high,cam_left_wrist,cam_right_wrist}.mp4
├── val/...
├── preprocessing_metadata.json      # 含 task_name、stride、源文件映射
└── t5_embeddings.pkl
episode_{i}.hdf5:
  attrs/task_name
  observations/qpos           (T', 20)
  observations/force          (T', 12)   # 去偏置 + 滤波后
  observations/force_abs      (T', 12)   # 原始绝对值（消融用）
  observations/video_paths/{cam_high, cam_left_wrist, cam_right_wrist}
  action                      (T', 20)
```

相机命名与 `ALOHADataset` 保持一致（cam_high / cam_left_wrist / cam_right_wrist），这样可以直接复用它的 MP4 读取逻辑。

---

## 4. 六维力如何加入 Dream-Tac

### 4.1 三种方案对比

| 方案 | 做法 | 改动量 | 优点 | 缺点 |
|---|---|---|---|---|
| A. 力拼进 proprio | proprio = [位姿 20, 力 12] = 32 维；未来 proprio 中自然包含未来力 | 只改数据，不改模型 | 最快跑通，模型可以“预测未来力” | 力被淹没在本体向量里；CASA 无法单独作用在力上 |
| **B. 独立力潜帧 + CASA（推荐）** | 新增“当前力”“未来力”两个潜帧，按 proprio 的方式注入；`tactile_latent_t_indices` 指向这两个潜帧；门控改由力的变化计算 | 新数据集类 + 模型侧少量通用改动 | 与 Dream-Tac 的设计（触觉独立槽位 + 未来触觉预测 + 接触门控）一一对应，便于做消融对比 | 需要改 4–5 处代码 |
| C. 力渲染成伪触觉图像 | 把 12 维力画成热图或柱状图，当成触觉图像 | 几乎零模型改动 | 能直接复用全部触觉代码 | VAE 编码“画出来的图”效率低、精度损失大，不推荐 |

**推荐路线**：先做 **Phase 1（无力基线）** 打通全链路，然后做 **Phase 2（方案 B）**；方案 A 作为一个低成本的对照实验。

### 4.2 方案 B 的潜序列布局（与 cut_banana 一样是 12 个潜帧，计算量相同）

| idx | 内容 | 角色 |
|---|---|---|
| 0 | blank | 条件 |
| 1 | proprio \(s_t\)（20 维） | 条件 |
| 2 | 左腕图像 | 条件 |
| 3 | 右腕图像 | 条件 |
| 4 | head 图像 | 条件 |
| **5** | **当前力**：\(f_t\)（12 维）；可选扩展为最近 K 步历史，例如 K=16 → 192 维 | 条件 + CASA 触觉 token |
| 6 | 动作块 (16×20) | 预测 |
| 7 | 未来 proprio \(s_{t+H}\) | 预测 |
| 8 | 未来左腕图像 | 预测 |
| 9 | 未来右腕图像 | 预测 |
| 10 | 未来 head 图像 | 预测 |
| **11** | **未来力**：\(f_{t+H}\)；可选为整段未来力轨迹 \(f_{t+1..t+H}\)（16×12=192 维） | 预测 + CASA 触觉 token |

对应配置：`state_t=12`、`tokenizer.chunk_duration=45`、`min/max_num_conditional_frames=6`、`tactile_latent_t_indices=(5, 11)`、`use_tactile_self_attn_bias=True`。
所有向量都远小于每个潜帧的 12544 个元素，平铺注入没有问题。

### 4.3 方案 B 需要的代码改动（全部是新增或可开关的改动，不影响原有实验）

1. **新预处理脚本** `cosmos_policy/experiments/robot/tienyi/preprocess_tienyi_force_data.py`：实现第 3 节的全部内容。
2. **新数据集类** `cosmos_policy/datasets/tienyi_dataset.py`（`TienyiForceDataset`）
   - 参考 `ALOHADataset` 的三视角组装方式，以及 `FrankaDataset` 插入额外槽位、平移下标的写法；
   - 输出新字段 `force`、`future_force`，以及 `current_force_latent_idx=5`、`future_force_latent_idx=11`；
   - 门控：`tactile_self_attn_gate = scalar_gate_from_raw(raw_force_t, median=stats.median, mad=stats.mad)`；
   - 参数 `use_force: bool`，用于消融（False 时退化为 10 槽的无力布局）；
   - 计算力的统计量并归一化，扩展 `calculate_dataset_statistics`，或在类内部单独处理 `force` 这个 key。
3. **模型侧**（通用改动：只要 batch 里有对应的 key 就生效）
   - `policy_video2world_model.py::get_data_and_condition`：
     - 用 `replace_latent_with_proprio` 把 `force` 注入 `gt_frames[current_force_latent_idx]`，并把条件掩码置 1；
     - 把 `future_force_latent_idx` 加入“需要预测”的掩码分支，写法与 `future_proprio` 相同。
   - `policy_text2world_model.py::training_step / compute_loss_with_epsilon_and_sigma`：
     - 在 `x0` 中注入 `future_force`；
     - 新增 `future_force` 的 MSE / L1 日志，并把它加入各个 loss mask 分支（写法与 `future_tactile_*` 相同）。
   - 推理阶段 `get_x0_fn_from_batch`：注入当前力作为条件。
4. **CASA**：`MiniTrainDIT` 和注意力内核**不需要改**，只改配置里的 `tactile_latent_t_indices` 即可。
5. **实验配置**：在 `cosmos_policy_experiment_configs.py` 中新增并注册：
   - `cosmos_predict2_2b_480p_tienyi_battery_no_force`（10 槽，Phase 1 基线）；
   - `cosmos_predict2_2b_480p_tienyi_battery_force`（12 槽 + CASA）；
   - `..._force_no_attn_bias`（12 槽，不加 CASA）；
   - 可选 `..._force_in_proprio`（方案 A）。
6. **推理 / 开环评估**（跑通训练后再做）：在 `cosmos_utils.get_action` 中加入 tienyi 布局分支，并从潜帧 11 还原未来力；新写 `run_tienyi_openloop.py`，输出动作曲线、未来图像、**预测力与真实力的对比曲线**。

### 4.4 方案 A 的改动（对照实验）

只需在预处理或数据集中设置 `qpos = concat(pose20, force12)`，并保持 `action` 为 20 维；用 10 槽布局，CASA 关闭。

---

## 5. 实施计划

### Phase 0：环境与资源

- [ ] 按 README 用 Docker 或 `uv sync --extra cu128 --group franka` 搭建环境。当前机器的 base Python 连 h5py 都没有。
- [ ] 下载 Cosmos-Predict2-2B-Video2World checkpoint（`model-480p-16fps.pt`）、Wan2.1 VAE、T5 编码器；在配置中修改 `checkpoint.load_path`。
- [ ] 多卡资源：见下面的“2 小时训练资源估算”。
- [ ] 用 `--dryrun` 校验配置。

#### 2 小时训练资源估算（理论估算，开训后要用实测速度校正）

- 每个样本 12 个潜帧 × 14×14 patch，约 2352 个 token。2B DiT 前向加反向约 3×10¹³ FLOPs；再算上 VAE 编码 45 帧和数据加载，**单张 A100 约 2.5–3.5 样本/秒**。
- 数据并行**不会让每步变快**：每步时间只取决于每卡的 batch，增加卡数只是增大全局 batch。

| 方案 | 每卡 batch | 每步时间（估计） | 2 小时步数 | 全局 batch | 评价 |
|---|---|---|---|---|---|
| 8×A100 | 16 | 约 5–6 s | 约 1.2k–1.4k | 128 | 能看出趋势 |
| **8×A100（推荐最低配置）** | 8 | 约 2.5–3 s | 约 2.4k–2.8k | 64 | 步数接近 README 示例中的 `iter_3000` |
| **16×A100（推荐）** | 8 | 约 2.5–3 s | 约 2.4k–2.8k | 128 | 步数和全局 batch 都接近原始配置，2 小时能看到可信的开环效果 |
| 32×A100 | 8 | 同上 | 同上 | 256 | 步数不再增加，收益有限 |

配合 2 小时训练必须修改的配置：

- `scheduler.warm_up_steps`：从 2000 改为约 100–200，否则整个 2 小时都还在 warmup；`cycle_lengths` 改为约 3000。
- `checkpoint.save_iter`：从 50 改为约 500。2B 模型加优化器状态，每个 checkpoint 约 20–30 GB。
- **内存**：`FrankaDataset` / `ALOHADataset` 的写法是**每个 rank 都把全部视频载入内存**，单节点 8 个 rank 就是 8 份。按每条约 440 MB（stride=2）、节点 RAM 约 900 GB 计算，单节点**最多约 150–200 条**；或者改用 stride=3，或者使用 memmap（多个 rank 通过 page cache 共享同一份数据）。
- 2 小时**只是纯训练时间**。第一次跑还要加上：搭环境、下载 checkpoint 和 T5、预处理约 150 条数据（网络盘，32 进程大约 10 分钟级）、调试。实际第一次出结果按半天预估比较稳妥。

2 小时能看到什么、看不到什么：

- **能看到**：action loss 明显收敛；开环动作曲线能跟上 GT 的趋势；未来图像预测大致合理；未来力预测开始学到接触段的变化。
- **还不够**：判断“加力或 CASA 是否带来提升”。这需要更多步数、更多数据，以及对照组同条件训练。建议 2 小时版本先跑 `no_force` 和 `force` 两组做初步对比。

### Phase 1：无力基线，打通全链路（建议先只用 1 个日期目录，约 150 条，受单节点内存限制）

- [ ] 写好预处理脚本，先在 2–3 条数据上跑通，用肉眼检查 MP4 的颜色（BGR/RGB）和动作曲线。
- [ ] 生成 T5 嵌入（修改指令来源，或直接复用 `save_franka_t5_text_embeddings` 读取 metadata 中的 task_name）。
- [ ] 数据集类以 `use_force=False` 跑通，可视化一个样本的 `video` 各槽位（可参考 `libero_dataset.create_augmentation_visualization`）。
- [ ] 跑 `DEBUGGING=True`（只加载 1 个文件）的单卡短训练，确认 loss 下降、checkpoint 能正常保存。

### Phase 2：加入六维力（方案 B）

- [ ] 预处理中加入力的去偏置、滤波、统计和门控统计。
- [ ] 数据集输出 force 槽位和门控；模型侧加入 force 注入和 loss 日志。
- [ ] 单元检查：注入后再提取，力向量应能还原；门控分布不应集中在 0.15 或 1.0 两端。
- [ ] 训练 `..._force`，同时训练 `..._no_force` / `..._force_no_attn_bias` / `..._force_in_proprio` 作为对照。

### Phase 3：扩展到全量数据

- [ ] 内存估算：224² 下三路相机每步约 451 KB；stride=2 时每条约 985 步，约 440 MB。250 条约 110 GB（本机 RAM 962 GB，可以全部载入）；**全部 2748 条约 1.2 TB，无法全部载入**。
- [ ] 改为懒加载：预处理时把每条 episode 的帧存成 uint8 `.npy`，训练时用 memmap 随机读取（单帧读取比 MP4 解码快得多）；或者在 `__getitem__` 中按需读取 JPEG 并解码。
- [ ] 预处理用多进程并行；源数据在网络盘上，读取较慢（读 82 个文件的小数组就用了约 3.5 分钟）。

### Phase 4：评估

- [ ] 开环评估：动作 MSE、未来图像 IoU、未来力的 MSE / 相关系数。
- [ ] 消融：无力 / 力在 proprio 中 / 独立力槽位 / 独立力槽位 + CASA。

---

## 6. 风险与待确认事项

| 项 | 说明 |
|---|---|
| 力的坐标系与重力补偿 | **已确认是 TCP 系**。拟合显示重力项只能解释约 10–20% 的力变化，因此采用“去零偏 + 滤波 + 差分”，不做完整补偿（见 1.4、1.5） |
| 四元数顺序 | 根据头部位姿判断为 xyzw（预处理脚本的默认值，可用 `--quat_order` 切换），最好再找采集方确认一次 |
| 动作空间 | EE 位姿还是关节角，需要和部署端的控制接口一致 |
| `_fail` 目录 | **已确认是失败数据**，训练 demo 时排除；以后可以作为 rollout 数据训练价值函数 |
| 语言指令 | 当前只有 `test01`，需要人工给出 |
| 算力 | 2 小时训练建议 16×A100（最低 8×A100），见 Phase 0 的估算 |
| 门控标定 | 力信号的门控常数要用数据统计来定，不能直接沿用图像版本的常数 |
