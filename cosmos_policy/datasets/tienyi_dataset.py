# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TienyiForceDataset：天翼双臂（3 相机 + 双臂 6 维力）数据的 Cosmos Policy 训练数据集。

输入：`experiments/robot/tienyi/preprocess_tienyi_force_data.py` 的输出目录
    <data_dir>/
        force_statistics.json            力归一化统计 + 门控统计（仅 train 集）
        preprocessing_metadata.json
        train/episode_XXXXX.hdf5         action (T,20), observations/{qpos (T,20), force (T,12), force_gate_raw (T,)}
        train/episode_XXXXX_cam_{high,left_wrist,right_wrist}.mp4   224x224
        val/...

潜变量帧（slot）布局。每个 slot 是 1 个 latent frame = 4 个像素帧（第 0 个 blank 只占 1 个像素帧）：

    use_force=True（12 slots，state_t=12，chunk_duration=45，条件帧 6）
        0 blank | 1 proprio | 2 左腕 | 3 右腕 | 4 头部 | 5 force_cur
        6 action | 7 future proprio | 8 future 左腕 | 9 future 右腕 | 10 future 头部 | 11 future force
    use_force=False（10 slots，state_t=10，chunk_duration=37，条件帧 5）
        0 blank | 1 proprio | 2 左腕 | 3 右腕 | 4 头部
        5 action | 6 future proprio | 7 future 左腕 | 8 future 右腕 | 9 future 头部

向量模态（proprio / force / action / future_*）在像素空间放全黑帧占位，
由模型在 latent 空间用 replace_latent_with_proprio / replace_latent_with_action_chunk 平铺注入。

力的归一化：
    force（当前力，已减起始零偏）：按 train 集 p01/p99 裁剪后线性映射到 [-1, 1]
    future_force（默认 delta 模式 = F[t+H] - F[t]）：除以 train 集 |ΔF| 的逐维 p99，再裁剪到 [-1, 1]
CASA 门控：tactile_self_attn_gate = scalar_gate_from_raw(force_gate_raw[t], median, mad)
"""

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.aloha_dataset import load_video_as_images
from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import get_hdf5_files, preprocess_image
from cosmos_policy.utils.tactile_self_attn_gate import scalar_gate_from_raw

CAMERA_NAMES = ("cam_left_wrist", "cam_right_wrist", "cam_high")
STATS_FILENAME = "dataset_statistics_tienyi.json"
# 实验继承自 cosmos_predict2_2b_480p_libero，LazyConfig 会把 LIBERODataset 的这些参数合并进来；本数据集不使用
_INHERITED_LIBERO_KWARGS = {
    "use_wrist_images", "use_third_person_images", "use_proprio", "rollout_data_dir",
    "demonstration_sampling_prob", "success_rollout_sampling_prob", "treat_success_rollouts_as_demos",
    "return_value_function_returns", "gamma",
}

# LazyConfig 会为 DataLoader 和 DistributedSampler 各实例化一次数据集；
# 进程内按加载参数缓存原始 episode，避免视频在内存里存两份。
_EPISODE_CACHE: dict = {}


def _read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """按需解码单帧（RGB uint8）。用于 load_videos_to_memory=False 模式。"""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"无法读取 {video_path} 第 {frame_idx} 帧")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _video_num_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _minmax_to_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """线性映射 [lo, hi] -> [-1, 1]；常量维（hi == lo）映射为 0，避免除零。"""
    span = hi - lo
    safe = np.where(span > 1e-8, span, 1.0)
    out = 2.0 * (x - lo) / safe - 1.0
    return np.where(span > 1e-8, out, 0.0).astype(np.float32)


def _atomic_write_json(path: str, obj: dict) -> None:
    """多卡同时初始化时每个 rank 都可能写统计文件，先写临时文件再 rename 保证读到的总是完整文件。"""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


class TienyiForceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        t5_text_embeddings_path: str,
        is_train: bool = True,
        chunk_size: int = 16,
        final_image_size: int = 224,
        num_duplicates_per_image: int = 4,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_force: bool = True,
        future_force_mode: str = "delta",
        action_key: str = "action",
        proprio_key: str = "observations/qpos",
        max_episodes: int = 0,
        load_videos_to_memory: bool = True,
        num_load_threads: int = 16,
        **inherited_kwargs,
    ):
        """
        Args:
            data_dir: 预处理输出目录（含 train/ val/ 与 force_statistics.json）
            t5_text_embeddings_path: save_franka_t5_text_embeddings.py 生成的 t5_embeddings.pkl
            chunk_size: 动作块长度 H（future 帧取 t+H）
            num_duplicates_per_image: Wan2.1 VAE 时间压缩倍数，必须为 4
            use_force: True -> 12 slot 布局（含当前力/未来力）；False -> 10 slot 无力基线
            future_force_mode: "delta"（F[t+H]-F[t]，推荐）或 "absolute"（F[t+H]，与当前力同归一化）
            action_key / proprio_key: 默认笛卡尔 + rot6d + 夹爪（20 维）；可改成 action_joint / observations/qpos_joint（16 维）
            max_episodes: >0 时只加载前 N 个 episode（调试 / 控制内存）
            load_videos_to_memory: True 时每个 rank 把全部视频解码进内存（约 450 MB/episode，取样最快）；
                False 时每个样本按需从 MP4 解码 6 帧（内存几乎为 0，适合全量 ~2700 episode，需更多 num_workers）
        """
        unknown = set(inherited_kwargs) - _INHERITED_LIBERO_KWARGS
        assert not unknown, f"TienyiForceDataset 不支持参数 {sorted(unknown)}"
        assert num_duplicates_per_image == 4, "Wan2.1 VAE 时间压缩为 4，num_duplicates_per_image 必须为 4"
        assert future_force_mode in ("delta", "absolute")
        self.data_dir = data_dir
        self.is_train = is_train
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_force = use_force
        self.future_force_mode = future_force_mode
        self.action_key = action_key
        self.proprio_key = proprio_key
        self.load_videos_to_memory = load_videos_to_memory

        with open(t5_text_embeddings_path, "rb") as f:
            self.t5_text_embeddings = pickle.load(f)

        with open(os.path.join(data_dir, "force_statistics.json")) as f:
            self.force_stats = json.load(f)

        hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        if max_episodes > 0:
            hdf5_files = hdf5_files[:max_episodes]
        assert len(hdf5_files) > 0, f"{data_dir} 下没有找到 {'train' if is_train else 'val'} episode"

        # ---------------- 加载 episode（同进程内同参数只加载一次） ----------------
        cache_key = (
            os.path.abspath(data_dir), is_train, max_episodes, action_key, proprio_key, final_image_size,
            load_videos_to_memory,
        )
        if cache_key not in _EPISODE_CACHE:
            with ThreadPoolExecutor(max_workers=num_load_threads) as pool:
                _EPISODE_CACHE[cache_key] = list(
                    tqdm(pool.map(self._load_episode, hdf5_files), total=len(hdf5_files), desc="Loading Tienyi episodes")
                )
        # 浅拷贝：图像数组共享，actions/proprio 在下面归一化时生成新数组
        self.data = {i: dict(ep) for i, ep in enumerate(_EPISODE_CACHE[cache_key])}
        self.num_episodes = len(self.data)

        # ---------------- 归一化统计 ----------------
        self.dataset_stats = self._load_or_compute_statistics()
        for ep in self.data.values():
            if normalize_actions:
                ep["actions"] = _minmax_to_unit(
                    ep["actions"], np.array(self.dataset_stats["actions_min"]), np.array(self.dataset_stats["actions_max"])
                )
            if normalize_proprio:
                ep["proprio"] = _minmax_to_unit(
                    ep["proprio"], np.array(self.dataset_stats["proprio_min"]), np.array(self.dataset_stats["proprio_max"])
                )

        # ---------------- 全局 step -> (episode, t) ----------------
        self._step_to_episode_map = []
        for ep_idx, ep in self.data.items():
            self._step_to_episode_map.extend((ep_idx, t) for t in range(ep["num_steps"]))
        self.num_steps = len(self._step_to_episode_map)

        # ---------------- slot 下标 ----------------
        self.layout = self._build_layout(use_force)
        print(
            f"[TienyiForceDataset] {'train' if is_train else 'val'}: {self.num_episodes} episodes, "
            f"{self.num_steps} steps, use_force={use_force}, slots={len(self.layout['slots'])}"
        )

    # ------------------------------------------------------------------
    # 加载 / 统计
    # ------------------------------------------------------------------
    def _load_episode(self, hdf5_path: str) -> dict:
        with h5py.File(hdf5_path, "r") as f:
            actions = f[self.action_key][:].astype(np.float32)
            proprio = f[self.proprio_key][:].astype(np.float32)
            force = f["observations/force"][:].astype(np.float32)
            gate_raw = f["observations/force_gate_raw"][:].astype(np.float32)
            command = f.attrs["task_name"]
            if isinstance(command, bytes):
                command = command.decode()
        stem = os.path.splitext(hdf5_path)[0]
        video_paths = {cam: f"{stem}_{cam}.mp4" for cam in CAMERA_NAMES}
        if self.load_videos_to_memory:
            images = {
                cam: load_video_as_images(path, resize_size=self.final_image_size) for cam, path in video_paths.items()
            }
            video_lens = [len(v) for v in images.values()]
        else:
            images = None
            video_lens = [_video_num_frames(path) for path in video_paths.values()]
        n = min(len(actions), len(proprio), len(force), *video_lens)
        return dict(
            file_path=hdf5_path,
            command=command,
            num_steps=n,
            actions=actions[:n],
            proprio=proprio[:n],
            force_raw=force[:n],
            force_gate_raw=gate_raw[:n],
            images=None if images is None else {k: v[:n] for k, v in images.items()},
            video_paths=video_paths,
        )

    def _get_frame(self, ep: dict, cam: str, t: int) -> np.ndarray:
        if ep["images"] is not None:
            return ep["images"][cam][t]
        frame = _read_video_frame(ep["video_paths"][cam], t)
        if frame.shape[0] != self.final_image_size or frame.shape[1] != self.final_image_size:
            frame = cv2.resize(frame, (self.final_image_size, self.final_image_size), interpolation=cv2.INTER_AREA)
        return frame

    def _load_or_compute_statistics(self) -> dict:
        """
        actions/proprio 的 min/max 及 future_force 缩放尺度。
        train 集首次运行时计算并写入 <data_dir>/dataset_statistics_tienyi.json，val 集与推理直接读取。
        future_force_scale 依赖 chunk_size，按 key f"future_force_scale_chunk{H}" 分别保存。
        """
        path = os.path.join(self.data_dir, STATS_FILENAME)
        stats = {}
        if os.path.exists(path):
            with open(path) as f:
                stats = json.load(f)

        ff_key = f"future_force_scale_chunk{self.chunk_size}"
        need = [f"{k}_{s}" for k in ("actions", "proprio") for s in ("min", "max")] + [ff_key]
        missing = [k for k in need if k not in stats]
        if missing and stats.get("action_key", self.action_key) != self.action_key:
            raise ValueError(f"{path} 由 action_key={stats['action_key']} 生成，与当前 {self.action_key} 不符，请删除后重算")
        if not missing:
            return stats
        if not self.is_train:
            print(f"[TienyiForceDataset] 警告：{path} 缺少 {missing}，val 集将用自身数据计算（应先构建 train 集）")

        all_actions = np.concatenate([ep["actions"] for ep in self.data.values()])
        all_proprio = np.concatenate([ep["proprio"] for ep in self.data.values()])
        for name, arr in (("actions", all_actions), ("proprio", all_proprio)):
            stats[f"{name}_min"] = arr.min(0).tolist()
            stats[f"{name}_max"] = arr.max(0).tolist()
            stats[f"{name}_mean"] = arr.mean(0).tolist()
            stats[f"{name}_std"] = arr.std(0).tolist()
            stats[f"{name}_median"] = np.median(arr, 0).tolist()

        # future force 缩放：|F[t+H] - F[t]| 的逐维 p99（末尾不足 H 的取最后一帧，与 __getitem__ 一致）
        deltas = []
        for ep in self.data.values():
            f = ep["force_raw"]
            fut = np.minimum(np.arange(len(f)) + self.chunk_size, len(f) - 1)
            deltas.append(f[fut] - f)
        deltas = np.abs(np.concatenate(deltas))
        stats[ff_key] = np.maximum(np.percentile(deltas, 99, axis=0), 1e-3).tolist()

        stats["action_key"] = self.action_key
        stats["proprio_key"] = self.proprio_key
        if self.is_train:
            _atomic_write_json(path, stats)
        return stats

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    @staticmethod
    def _build_layout(use_force: bool) -> dict:
        slots = ["blank", "proprio", "wrist_left", "wrist_right", "head"]
        if use_force:
            slots.append("force")
        slots += ["action", "future_proprio", "future_wrist_left", "future_wrist_right", "future_head"]
        if use_force:
            slots.append("future_force")
        idx = {name: i for i, name in enumerate(slots)}
        return dict(slots=slots, idx=idx)

    @property
    def num_latent_frames(self) -> int:
        return len(self.layout["slots"])

    @property
    def num_pixel_frames(self) -> int:
        return 1 + (self.num_latent_frames - 1) * self.num_duplicates_per_image

    # ------------------------------------------------------------------
    # 力归一化
    # ------------------------------------------------------------------
    def normalize_force(self, force: np.ndarray) -> np.ndarray:
        lo = np.array(self.force_stats["force_p01"], dtype=np.float32)
        hi = np.array(self.force_stats["force_p99"], dtype=np.float32)
        return np.clip(_minmax_to_unit(np.clip(force, lo, hi), lo, hi), -1.0, 1.0)

    def normalize_future_force(self, future: np.ndarray, current: np.ndarray) -> np.ndarray:
        if self.future_force_mode == "absolute":
            return self.normalize_force(future)
        scale = np.array(self.dataset_stats[f"future_force_scale_chunk{self.chunk_size}"], dtype=np.float32)
        return np.clip((future - current) / scale, -1.0, 1.0).astype(np.float32)

    def unnormalize_future_force(self, pred: np.ndarray, current_raw: np.ndarray) -> np.ndarray:
        """推理/评估用：把模型预测的 future_force（[-1,1]）还原为牛顿/牛米。"""
        if self.future_force_mode == "absolute":
            lo = np.array(self.force_stats["force_p01"], dtype=np.float32)
            hi = np.array(self.force_stats["force_p99"], dtype=np.float32)
            return (pred + 1.0) / 2.0 * (hi - lo) + lo
        scale = np.array(self.dataset_stats[f"future_force_scale_chunk{self.chunk_size}"], dtype=np.float32)
        return current_raw + pred * scale

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self._step_to_episode_map[idx]
        ep = self.data[ep_idx]
        n = ep["num_steps"]
        future_t = min(t + self.chunk_size, n - 1)
        L = self.layout["idx"]

        # ---------- 像素帧：先构造 unique 帧 + 重复次数，一次性做增广后再展开 ----------
        H = W = self.final_image_size
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        frame_by_slot = {
            "blank": blank,
            "proprio": blank,
            "wrist_left": self._get_frame(ep, "cam_left_wrist", t),
            "wrist_right": self._get_frame(ep, "cam_right_wrist", t),
            "head": self._get_frame(ep, "cam_high", t),
            "force": blank,
            "action": blank,
            "future_proprio": blank,
            "future_wrist_left": self._get_frame(ep, "cam_left_wrist", future_t),
            "future_wrist_right": self._get_frame(ep, "cam_right_wrist", future_t),
            "future_head": self._get_frame(ep, "cam_high", future_t),
            "future_force": blank,
        }
        frames = np.stack([frame_by_slot[s] for s in self.layout["slots"]])
        repeats = torch.tensor([1] + [self.num_duplicates_per_image] * (self.num_latent_frames - 1))
        images = preprocess_image(
            frames,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )  # (C, num_latent_frames, H, W)
        images = torch.repeat_interleave(images, repeats, dim=1)  # (C, num_pixel_frames, H, W)
        assert images.shape[1] == self.num_pixel_frames

        # ---------- 向量模态 ----------
        action_chunk = get_action_chunk_with_padding(
            actions=ep["actions"], relative_step_idx=t, chunk_size=self.chunk_size, num_steps=n
        )
        proprio = ep["proprio"][t]
        future_proprio = ep["proprio"][future_t]

        sample = {
            "video": images,
            "actions": torch.as_tensor(action_chunk, dtype=torch.float32),
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[ep["command"]]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, H, W),
            "image_size": H * torch.ones(4),
            "proprio": torch.as_tensor(proprio, dtype=torch.float32),
            "future_proprio": torch.as_tensor(future_proprio, dtype=torch.float32),
            "__key__": idx,
            # 只用演示数据训练策略 + 世界模型；不训练价值函数
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "value_function_return": float("-100"),
            "next_action_chunk": torch.zeros_like(torch.as_tensor(action_chunk, dtype=torch.float32)),
            "next_value_function_return": float("-100"),
            "value_latent_idx": -1,
            "action_latent_idx": L["action"],
            "current_proprio_latent_idx": L["proprio"],
            "current_wrist_image_latent_idx": L["wrist_left"],
            "current_wrist_image2_latent_idx": L["wrist_right"],
            "current_image_latent_idx": L["head"],
            "future_proprio_latent_idx": L["future_proprio"],
            "future_wrist_image_latent_idx": L["future_wrist_left"],
            "future_wrist_image2_latent_idx": L["future_wrist_right"],
            "future_image_latent_idx": L["future_head"],
        }

        if self.use_force:
            f_raw = ep["force_raw"]
            sample["force"] = torch.from_numpy(self.normalize_force(f_raw[t]))
            sample["future_force"] = torch.from_numpy(self.normalize_future_force(f_raw[future_t], f_raw[t]))
            sample["current_force_latent_idx"] = L["force"]
            sample["future_force_latent_idx"] = L["future_force"]
            gate_stats = self.force_stats["gate"]
            sample["tactile_self_attn_gate"] = torch.tensor(
                scalar_gate_from_raw(
                    float(ep["force_gate_raw"][t]), median=gate_stats["raw_median"], mad=gate_stats["raw_mad"]
                ),
                dtype=torch.float32,
            )
        return sample
