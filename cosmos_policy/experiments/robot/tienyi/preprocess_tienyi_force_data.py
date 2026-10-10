# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
把 tienyi 双臂（3 相机 + 左右六维力）原始 HDF5 转换为 Dream-Tac / Cosmos Policy 训练用的预处理格式。

输入（原始数据）
----------------
    <raw_root>/<date_dir>/success_episodes/<episode_id>/data/trajectory.hdf5
    其中 <date_dir> 以 "_fail" 结尾的是失败数据，默认排除。
    使用到的键（*_align 已与相机时间戳对齐，长度均为 T，约 29 Hz）：
        camera_observations/color_images/{camera_head,camera_left,camera_right}   (T,) JPEG bytes, 640x480
        camera_observations/timestamp                                            (T,)
        camera_observations/is_intervene                                         (T,) bool
        puppet/end_effector_{left,right}_pose_align/data                         (T, 7) xyz + 四元数
        puppet/end_effector_{left,right}_position_align/data                     (T, 1) 夹爪开度
        puppet/arm_{left,right}_position_align/data                              (T, 7) 从臂关节角
        master/arm_{left,right}_position_align/data                              (T, 7) 主臂关节角（遥操作指令）
        master/end_effector_{left,right}_position_align/data                     (T,)   主臂夹爪指令
        force_observations/force_{left,right}_end_effector_align/data            (T, 6) TCP 系 [Fx,Fy,Fz,Tx,Ty,Tz]
        force_observations/force_{left,right}_end_effector_raw/{data,timestamp}  (~2T, 6) 原始读数，约 59 Hz

输出（预处理数据）
------------------
    <out_dir>/
    ├── train/episode_XXXXX.hdf5 + episode_XXXXX_{cam_high,cam_left_wrist,cam_right_wrist}.mp4
    ├── val/...
    ├── preprocessing_metadata.json   # dataset_info.task_name 供 save_franka_t5_text_embeddings 生成 T5 嵌入
    └── force_statistics.json         # 力的逐维统计 + 接触门控 raw 值的 median / MAD（仅用 train 统计）

    每个 episode_XXXXX.hdf5（T' = ceil(T / stride)）：
        attrs: task_name（= 语言指令，也是 T5 嵌入的 key）, source_path, stride, fps, quat_order,
               force_window_s, force_baseline_s
        observations/qpos            (T', 20) float32  当前从臂状态，见 STATE_LAYOUT
        observations/qpos_joint      (T', 16) float32  [左 7 关节, 左夹爪, 右 7 关节, 右夹爪]（从臂）
        observations/force           (T', 12) float32  [左 Fx..Tz, 右 Fx..Tz]，原始读数在 (t - 一个控制周期, t]
                                                       内取平均后减去 episode 起始零偏（因果，见 force_filter.py）
        observations/force_abs       (T', 12) float32  *_align 读数（未滤波、未去零偏），供消融用
        observations/force_baseline  (12,)    float32  本条 episode 被减掉的零偏
        observations/force_gate_raw  (T',)    float32  接触事件强度（第二遍写入），见 compute_gate_raw
        observations/timestamp       (T',)    float64
        observations/video_paths/{cam_high,cam_left_wrist,cam_right_wrist}  同目录下 MP4 的相对路径
        action                       (T', 20) float32  action[t] = qpos[t+1]（下一时刻的绝对位姿 + 夹爪），最后一步重复
        action_joint                 (T', 16) float32  主臂关节 + 主臂夹爪指令（备选的关节空间动作）

相机映射（与 ALOHADataset 的命名一致）：
    camera_head  -> cam_high        （主视角 primary）
    camera_left  -> cam_left_wrist  （左腕）
    camera_right -> cam_right_wrist （右腕）

用法示例：
    # 冒烟测试：只转换 3 条
    python -m cosmos_policy.experiments.robot.tienyi.preprocess_tienyi_force_data \
        --raw_root /media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover \
        --out_dir /path/to/tienyi_preprocessed/back_battery_and_cover_smoke \
        --max_episodes 3 --num_workers 3

    # 2 小时实验规模：从全部成功数据中随机抽 160 条
    python -m cosmos_policy.experiments.robot.tienyi.preprocess_tienyi_force_data \
        --raw_root <同上> --out_dir /path/to/tienyi_preprocessed/back_battery_and_cover_160 \
        --max_episodes 160 --num_workers 32
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from cosmos_policy.experiments.robot.tienyi.force_filter import force_baseline, force_window_mean

DEFAULT_INSTRUCTION = "install the small battery on the back and then install the small battery cover"

# 原始相机键 -> 输出相机键
CAMERA_MAP = {
    "camera_head": "cam_high",
    "camera_left": "cam_left_wrist",
    "camera_right": "cam_right_wrist",
}

ARMS = ("left", "right")

# qpos / action 的 20 维布局（每只臂 10 维，左臂在前）
STATE_LAYOUT = [
    f"{arm}_{name}"
    for arm in ARMS
    for name in ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32", "gripper")
]
FORCE_LAYOUT = [f"{arm}_{name}" for arm in ARMS for name in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")]


# ----------------------------------------------------------------------------------------------
# 数学工具
# ----------------------------------------------------------------------------------------------
def quat_to_rotmat(q: np.ndarray, order: str) -> np.ndarray:
    """(N, 4) 四元数 -> (N, 3, 3) 旋转矩阵。先归一化；q 与 -q 得到相同的矩阵，因此符号翻转不影响结果。"""
    if order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unknown quat order: {order}")
    n = np.sqrt(x * x + y * y + z * z + w * w) + 1e-12
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 6)：取旋转矩阵前两列 [r11,r21,r31,r12,r22,r32]，连续且无奇异，适合回归。"""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def compute_gate_raw(force: np.ndarray, force_std: np.ndarray, k: int) -> np.ndarray:
    """接触事件强度：raw_t = max_arm RMS_dim((F_t - F_{t-k}) / std)。

    在 k 步（下采样后）的窗口上做差分，消掉零偏和缓慢变化的重力分量，同时压住逐帧噪声；
    t < k 时与第 0 帧做差分。Dream-Tac 在数据集中用 scalar_gate_from_raw(raw, median, mad) 把它映射为门控 g_t。
    """
    prev = force[np.maximum(np.arange(len(force)) - k, 0)]
    d = (force - prev) / (force_std[None, :] + 1e-6)
    left = np.sqrt((d[:, :6] ** 2).mean(axis=1))
    right = np.sqrt((d[:, 6:] ** 2).mean(axis=1))
    return np.maximum(left, right).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# 单条 episode 处理（在子进程中运行）
# ----------------------------------------------------------------------------------------------
def _write_video(frames_bgr, path: str, fps: float) -> None:
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()


def process_episode(job: dict) -> dict:
    """转换一条 episode。返回元信息 dict；跳过时返回 {"skipped": 原因}。"""
    cv2.setNumThreads(1)
    args = job["args"]
    src = job["src"]
    out_dir = job["out_dir"]
    name = job["name"]
    stride = args["stride"]

    try:
        with h5py.File(src, "r") as f:
            T = int(f["camera_observations/timestamp"].shape[0])
            if T < args["min_len"]:
                return {"skipped": f"too short (T={T})", "src": src}
            if args["drop_intervened"] and bool(np.any(f["camera_observations/is_intervene"][:])):
                return {"skipped": "has intervention", "src": src}

            ts_full = f["camera_observations/timestamp"][:].astype(np.float64)
            fs = 1.0 / float(np.median(np.diff(ts_full)))  # 相机帧率，约 29 Hz
            idx = np.arange(0, T, stride)

            # ---------- 状态：xyz + rot6d + 夹爪，左右各 10 维 ----------
            state_parts, joint_parts, master_parts = [], [], []
            for arm in ARMS:
                pose = f[f"puppet/end_effector_{arm}_pose_align/data"][:].astype(np.float64)  # (T, 7)
                rot6d = rotmat_to_rot6d(quat_to_rotmat(pose[:, 3:7], args["quat_order"]))
                grip = f[f"puppet/end_effector_{arm}_position_align/data"][:].astype(np.float64).reshape(T, 1)
                state_parts += [pose[:, :3], rot6d, grip]

                joints = f[f"puppet/arm_{arm}_position_align/data"][:].astype(np.float64)  # (T, 7)
                joint_parts += [joints, grip]

                m_joints = f[f"master/arm_{arm}_position_align/data"][:].astype(np.float64)
                m_grip = f[f"master/end_effector_{arm}_position_align/data"][:].astype(np.float64).reshape(T, 1)
                master_parts += [m_joints, m_grip]
            state_full = np.concatenate(state_parts, axis=1)  # (T, 20)
            qpos_joint_full = np.concatenate(joint_parts, axis=1)  # (T, 16)
            master_full = np.concatenate(master_parts, axis=1)  # (T, 16)

            # ---------- 六维力（TCP 系）：按控制周期因果平均 -> 减去 episode 起始零偏 ----------
            # 观测时刻 = 该步图像的相机时间戳；窗口 = 一个控制周期。部署端用 force_filter.OnlineForceFilter 复现。
            # 不做完整重力补偿：重力项只能解释约 10-20% 的力变化（见 docs/tienyi_force_integration_plan.md 1.4）。
            ts = ts_full[idx]
            window_s = args["force_window_s"] if args["force_window_s"] > 0 else stride / fs
            force_parts, baseline_parts, abs_parts = [], [], []
            for arm in ARMS:
                g = f"force_observations/force_{arm}_end_effector_"
                raw_ts = f[g + "raw/timestamp"][:].astype(np.float64)
                raw_data = f[g + "raw/data"][:].astype(np.float64)
                order = np.argsort(raw_ts, kind="stable")
                raw_ts, raw_data = raw_ts[order], raw_data[order]
                force_parts.append(force_window_mean(raw_ts, raw_data, ts, window_s))
                baseline_parts.append(force_baseline(raw_ts, raw_data, ts_full[0], args["force_baseline_s"]))
                abs_parts.append(f[g + "align/data"][:].astype(np.float64)[idx])
            force_baseline_12 = np.concatenate(baseline_parts)
            force = (np.concatenate(force_parts, axis=1) - force_baseline_12[None, :]).astype(np.float32)
            force_abs = np.concatenate(abs_parts, axis=1).astype(np.float32)

            # ---------- 时间下采样 ----------
            qpos = state_full[idx].astype(np.float32)
            action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)  # action[t] = qpos[t+1]
            qpos_joint = qpos_joint_full[idx].astype(np.float32)
            action_joint = master_full[idx].astype(np.float32)  # 主臂指令本身就是 t 时刻的动作，不需要平移

            # ---------- 图像：JPEG 解码 -> resize -> MP4 ----------
            # 原始 JPEG 由 BGR 数组编码（camera_color_channel=bgr），cv2.imdecode 直接得到正确的 BGR，
            # 正好是 cv2.VideoWriter 需要的格式；训练端 load_video_as_images 会再转换为 RGB。
            video_paths = {}
            out_fps = fs / stride
            for raw_key, out_key in CAMERA_MAP.items():
                jpgs = f[f"camera_observations/color_images/{raw_key}"][idx]
                frames = []
                for buf in jpgs:
                    img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError(f"JPEG decode failed: {raw_key}")
                    frames.append(
                        cv2.resize(img, (args["img_size"], args["img_size"]), interpolation=cv2.INTER_AREA)
                    )
                fname = f"{name}_{out_key}.mp4"
                _write_video(frames, os.path.join(out_dir, fname), out_fps)
                video_paths[out_key] = fname

        # ---------- 写出预处理 HDF5 ----------
        out_path = os.path.join(out_dir, f"{name}.hdf5")
        with h5py.File(out_path, "w") as g:
            g.attrs["task_name"] = args["instruction"]
            g.attrs["source_path"] = src
            g.attrs["stride"] = stride
            g.attrs["fps"] = out_fps
            g.attrs["quat_order"] = args["quat_order"]
            g.attrs["force_window_s"] = window_s
            g.attrs["force_baseline_s"] = args["force_baseline_s"]
            obs = g.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            obs.create_dataset("qpos_joint", data=qpos_joint)
            obs.create_dataset("force", data=force)
            obs.create_dataset("force_abs", data=force_abs)
            obs.create_dataset("force_baseline", data=force_baseline_12.astype(np.float32))
            obs.create_dataset("timestamp", data=ts)
            vp = obs.create_group("video_paths")
            for k, v in video_paths.items():
                vp.create_dataset(k, data=v.encode("utf-8"))
            g.create_dataset("action", data=action)
            g.create_dataset("action_joint", data=action_joint)

        return {
            "src": src,
            "out": out_path,
            "name": name,
            "split": job["split"],
            "raw_len": T,
            "episode_length": int(len(idx)),
            "fps": out_fps,
            "force_window_s": window_s,
            "video_files": list(video_paths.values()),
        }
    except Exception as e:  # 单条失败不影响整体，记录到 metadata
        return {"skipped": f"error: {type(e).__name__}: {e}", "src": src}


# ----------------------------------------------------------------------------------------------
# 第二遍：全局力统计 + 门控 raw 值
# ----------------------------------------------------------------------------------------------
def finalize_force_statistics(out_dir: str, episodes: list, gate_k: int) -> dict:
    """用 train 集的力计算逐维统计；给每条 episode 写入 observations/force_gate_raw；返回统计 dict。"""
    train_eps = [e for e in episodes if e["split"] == "train"] or episodes
    all_force = []
    for e in train_eps:
        with h5py.File(e["out"], "r") as g:
            all_force.append(g["observations/force"][:])
    F = np.concatenate(all_force, axis=0)
    std = F.std(axis=0)
    # 门控用的 std 设下限：某些维度几乎不动（例如右臂 Tz 的 std 约 0.02），直接除会把噪声放大成假“接触”
    gate_std = std.copy()
    for group in ([0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]):  # 力 / 力矩分别处理
        gate_std[group] = np.maximum(std[group], 0.25 * np.median(std[group]))

    train_names = {e["name"] for e in train_eps}
    gate_train = []
    for e in tqdm(episodes, desc="Writing force_gate_raw"):
        with h5py.File(e["out"], "a") as g:
            raw = compute_gate_raw(g["observations/force"][:], gate_std, gate_k)
            if "observations/force_gate_raw" in g:
                del g["observations/force_gate_raw"]
            g["observations"].create_dataset("force_gate_raw", data=raw)
        if e["name"] in train_names:
            gate_train.append(raw)
    G = np.concatenate(gate_train)
    med = float(np.median(G))
    mad = float(np.median(np.abs(G - med)))

    def _v(a):
        return [float(x) for x in a]

    return {
        "layout": FORCE_LAYOUT,
        "frame": "TCP",
        "note": "observations/force = 原始读数按控制周期因果平均（force_filter.py）后减去 episode 起始零偏；统计只用 train 集",
        "num_steps": int(F.shape[0]),
        "force_mean": _v(F.mean(axis=0)),
        "force_std": _v(std),
        "force_min": _v(F.min(axis=0)),
        "force_max": _v(F.max(axis=0)),
        "force_p01": _v(np.percentile(F, 1, axis=0)),
        "force_p99": _v(np.percentile(F, 99, axis=0)),
        "gate": {
            "k_steps": gate_k,
            "std_used": _v(gate_std),
            "raw_median": med,
            "raw_mad": mad,
            "raw_p90": float(np.percentile(G, 90)),
            "raw_p99": float(np.percentile(G, 99)),
            "usage": "g_t = scalar_gate_from_raw(force_gate_raw[t], median=raw_median, mad=raw_mad)",
        },
    }


# ----------------------------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------------------------
def list_source_episodes(raw_root: str, include_fail: bool, date_filter: list) -> list:
    """枚举原始 trajectory.hdf5，按日期目录过滤。"""
    files = sorted(glob.glob(os.path.join(raw_root, "*", "success_episodes", "*", "data", "trajectory.hdf5")))
    out = []
    for p in files:
        date_dir = p.split(os.sep)[-5]
        if date_dir.endswith("_fail") and not include_fail:
            continue
        if date_filter and not any(s in date_dir for s in date_filter):
            continue
        out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="tienyi 双臂 + 六维力 -> Dream-Tac 预处理格式")
    parser.add_argument("--raw_root", required=True, help="任务目录，例如 .../tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover")
    parser.add_argument("--out_dir", required=True, help="输出目录（即训练配置中的 data_dir）")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="语言指令（原始数据里只有 test01）")
    parser.add_argument("--include_fail", action="store_true", help="包含 *_fail 目录（失败数据，默认排除）")
    parser.add_argument("--date_filter", default="", help="只用名称包含这些子串的日期目录，逗号分隔，例如 20260916,20260917")
    parser.add_argument("--max_episodes", type=int, default=0, help="随机抽取的最大条数，0 表示全部")
    parser.add_argument("--percent_val", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--stride", type=int, default=2, help="时间下采样步长（29 Hz / 2 约等于 14.5 Hz）")
    parser.add_argument("--img_size", type=int, default=224, help="输出图像边长（方形）")
    parser.add_argument("--min_len", type=int, default=300, help="原始帧数少于该值的 episode 跳过")
    parser.add_argument("--drop_intervened", action="store_true", help="跳过含人工干预帧的 episode")
    parser.add_argument("--quat_order", default="xyzw", choices=["xyzw", "wxyz"],
                        help="pose 中四元数的顺序；头部位姿按 xyzw 解读为向下俯视约 38 度，符合实际")
    parser.add_argument("--force_window_s", type=float, default=0.0,
                        help="力的因果平均窗口（秒）；0 表示一个控制周期 = stride / 相机帧率（约 0.069 s）")
    parser.add_argument("--force_baseline_s", type=float, default=0.5, help="用开头多少秒估计零偏")
    parser.add_argument("--gate_k", type=int, default=4, help="门控差分窗口（下采样后的步数）")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    date_filter = [s for s in args.date_filter.split(",") if s]
    srcs = list_source_episodes(args.raw_root, args.include_fail, date_filter)
    if not srcs:
        raise SystemExit(f"No trajectory.hdf5 found under {args.raw_root}")
    rng = random.Random(args.seed)
    rng.shuffle(srcs)
    if args.max_episodes > 0:
        srcs = srcs[: args.max_episodes]
    n_val = int(len(srcs) * args.percent_val)
    if args.percent_val > 0 and len(srcs) >= 2:
        n_val = max(1, n_val)

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    shared = {k: getattr(args, k) for k in (
        "instruction", "stride", "img_size", "min_len", "drop_intervened", "quat_order",
        "force_window_s", "force_baseline_s",
    )}
    jobs = []
    for i, src in enumerate(srcs):
        split = "val" if i < n_val else "train"
        jobs.append(dict(
            args=shared, src=src, split=split, name=f"episode_{i:05d}",
            out_dir=os.path.join(args.out_dir, split),
        ))
    print(f"Episodes selected: {len(jobs)} (train {len(jobs) - n_val}, val {n_val}); workers={args.num_workers}")

    t0 = time.time()
    results = []
    with mp.get_context("fork").Pool(args.num_workers) as pool:
        for r in tqdm(pool.imap_unordered(process_episode, jobs), total=len(jobs), desc="Converting"):
            results.append(r)
    done = sorted([r for r in results if "skipped" not in r], key=lambda r: r["name"])
    skipped = [r for r in results if "skipped" in r]
    print(f"Converted {len(done)}, skipped {len(skipped)} in {time.time() - t0:.0f}s")
    for s in skipped:
        print("  skipped:", s["skipped"], s["src"])
    if not done:
        raise SystemExit("Nothing converted.")

    force_stats = finalize_force_statistics(args.out_dir, done, args.gate_k)
    with open(os.path.join(args.out_dir, "force_statistics.json"), "w") as fp:
        json.dump(force_stats, fp, indent=2, ensure_ascii=False)

    metadata = {
        "dataset_info": {
            "original_dataset_path": args.raw_root,
            "preprocessed_dataset_path": args.out_dir,
            "task_name": args.instruction,
            "total_episodes": len(done),
            "train_episodes": sum(r["split"] == "train" for r in done),
            "val_episodes": sum(r["split"] == "val" for r in done),
            "total_steps": int(sum(r["episode_length"] for r in done)),
            "state_layout": STATE_LAYOUT,
            "force_layout": FORCE_LAYOUT,
            "camera_map": CAMERA_MAP,
            "preprocessing_settings": {k: v for k, v in vars(args).items()},
        },
        "episode_mapping": done,
        "skipped": skipped,
    }
    with open(os.path.join(args.out_dir, "preprocessing_metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)
    print(f"Done. Output: {args.out_dir}")
    print(f"Gate stats: median={force_stats['gate']['raw_median']:.4f} mad={force_stats['gate']['raw_mad']:.4f}")


if __name__ == "__main__":
    main()
