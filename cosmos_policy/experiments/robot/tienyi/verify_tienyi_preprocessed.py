# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
检查 preprocess_tienyi_force_data.py 的输出是否完整、一致，并导出一张可视化图供人工检查。

检查项：
    - 每条 episode 的 qpos / action / force / force_gate_raw / timestamp 长度一致，并且没有 NaN/Inf
    - 三路 MP4 都能打开，帧数与 T' 一致，分辨率正确
    - rot6d 两列是单位正交向量（验证四元数转换正确）
    - action[t] == qpos[t+1]
导出（--vis_dir）：
    - <name>_frames.png：首 / 中 / 尾三个时刻的三路相机图像（用于检查颜色是否正常、相机映射是否正确）
    - <name>_signals.png：力（左右 Fz、|F|）、门控 raw 值、夹爪开度随时间变化的曲线

用法：
    python -m cosmos_policy.experiments.robot.tienyi.verify_tienyi_preprocessed \
        --data_dir /path/to/tienyi_preprocessed/back_battery_and_cover_160 --vis_dir ./tienyi_verify --num_vis 2
"""

import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _read_str(ds):
    v = ds[()]
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = 0
    shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape = frame.shape
        n += 1
    cap.release()
    return n, shape


def _read_frames(path, indices):
    cap = cv2.VideoCapture(path)
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None)
    cap.release()
    return out


def check_episode(path, img_size):
    errs = []
    d = os.path.dirname(path)
    with h5py.File(path, "r") as f:
        qpos = f["observations/qpos"][:]
        action = f["action"][:]
        force = f["observations/force"][:]
        gate = f["observations/force_gate_raw"][:] if "observations/force_gate_raw" in f else None
        ts = f["observations/timestamp"][:]
        T = len(qpos)
        if gate is None:
            errs.append("missing force_gate_raw")
        for k, a in [("action", action), ("force", force), ("timestamp", ts)] + ([("gate", gate)] if gate is not None else []):
            if len(a) != T:
                errs.append(f"{k} len {len(a)} != {T}")
            if not np.all(np.isfinite(a)):
                errs.append(f"{k} has NaN/Inf")
        if qpos.shape[1] != 20 or action.shape[1] != 20 or force.shape[1] != 12:
            errs.append(f"bad dims qpos{qpos.shape} action{action.shape} force{force.shape}")
        if not np.allclose(action[:-1], qpos[1:]):
            errs.append("action[t] != qpos[t+1]")
        for arm in range(2):
            c0 = qpos[:, arm * 10 + 3: arm * 10 + 6]
            c1 = qpos[:, arm * 10 + 6: arm * 10 + 9]
            if np.abs(np.linalg.norm(c0, axis=1) - 1).max() > 1e-3 or np.abs((c0 * c1).sum(1)).max() > 1e-3:
                errs.append(f"rot6d not orthonormal (arm {arm})")
        vids = {c: os.path.join(d, _read_str(f["observations/video_paths"][c])) for c in CAMS}
    for c, p in vids.items():
        info = _video_info(p)
        if info is None:
            errs.append(f"{c}: cannot open {p}")
            continue
        n, shape = info
        if n != T:
            errs.append(f"{c}: {n} frames != {T}")
        if shape[:2] != (img_size, img_size):
            errs.append(f"{c}: frame shape {shape}")
    return T, errs, vids, qpos, force, gate


def visualize(name, vids, qpos, force, gate, vis_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(qpos)
    idx = [0, T // 2, T - 1]
    rows = [np.concatenate([fr for fr in _read_frames(vids[c], idx)], axis=1) for c in CAMS]
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(os.path.join(vis_dir, f"{name}_frames.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(force[:, 2], label="left Fz")
    ax[0].plot(force[:, 8], label="right Fz")
    ax[0].set_ylabel("Fz - baseline (N)")
    ax[0].legend()
    ax[1].plot(np.linalg.norm(force[:, 0:3], axis=1), label="left |F|")
    ax[1].plot(np.linalg.norm(force[:, 6:9], axis=1), label="right |F|")
    ax[1].set_ylabel("|F| (N)")
    ax[1].legend()
    if gate is not None:
        ax[2].plot(gate)
    ax[2].set_ylabel("gate raw")
    ax[3].plot(qpos[:, 9], label="left gripper")
    ax[3].plot(qpos[:, 19], label="right gripper")
    ax[3].set_ylabel("gripper")
    ax[3].set_xlabel("step (downsampled)")
    ax[3].legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, f"{name}_signals.png"), dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--vis_dir", default="")
    parser.add_argument("--num_vis", type=int, default=1)
    args = parser.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "preprocessing_metadata.json")))
    img_size = meta["dataset_info"]["preprocessing_settings"]["img_size"]
    files = sorted(glob.glob(os.path.join(args.data_dir, "*", "episode_*.hdf5")))
    print(f"task_name: {meta['dataset_info']['task_name']}")
    print(f"episodes: {len(files)} (metadata says {meta['dataset_info']['total_episodes']}), "
          f"skipped during preprocessing: {len(meta['skipped'])}")
    assert os.path.exists(os.path.join(args.data_dir, "force_statistics.json")), "force_statistics.json missing"

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
    n_bad, total = 0, 0
    for i, p in enumerate(files):
        T, errs, vids, qpos, force, gate = check_episode(p, img_size)
        total += T
        name = os.path.splitext(os.path.basename(p))[0]
        if errs:
            n_bad += 1
            print(f"[BAD] {p}: {errs}")
        if args.vis_dir and i < args.num_vis:
            visualize(name, vids, qpos, force, gate, args.vis_dir)
    print(f"checked {len(files)} episodes, total steps {total}, bad {n_bad}")
    if args.vis_dir:
        print(f"visualizations written to {args.vis_dir}")


if __name__ == "__main__":
    main()
