# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
六维力的因果滤波：按控制周期取平均。训练数据转换与真机部署共用这一份实现。

定义（只用过去的数据，实时可算）：
    F_t = mean{ 原始读数 f_i : t - window_s < ts_i <= t }
    t 是观测时刻（与该步图像的相机时间戳一致），window_s 是一个控制周期（stride / 相机帧率，约 0.069 s）。
    窗口内没有读数时，退化为 t 之前最近的一个读数；t 之前完全没有读数时，用第一个读数。

零偏：
    baseline = mean{ f_i : t_start <= ts_i <= t_start + baseline_s }
    训练时 t_start 是 episode 第一帧相机时间；部署时是机械臂静止、开始执行前的那段时间。

训练端：preprocess_tienyi_force_data.py 调用 force_window_mean / force_baseline。
部署端：用 OnlineForceFilter 逐条 push 传感器读数，在每个控制步用图像时间戳调用 get()。
"""

from collections import deque

import numpy as np


def force_window_mean(
    sample_ts: np.ndarray, sample_data: np.ndarray, query_ts: np.ndarray, window_s: float
) -> np.ndarray:
    """
    Args:
        sample_ts: (N,) 原始读数时间戳（秒），单调不减
        sample_data: (N, D) 原始读数
        query_ts: (M,) 观测时刻
        window_s: 窗口长度（秒）
    Returns:
        (M, D) 每个观测时刻的窗口平均
    """
    sample_ts = np.asarray(sample_ts, dtype=np.float64)
    sample_data = np.asarray(sample_data, dtype=np.float64)
    query_ts = np.atleast_1d(np.asarray(query_ts, dtype=np.float64))
    assert np.all(np.diff(sample_ts) >= 0), "sample_ts 必须单调不减"
    assert len(sample_ts) > 0

    csum = np.concatenate([np.zeros((1, sample_data.shape[1])), np.cumsum(sample_data, axis=0)], axis=0)
    hi = np.searchsorted(sample_ts, query_ts, side="right")  # 窗口 (lo, hi] 对应样本下标 [lo, hi)
    lo = np.searchsorted(sample_ts, query_ts - window_s, side="right")
    count = hi - lo
    out = (csum[hi] - csum[lo]) / np.maximum(count, 1)[:, None]

    empty = count == 0
    if np.any(empty):
        last = np.clip(hi[empty] - 1, 0, len(sample_ts) - 1)
        out[empty] = sample_data[last]
    return out


def force_baseline(sample_ts: np.ndarray, sample_data: np.ndarray, t_start: float, baseline_s: float) -> np.ndarray:
    """(D,) 零偏：[t_start, t_start + baseline_s] 内读数的均值；没有读数时用 t_start 之后的第一个读数。"""
    sample_ts = np.asarray(sample_ts, dtype=np.float64)
    sample_data = np.asarray(sample_data, dtype=np.float64)
    mask = (sample_ts >= t_start) & (sample_ts <= t_start + baseline_s)
    if np.any(mask):
        return sample_data[mask].mean(axis=0)
    i = min(np.searchsorted(sample_ts, t_start, side="left"), len(sample_ts) - 1)
    return sample_data[i].copy()


class OnlineForceFilter:
    """
    部署端在线版本，与 force_window_mean / force_baseline 逐值一致。

    用法（左右臂各一个实例，结果按 [左 6 维, 右 6 维] 拼成 12 维）：
        filt = OnlineForceFilter(window_s=meta["force_window_s"])
        # 传感器回调里：
        filt.push(ts, wrench6)
        # 开始执行前，机械臂静止约 baseline_s 秒后：
        filt.capture_baseline(t_start, baseline_s=0.5)
        # 每个控制步，用本步图像的时间戳：
        f_rel = filt.get(t_obs)          # 已减零偏，单位 N / N·m
    """

    def __init__(self, window_s: float, buffer_s: float = 5.0):
        self.window_s = window_s
        self.buffer_s = max(buffer_s, window_s)
        self._ts = deque()
        self._data = deque()
        self._baseline = None
        self._baseline_samples = []

    def push(self, ts: float, wrench) -> None:
        if self._ts and ts < self._ts[-1]:
            raise ValueError("传感器时间戳倒退")
        self._ts.append(float(ts))
        self._data.append(np.asarray(wrench, dtype=np.float64))
        if self._baseline is None:
            self._baseline_samples.append((float(ts), self._data[-1]))
        while self._ts and self._ts[0] < ts - self.buffer_s:
            self._ts.popleft()
            self._data.popleft()

    def capture_baseline(self, t_start: float, baseline_s: float = 0.5) -> np.ndarray:
        ts = np.array([t for t, _ in self._baseline_samples])
        data = np.stack([d for _, d in self._baseline_samples])
        self._baseline = force_baseline(ts, data, t_start, baseline_s)
        self._baseline_samples = []
        return self._baseline

    def get(self, t_obs: float, subtract_baseline: bool = True) -> np.ndarray:
        assert self._ts, "还没有收到任何读数"
        f = force_window_mean(np.array(self._ts), np.stack(self._data), np.array([t_obs]), self.window_s)[0]
        if subtract_baseline:
            assert self._baseline is not None, "先调用 capture_baseline"
            f = f - self._baseline
        return f
