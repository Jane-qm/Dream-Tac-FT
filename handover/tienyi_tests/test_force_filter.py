import h5py
import numpy as np

from cosmos_policy.experiments.robot.tienyi.force_filter import OnlineForceFilter, force_baseline, force_window_mean

P = "/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover/tienyi_prod2_dualArm-gripper-3cameras_515_Back_small_battery_installation_and_small_battery_cover_installation_20260916/success_episodes/3858544/data/trajectory.hdf5"
with h5py.File(P) as f:
    cam_ts = f["camera_observations/timestamp"][:]
    rts = f["force_observations/force_left_end_effector_raw/timestamp"][:]
    rd = f["force_observations/force_left_end_effector_raw/data"][:]
q = cam_ts[::2]
window = 2 / (1 / np.median(np.diff(cam_ts)))
print(f"window {window*1000:.1f} ms")

off = force_window_mean(rts, rd, q, window)
base = force_baseline(rts, rd, cam_ts[0], 0.5)

# 1. 在线版本逐条 push，与离线结果逐值一致
filt = OnlineForceFilter(window_s=window)
on, j = [], 0
for i, t in enumerate(q):
    while j < len(rts) and rts[j] <= t:
        filt.push(rts[j], rd[j])
        j += 1
    if i == 0:
        # 训练里零偏用的是 [t0, t0+0.5s]，在线要等这段读数到齐后再取
        pass
    on.append(filt.get(t, subtract_baseline=False))
on = np.array(on)
print("online == offline:", np.allclose(on, off), "max diff", np.abs(on - off).max())
filt2 = OnlineForceFilter(window_s=window)
for t, d in zip(rts, rd):
    if t > cam_ts[0] + 0.6:
        break
    filt2.push(t, d)
print("baseline online == offline:", np.allclose(filt2.capture_baseline(cam_ts[0], 0.5), base))

# 2. 因果性：把观测时刻之后的读数全部改掉，结果不变
k = len(q) // 2
rd2 = rd.copy()
rd2[rts > q[k]] += 1000.0
off2 = force_window_mean(rts, rd2, q[: k + 1], window)
print("causal (future change has no effect):", np.allclose(off2, off[: k + 1]))

# 3. 每个窗口的读数个数
hi = np.searchsorted(rts, q, side="right")
lo = np.searchsorted(rts, q - window, side="right")
c = hi - lo
print("samples per window: min %d median %d max %d, empty %d" % (c.min(), np.median(c), c.max(), (c == 0).sum()))
