# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tienyi 开环评估：在 val 集上用训练好的 checkpoint 预测动作块（以及未来力），与真值对比。

样本直接由 TienyiForceDataset(is_train=False, 不做增广) 构造，与训练输入完全一致；
送入模型前把真值目标（actions / future_proprio / future_force）清零，只保留条件帧信息，避免泄漏。

输出 <out_dir>/：
    metrics.json            归一化空间与物理空间的 MSE / L1（整体、逐维、逐 horizon 步）
    ep<k>_actions.png       每条 val episode：每个时刻预测动作块第 0 步 vs 真值 的逐维曲线
    ep<k>_future_force.png  （use_force 时）预测 F[t+H]-F[t] vs 真值

用法:
    PYTHONPATH=. python -m cosmos_policy.experiments.robot.tienyi.openloop_eval \
        --experiment cosmos_predict2_2b_480p_tienyi_battery_force \
        --ckpt_dir $IMAGINAIRE_OUTPUT_ROOT/cosmos_policy_tienyi_battery/cosmos_v2_finetune/<name>/checkpoints/iter_000003000 \
        --data_dir /media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160 \
        --out_dir /media/jushen/zhixuan/tienyi_eval/<name>
"""

import argparse
import json
import os

import numpy as np
import torch

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.datasets.tienyi_dataset import TienyiForceDataset
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence

TARGET_KEYS = ("actions", "future_proprio", "future_force")


def collate(samples, device):
    """与 cosmos_utils.get_action 一致：浮点输入用 bfloat16。"""
    batch = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            batch[k] = torch.stack(vals).to(device)
        elif isinstance(vals[0], (int, float)):
            batch[k] = torch.tensor(vals, device=device)
        else:
            continue
        if batch[k].is_floating_point():
            batch[k] = batch[k].to(torch.bfloat16)
    return batch


def unnormalize_minmax(x, lo, hi):
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    return 0.5 * (x + 1.0) * (hi - lo) + lo


def plot_episode(path, t_list, pred, gt, title, dim_names=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = gt.shape[-1]
    cols = 4
    rows = (D + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows), squeeze=False)
    for d in range(D):
        ax = axes[d // cols][d % cols]
        ax.plot(t_list, gt[:, d], "k-", lw=1.2, label="gt")
        ax.plot(t_list, pred[:, d], "r--", lw=1.0, label="pred")
        ax.set_title(dim_names[d] if dim_names else f"dim {d}", fontsize=8)
        ax.tick_params(labelsize=6)
    for d in range(D, rows * cols):
        axes[d // cols][d % cols].axis("off")
    axes[0][0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--ckpt_dir", required=True, help=".../checkpoints/iter_XXXXXXXXX（DCP 目录）")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--num_episodes", type=int, default=8, help="评估的 val episode 数（0 = 全部）")
    p.add_argument("--step_stride", type=int, default=8, help="每条 episode 每隔多少步取一个起点")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=5, help="去噪步数（与部署一致）")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.ckpt_dir,
        config_file=args.config_file,
        load_ema_to_reg=False,
    )
    model.eval()
    device = "cuda"

    use_force = model.config.state_t == 12
    num_cond = model.config.min_num_conditional_frames
    ds = TienyiForceDataset(
        data_dir=args.data_dir,
        t5_text_embeddings_path=os.path.join(args.data_dir, "t5_embeddings.pkl"),
        is_train=False,
        use_image_aug=False,
        use_stronger_image_aug=False,
        use_force=use_force,
        max_episodes=args.num_episodes,
    )
    H = ds.chunk_size
    stats = ds.dataset_stats
    action_dim = len(stats["actions_min"])

    # 按 episode 收集样本下标
    ep_to_idx = {}
    for i, (ep_idx, t) in enumerate(ds._step_to_episode_map):
        if t % args.step_stride == 0:
            ep_to_idx.setdefault(ep_idx, []).append((t, i))

    all_pred_a, all_gt_a, all_pred_ff, all_gt_ff = [], [], [], []
    per_ep = {}
    for ep_idx, items in sorted(ep_to_idx.items()):
        ep_pred_a, ep_gt_a, ep_pred_ff, ep_gt_ff, ts = [], [], [], [], []
        for b0 in range(0, len(items), args.batch_size):
            chunk = items[b0 : b0 + args.batch_size]
            samples = [ds[i] for _, i in chunk]
            gt_actions = torch.stack([s["actions"] for s in samples]).numpy()
            gt_ff = torch.stack([s["future_force"] for s in samples]).numpy() if use_force else None
            batch = collate(samples, device)
            for k in TARGET_KEYS:
                if k in batch:
                    batch[k] = torch.zeros_like(batch[k])
            batch["num_conditional_frames"] = num_cond
            B = len(samples)
            with torch.inference_mode():
                latent = model.generate_samples_from_batch(
                    batch, n_sample=B, num_steps=args.num_steps, seed=args.seed, is_negative_prompt=False
                )
            a_idx = torch.full((B,), int(samples[0]["action_latent_idx"]), dtype=torch.int64, device=latent.device)
            pred_a = extract_action_chunk_from_latent_sequence(latent, (H, action_dim), a_idx).float().cpu().numpy()
            ep_pred_a.append(pred_a)
            ep_gt_a.append(gt_actions)
            if use_force:
                f_idx = torch.full((B,), int(samples[0]["future_force_latent_idx"]), dtype=torch.int64, device=latent.device)
                pred_ff = extract_action_chunk_from_latent_sequence(latent, (1, gt_ff.shape[-1]), f_idx)[:, 0]
                ep_pred_ff.append(pred_ff.float().cpu().numpy())
                ep_gt_ff.append(gt_ff)
            ts += [t for t, _ in chunk]
        pa, ga = np.concatenate(ep_pred_a), np.concatenate(ep_gt_a)
        all_pred_a.append(pa)
        all_gt_a.append(ga)
        name = os.path.basename(ds.data[ep_idx]["file_path"])
        per_ep[name] = {"action_mse_norm": float(((pa - ga) ** 2).mean()), "num_samples": len(ts)}
        pa_u = unnormalize_minmax(pa[:, 0], stats["actions_min"], stats["actions_max"])
        ga_u = unnormalize_minmax(ga[:, 0], stats["actions_min"], stats["actions_max"])
        plot_episode(
            os.path.join(args.out_dir, f"ep{ep_idx}_actions.png"), ts, pa_u, ga_u, f"{name}: action[t] pred vs gt"
        )
        if use_force:
            pf, gf = np.concatenate(ep_pred_ff), np.concatenate(ep_gt_ff)
            all_pred_ff.append(pf)
            all_gt_ff.append(gf)
            per_ep[name]["future_force_mse_norm"] = float(((pf - gf) ** 2).mean())
            scale = np.asarray(stats[f"future_force_scale_chunk{H}"], np.float32)
            plot_episode(
                os.path.join(args.out_dir, f"ep{ep_idx}_future_force.png"),
                ts, pf * scale, gf * scale, f"{name}: F[t+{H}]-F[t] (N, Nm) pred vs gt",
            )
        print(f"[eval] {name}: {per_ep[name]}", flush=True)

    pa, ga = np.concatenate(all_pred_a), np.concatenate(all_gt_a)
    span = np.asarray(stats["actions_max"], np.float32) - np.asarray(stats["actions_min"], np.float32)
    err_u = (pa - ga) * 0.5 * span  # 物理单位误差
    metrics = {
        "experiment": args.experiment,
        "ckpt_dir": args.ckpt_dir,
        "num_samples": int(len(pa)),
        "action_mse_norm": float(((pa - ga) ** 2).mean()),
        "action_l1_norm": float(np.abs(pa - ga).mean()),
        "action_mse_norm_per_horizon": ((pa - ga) ** 2).mean(axis=(0, 2)).tolist(),
        "action_l1_phys_per_dim": np.abs(err_u).mean(axis=(0, 1)).tolist(),
        "action_l1_phys_first_step_per_dim": np.abs(err_u[:, 0]).mean(axis=0).tolist(),
        "per_episode": per_ep,
    }
    if use_force:
        pf, gf = np.concatenate(all_pred_ff), np.concatenate(all_gt_ff)
        scale = np.asarray(stats[f"future_force_scale_chunk{H}"], np.float32)
        metrics["future_force_mse_norm"] = float(((pf - gf) ** 2).mean())
        metrics["future_force_l1_phys_per_dim"] = np.abs((pf - gf) * scale).mean(axis=0).tolist()
        # 参考：预测"力不变"（ΔF=0）的基线误差
        metrics["future_force_mse_norm_zero_baseline"] = float((gf**2).mean())
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
