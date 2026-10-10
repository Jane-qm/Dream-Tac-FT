"""CPU 测试：从源码中抽出真实函数，验证力注入、未来力 loss 掩码与日志。"""
import ast
import types
from typing import Optional

import torch
from einops import rearrange

SRC = "/media/jushen/zhixuan/Dream-Tac/cosmos_policy/models/policy_text2world_model.py"
tree = ast.parse(open(SRC).read())
wanted = {"replace_latent_with_action_chunk", "replace_latent_with_proprio", "compute_loss_with_epsilon_and_sigma"}
nodes = []
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name in wanted:
        nodes.append(n)
ns = {"torch": torch, "rearrange": rearrange, "Optional": Optional, "Text2WorldCondition": object}
exec(compile(ast.Module(body=nodes, type_ignores=[]), SRC, "exec"), ns)
replace_proprio = ns["replace_latent_with_proprio"]
compute_loss = ns["compute_loss_with_epsilon_and_sigma"]

B, C, T, H, W = 2, 16, 12, 28, 28
torch.manual_seed(0)

# ---------- 1. replace_latent_with_proprio 平铺正确 ----------
x0 = torch.randn(B, C, T, H, W)
force = torch.rand(B, 12) * 2 - 1
idx = torch.tensor([5, 5])
out = replace_proprio(x0.clone(), force, idx)
flat = out[:, :, 5].reshape(B, -1)
assert torch.allclose(flat[:, :12], force) and torch.allclose(flat[:, 12:24], force)
assert torch.equal(out[:, :, :5], x0[:, :, :5]) and torch.equal(out[:, :, 6:], x0[:, :, 6:])
print("[OK] replace_latent_with_proprio: 12 维力平铺到 slot 5，其他 slot 不变")


# ---------- 2. compute_loss：力注入 x0 + 未来力 loss ----------
class Cfg:
    mask_current_state_action_for_value_prediction = False
    mask_future_state_for_qvalue_prediction = False
    mask_loss_for_action_future_state_prediction = False
    mask_value_prediction_loss_for_policy_prediction = False
    action_loss_multiplier = 1


def make_self(perfect_except=None):
    """denoise 返回 x0 本身（完美预测），perfect_except 指定的 slot 加噪声。"""
    captured = {}

    def marginal_prob(x0, sigma):
        captured["x0_injected"] = x0.clone()
        return x0, torch.ones_like(sigma)

    def denoise(xt, sigma, condition):
        pred = captured["x0_injected"].clone()
        if perfect_except is not None:
            pred[:, :, perfect_except] += 1.0
        return types.SimpleNamespace(x0=pred)

    self = types.SimpleNamespace(
        sde=types.SimpleNamespace(marginal_prob=marginal_prob),
        denoise=denoise,
        get_per_sigma_loss_weights=lambda sigma: torch.ones_like(sigma),
        config=Cfg(),
    )
    return self, captured


def run(self, with_force: bool):
    x0 = torch.randn(B, C, T, H, W)
    const = lambda v: torch.full((B,), v, dtype=torch.long)
    kw = dict(
        action_chunk=torch.rand(B, 16, 20),
        action_indices=const(6),
        proprio=torch.rand(B, 20),
        current_proprio_indices=const(1),
        future_proprio=torch.rand(B, 20),
        future_proprio_indices=const(7),
        future_wrist_image_indices=const(8),
        future_wrist_image2_indices=const(9),
        future_image_indices=const(10),
        future_image2_indices=None,
        future_tactile_left_indices=None,
        future_tactile_right_indices=None,
        rollout_data_mask=torch.zeros(B, dtype=torch.long),
        world_model_sample_mask=torch.zeros(B, dtype=torch.long),
        value_function_sample_mask=torch.zeros(B, dtype=torch.long),
        value_function_return=torch.full((B,), -100.0),
        value_indices=const(-1),
    )
    if with_force:
        kw.update(force=force, current_force_indices=const(5), future_force=-force, future_force_indices=const(11))
    cond = types.SimpleNamespace()
    return compute_loss(self, x0, cond, torch.zeros_like(x0), torch.ones(B, T), **kw)


self, cap = make_self(perfect_except=11)
out, *_ = run(self, with_force=True)
x0_inj = cap["x0_injected"]
assert torch.allclose(x0_inj[:, :, 5].reshape(B, -1)[:, :12], force)
assert torch.allclose(x0_inj[:, :, 11].reshape(B, -1)[:, :12], -force)
assert torch.allclose(out["demo_sample_future_force_mse_loss"], torch.tensor(1.0))
assert out["demo_sample_action_mse_loss"] == 0
print("[OK] 力注入 x0 的 slot 5（当前）/ slot 11（未来）；未来力误差只体现在 future_force loss 中")

self, _ = make_self()
out, *_ = run(self, with_force=False)
assert torch.isnan(out["demo_sample_future_force_mse_loss"])
print("[OK] 无力键时（LIBERO/ALOHA/Franka/no_force）future_force loss = NaN，行为不变")

# ---------- 3. 掩码分支包含未来力 slot ----------
Cfg.mask_value_prediction_loss_for_policy_prediction = True
self, _ = make_self(perfect_except=11)
out, kendall, *_ = run(self, with_force=True)
per_frame = kendall.mean(dim=(1, 3, 4))[0]
assert per_frame[11] > 0, per_frame
print("[OK] mask_value_prediction_loss_for_policy_prediction 分支下 slot 11 仍计入 loss")
print("ALL PASSED")
