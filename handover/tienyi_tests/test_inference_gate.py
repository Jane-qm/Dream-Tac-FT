"""CPU 测试：get_x0_fn_from_batch 推理时的门控与当前力注入（从源码抽出真实函数，mock self / condition）。"""
import ast
import types
from dataclasses import dataclass, fields
from typing import Optional

import torch

SRC_V2W = "/media/jushen/zhixuan/Dream-Tac/cosmos_policy/models/policy_video2world_model.py"
SRC_T2W = "/media/jushen/zhixuan/Dream-Tac/cosmos_policy/models/policy_text2world_model.py"


def extract(path, names):
    tree = ast.parse(open(path).read())
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]


@dataclass
class MockCondition:
    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None
    tactile_self_attn_gate_B: Optional[torch.Tensor] = None

    def edit_data_type(self, _):
        return self

    def set_video_condition(self, gt_frames, num_conditional_frames, **_):
        self.gt_frames = gt_frames.clone()
        B, _, T, H, W = gt_frames.shape
        self.condition_video_input_mask_B_C_T_H_W = torch.zeros(B, 1, T, H, W)
        self.condition_video_input_mask_B_C_T_H_W[:, :, :num_conditional_frames] = 1
        return self

    def edit_for_inference(self, **_):
        return self

    def to_dict(self, skip_underscore=True):
        return {f.name: getattr(self, f.name) for f in fields(self)}


ns = {
    "torch": torch,
    "Dict": dict,
    "Callable": object,
    "NUM_CONDITIONAL_FRAMES_KEY": "num_conditional_frames",
    "DataType": types.SimpleNamespace(IMAGE="image", VIDEO="video"),
    "parallel_state": types.SimpleNamespace(is_initialized=lambda: True),
}
code = extract(SRC_T2W, {"replace_latent_with_proprio"}) + extract(SRC_V2W, {"get_x0_fn_from_batch"})
exec(compile(ast.Module(body=code, type_ignores=[]), SRC_V2W, "exec"), ns)
get_x0_fn_from_batch = ns["get_x0_fn_from_batch"]

B, C, T, H, W = 2, 16, 12, 28, 28
x0 = torch.randn(B, C, T, H, W)
holder = {}


def make_self():
    return types.SimpleNamespace(
        conditioner=types.SimpleNamespace(get_condition_uncondition=lambda b: (holder.setdefault("c", MockCondition()), None)),
        is_image_batch=lambda b: False,
        get_data_and_condition=lambda b: (None, x0.clone(), None),
        config=types.SimpleNamespace(min_num_conditional_frames=5, max_num_conditional_frames=5, conditional_frames_probs=None),
        broadcast_split_for_model_parallelsim=lambda x, c, e, s: (x, c, e, s),
        net=types.SimpleNamespace(is_context_parallel_enabled=False),
    )


def base_batch():
    idx = lambda v: torch.full((B,), v, dtype=torch.long)
    return {
        "num_conditional_frames": 5,
        "proprio": torch.rand(B, 20),
        "current_proprio_latent_idx": idx(1),
        "current_wrist_image_latent_idx": idx(2),
        "current_image_latent_idx": idx(4),
    }


holder.clear()
get_x0_fn_from_batch(make_self(), base_batch())
c = holder["c"]
assert c.tactile_self_attn_gate_B is None
assert torch.all(c.condition_video_input_mask_B_C_T_H_W[:, :, 5] == 0)
print("[OK] 无门控 / 无力键（原 LIBERO/ALOHA/Franka 推理）：condition 不变")

batch = base_batch()
force = torch.rand(B, 12) * 2 - 1
batch.update(force=force, current_force_latent_idx=torch.full((B,), 5), tactile_self_attn_gate=torch.tensor([0.3, 0.9]))
holder.clear()
_, orig = get_x0_fn_from_batch(make_self(), batch, return_orig_clean_latent_frames=True)
c = holder["c"]
assert torch.allclose(c.tactile_self_attn_gate_B, torch.tensor([0.3, 0.9]))
assert orig is not None and orig.shape == x0.shape
assert torch.all(c.condition_video_input_mask_B_C_T_H_W[:, :, 5] == 1)
assert torch.allclose(c.gt_frames[:, :, 5].reshape(B, -1)[:, :12], force)
print("[OK] 有门控：写入 condition.tactile_self_attn_gate_B，orig_gt_frames 保留")
print("[OK] 当前力槽位 5：条件掩码置 1 并注入力")
print("ALL PASSED")
