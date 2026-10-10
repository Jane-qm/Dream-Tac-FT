#!/bin/bash
# 第 1 步：conda 环境（Python 3.10）+ torch 2.7.0 cu128 + Dream-Tac 依赖（不含 transformer-engine / flash-attn）
set -euxo pipefail

ENV=/media/jushen/zhixuan/envs/dreamtac
REPO=/media/jushen/zhixuan/Dream-Tac
CONDA_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

if [ ! -x "$ENV/bin/python" ]; then
  /root/miniforge/bin/conda create -y -p "$ENV" python=3.10 --override-channels -c "$CONDA_MIRROR"
fi
PY="$ENV/bin/python"

$PY -m pip install -U pip wheel setuptools
WHEELS=/media/jushen/zhixuan/tienyi_setup/downloads/wheels   # dl_torch.sh 多连接下载的 wheel（pip 直连太慢）
if ls "$WHEELS"/torch-2.7.0+cu128-*.whl >/dev/null 2>&1; then
  $PY -m pip install "$WHEELS"/torch-2.7.0+cu128-*.whl "$WHEELS"/torchvision-0.22.0+cu128-*.whl
else
  $PY -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
fi

# 防止后续依赖把 torch 换成别的版本
CONSTRAINTS=/media/jushen/zhixuan/tienyi_setup/constraints.txt
printf "torch==2.7.0\ntorchvision==0.22.0\n" > "$CONSTRAINTS"

# cosmos_policy 目录结构不适合 pip install -e（训练用 PYTHONPATH=<repo>），直接装依赖
$PY -m pip install -r /media/jushen/zhixuan/tienyi_setup/requirements_dreamtac.txt -c "$CONSTRAINTS"
$PY -m pip install -c "$CONSTRAINTS" ninja packaging pybind11 "nvidia-cudnn-cu12>=9.5,<10"

$PY - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (x @ x).float().abs().mean().item())
import megatron.core, transformers, diffusers  # noqa
print("megatron / transformers / diffusers import ok")
EOF
echo "STEP1_DONE"
