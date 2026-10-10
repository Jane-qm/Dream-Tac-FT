#!/bin/bash
# 第 2c 步：源码编译 flash-attn 2.7.3（只编 sm_80）。
# 训练本身不用 flash-attn（attention 走 TE / SDPA），但 config 导入链上的 _src/reason1/networks/qwen2_5_vl.py
# 在 import 时 assert flash_attn 可用，所以必须装。GitHub 不通，不能用预编译 wheel。
set -euxo pipefail

source /media/jushen/zhixuan/tienyi_setup/env.sh
export CUDA_HOME=/media/jushen/zhixuan/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=64
command -v g++ >/dev/null || apt-get install -y -q g++

$ENV_PY -m pip install --no-build-isolation -c /media/jushen/zhixuan/tienyi_setup/constraints.txt \
    /media/jushen/zhixuan/tienyi_setup/downloads/flash_attn-2.7.3.tar.gz

$ENV_PY - <<'EOF'
import torch
from flash_attn import flash_attn_func
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
print("flash_attn ok", flash_attn_func(q, q, q).shape)
EOF
echo "STEP2C_DONE"
