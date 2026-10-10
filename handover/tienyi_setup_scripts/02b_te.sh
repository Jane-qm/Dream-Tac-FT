#!/bin/bash
# 第 2b 步：用本地 CUDA 12.8 编译 transformer-engine 2.2.0（只编 A100 / sm_80）
set -euxo pipefail

ENV=/media/jushen/zhixuan/envs/dreamtac
PY="$ENV/bin/python"
export CUDA_HOME=/media/jushen/zhixuan/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDNN_PATH=$($PY -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")
export CPATH="$CUDNN_PATH/include:${CPATH:-}"
export LIBRARY_PATH="$CUDNN_PATH/lib:${LIBRARY_PATH:-}"
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=80
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=32

command -v g++ >/dev/null || apt-get install -y -q g++   # 容器里默认没有编译器
nvcc --version
$PY -m pip install --no-build-isolation -c /media/jushen/zhixuan/tienyi_setup/constraints.txt \
    "transformer_engine[pytorch]==2.2.0"

$PY - <<'EOF'
import torch, transformer_engine.pytorch as te
m = te.Linear(256, 256).cuda()
y = m(torch.randn(8, 256, device="cuda"))
print("transformer_engine ok", y.shape)
EOF
echo "STEP2B_DONE"
