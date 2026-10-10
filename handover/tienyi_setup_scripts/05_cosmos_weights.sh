#!/bin/bash
# 第 5 步：下载 Cosmos-Predict2-2B-Video2World 的 480p/16fps 权重和 Wan2.1 VAE（gated，需要 HF token）
# token 来源：环境变量 HF_TOKEN，或文件 /media/jushen/zhixuan/tienyi_setup/hf_token（chmod 600）
# 代码里写死的占位路径 /path/to/Cosmos-Predict2-2B-Video2World 用软链接指向实际目录，不改代码。
set -uo pipefail

DST=/media/jushen/zhixuan/checkpoints/Cosmos-Predict2-2B-Video2World
export HF_HOME=/media/jushen/zhixuan/hf_home
export HF_ENDPOINT=https://hf-mirror.com
if [ -z "${HF_TOKEN:-}" ] && [ -f /media/jushen/zhixuan/tienyi_setup/hf_token ]; then
  export HF_TOKEN=$(cat /media/jushen/zhixuan/tienyi_setup/hf_token)
fi
[ -n "${HF_TOKEN:-}" ] || { echo "缺少 HF_TOKEN"; exit 1; }
PY=/media/jushen/zhixuan/.venv_cpu_test/bin/python

for i in $(seq 1 30); do
  $PY - <<EOF && break
from huggingface_hub import hf_hub_download
for f in ["model-480p-16fps.pt", "tokenizer/tokenizer.pth", "config.json"]:
    p = hf_hub_download("nvidia/Cosmos-Predict2-2B-Video2World", f, local_dir="$DST")
    print("ok", p)
EOF
  echo "retry $i"; sleep 20
done
[ -f "$DST/model-480p-16fps.pt" ] && [ -f "$DST/tokenizer/tokenizer.pth" ] || { echo "下载失败"; exit 1; }

mkdir -p /path/to
ln -sfn "$DST" /path/to/Cosmos-Predict2-2B-Video2World
ls -la /path/to/Cosmos-Predict2-2B-Video2World/ /path/to/Cosmos-Predict2-2B-Video2World/tokenizer/
echo "STEP5_DONE"
