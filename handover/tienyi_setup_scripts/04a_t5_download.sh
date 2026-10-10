#!/bin/bash
# 第 4a 步：下载 T5-11B 到 HF 缓存（GPFS），只下 PyTorch 权重 + tokenizer，失败自动重试
set -uxo pipefail

export HF_HOME=/media/jushen/zhixuan/hf_home
export HF_ENDPOINT=https://hf-mirror.com
PY=/media/jushen/zhixuan/.venv_cpu_test/bin/python

for i in $(seq 1 30); do
  $PY - <<'EOF' && { echo "STEP4A_DONE"; exit 0; }
from huggingface_hub import list_repo_files, snapshot_download
files = list_repo_files("google-t5/t5-11b")
print(files)
weights = [f for f in files if f.endswith(".safetensors") or f == "model.safetensors.index.json"]
if not weights:
    weights = [f for f in files if f.startswith("pytorch_model") and (f.endswith(".bin") or f.endswith(".json"))]
allow = ["config.json", "spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"] + weights
print("allow:", allow)
p = snapshot_download("google-t5/t5-11b", allow_patterns=allow, max_workers=8)
print("downloaded to", p)
EOF
  echo "retry $i"; sleep 20
done
exit 1
