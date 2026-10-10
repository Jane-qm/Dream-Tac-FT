#!/bin/bash
# 第 4/5 步：从 ModelScope（nv-community/Cosmos-Predict2-2B-Video2World，与 HF 官方仓库同内容，无需 token）下载
#   - model-480p-16fps.pt + tokenizer/tokenizer.pth（Wan2.1 VAE）：训练用
#   - text_encoder/（T5-11B encoder）+ tokenizer/ 里的 T5 分词器：生成 t5_embeddings.pkl 用
# 代码里写死的占位路径 /path/to/Cosmos-Predict2-2B-Video2World 用软链接指向实际目录，不改代码。
set -uxo pipefail

DST=/media/jushen/zhixuan/checkpoints/Cosmos-Predict2-2B-Video2World
PY=/media/jushen/zhixuan/.venv_cpu_test/bin/python
mkdir -p "$DST"

FILES="config.json model-480p-16fps.pt tokenizer/tokenizer.pth
tokenizer/spiece.model tokenizer/tokenizer.json tokenizer/tokenizer_config.json tokenizer/special_tokens_map.json
text_encoder/config.json text_encoder/model.safetensors.index.json
text_encoder/model-00001-of-00002.safetensors text_encoder/model-00002-of-00002.safetensors"

for i in $(seq 1 20); do
  $PY /media/jushen/zhixuan/tienyi_setup/ms_dl.py nv-community/Cosmos-Predict2-2B-Video2World "$DST" $FILES && break
  echo "retry $i"; sleep 10
done
for f in $FILES; do [ -s "$DST/$f" ] || { echo "缺文件 $f"; exit 1; }; done

mkdir -p /path/to
ln -sfn "$DST" /path/to/Cosmos-Predict2-2B-Video2World

# T5 本地目录：encoder 权重 + 分词器放一起，供 CosmosT5TextEncoder(model_name=本地目录) 加载
T5=/media/jushen/zhixuan/checkpoints/t5-11b-encoder
mkdir -p "$T5"
for f in "$DST"/text_encoder/* "$DST"/tokenizer/spiece.model "$DST"/tokenizer/tokenizer.json \
         "$DST"/tokenizer/tokenizer_config.json "$DST"/tokenizer/special_tokens_map.json; do
  ln -sfn "$f" "$T5/$(basename "$f")"
done
ls -laL /path/to/Cosmos-Predict2-2B-Video2World/ "$T5"
echo "STEP4_DONE"
