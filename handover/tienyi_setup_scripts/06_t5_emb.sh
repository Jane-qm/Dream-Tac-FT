#!/bin/bash
# 第 4b 步：用本地 T5-11B encoder 生成 $DATA/t5_embeddings.pkl（与 save_franka_t5_text_embeddings 输出格式相同）
set -euxo pipefail

REPO=/media/jushen/zhixuan/Dream-Tac
DATA=/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160
export PYTHONPATH=$REPO
cd "$REPO"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /media/jushen/zhixuan/envs/dreamtac/bin/python - <<EOF
import json, os, torch
from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder, get_text_embedding
from cosmos_policy.datasets.t5_embedding_utils import save_embeddings

task = json.load(open(os.path.join("$DATA", "preprocessing_metadata.json")))["dataset_info"]["task_name"]
print("task:", repr(task))
enc = CosmosT5TextEncoder(model_name="/media/jushen/zhixuan/checkpoints/t5-11b-encoder", local_files_only=True)
emb = get_text_embedding(task, encoder=enc).to(dtype=torch.bfloat16).cpu()
print("emb", tuple(emb.shape), "nonzero tokens", int((emb.abs().sum(-1) > 0).sum()))
save_embeddings({task: emb}, "$DATA")
EOF
echo "STEP4B_DONE"
