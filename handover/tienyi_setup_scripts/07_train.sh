#!/bin/bash
# 训练启动（冒烟与正式共用）。用法：
#   NGPU=1 BATCH=8 MAX_ITER=20 JOB=smoke_b8 ./07_train.sh
#   NGPU=2 BATCH=16 MAX_ITER=12000 JOB=force_2gpu ./07_train.sh
# 其余参数：EXP（实验名，默认加力主实验）、SAVE_ITER、WARMUP、GPUS（CUDA_VISIBLE_DEVICES）、PORT
set -euo pipefail
source /media/jushen/zhixuan/tienyi_setup/env.sh

EXP=${EXP:-cosmos_predict2_2b_480p_tienyi_battery_force}
NGPU=${NGPU:-2}
BATCH=${BATCH:-8}
MAX_ITER=${MAX_ITER:-3000}
SAVE_ITER=${SAVE_ITER:-500}
WARMUP=${WARMUP:-$(( MAX_ITER / 20 > 1 ? MAX_ITER / 20 : 1 ))}
JOB=${JOB:-$EXP}
PORT=${PORT:-12341}
export CUDA_VISIBLE_DEVICES=${GPUS:-$(seq -s, 0 $((NGPU - 1)))}

cd "$REPO"
torchrun --nproc_per_node=$NGPU --master_port=$PORT -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=$EXP \
  job.name=$JOB job.wandb_id=$JOB job.wandb_mode=offline \
  model.config.fsdp_shard_size=$NGPU \
  dataloader_train.batch_size=$BATCH \
  trainer.max_iter=$MAX_ITER \
  "scheduler.cycle_lengths=[$MAX_ITER,100000000000000]" \
  "scheduler.warm_up_steps=[$WARMUP,0]" \
  checkpoint.save_iter=$SAVE_ITER \
  ${EXTRA:-}
