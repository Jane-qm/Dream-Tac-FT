#!/bin/bash
# 无人值守：双卡正式训练（失败自动从最新 checkpoint 续训，最多 3 次）→ val 集开环评估（最终 + 中间 checkpoint）
set -uo pipefail
cd /media/jushen/zhixuan/tienyi_setup
source env.sh

export EXP=${EXP:-cosmos_predict2_2b_480p_tienyi_battery_force}
export JOB=${JOB:-force_2gpu_b16_4k}
export NGPU=2 BATCH=${BATCH:-16} MAX_ITER=${MAX_ITER:-4000} SAVE_ITER=${SAVE_ITER:-500}
CKPT_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy_tienyi_battery/cosmos_v2_finetune/$JOB/checkpoints
FINAL=$(printf "iter_%09d" "$MAX_ITER")

for attempt in 1 2 3; do
  echo "[pipeline] $(date '+%F %T') train attempt $attempt"
  ./07_train.sh > "logs/${JOB}_train_try${attempt}.log" 2>&1
  [ -d "$CKPT_ROOT/$FINAL" ] && break
  echo "[pipeline] $(date '+%F %T') attempt $attempt ended without $FINAL, retrying in 60s"
  sleep 60
done
[ -d "$CKPT_ROOT/$FINAL" ] || { echo "[pipeline] training failed"; exit 1; }
echo "[pipeline] $(date '+%F %T') training done"

run_eval() {  # $1 = GPU, $2 = checkpoint dir name
  CUDA_VISIBLE_DEVICES=$1 $ENV_PY -m cosmos_policy.experiments.robot.tienyi.openloop_eval \
    --experiment "$EXP" --ckpt_dir "$CKPT_ROOT/$2" --data_dir "$TIENYI_DATA_DIR" \
    --out_dir "/media/jushen/zhixuan/tienyi_eval/$JOB/$2" --num_episodes 0 --step_stride 8 \
    > "/media/jushen/zhixuan/tienyi_setup/logs/${JOB}_eval_$2.log" 2>&1
}
cd "$REPO"
MID=$(printf "iter_%09d" $(( MAX_ITER / 2 )))
run_eval 0 "$FINAL" &
[ -d "$CKPT_ROOT/$MID" ] && run_eval 1 "$MID" &
wait
echo "[pipeline] $(date '+%F %T') eval done"
echo PIPELINE_DONE
