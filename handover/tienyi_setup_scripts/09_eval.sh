#!/bin/bash
# 仅评估：对已训练好的 checkpoint 跑 val 集开环评估（最终 + 中间）
set -uo pipefail
cd /media/jushen/zhixuan/tienyi_setup
source env.sh

export EXP=${EXP:-cosmos_predict2_2b_480p_tienyi_battery_force}
export JOB=${JOB:-force_2gpu_b16_4k}
MAX_ITER=${MAX_ITER:-4000}
CKPT_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy_tienyi_battery/cosmos_v2_finetune/$JOB/checkpoints
LOGDIR=/media/jushen/zhixuan/tienyi_setup/logs
FINAL=$(printf "iter_%09d" "$MAX_ITER")
MID=$(printf "iter_%09d" $(( MAX_ITER / 2 )))

run_eval() {  # $1 = GPU, $2 = checkpoint dir name
  CUDA_VISIBLE_DEVICES=$1 $ENV_PY -m cosmos_policy.experiments.robot.tienyi.openloop_eval \
    --experiment "$EXP" --ckpt_dir "$CKPT_ROOT/$2" --data_dir "$TIENYI_DATA_DIR" \
    --out_dir "/media/jushen/zhixuan/tienyi_eval/$JOB/$2" --num_episodes 0 --step_stride 8 \
    > "$LOGDIR/${JOB}_eval_$2.log" 2>&1
  echo "[eval] $(date '+%F %T') $2 exit=$?"
}
cd "$REPO"
echo "[eval] $(date '+%F %T') start"
run_eval 0 "$FINAL" &
run_eval 1 "$MID" &
wait
echo "[eval] $(date '+%F %T') EVAL_DONE"
