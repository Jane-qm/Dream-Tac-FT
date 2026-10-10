#!/bin/bash
# 第 3 步：转换 160 条成功数据（因果力滤波）并校验
set -euxo pipefail

REPO=/media/jushen/zhixuan/Dream-Tac
RAW=/media/jushen/project-rl-dataset/raw_data_0909_autoupdate/tienyi_prod2_dualArm-gripper-3cameras_515/back_battery_and_cover
DATA=/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160
export PYTHONPATH=$REPO

cd "$REPO"
python3 -m cosmos_policy.experiments.robot.tienyi.preprocess_tienyi_force_data \
  --raw_root "$RAW" --out_dir "$DATA" --max_episodes 160 --num_workers 32
python3 -m cosmos_policy.experiments.robot.tienyi.verify_tienyi_preprocessed \
  --data_dir "$DATA" --vis_dir "$DATA/../back_battery_and_cover_160_verify" --num_vis 3
echo "STEP3_DONE"
