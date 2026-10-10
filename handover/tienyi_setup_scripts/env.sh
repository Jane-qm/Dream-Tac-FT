# 训练 / 评估统一环境：source /media/jushen/zhixuan/tienyi_setup/env.sh
export ENV_PY=/media/jushen/zhixuan/envs/dreamtac/bin/python
export REPO=/media/jushen/zhixuan/Dream-Tac
export PYTHONPATH=$REPO
export TIENYI_DATA_DIR=/media/jushen/zhixuan/tienyi_preprocessed/back_battery_and_cover_160
export HF_HOME=/media/jushen/zhixuan/hf_home
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
export IMAGINAIRE_OUTPUT_ROOT=/media/jushen/zhixuan/checkpoints
# transformer-engine 在 import 时直接 dlopen cuDNN 等库，需要能找到 pip 装的 nvidia-* 库
_NV=/media/jushen/zhixuan/envs/dreamtac/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$(ls -d $_NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
export PATH=/media/jushen/zhixuan/envs/dreamtac/bin:$PATH
