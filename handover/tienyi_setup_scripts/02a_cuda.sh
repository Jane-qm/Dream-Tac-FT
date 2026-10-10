#!/bin/bash
# 第 2a 步：下载并安装 CUDA 12.8.1 toolkit 到 GPFS（只装 toolkit，不装驱动），用于编译 transformer-engine
set -euxo pipefail

DST=/media/jushen/zhixuan/cuda-12.8
DL=/media/jushen/zhixuan/tienyi_setup/downloads
RUN=cuda_12.8.1_570.124.06_linux.run
mkdir -p "$DL" "$DL/tmp"

if [ ! -x "$DST/bin/nvcc" ]; then
  if [ ! -f "$DL/$RUN.done" ]; then
    wget -c -q --show-progress --progress=dot:giga -O "$DL/$RUN" \
      "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/$RUN"
    touch "$DL/$RUN.done"
  fi
  sh "$DL/$RUN" --silent --toolkit --toolkitpath="$DST" --no-man-page --override --tmpdir="$DL/tmp"
fi
"$DST/bin/nvcc" --version
echo "STEP2A_DONE"
