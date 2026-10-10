#!/bin/bash
# torch / torchvision cu128 wheel 多连接下载（pip 单连接只有 ~200 KB/s）
set -euxo pipefail
W=/media/jushen/zhixuan/tienyi_setup/downloads/wheels
mkdir -p $W
cd /media/jushen/zhixuan/tienyi_setup
/media/jushen/zhixuan/.venv_cpu_test/bin/python - <<'PY'
import ms_dl
W = "/media/jushen/zhixuan/tienyi_setup/downloads/wheels"
for name in ["torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl",
             "torchvision-0.22.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"]:
    ms_dl.download_url(f"https://download.pytorch.org/whl/cu128/{name}", f"{W}/{name.replace('%2B', '+')}")
PY
echo DL_TORCH_DONE
