# 项目交接说明 (Dream-Tac-FT)

## 📌 交接情况
- 交接日期：2026-10-10
- 原开发环境：百度云 A100 主机 (`/media/jushen/zhixuan`)

## 📂 已提交到仓库的内容
| 文件夹 | 说明 |
|--------|------|
| `Dream-Tac/` (主分支) | 核心训练/推理代码 |
| `handover/tienyi_tests/` | 测试代码与验证脚本 |
| `handover/tienyi_eval/` | 实验结果（metrics.json、可视化图片） |
| `handover/tienyi_setup_scripts/` | 环境搭建与训练启动脚本 |

## 🚫 未提交的大型文件（接手人需自行处理）
| 文件/文件夹 | 大小 | 处理方式 |
|-------------|------|----------|
| `tienyi_setup/` | 6.1G | 见 `handover/tienyi_setup_scripts/` 中的脚本重新搭建 |
| `tienyi_preprocessed/` | 1.7G | 需运行预处理代码重新生成 |
| `checkpoints/` | - | 模型权重，需重新训练或单独下载 |

## 🚀 接手人快速开始
1. **克隆仓库**：
   ```bash
   git clone git@github.com:Jane-qm/Dream-Tac-FT.git
   cd Dream-Tac-FT
   git checkout 1010FT_as_input

