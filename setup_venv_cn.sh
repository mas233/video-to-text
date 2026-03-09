#!/bin/bash
# Video Summary 项目 - Python 虚拟环境自动配置脚本
# 使用清华 pip 镜像源 / 优先 ModelScope 下载模型

# ─── 平台配置 ────────────────────────────────────────────────────────────────
PIP_INDEX_ARGS="-i https://pypi.tuna.tsinghua.edu.cn/simple"  # 清华 pip 镜像
PREFERRED_PLATFORM="modelscope"  # Whisper 模型优先从 ModelScope 下载
# ─────────────────────────────────────────────────────────────────────────────

# shellcheck source=setup_common.sh
source "$(dirname "$0")/setup_common.sh"
run_setup
