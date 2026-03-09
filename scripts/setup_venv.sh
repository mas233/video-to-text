#!/bin/bash
# Video Summary 项目 - Python 虚拟环境自动配置脚本
# 使用默认 pip 源 / 优先 HuggingFace 下载模型

# ─── 平台配置 ────────────────────────────────────────────────────────────────
PIP_INDEX_ARGS=""               # 使用 pip 默认源
PREFERRED_PLATFORM="huggingface" # Whisper 模型优先从 HuggingFace 下载
# ─────────────────────────────────────────────────────────────────────────────

# shellcheck source=scripts/setup_common.sh
source "$(dirname "$0")/setup_common.sh"
run_setup
