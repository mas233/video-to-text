#!/bin/bash
# Video Summary - 公共环境配置逻辑
# 此文件不可直接运行，请通过 setup_venv.sh 或 setup_venv_cn.sh 调用
#
# 调用前须设置以下变量：
#   PIP_INDEX_ARGS      pip 额外参数，如 "-i https://pypi.tuna.tsinghua.edu.cn/simple"
#                       或空字符串（使用默认源）
#   PREFERRED_PLATFORM  "modelscope" 或 "huggingface"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "此脚本不可直接运行，请使用 setup_venv.sh 或 setup_venv_cn.sh"
    exit 1
fi

# ─── 颜色 ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ─── Python 最低版本要求 ──────────────────────────────────────────────────────
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ─── 全局状态（由步骤函数填写） ───────────────────────────────────────────────
VENV_EXISTS=false
PYTHON_CMD=""
WHISPER_NAME=""
WHISPER_HF_ID=""
WHISPER_MS_ID=""
WHISPER_MEM=""

# ═════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════

# 获取 Python 版本字符串（失败返回空）
_get_python_version() {
    local py_cmd=$1
    if command -v "$py_cmd" &>/dev/null; then
        $py_cmd --version 2>&1 | awk '{print $2}'
    fi
}

# 判断版本是否满足最低要求
_version_ge() {
    local major minor
    major=$(echo "$1" | cut -d. -f1)
    minor=$(echo "$1" | cut -d. -f2)
    [ "$major" -gt "$MIN_PYTHON_MAJOR" ] && return 0
    [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ] && return 0
    return 1
}

# 获取系统可用内存（GB）
_detect_memory_gb() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Apple Silicon GPU/CPU 共享内存，取总量
        echo $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    else
        # Linux: 取 MemAvailable
        echo $(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    fi
}

# 获取 NVIDIA GPU 显存（GB），不支持时返回 0
_detect_gpu_memory_gb() {
    if command -v nvidia-smi &>/dev/null; then
        local gpu_mem
        gpu_mem=$(nvidia-smi --query-gpu=memory.total \
                      --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_mem" ] && [ "$gpu_mem" -gt 0 ]; then
            echo $(( gpu_mem / 1024 ))
            return
        fi
    fi
    echo "0"
}

# 根据系统资源选择 Whisper 模型
# 阈值与 whisper_parser.py select_model_by_resources() 保持一致:
#   >=8 GB → large-v3-turbo  |  >=4 GB → medium  |  其余 → small
# 结果写入全局变量 WHISPER_NAME / WHISPER_HF_ID / WHISPER_MS_ID / WHISPER_MEM
_select_whisper_model() {
    local mem_gb gpu_gb total_gb
    mem_gb=$(_detect_memory_gb)
    gpu_gb=$(_detect_gpu_memory_gb)
    total_gb=$(( mem_gb > gpu_gb ? mem_gb : gpu_gb ))
    WHISPER_MEM="${total_gb}GB"

    if [ "$total_gb" -ge 8 ]; then
        WHISPER_NAME="large-v3-turbo"
        WHISPER_HF_ID="openai/whisper-large-v3-turbo"
        WHISPER_MS_ID="AI-ModelScope/whisper-large-v3-turbo"
    elif [ "$total_gb" -ge 4 ]; then
        WHISPER_NAME="medium"
        WHISPER_HF_ID="openai/whisper-medium"
        WHISPER_MS_ID="AI-ModelScope/whisper-medium"
    else
        WHISPER_NAME="small"
        WHISPER_HF_ID="openai/whisper-small"
        WHISPER_MS_ID="AI-ModelScope/whisper-small"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# 步骤函数（每步独立，出错才 exit 1；已满足条件则直接 return）
# ═════════════════════════════════════════════════════════════════════════════

# 步骤 1：检查虚拟环境是否已存在
# - 存在且完整 → 设置 VENV_EXISTS=true，返回（后续步骤会跳过安装）
# - 存在但损坏 → 删除后继续
# - 不存在     → 继续创建
step1_check_venv() {
    echo "【步骤 1/6】检查虚拟环境..."
    if [ ! -d ".venv" ]; then
        echo "  未检测到 .venv，将新建虚拟环境"
        return
    fi

    if [ -f ".venv/bin/activate" ]; then
        echo -e "${GREEN}✓${NC} 虚拟环境 .venv 已存在且完整，跳过创建与依赖安装"
        VENV_EXISTS=true
    else
        echo -e "${YELLOW}⚠${NC}  虚拟环境损坏，正在清理并重建..."
        rm -rf .venv
    fi
}

# 步骤 2：查找合适的 Python（VENV_EXISTS 时跳过）
step2_find_python() {
    if [ "$VENV_EXISTS" = true ]; then
        echo "【步骤 2/6】跳过（虚拟环境已存在）"
        return
    fi

    echo "【步骤 2/6】查找 Python (>= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})..."
    local py_commands=("python3" "python" "python3.13" "python3.12" "python3.11" "python3.10")
    for cmd in "${py_commands[@]}"; do
        local ver
        ver=$(_get_python_version "$cmd")
        if [ -n "$ver" ] && _version_ge "$ver"; then
            PYTHON_CMD="$cmd"
            echo -e "${GREEN}✓${NC} 找到 Python: $cmd（版本 $ver）"
            return
        fi
    done

    echo -e "${RED}✗${NC} 未找到 Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"
    echo "  安装方式："
    echo "    macOS:  brew install python@3.11"
    echo "    Ubuntu: sudo apt install python3.11 python3.11-venv"
    echo "    官方:   https://www.python.org/downloads/"
    exit 1
}

# 步骤 3：创建虚拟环境（VENV_EXISTS 时跳过）
step3_create_venv() {
    if [ "$VENV_EXISTS" = true ]; then
        echo "【步骤 3/6】跳过（虚拟环境已存在）"
        return
    fi

    echo "【步骤 3/6】创建虚拟环境..."
    if ! "$PYTHON_CMD" -m venv .venv; then
        echo -e "${RED}✗${NC} 虚拟环境创建失败"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} 虚拟环境创建成功"
}

# 步骤 4：激活环境并安装依赖（已存在时只激活，不重装）
step4_activate_and_install() {
    echo "【步骤 4/6】激活虚拟环境..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} 虚拟环境已激活"

    if [ "$VENV_EXISTS" = true ]; then
        echo "  跳过依赖安装（虚拟环境已就绪）"
        return
    fi

    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}✗${NC} 未找到 requirements.txt"
        exit 1
    fi

    echo "  → 升级 pip..."
    # shellcheck disable=SC2086
    python -m pip install --upgrade pip -q ${PIP_INDEX_ARGS}

    echo "  → 安装依赖包（这可能需要几分钟）..."
    # shellcheck disable=SC2086
    pip install -r requirements.txt -q ${PIP_INDEX_ARGS}

    echo -e "${GREEN}✓${NC} 依赖包安装完成"
}

# 步骤 5：验证核心库
step5_verify() {
    echo "【步骤 5/6】验证核心库..."
    python -c "
import torch, transformers
print(f'  → PyTorch:        {torch.__version__}')
print(f'  → Transformers:   {transformers.__version__}')
" 2>/dev/null \
        && echo -e "${GREEN}✓${NC} 核心库验证通过" \
        || echo -e "${YELLOW}⚠${NC}  部分库可能未正确安装，请检查依赖"
}

# 步骤 6：检测硬件，选择并可选预下载 Whisper 模型
step6_whisper() {
    echo "【步骤 6/6】检测硬件，选择 Whisper 语音识别模型..."
    _select_whisper_model

    echo -e "${GREEN}✓${NC} 可用内存/显存: ${WHISPER_MEM} → 推荐模型: ${WHISPER_NAME}"

    # 根据 PREFERRED_PLATFORM 决定展示顺序
    if [ "$PREFERRED_PLATFORM" = "modelscope" ]; then
        echo "  ★ 推荐 ModelScope : ${WHISPER_MS_ID}"
        echo "    备用 HuggingFace : ${WHISPER_HF_ID}"
    else
        echo "  ★ 推荐 HuggingFace : ${WHISPER_HF_ID}"
        echo "    备用 ModelScope  : ${WHISPER_MS_ID}"
    fi
    echo ""

    read -r -p "是否现在预下载 Whisper 模型 ${WHISPER_NAME}？(y/N): " DOWNLOAD_NOW
    if [[ ! "$DOWNLOAD_NOW" =~ ^[Yy]$ ]]; then
        echo "  已跳过，首次转录时将自动下载。"
        return
    fi

    echo "  → 正在下载 ${WHISPER_NAME}（可能需要数分钟）..."

    # PYEOF 不加引号，允许 shell 展开变量
    python - <<PYEOF 2>&1 || echo -e "${YELLOW}⚠${NC}  下载失败，首次转录时将重试"
import os, sys, torch
sys.path.insert(0, '.')

preferred = "${PREFERRED_PLATFORM}"
hf_id     = "${WHISPER_HF_ID}"
ms_id     = "${WHISPER_MS_ID}"
cache     = os.getenv("TRANSFORMERS_CACHE", "./models")

def try_modelscope():
    from modelscope.hub.snapshot_download import snapshot_download
    d = snapshot_download(ms_id, cache_dir=cache)
    print(f"  ModelScope 下载完成: {d}")

def try_huggingface():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    AutoProcessor.from_pretrained(hf_id, cache_dir=cache)
    AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_id, cache_dir=cache,
        low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=dtype
    )
    print(f"  HuggingFace 下载完成: {hf_id}")

# 按 PREFERRED_PLATFORM 确定尝试顺序
first, second = (try_modelscope, try_huggingface) \
    if preferred == "modelscope" else (try_huggingface, try_modelscope)

try:
    first()
except Exception as e:
    print(f"  主平台失败（{e}），尝试备用平台...")
    second()
PYEOF

    echo -e "${GREEN}✓${NC} Whisper 模型已缓存至 ./models/"
}

# ═════════════════════════════════════════════════════════════════════════════
# 输出辅助
# ═════════════════════════════════════════════════════════════════════════════

_print_header() {
    echo "======================================"
    echo "  Video Summary - 环境配置"
    echo "======================================"
    echo ""
}

_print_summary() {
    local whisper_link
    if [ "$PREFERRED_PLATFORM" = "modelscope" ]; then
        whisper_link="ModelScope: ${WHISPER_MS_ID}"
    else
        whisper_link="HuggingFace: ${WHISPER_HF_ID}"
    fi

    echo ""
    echo "======================================"
    echo -e "${GREEN}✓ 环境配置完成！${NC}"
    echo "======================================"
    echo ""
    echo -e "${YELLOW}下一步:${NC}"
    echo -e "  1. 激活环境:  ${GREEN}source .venv/bin/activate${NC}"
    echo "  2. 运行 Agent: ${GREEN}python agent/video_agent.py --interactive${NC}"
    echo ""
    echo -e "${YELLOW}硬件提示:${NC}"
    echo "  - 推荐 Whisper 模型: ${WHISPER_NAME}（${whisper_link}）"
    echo "  - 推荐使用 Apple Silicon MPS 或 NVIDIA GPU 加速"
    echo "  - CPU 模式较慢但也可用"
    echo ""
}

# ═════════════════════════════════════════════════════════════════════════════
# 主流程（由各入口脚本调用）
# ═════════════════════════════════════════════════════════════════════════════

run_setup() {
    _print_header
    step1_check_venv
    step2_find_python
    step3_create_venv
    step4_activate_and_install
    step5_verify
    step6_whisper
    _print_summary
}
