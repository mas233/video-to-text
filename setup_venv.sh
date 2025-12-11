#!/bin/bash

# Video Summary 项目 - Python 虚拟环境自动配置脚本
# 功能：检查并创建 Python 虚拟环境，安装依赖

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python 最低版本要求
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

echo "======================================"
echo "  Video Summary - 环境配置"
echo "======================================"
echo ""

# 函数: 获取 Python 版本
get_python_version() {
    local py_cmd=$1
    if command -v "$py_cmd" &> /dev/null; then
        # 获取版本号，例如: "Python 3.11.5" -> "3.11.5"
        local version=$($py_cmd --version 2>&1 | awk '{print $2}')
        echo "$version"
    else
        echo ""
    fi
}

# 函数: 比较版本号
version_ge() {
    local version=$1
    local major=$(echo "$version" | cut -d. -f1)
    local minor=$(echo "$version" | cut -d. -f2)
    
    if [ "$major" -gt "$MIN_PYTHON_MAJOR" ]; then
        return 0
    elif [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
        return 0
    else
        return 1
    fi
}

# 函数: 查找合适的 Python 命令
find_python() {
    # 尝试的 Python 命令列表
    local py_commands=("python3" "python" "python3.13" "python3.12" "python3.11" "python3.10")
    
    for cmd in "${py_commands[@]}"; do
        local version=$(get_python_version "$cmd")
        if [ -n "$version" ]; then
            if version_ge "$version"; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    
    return 1
}

# 步骤 1: 检查 .venv 是否已存在
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} 发现已存在的虚拟环境: .venv"
    
    # 检查虚拟环境是否可用
    if [ -f ".venv/bin/activate" ]; then
        echo ""
        echo "虚拟环境已就绪！"
        echo ""
        echo -e "${YELLOW}请运行以下命令激活环境:${NC}"
        echo -e "  ${GREEN}source .venv/bin/activate${NC}"
        echo ""
        echo "激活后可运行:"
        echo "  python agent/video_agent.py --interactive"
        echo ""
        exit 0
    else
        echo -e "${YELLOW}⚠${NC}  虚拟环境损坏，将重新创建..."
        rm -rf .venv
    fi
fi

# 步骤 2: 查找合适的 Python
echo "正在查找 Python (>= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})..."
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}✗${NC} 未找到合适的 Python 版本"
    echo ""
    echo "要求: Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}"
    echo ""
    echo "请先安装 Python:"
    echo "  - macOS:   brew install python@3.11"
    echo "  - Ubuntu:  sudo apt install python3.11 python3.11-venv"
    echo "  - 官方:    https://www.python.org/downloads/"
    echo ""
    exit 1
fi

PYTHON_VERSION=$(get_python_version "$PYTHON_CMD")
echo -e "${GREEN}✓${NC} 找到 Python: $PYTHON_CMD (版本 $PYTHON_VERSION)"

# 步骤 3: 创建虚拟环境
echo ""
echo "正在创建虚拟环境..."
$PYTHON_CMD -m venv .venv

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗${NC} 虚拟环境创建失败"
    exit 1
fi

echo -e "${GREEN}✓${NC} 虚拟环境创建成功"

# 步骤 4: 激活虚拟环境并安装依赖
echo ""
echo "正在安装依赖包..."

# 激活虚拟环境
source .venv/bin/activate

# 升级 pip
echo "  → 升级 pip..."
python -m pip install --upgrade pip -q

# 安装 requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗${NC} 未找到 requirements.txt"
    exit 1
fi

echo "  → 安装依赖包 (这可能需要几分钟)..."
pip install -r requirements.txt -q

echo -e "${GREEN}✓${NC} 依赖包安装完成"

# 步骤 5: 验证关键包
echo ""
echo "验证安装..."
python -c "import torch; import transformers; print('  → PyTorch:', torch.__version__); print('  → Transformers:', transformers.__version__)" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} 核心库验证通过" || \
    echo -e "${YELLOW}⚠${NC}  部分库可能未正确安装"

# 完成
echo ""
echo "======================================"
echo -e "${GREEN}✓ 环境配置完成！${NC}"
echo "======================================"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo -e "  1. 激活环境:  ${GREEN}source .venv/bin/activate${NC}"
echo "  2. 运行 Agent: ${GREEN}python agent/video_agent.py --interactive${NC}"
echo ""
echo -e "${YELLOW}提示:${NC}"
echo "  - 首次运行会下载模型 (~14.5 GB)"
echo "  - 推荐使用 Apple Silicon MPS 或 NVIDIA GPU 加速"
echo "  - CPU 模式较慢但也可用"
echo ""
