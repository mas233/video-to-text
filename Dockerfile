# Video Summary - Agent 模式 Docker 镜像
# 使用 Python 3.11 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
# - ffmpeg: 视频/音频处理
# - git: 克隆模型仓库
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p /app/models /app/data /app/output

# 复制应用代码
COPY core/ /app/core/
COPY agent/ /app/agent/
COPY mcp/ /app/mcp/

# 设置模型缓存目录
# 建议通过 volume 挂载持久化模型 (~14.5GB)
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# 工作目录
WORKDIR /app

# 默认命令：交互模式
# 用户可以覆盖此命令
CMD ["python", "agent/video_agent.py", "--interactive"]
