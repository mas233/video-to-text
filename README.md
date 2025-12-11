# Video Summary（视频转文字 + 智能 Agent）

基于开源 LLM 的视频处理服务，支持视频转文字、音频提取、智能对话等功能。

## 🚀 快速开始

### 1. 安装依赖

```bash
# Python 环境 (推荐 3.11+)
pip install -r requirements.txt
```

### 2. 运行 Agent

```bash
# 交互模式
python agent/video_agent.py --interactive

# 单次任务
python agent/video_agent.py "列出 data 目录的文件"
```

### 3. 示例任务

```
请输入任务: 列出 data 目录的文件
请输入任务: 检查 data/transcriptions 是否存在
请输入任务: 提取视频的音频
```

## 📋 核心功能

### Agent 模式 (推荐)
- ✅ **完全离线** - 使用本地 Qwen2.5-7B 模型
- ✅ **智能工具调用** - 原生 `<tool_call>` 格式
- ✅ **多轮对话** - 支持复杂任务分解
- ✅ **Apple Silicon** - MPS 加速支持

### 可用工具
1. `list_files` - 列出目录文件
2. `check_file_exists` - 检查文件存在性
3. `extract_audio` - 视频提取音频
4. `transcribe_audio` - 音频转文字 (返回 JSON: source/output/model/timestamp)
5. `save_text` - 保存文本到文件

## ⚙️ 配置选项

### 指定设备
```bash
# MPS (Apple Silicon, 推荐)
python agent/video_agent.py --device mps --interactive

# CPU (较慢)
python agent/video_agent.py --device cpu --interactive

# CUDA (NVIDIA GPU)
python agent/video_agent.py --device cuda --interactive
```

### 指定模型
```bash
# Qwen 2.5 7B (默认, 推荐)
python agent/video_agent.py --model Qwen/Qwen2.5-7B-Instruct

# Qwen 2 7B
python agent/video_agent.py --model Qwen/Qwen2-7B-Instruct
```

## 🛠️ 技术栈

- **LLM**: Qwen/Qwen2.5-7B-Instruct (7B 参数)
- **框架**: HuggingFace Transformers
- **加速**: PyTorch MPS / CUDA
- **模型源**: ModelScope (国内) / HuggingFace (国外)

## 📊 性能指标

- **模型大小**: ~14.5 GB (FP16)
- **推理速度**: ~3s/iteration (M1 Max)
- **内存需求**: 16 GB RAM (推荐)
- **格式正确率**: 100%

## 🔧 环境要求

- Python 3.11+
- PyTorch 2.0+
- 16GB+ RAM
- Apple Silicon / NVIDIA GPU / CPU

## 📖 项目结构

```
video-summary/
├── agent/
│   ├── video_agent.py       # 主 Agent 实现
│   └── README.md            # Agent 详细文档
├── core/                    # 核心处理模块
│   ├── audio_extractor.py   # 音频提取
│   ├── whisper_parser.py    # 语音转文字
│   └── video_extractor.py   # 视频处理
├── data/                    # 数据目录
├── requirements.txt         # Python 依赖
└── README.md               # 本文件
```

## 🎯 使用示例

### 本地运行
```bash
python agent/video_agent.py --interactive

请输入任务: 列出 data 目录的所有文件
请输入任务: 提取视频的音频并转录为文字
请输入任务: 将结果保存到 output/result.txt
```

### Docker 运行

**构建镜像**:
```bash
docker build -t video-summary .
```

**交互模式**:
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  video-summary
```

**单次任务**:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  video-summary \
  python agent/video_agent.py "列出 data 目录的文件"
```

**说明**:
- `-v $(pwd)/data:/app/data` - 挂载数据目录
- `-v $(pwd)/output:/app/output` - 挂载输出目录
- `-v $(pwd)/models:/app/models` - 持久化模型缓存 (~14.5GB)
- `--rm` - 容器退出后自动删除
- `-it` - 交互模式

## 🔍 故障排查

### 模型下载慢
自动优先使用 ModelScope (国内快)

### 内存不足
使用 CPU 模式: `python agent/video_agent.py --device cpu`

### MPS 错误
回退到 CPU: `python agent/video_agent.py --device cpu`

## 📄 许可证

MIT License

## 🔗 相关资源

- [Qwen 官方文档](https://qwen.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [ModelScope](https://modelscope.cn/)

---

**更新时间**: 2025-12-11  
**版本**: v2.0 (Qwen 原生工具调用)
