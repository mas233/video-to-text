# 视频处理智能 Agent (开源模型版本)

基于 LangChain + HuggingFace 的自主视频处理 Agent，使用开源 LLM，具备 Chain of Thought 推理和自我纠错能力。

## ✨ 特性

- ✅ **开源模型**: 使用 DeepSeek-R1/Qwen3.5 等开源 LLM，无需 API 费用
- ✅ **离线运行**: 模型下载后可完全离线使用
- ✅ **文件系统访问**: 浏览和检查文件
- ✅ **MCP 工具调用**: 调用视频处理工具
- ✅ **Chain of Thought**: ReAct 推理模式
- ✅ **自我纠错**: 分析错误并尝试修复
- ✅ **错误日志**: 完整的错误追踪和记录
- ✅ **交互模式**: 支持对话式操作
- ✅ **量化支持**: 8-bit/4-bit 量化节省显存

## 可用工具

Agent 可以使用以下工具:

### 1. list_files
列出目录下的文件

### 2. check_file_exists
检查文件是否存在

### 3. extract_audio_from_video
从视频中提取音频

### 4. transcribe_audio_to_text
将音频转录为文字

### 5. video_to_text_complete
完整的视频转文字流程

## 支持的模型

| 模型 | 参数量 | 中文能力 | 推荐度 | 备注 |
|------|--------|---------|--------|------|
| **deepseek-ai/DeepSeek-R1-0528-Qwen3-8B** | 8B | ⭐⭐⭐⭐⭐ | ✅ 强烈推荐 | 推理能力极强，基于 Qwen3-8B 蒸馏 |
| Qwen/Qwen3.5-9B | 9.65B | ⭐⭐⭐⭐⭐ | ✅ 推荐 | 多模态，支持 262K 上下文 |
| Qwen/Qwen3.5-27B | 27.78B | ⭐⭐⭐⭐⭐ | 可选（需较多显存）| 多模态旗舰，综合能力最强 |

### 模型下载链接

| 模型 | ModelScope | HuggingFace |
|------|-----------|-------------|
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | [ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) |
| Qwen/Qwen3.5-9B | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B) | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen/Qwen3.5-27B | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B) | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B) |

## 使用方法

### 前置要求

```bash
# 安装依赖
pip install -r requirements.txt

# 首次运行会自动下载模型（~14GB，需要时间）
# 模型会缓存到 ./models/ 目录
```

**硬件要求:**
- **GPU (CUDA/MPS)**: 推荐，速度快
  - 16GB+ 显存: 可直接运行 7B 模型
  - 8-12GB 显存: 使用 `--8bit` 量化
  - 4-6GB 显存: 使用 `--4bit` 量化
- **CPU**: 可用但很慢（~5-10 分钟/次推理）
  - 需要 16GB+ 内存

### 快速开始

#### 1. 交互模式（推荐）

```bash
# 使用默认模型 (DeepSeek-R1-0528-Qwen3-8B)
python agent/video_agent.py --interactive

# 使用 8-bit 量化（节省显存）
python agent/video_agent.py --interactive --8bit

# 使用 4-bit 量化（更省显存）
python agent/video_agent.py --interactive --4bit

# 指定 Qwen3.5-9B
python agent/video_agent.py --interactive --model Qwen/Qwen3.5-9B

# 指定 Qwen3.5-27B（需要较多显存）
python agent/video_agent.py --interactive --model Qwen/Qwen3.5-27B
```

示例对话:
```
正在初始化 Agent (开源模型版本)...
模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
设备: mps
缓存目录: ./models
正在加载模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
✓ 模型加载完成
Agent 初始化完成!

请输入任务: 列出 data 目录下的所有视频文件

执行中...
结果:
目录 data 的内容:
文件: video1.mp4 (8.40 MB)
文件: video2.mp4 (15.23 MB)

请输入任务: 把 data/video1.mp4 转为文字，语言是中文
```

#### 2. 单次执行

```bash
# 执行单个任务
python agent/video_agent.py "列出 data 目录的所有文件"

# 使用量化
python agent/video_agent.py --8bit "将 data/sample.mp4 转为文字"
```

#### 3. 命令行参数

```bash
python agent/video_agent.py --help

# 常用参数:
# --interactive, -i     交互模式
# --model MODEL         模型名称
# --cache-dir DIR       模型缓存目录 (默认: ./models)
# --device DEVICE       设备 (cuda/mps/cpu)，默认自动选择
# --8bit                使用 8-bit 量化
# --4bit                使用 4-bit 量化
# --max-length N        最大生成长度 (默认: 2048)
```

#### 4. 编程使用

```python
from agent.video_agent import VideoProcessingAgent

# 创建 Agent (默认 DeepSeek-R1-0528-Qwen3-8B)
agent = VideoProcessingAgent()

# 或者自定义配置
agent = VideoProcessingAgent(
    model_name="Qwen/Qwen3.5-9B",
    model_cache_dir="./models",
    temperature=0.1,
    max_length=2048,
    load_in_8bit=True,  # 使用 8-bit 量化
    device="cuda"       # 或 "mps", "cpu"
)

# 执行任务
result = agent.run("列出 data 目录下的视频，然后转录第一个")

if result["success"]:
    print(result["output"])
else:
    print(f"错误: {result['error']}")
    if "log_file" in result:
        print(f"日志: {result['log_file']}")
```

## 模型下载

首次运行时，模型会自动从 HuggingFace 下载到本地缓存（默认 `./models/`）。

### 下载过程

```bash
python agent/video_agent.py --interactive

# 输出示例:
正在初始化 Agent (开源模型版本)...
模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
设备: cuda
缓存目录: ./models
正在加载模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
Downloading (…)lve/main/config.json: 100%|████████| 615/615 [00:00<00:00, 38.6kB/s]
Downloading pytorch_model.bin: 100%|████████| 16.4G/16.4G [16:24<00:00, 16.6MB/s]
Downloading (…)okenizer_config.json: 100%|████████| 354/354 [00:00<00:00, 28.1kB/s]
✓ 模型加载完成
```

### 预下载模型（推荐）

通过 ModelScope 下载（国内推荐）:
```bash
# 使用 modelscope 命令行工具预先下载
modelscope download --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local_dir ./models/DeepSeek-R1-0528-Qwen3-8B
modelscope download --model Qwen/Qwen3.5-9B --local_dir ./models/Qwen3.5-9B
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./models/Qwen3.5-27B
```

通过 HuggingFace 下载（海外）:
```bash
# 需要先安装 huggingface_hub
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local-dir ./models/DeepSeek-R1-0528-Qwen3-8B
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ./models/Qwen3.5-9B
```

### 模型文件大小

| 模型 | 大小 | 下载时间 (估算) |
|------|------|----------------|
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | ~16.4 GB | 100Mbps: ~22分钟 |
| Qwen/Qwen3.5-9B | ~19.3 GB | 100Mbps: ~26分钟 |
| Qwen/Qwen3.5-27B | ~55.6 GB | 100Mbps: ~75分钟 |

**提示**: 模型只需下载一次，后续使用会从本地缓存加载（~30秒）。

## Agent 工作流程

```
用户输入
  ↓
[Chain of Thought 推理]
  ↓
选择合适的工具
  ↓
执行工具
  ↓
分析结果
  ↓
是否成功? 
  ├─ 是 → 返回结果
  └─ 否 → 分析错误 → 重试或报告
```

## 自我纠错机制

Agent 能够:

1. **文件路径验证**: 自动检查文件是否存在
2. **参数修正**: 分析错误并调整参数重试
3. **错误分类**: 区分不同类型的错误
4. **日志记录**: 保存详细的错误信息到 `agent/logs/`

## 错误日志

所有错误都会记录到 `agent/logs/` 目录:

```
agent/logs/
├── error_extract_audio_20251210_153045.log
├── error_transcribe_audio_20251210_153112.log
└── error_agent_run_20251210_153130.log
```

日志包含:
- 时间戳
- 工具名称
- 错误类型
- 错误信息
- 输入参数
- 完整堆栈追踪

## 示例任务

### 简单任务

```bash
# 列出文件
"列出 data 目录下的所有文件"

# 检查文件
"检查 data/video.mp4 是否存在"

# 提取音频
"从 data/video.mp4 提取音频"
```

### 复杂任务

```bash
# 完整流程
"找到 data 目录下最大的视频文件，将其转录为中文文本"

# 批量处理
"列出 data 目录的所有 mp4 文件，并逐个转录"

# 条件处理
"如果 data/important.mp4 存在，就转录它；否则告诉我文件不存在"
```

## 配置

### 模型选择

```bash
# 使用 DeepSeek-R1-0528-Qwen3-8B (默认，推理能力强)
python agent/video_agent.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# 使用 Qwen3.5-9B (多模态，性能均衡)
python agent/video_agent.py --model Qwen/Qwen3.5-9B

# 使用 Qwen3.5-27B (旗舰，需要较大显存)
python agent/video_agent.py --model Qwen/Qwen3.5-27B
```

### 温度参数

编程使用时可调整:
```python
agent = VideoProcessingAgent(
    temperature=0.0  # 0=确定性, 1=创造性
)
```

## 性能

| 操作 | 时间 | 成本 |
|------|------|------|
| 简单任务 (列出文件) | 2-5秒 | ~$0.01 |
| 视频转文字 | 3-5分钟 | ~$0.05-0.10 |
| 复杂推理 | 10-30秒 | ~$0.02-0.05 |

## 故障排查

### Q: 模型下载太慢
**A**: 国内用户优先使用 ModelScope 下载，速度更快:
```bash
pip install modelscope
modelscope download --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local_dir ./models/DeepSeek-R1-0528-Qwen3-8B
```

### Q: Agent 陷入循环
**A**: 设置更低的温度 (temperature=0) 或切换为推理能力更强的模型 (Qwen3.5-27B)

### Q: 工具调用失败
**A**: 检查 `agent/logs/` 目录下的错误日志

### Q: 显存不足 (OOM)
**A**: 使用量化参数或选择更小的模型:
```bash
# 使用 4-bit 量化
python agent/video_agent.py --4bit --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# 换用更小的模型
python agent/video_agent.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

## 扩展

### 添加新工具

```python
def _my_custom_tool(self, input: str) -> str:
    # 实现你的工具逻辑
    return "结果"

# 在 _create_tools() 中添加
Tool(
    name="my_tool",
    func=self._my_custom_tool,
    description="工具描述"
)
```

### 自定义 Prompt

修改 `_create_agent()` 中的 `template` 变量。

## 限制

- 首次运行需要下载大型模型文件（8~56 GB）
- 推理速度受硬件和模型大小限制
- Qwen3.5-27B 需要 48GB+ 显存全精度运行

## 未来改进

- [ ] 多轮对话记忆持久化
- [ ] 工具调用缓存
- [ ] 并行任务执行
- [ ] Web UI 界面
- [ ] 支持更多新一代开源模型
