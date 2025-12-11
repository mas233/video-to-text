# MCP 服务器

Model Context Protocol (MCP) 服务器，为视频处理工具提供标准化接口。

## 可用工具

### 1. extract_audio
从视频文件中提取音频

**参数**:
- `video_path` (必需): 视频文件路径
- `output_path` (可选): 输出音频路径
- `audio_format` (可选): 音频格式，默认 mp3
- `bitrate` (可选): 比特率，默认 128k

**示例**:
```json
{
  "video_path": "data/video.mp4",
  "audio_format": "mp3",
  "bitrate": "192k"
}
```

### 2. transcribe_audio
将音频转录为文字

**参数**:
- `audio_path` (必需): 音频文件路径
- `language` (可选): 语言代码，默认 zh
- `output_dir` (可选): 输出目录

**示例**:
```json
{
  "audio_path": "data/audio.mp3",
  "language": "zh"
}
```

### 3. video_to_text
完整的视频转文字流程

**参数**:
- `video_path` (必需): 视频文件路径
- `language` (可选): 语言代码，默认 zh
- `output_dir` (可选): 输出目录
- `keep_audio` (可选): 是否保留音频，默认 false

**示例**:
```json
{
  "video_path": "data/video.mp4",
  "language": "zh",
  "keep_audio": true
}
```

## 启动服务器

```bash
python mcp/server.py
```

## 配置

修改 `config.json` 以配置服务器路径和环境变量。
