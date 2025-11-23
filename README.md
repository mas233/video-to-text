# Video Summary（视频转文字服务）

基于 Spring Boot 的视频转文字服务，调用千问（DashScope）OpenAI 兼容接口完成转写。项目提供 REST 和 SSE 两种返回方式，并在服务端自动提取音频并控制模型输入长度。

**输入文件**
- 支持：带音轨的 MP4 视频，或 MP3 音频
- 服务端会将音频统一转码为 `mp3(24kHz/mono/32kbps)`，必要时自动降码以满足接口的最大输入限制

## 环境要求
- JDK 8 及以上（推荐 JDK 17）
- Maven 3.6+（用于构建）

## 配置 API Key
千问 API Key 通过环境变量或配置文件注入，二选一即可：
- 环境变量：`DASHSCOPE_API_KEY`（推荐方式）
- 配置文件：`src/main/resources/application.properties` 中的 `qwen.api.key`

示例（本地终端设置环境变量并启动）：
```bash
export DASHSCOPE_API_KEY=sk-xxxx
java -jar target/video-summary-1.0.0.jar
```

或将 `.env.example` 中的 `DASHSCOPE_API_KEY` 替换为真实值后，在同一终端加载再启动：
```bash
set -a; . ./.env.example; set +a
java -jar target/video-summary-1.0.0.jar
```

## 构建与运行
```bash
mvn clean package
java -jar target/video-summary-1.0.0.jar
```

（可选）Docker 部署：
```bash
docker build -t video-summary .
docker run -d -p 8080:8080 -e DASHSCOPE_API_KEY=sk-xxxx video-summary
```
或使用 Compose：
```bash
docker-compose up -d
```

## 接口说明

### 1. REST：视频转文字
`POST /api/video-to-text`
- 参数：
  - `file`（必填）：MP4 视频或 MP3 音频
  - `language`（可选，默认 `auto`）
  - `enableSpeakerDiarization`（可选，默认 `false`）
- 返回：JSON，`data` 字段为完整文本

示例（使用示例路径）：
```bash
curl -sS -X POST -H "Expect:" \
  -F "file=@/Users/user/Downloads/example.mp4" \
  -F "language=auto" \
  -F "enableSpeakerDiarization=false" \
  http://localhost:8080/api/video-to-text
```

### 2. SSE：视频转文字（流式）
`POST /api/video-to-text-sse`
- 行为：服务端先缓冲文本，再以 200 字符分片通过 SSE 输出
- 事件：
  - `start`：开始处理
  - 若干 `partial`：分片文本
  - `complete`：合并文本
  - `error`：错误信息

示例（使用示例路径）：
```bash
curl -N -X POST -H "Expect:" \
  -F "file=@/Users/user/Downloads/example.mp4" \
  -F "language=auto" \
  -F "enableSpeakerDiarization=false" \
  http://localhost:8080/api/video-to-text-sse
```

 

## 运行时行为与限制
- 最大上传大小：100MB（可在 `application.properties` 中调整）
- 模型输入长度：服务端自动控制并在超限时提示“音频过长”
- 健康检查：`GET /actuator/health`

## 许可协议
本项目采用 **Apache License 2.0**。