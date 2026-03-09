# Video Summary UI（视频转文字 Web 界面）

当前文档仅保留 UI 部署与运行说明。

## 环境要求

- Python 3.11+
- ffmpeg（本地运行时需要）
- Docker（可选，用于容器部署）

## 本地直接运行

1) 一键配置环境（跨平台）

```bash
python setup_env.py
```

Windows 也可使用：

```powershell
py setup_env.py
```

2) 启动 UI

```bash
python ui/ui.py
```

3) 访问页面

```text
http://127.0.0.1:5000
```

## Docker 部署（UI）

```bash
docker build -f Dockerfile-ui -t video-summary-ui .
docker run --rm -p 5000:5000 -v /tmp/models:/tmp/models -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" video-summary-ui
```

说明：
- `/tmp/models` 使用宿主机映射，模型下载与加载都走宿主机缓存。
- `data` 与 `output` 挂载到当前工作区，便于持久化上传和产物。
