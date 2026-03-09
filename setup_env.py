#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        input=input_text,
        check=check,
    )


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("[ffmpeg] 已安装，跳过")
        return

    system = platform.system().lower()
    print(f"[ffmpeg] 未检测到，尝试安装（{system}）")

    if system == "darwin":
        if not shutil.which("brew"):
            raise RuntimeError("macOS 未检测到 brew，请先安装 Homebrew 后重试")
        run(["brew", "install", "ffmpeg"])
    elif system == "linux":
        if shutil.which("apt-get"):
            prefix = ["sudo"] if os.geteuid() != 0 and shutil.which("sudo") else []
            run(prefix + ["apt-get", "update"])
            run(prefix + ["apt-get", "install", "-y", "ffmpeg"])
        elif shutil.which("dnf"):
            prefix = ["sudo"] if os.geteuid() != 0 and shutil.which("sudo") else []
            run(prefix + ["dnf", "install", "-y", "ffmpeg"])
        elif shutil.which("yum"):
            prefix = ["sudo"] if os.geteuid() != 0 and shutil.which("sudo") else []
            run(prefix + ["yum", "install", "-y", "ffmpeg"])
        elif shutil.which("pacman"):
            prefix = ["sudo"] if os.geteuid() != 0 and shutil.which("sudo") else []
            run(prefix + ["pacman", "-Sy", "--noconfirm", "ffmpeg"])
        else:
            raise RuntimeError("Linux 未识别可用包管理器（apt/dnf/yum/pacman）")
    elif system == "windows":
        if shutil.which("winget"):
            run(
                [
                    "winget",
                    "install",
                    "--id",
                    "Gyan.FFmpeg",
                    "-e",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                ]
            )
        elif shutil.which("choco"):
            run(["choco", "install", "ffmpeg", "-y"])
        elif shutil.which("scoop"):
            run(["scoop", "install", "ffmpeg"])
        else:
            raise RuntimeError("Windows 未检测到 winget/choco/scoop，无法自动安装 ffmpeg")
    else:
        raise RuntimeError(f"不支持的系统: {system}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 安装后仍未检测到，请手动检查 PATH")


def setup_venv() -> Path:
    system = platform.system().lower()

    if system in ("darwin", "linux"):
        setup_script = ROOT / "scripts" / "setup_venv.sh"
        if not setup_script.exists():
            raise FileNotFoundError(f"未找到脚本: {setup_script}")
        run(["bash", str(setup_script)], input_text="n\n")
        py = ROOT / ".venv" / "bin" / "python"
    elif system == "windows":
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        if not py.exists():
            run([sys.executable, "-m", "venv", ".venv"])
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        raise RuntimeError(f"不支持的系统: {system}")

    if not py.exists():
        raise RuntimeError("venv 创建失败，未找到 Python 可执行文件")

    return py


def install_whisper_model(venv_python: Path) -> None:
    print("[whisper] 检测硬件并预下载模型（不加载）")

    code = r'''
import os
from pathlib import Path
import psutil
import torch

cache_dir = os.getenv("TRANSFORMERS_CACHE")
if not cache_dir:
    cache_dir = str(Path("models").resolve())

os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["MODELSCOPE_CACHE"] = cache_dir
Path(cache_dir).mkdir(parents=True, exist_ok=True)

available_mem = psutil.virtual_memory().available / (1024 ** 3)
gpu_mem = 0
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

total = max(available_mem, gpu_mem)
if total >= 6:
    model_id, model_name = "openai/whisper-large-v3-turbo", "large-v3-turbo"
elif total >= 4:
    model_id, model_name = "openai/whisper-medium", "medium"
else:
    model_id, model_name = "openai/whisper-small", "small"

ms_map = {
    "large-v3-turbo": "AI-ModelScope/whisper-large-v3-turbo",
    "medium": "AI-ModelScope/whisper-medium",
    "small": "AI-ModelScope/whisper-small",
}
ms_id = ms_map[model_name]

print(f"[whisper] 资源检测: total={total:.2f}GB -> {model_name}")
print(f"[whisper] cache: {cache_dir}")

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot
    path = ms_snapshot(ms_id, cache_dir=cache_dir)
    print(f"[whisper] ModelScope 下载完成: {path}")
except Exception as e:
    print(f"[whisper] ModelScope 失败: {e}")
    from huggingface_hub import snapshot_download as hf_snapshot
    path = hf_snapshot(repo_id=model_id, cache_dir=cache_dir)
    print(f"[whisper] HuggingFace 下载完成: {path}")
'''

    run([str(venv_python), "-c", code])


def main() -> int:
    try:
        venv_python = setup_venv()
        ensure_ffmpeg()
        install_whisper_model(venv_python)
        print("\n环境配置完成。")
        print("启动 UI: python ui/ui.py")
        return 0
    except Exception as exc:
        print(f"\n配置失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
