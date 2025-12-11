#!/usr/bin/env python3
"""
预下载 Whisper 模型到本地
支持离线使用

使用方法:
    python download_model.py                    # 下载默认模型（turbo）
    python download_model.py --model large-v3   # 下载指定模型
    python download_model.py --list             # 列出所有可用模型
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# 可用的 Whisper 模型
MODELS = {
    "tiny": {
        "id": "openai/whisper-tiny",
        "size": "39M",
        "description": "最小模型，速度最快，精度较低"
    },
    "base": {
        "id": "openai/whisper-base",
        "size": "74M",
        "description": "基础模型，平衡速度和精度"
    },
    "small": {
        "id": "openai/whisper-small",
        "size": "244M",
        "description": "小型模型，较好的精度"
    },
    "medium": {
        "id": "openai/whisper-medium",
        "size": "769M",
        "description": "中型模型，高精度"
    },
    "large-v3": {
        "id": "openai/whisper-large-v3",
        "size": "1.5GB",
        "description": "大型模型 V3，最高精度"
    },
    "turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "size": "809M",
        "description": "Turbo 版本，速度更快（推荐）"
    }
}

def list_models():
    """列出所有可用模型"""
    print("\n可用的 Whisper 模型：\n")
    print(f"{'名称':<12} {'大小':<10} {'描述'}")
    print("-" * 60)
    for name, info in MODELS.items():
        print(f"{name:<12} {info['size']:<10} {info['description']}")
    print()

def download_model(model_name: str, cache_dir: str = "./models"):
    """下载指定模型"""
    if model_name not in MODELS:
        print(f"错误: 未知模型 '{model_name}'")
        print("运行 'python download_model.py --list' 查看可用模型")
        sys.exit(1)
    
    model_info = MODELS[model_name]
    model_id = model_info["id"]
    
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_name}")
    print(f"模型 ID: {model_id}")
    print(f"预计大小: {model_info['size']}")
    print(f"保存路径: {Path(cache_dir).absolute()}")
    print(f"{'='*60}\n")
    
    # 创建缓存目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载模型
        print("正在下载模型文件...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        print("✓ 模型文件下载完成")
        
        # 下载处理器
        print("\n正在下载处理器...")
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        print("✓ 处理器下载完成")
        
        print(f"\n{'='*60}")
        print(f"✓ 模型 '{model_name}' 下载成功！")
        print(f"{'='*60}\n")
        
        print("验证下载的文件:")
        model_dir = Path(cache_dir)
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        print(f"总大小: {total_size / (1024**3):.2f} GB")
        print(f"文件数: {sum(1 for _ in model_dir.rglob('*') if _.is_file())}")
        
        print("\n现在可以离线使用该模型了！")
        
    except Exception as e:
        print(f"\n错误: 下载失败")
        print(f"详细信息: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="预下载 Whisper 模型到本地，支持离线使用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                        # 下载默认模型（turbo）
  %(prog)s --model small          # 下载 small 模型
  %(prog)s --cache-dir /data/models  # 指定保存目录
  %(prog)s --list                 # 列出所有模型
        """
    )
    
    parser.add_argument(
        "--model",
        default="turbo",
        help="模型名称 (默认: turbo)"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./models",
        help="模型保存目录 (默认: ./models)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用模型"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        sys.exit(0)
    
    download_model(args.model, args.cache_dir)

if __name__ == "__main__":
    main()
