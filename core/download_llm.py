#!/usr/bin/env python3
"""
预下载 LLM 模型到本地（用于 Agent）
支持离线使用

使用方法:
    python download_llm.py                              # 下载默认模型（Qwen2.5-7B）
    python download_llm.py --model qwen2-7b            # 下载指定模型
    python download_llm.py --list                      # 列出所有可用模型
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 可用的 LLM 模型
MODELS = {
    "qwen2.5-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "size": "~14.5GB",
        "description": "通义千问 2.5（推荐），中文能力最强",
        "params": "7B",
        "languages": "中文/英文"
    },
    "qwen2-7b": {
        "id": "Qwen/Qwen2-7B-Instruct",
        "size": "~14.2GB",
        "description": "通义千问 2.0，中文能力优秀",
        "params": "7B",
        "languages": "中文/英文"
    },
    "llama3.1-8b": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "~16.0GB",
        "description": "Meta LLaMA 3.1，英文能力强",
        "params": "8B",
        "languages": "英文/多语言"
    },
}

def list_models():
    """列出所有可用模型"""
    print("\n可用的 LLM 模型（用于 Agent）：\n")
    print(f"{'名称':<15} {'参数量':<8} {'大小':<12} {'语言':<12} {'描述'}")
    print("-" * 90)
    for name, info in MODELS.items():
        print(f"{name:<15} {info['params']:<8} {info['size']:<12} {info['languages']:<12} {info['description']}")
    
    print("\n推荐使用:")
    print("  - 中文视频: qwen2.5-7b (默认)")
    print("  - 英文视频: llama3.1-8b")
    print("  - 资源有限: qwen2-7b\n")

def download_model(model_name: str, cache_dir: str = "./models", use_8bit: bool = False):
    """下载指定模型，优先使用 ModelScope 源"""
    if model_name not in MODELS:
        print(f"错误: 未知模型 '{model_name}'")
        print("运行 'python download_llm.py --list' 查看可用模型")
        sys.exit(1)
    
    model_info = MODELS[model_name]
    model_id = model_info["id"]
    
    print(f"\n{'='*70}")
    print(f"开始下载模型: {model_name}")
    print(f"模型 ID: {model_id}")
    print(f"参数量: {model_info['params']}")
    print(f"预计大小: {model_info['size']}")
    print(f"保存路径: {Path(cache_dir).absolute()}")
    print(f"{'='*70}\n")
    
    print("⚠️  注意:")
    print("  1. 优先从 ModelScope 下载（国内更快）")
    print("  2. ModelScope 失败会自动切换到 HuggingFace")
    print("  3. 首次下载需要时间（~14GB，100Mbps 约 20 分钟）")
    print("  4. 确保有足够的磁盘空间（至少 20GB）")
    print("  5. 下载完成后可以完全离线使用\n")
    
    # 创建缓存目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # 将 HuggingFace 格式转换为 ModelScope 格式
    modelscope_id = model_id.replace("/", "/").lower()
    
    # 尝试 ModelScope
    success = _try_download_from_modelscope(modelscope_id, model_id, cache_dir, use_8bit)
    
    # 失败则尝试 HuggingFace
    if not success:
        print("\n" + "="*70)
        print("切换到 HuggingFace 源")
        print("="*70 + "\n")
        _try_download_from_huggingface(model_id, cache_dir, use_8bit)
    
    # 验证下载
    print("\n验证下载的文件:")
    model_dir = Path(cache_dir)
    if model_dir.exists():
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        file_count = sum(1 for _ in model_dir.rglob('*') if _.is_file())
        print(f"  总大小: {total_size / (1024**3):.2f} GB")
        print(f"  文件数: {file_count}")
    
    print("\n✅ 模型已准备就绪！")
    print("\n使用方法:")
    print(f"  python agent/video_agent.py --interactive --model {model_id}")
    if use_8bit:
        print(f"  python agent/video_agent.py --interactive --model {model_id} --8bit")


def _try_download_from_modelscope(modelscope_id: str, original_id: str, cache_dir: str, use_8bit: bool) -> bool:
    """尝试从 ModelScope 下载"""
    try:
        print("步骤 1/3: 尝试从 ModelScope 下载...")
        
        # 检查是否安装了 modelscope
        try:
            from modelscope import AutoTokenizer as MSAutoTokenizer
            from modelscope import AutoModelForCausalLM as MSAutoModelForCausalLM
        except ImportError:
            print("⚠️  modelscope 未安装，跳过 ModelScope 源")
            print("   提示: pip install modelscope")
            return False
        
        # 下载分词器
        print(f"  从 ModelScope 下载分词器: {modelscope_id}")
        tokenizer = MSAutoTokenizer.from_pretrained(
            modelscope_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("  ✓ 分词器下载完成\n")
        
        # 下载模型
        print("步骤 2/3: 从 ModelScope 下载模型文件（这可能需要一些时间）...")
        
        # 配置下载参数
        download_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # 如果使用量化，添加相应配置
        if use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                download_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                print("  (使用 8-bit 量化下载)")
            except ImportError:
                print("  ⚠️  transformers 或 bitsandbytes 未安装，跳过量化")
        
        model = MSAutoModelForCausalLM.from_pretrained(
            modelscope_id,
            **download_kwargs
        )
        print("  ✓ 模型文件下载完成\n")
        
        print(f"{'='*70}")
        print(f"✓ 成功从 ModelScope 下载模型！")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ModelScope 下载失败: {type(e).__name__}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        return False


def _try_download_from_huggingface(model_id: str, cache_dir: str, use_8bit: bool):
    """从 HuggingFace 下载（备用）"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 下载分词器
        print("步骤 1/3: 从 HuggingFace 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ 分词器下载完成\n")
        
        # 下载模型
        print("步骤 2/3: 从 HuggingFace 下载模型文件（这可能需要一些时间）...")
        
        # 配置下载参数
        download_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # 如果使用量化，添加相应配置
        if use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                download_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                print("(使用 8-bit 量化下载)")
            except ImportError:
                pass
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **download_kwargs
        )
        print("✓ 模型文件下载完成\n")
        
        print(f"{'='*70}")
        print(f"✓ 成功从 HuggingFace 下载模型！")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 错误: HuggingFace 下载也失败")
        print(f"详细信息: {e}")
        print("\n排查建议:")
        print("  1. 检查网络连接")
        print("  2. ModelScope: 可能模型名称不匹配")
        print("  3. HuggingFace: 需要访问 huggingface.co")
        print("  4. 如果是 LLaMA 模型，可能需要申请访问权限")
        print("  5. 尝试使用代理或 VPN")
        print("  6. 设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="预下载 LLM 模型到本地（用于视频处理 Agent）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                              # 下载默认模型（Qwen2.5-7B）
  %(prog)s --model qwen2-7b            # 下载 Qwen 2.0 模型
  %(prog)s --model llama3.1-8b         # 下载 LLaMA 3.1 模型
  %(prog)s --cache-dir /data/models    # 指定保存目录
  %(prog)s --list                      # 列出所有模型
  %(prog)s --8bit                      # 使用 8-bit 量化下载

推荐配置:
  中文视频: qwen2.5-7b (默认)
  英文视频: llama3.1-8b
  显存有限: 添加 --8bit 参数
        """
    )
    
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
        help="模型名称 (默认: qwen2.5-7b)"
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
    
    parser.add_argument(
        "--8bit",
        action="store_true",
        help="使用 8-bit 量化下载（节省空间）"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        sys.exit(0)
    
    download_model(
        model_name=args.model,
        cache_dir=args.cache_dir,
        use_8bit=args.__dict__.get('8bit', False)
    )

if __name__ == "__main__":
    main()
