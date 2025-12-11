#!/usr/bin/env python3
"""
Whisper 语音转文字模块
使用 OpenAI Whisper 模型将音频转录为文字

支持的语言: 中文(zh), 英语(en), 日语(ja), 韩语(ko) 等 99+ 种语言
"""

import torch
import os
import json
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
from typing import Union, Optional

# 设置本地模型缓存目录（支持离线使用）
# 优先使用环境变量，否则使用当前目录下的 models 文件夹
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "./models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# 全局模型变量（延迟加载）
_model = None
_processor = None
_pipe = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"


def load_model(verbose: bool = True):
    """
    加载 Whisper 模型（延迟加载）
    
    Args:
        verbose: 是否显示加载信息
    """
    global _model, _processor, _pipe
    
    if _pipe is not None:
        return  # 模型已加载
    
    # 设置缓存环境变量（确保 HuggingFace 使用项目缓存目录）
    cache_dir_abs = str(Path(CACHE_DIR).resolve())
    os.environ['HF_HOME'] = cache_dir_abs
    os.environ['TRANSFORMERS_CACHE'] = cache_dir_abs
    
    if verbose:
        print(f"设备: {device}")
        print(f"模型缓存目录: {CACHE_DIR}")
        print(f"正在加载模型 {model_id}...")
    
    # 从本地缓存加载模型（首次运行会自动下载到 cache_dir）
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        cache_dir=cache_dir_abs
    )
    _model.to(device)
    
    _processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir_abs
    )
    
    _pipe = pipeline(
        "automatic-speech-recognition",
        model=_model,
        tokenizer=_processor.tokenizer,
        feature_extractor=_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    if verbose:
        print("✓ 模型加载完成！\n")


def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "zh",
    output_dir: Optional[Union[str, Path]] = None,
    save_txt: bool = True,
    save_json: bool = True,
    verbose: bool = True
) -> dict:
    """
    转录音频文件并保存结果
    
    Args:
        audio_path: 音频文件路径
        language: 语言代码，默认中文 (zh)，可选 en, ja, ko 等
        output_dir: 输出目录，默认为 data/transcriptions/
        save_txt: 是否保存为文本文件
        save_json: 是否保存为 JSON 文件（包含元数据）
        verbose: 是否显示详细信息
    
    Returns:
        包含转录文本和元数据的字典，格式:
        {
            "text": "转录文本",
            "language": "zh",
            "audio_file": "音频文件名",
            "timestamp": "2025-12-10T15:30:00",
            "model": "openai/whisper-large-v3-turbo",
            "output_files": {
                "txt": "输出文本文件路径",
                "json": "输出JSON文件路径"
            }
        }
    """
    # 确保模型已加载
    load_model(verbose=verbose)
    
    audio_path = Path(audio_path).resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    if verbose:
        print(f"正在转录音频: {audio_path.name}")
        print(f"语言: {language}")
    
    # 转录音频（支持长音频）
    result = _pipe(
        str(audio_path),
        generate_kwargs={"language": language},
        return_timestamps=True  # 支持长音频（>30秒）
    )
    
    # 构建结果字典
    transcription_result = {
        "text": result["text"],
        "language": language,
        "audio_file": audio_path.name,
        "audio_path": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "device": device,
        "output_files": {}
    }
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path("data/transcriptions")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名（基于音频文件名）
    base_name = audio_path.stem
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为文本文件
    if save_txt:
        txt_path = output_dir / f"{base_name}_transcription.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"# 音频转录结果\n")
            f.write(f"# 音频文件: {audio_path.name}\n")
            f.write(f"# 转录时间: {transcription_result['timestamp']}\n")
            f.write(f"# 语言: {language}\n")
            f.write(f"# 模型: {model_id}\n")
            f.write(f"\n{result['text']}\n")
        
        transcription_result["output_files"]["txt"] = str(txt_path)
        
        if verbose:
            print(f"✓ 文本已保存: {txt_path}")
    
    # 保存为 JSON 文件
    if save_json:
        json_path = output_dir / f"{base_name}_transcription.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, ensure_ascii=False, indent=2)
        
        transcription_result["output_files"]["json"] = str(json_path)
        
        if verbose:
            print(f"✓ JSON 已保存: {json_path}")
    
    if verbose:
        print(f"\n转录完成! 文本长度: {len(result['text'])} 字符")
    
    return transcription_result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python whisper_parser.py <音频文件路径> [语言代码] [输出目录]")
        print("\n示例:")
        print("  python whisper_parser.py sample.mp3")
        print("  python whisper_parser.py sample.mp3 zh")
        print("  python whisper_parser.py sample.mp3 en data/output")
        print("\n支持的语言代码:")
        print("  zh (中文), en (英语), ja (日语), ko (韩语), 等")
        print("\n如需预下载模型（离线使用），请先联网运行一次脚本")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "zh"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result = transcribe_audio(
            audio_file,
            language=lang,
            output_dir=out_dir,
            save_txt=True,
            save_json=True
        )
        
        print("\n" + "="*60)
        print("转录文本:")
        print("="*60)
        print(result["text"])
        print("="*60)
        
        if result.get("output_files"):
            print("\n输出文件:")
            for file_type, file_path in result["output_files"].items():
                print(f"  {file_type.upper()}: {file_path}")
        
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)
