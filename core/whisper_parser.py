#!/usr/bin/env python3
"""
Whisper 语音转文字模块
使用 OpenAI Whisper 模型将音频转录为文字

支持的语言: 中文(zh), 英语(en), 日语(ja), 韩语(ko) 等 99+ 种语言
"""

import torch
import os
import json
import re
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
from typing import Union, Optional, List, Dict

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


def split_long_chunk_by_sentences(text: str, timestamp: list, max_chars: int = 80) -> List[Dict]:
    """
    将超长文本按句子切分成多个小段
    
    Args:
        text: 要切分的文本
        timestamp: [start, end] 时间范围
        max_chars: 每段最大字符数
    
    Returns:
        切分后的 chunks
    """
    # 句子结束标点
    sentence_pattern = re.compile(r'([^。！？\.!\?]+[。！？\.!\?]+)')
    sentences = sentence_pattern.findall(text)
    
    # 如果没找到句子边界，按字符数硬切
    if not sentences:
        sentences = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    
    if len(sentences) <= 1:
        return [{"timestamp": timestamp, "text": text}]
    
    # 计算每个句子的时间占比
    total_chars = sum(len(s) for s in sentences)
    start_time, end_time = timestamp
    duration = end_time - start_time
    
    chunks = []
    current_time = start_time
    current_text = ""
    
    for sentence in sentences:
        sentence_duration = (len(sentence) / total_chars) * duration
        
        # 累积句子
        current_text += sentence
        
        # 如果达到字符数限制或是最后一句，保存当前段
        if len(current_text) >= max_chars or sentence == sentences[-1]:
            next_time = current_time + sentence_duration if sentence != sentences[-1] else end_time
            chunks.append({
                "timestamp": [current_time, next_time],
                "text": current_text
            })
            current_time = next_time
            current_text = ""
        else:
            # 继续累积，更新时间
            current_time += sentence_duration
    
    return chunks


def merge_word_chunks_to_sentences(chunks: List[Dict], max_sentences: int = 2, max_chars: int = 150) -> List[Dict]:
    """
    将词级别的 chunks 合并为句子级别，每段包含最多 max_sentences 个句子或 max_chars 个字符
    
    策略：
    1. 累积 chunk 直到满足以下任一条件：
       - 达到 max_sentences 个句子结束标点
       - 累积文本长度超过 max_chars 字符
    2. 如果一个 chunk 本身就很长（>max_chars），直接作为独立段落
    
    Args:
        chunks: 词级别的 chunks（每个包含 timestamp 和 text）
        max_sentences: 每段最多包含的句子数（默认 2）
        max_chars: 每段最多包含的字符数（默认 150）
    
    Returns:
        合并后的句子级别 chunks
    """
    if not chunks:
        return []
    
    # 句子结束标点（中英文）
    sentence_end_pattern = re.compile(r'[。！？\.!\?]')
    
    merged_chunks = []
    current_text = ""
    current_start = None
    current_end = None
    sentence_count = 0
    
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        chunk_start = chunk["timestamp"][0]
        chunk_end = chunk["timestamp"][1]
        
        # 初始化起始时间
        if current_start is None:
            current_start = chunk_start
        
        # 如果当前 chunk 本身就很长，进行二次切分
        if len(text) > max_chars:
            # 先保存之前累积的内容（如果有）
            if current_text.strip():
                merged_chunks.append({
                    "timestamp": [current_start, current_end],
                    "text": current_text
                })
            
            # 将长 chunk 切分成多个小段
            sub_chunks = split_long_chunk_by_sentences(
                text, 
                [chunk_start, chunk_end], 
                max_chars
            )
            merged_chunks.extend(sub_chunks)
            
            # 重置
            current_text = ""
            current_start = None
            sentence_count = 0
        else:
            # 累积文本
            current_text += text
            current_end = chunk_end
            
            # 统计句子数
            sentence_matches = sentence_end_pattern.findall(text)
            sentence_count += len(sentence_matches)
            
            # 检查是否需要切分（达到句子数或字符数限制）
            should_split = (
                sentence_count >= max_sentences or 
                len(current_text) >= max_chars
            )
            
            if should_split:
                merged_chunks.append({
                    "timestamp": [current_start, current_end],
                    "text": current_text
                })
                # 重置
                sentence_count = 0
                current_text = ""
                current_start = None
    
    # 添加最后一段（如果有剩余内容）
    if current_text.strip():
        merged_chunks.append({
            "timestamp": [current_start, current_end],
            "text": current_text
        })
    
    return merged_chunks


def load_model(verbose: bool = True):
    """
    加载 Whisper 模型（延迟加载）
    优先从 ModelScope 下载（国内快），失败则使用 HuggingFace
    
    Args:
        verbose: 是否显示加载信息
    """
    global _model, _processor, _pipe
    
    if _pipe is not None:
        return  # 模型已加载
    
    # 设置缓存环境变量
    cache_dir_abs = str(Path(CACHE_DIR).resolve())
    os.environ['HF_HOME'] = cache_dir_abs
    os.environ['TRANSFORMERS_CACHE'] = cache_dir_abs
    os.environ['MODELSCOPE_CACHE'] = cache_dir_abs
    
    if verbose:
        print(f"设备: {device}")
        print(f"模型缓存目录: {CACHE_DIR}")
        print(f"正在加载模型 {model_id}...")
    
    model_loaded = False
    
    # 方案 1: 尝试从 ModelScope 加载（国内快）
    try:
        if verbose:
            print("尝试从 ModelScope 下载...")
        
        from modelscope import snapshot_download
        
        # ModelScope 上的 Whisper 模型
        ms_model_id = "openai-mirror/whisper-large-v3-turbo"
        
        model_dir = snapshot_download(
            ms_model_id,
            cache_dir=cache_dir_abs
        )
        
        if verbose:
            print(f"ModelScope 下载完成: {model_dir}")
        
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        _model.to(device)
        
        _processor = AutoProcessor.from_pretrained(model_dir)
        
        if verbose:
            print("✓ 从 ModelScope 加载成功")
        
        model_loaded = True
        
    except ImportError:
        if verbose:
            print("modelscope 库未安装，跳过 ModelScope")
    except Exception as e:
        if verbose:
            print(f"ModelScope 加载失败: {e}")
            print("尝试从 HuggingFace 下载...")
    
    # 方案 2: 如果 ModelScope 失败，从 HuggingFace 加载
    if not model_loaded:
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
        
        if verbose:
            print("✓ 从 HuggingFace 加载成功")
    
    # 创建 pipeline
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
    
    # 转录音频（启用词级别时间戳，然后合并为句子）
    result = _pipe(
        str(audio_path),
        generate_kwargs={
            "language": language,
            "task": "transcribe",  # 转录模式（带标点）
        },
        return_timestamps="word",  # 返回词级别时间戳（更细粒度）
        chunk_length_s=30,         # 30秒音频分块处理
    )
    
    # 将词级别的分段合并为句子级别（每段最多 2 个句子，最多 80 字符）
    raw_chunks = result.get("chunks", [])
    sentence_chunks = merge_word_chunks_to_sentences(
        raw_chunks, 
        max_sentences=2,  # 最多2个句子
        max_chars=80      # 或最多80个字符
    )
    
    if verbose:
        print(f"分段数: {len(sentence_chunks)} ({len(raw_chunks)} → {len(sentence_chunks)})")
    
    # 构建结果字典
    transcription_result = {
        "text": result["text"],
        "chunks": sentence_chunks,  # 句子级别的分段（每段 1-2 句话，最多 80 字符）
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
    
    # 保存为文本文件（带时间戳）
    if save_txt:
        txt_path = output_dir / f"{base_name}_transcription.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"# 音频转录结果\n")
            f.write(f"# 音频文件: {audio_path.name}\n")
            f.write(f"# 转录时间: {transcription_result['timestamp']}\n")
            f.write(f"# 语言: {language}\n")
            f.write(f"# 模型: {model_id}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"完整文本:\n")
            f.write(f"{'='*60}\n")
            f.write(f"{result['text']}\n\n")
            
            # 如果有分段时间戳，输出带时间戳的文本
            if result.get("chunks"):
                f.write(f"{'='*60}\n")
                f.write(f"分段文本（带时间戳）:\n")
                f.write(f"{'='*60}\n")
                for chunk in result["chunks"]:
                    timestamp = chunk.get("timestamp", (None, None))
                    text = chunk.get("text", "")
                    
                    if timestamp[0] is not None and timestamp[1] is not None:
                        start_time = format_timestamp(timestamp[0])
                        end_time = format_timestamp(timestamp[1])
                        f.write(f"[{start_time} --> {end_time}] {text}\n")
                    else:
                        f.write(f"{text}\n")
        
        transcription_result["output_files"]["txt"] = str(txt_path)
        
        if verbose:
            print(f"✓ 文本已保存: {txt_path}")
    
    # 保存为 JSON 文件（包含完整的时间戳信息）
    if save_json:
        json_path = output_dir / f"{base_name}_transcription.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, ensure_ascii=False, indent=2)
        
        transcription_result["output_files"]["json"] = str(json_path)
        
        if verbose:
            print(f"✓ JSON 已保存: {json_path}")
    
    if verbose:
        print(f"\n转录完成! 文本长度: {len(result['text'])} 字符")
        if result.get("chunks"):
            print(f"分段数: {len(result['chunks'])}")
    
    return transcription_result


def format_timestamp(seconds: float) -> str:
    """
    将秒数转换为 时:分:秒 格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串 (HH:MM:SS.mmm)
    """
    if seconds is None:
        return "00:00:00.000"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

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
