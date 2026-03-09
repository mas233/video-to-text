#!/usr/bin/env python3
"""
Whisper 语音转文字模块
使用 OpenAI Whisper 模型将音频转录为文字

"""

import torch
import os
import json
import re
import psutil
from datetime import datetime
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

# 设置本地模型缓存目录
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "./models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# 全局变量
_model = None
_processor = None
_pipe = None


def get_optimal_device():
    """获取最优推理设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            arch = f"sm_{props.major}{props.minor}"
            if arch in torch.cuda.get_arch_list():
                return "cuda:0", torch.float16
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def select_model_by_resources():
    """根据系统资源自动选择合适的 Whisper 模型
    
    Returns:
        tuple: (model_id, model_name)
    """
    # 获取可用内存（GB）
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    
    # 获取 GPU 内存（如果有）
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # 选择模型的内存需求（估算值，包含模型+推理开销）
    # large-v3: ~6GB, large-v3-turbo: ~4GB, medium: ~3GB, small: ~2GB
    total_memory = max(available_memory, gpu_memory)
    
    if total_memory >= 8:
        return "openai/whisper-large-v3-turbo", "large-v3-turbo"
    elif total_memory >= 6:
        return "openai/whisper-large-v3-turbo", "large-v3-turbo"
    elif total_memory >= 4:
        return "openai/whisper-medium", "medium"
    else:
        return "openai/whisper-small", "small"


device, torch_dtype = get_optimal_device()
model_id, model_name = select_model_by_resources()


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
    3. 特殊处理：数字、小数点、百分号等不能被切分
    
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
    # 数字相关字符（数字、小数点、逗号、百分号等）
    numeric_pattern = re.compile(r'^[\d\.\,\%\+\-]+$')
    
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
            
            # 检查下一个 chunk 是否是数字相关（小数点、百分号等）
            is_next_numeric = False
            if i + 1 < len(chunks):
                next_text = chunks[i + 1]["text"].strip()
                is_next_numeric = bool(numeric_pattern.match(next_text))
            
            # 检查当前 chunk 是否是数字相关
            is_current_numeric = bool(numeric_pattern.match(text.strip()))
            
            # 检查是否需要切分（达到句子数或字符数限制）
            # 但如果下一个是数字相关，或当前是数字相关，不切分
            should_split = (
                (sentence_count >= max_sentences or len(current_text) >= max_chars)
                and not is_next_numeric
                and not is_current_numeric
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


def _get_modelscope_id(model_name: str) -> str:
    """根据模型名称获取 ModelScope ID"""
    modelscope_ids = {
        'large-v3': 'AI-ModelScope/whisper-large-v3',
        'large-v3-turbo': 'AI-ModelScope/whisper-large-v3-turbo',
        'medium': 'AI-ModelScope/whisper-medium',
        'small': 'AI-ModelScope/whisper-small',
        'tiny': 'AI-ModelScope/whisper-tiny'
    }
    return modelscope_ids.get(model_name, "AI-ModelScope/whisper-large-v3-turbo")


def _load_from_modelscope(modelscope_id: str, cache_dir: str, max_retries: int = 3):
    """从 ModelScope 加载模型"""
    from modelscope.hub.snapshot_download import snapshot_download
    
    for attempt in range(max_retries):
        try:
            model_dir = snapshot_download(modelscope_id, cache_dir=cache_dir)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_dir)
            return model, processor
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
    return None, None


def load_model():
    """加载 Whisper 模型（延迟加载）"""
    global _model, _processor, _pipe
    
    if _pipe is not None:
        return
    
    # 设置缓存目录
    cache_dir_abs = str(Path(CACHE_DIR).resolve())
    os.environ['MODELSCOPE_CACHE'] = cache_dir_abs
    os.environ['HF_HOME'] = cache_dir_abs
    os.environ['TRANSFORMERS_CACHE'] = cache_dir_abs
    
    # 尝试从 ModelScope 加载
    modelscope_id = _get_modelscope_id(model_name)
    try:
        _model, _processor = _load_from_modelscope(modelscope_id, cache_dir_abs)
    except Exception:
        # ModelScope 失败，使用 HuggingFace
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir_abs
        )
        _model.to(device)
        _processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir_abs)
    
    # 创建 pipeline（配置 tokenizer 输出标点）
    _pipe = pipeline(
        "automatic-speech-recognition",
        model=_model,
        tokenizer=_processor.tokenizer,
        feature_extractor=_processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
        model_kwargs={"use_cache": True},
    )


def _build_transcription_result(result: dict, audio_path: Path, language: str, sentence_chunks: List[Dict]) -> Dict[str, Any]:
    """构建转录结果字典"""
    return {
        "text": result["text"],
        "chunks": sentence_chunks,
        "language": language,
        "audio_file": audio_path.name,
        "audio_path": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "device": device,
        "output_files": {}
    }


def _save_transcription_files(transcription_result: Dict[str, Any], output_dir: Path, base_name: str, 
                               save_txt: bool, save_json: bool):
    """保存转录结果文件"""
    if save_txt:
        txt_path = output_dir / f"{base_name}_transcription.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"# 音频: {transcription_result['audio_file']} | 语言: {transcription_result['language']} | 模型: {model_name}\n")
            f.write(f"# 设备: {device} | 时间: {transcription_result['timestamp']}\n\n")
            f.write(f"{'='*60}\n完整文本:\n{'='*60}\n{transcription_result['text']}\n\n")
            
            if transcription_result['chunks']:
                f.write(f"{'='*60}\n分段文本（带时间戳）:\n{'='*60}\n")
                for chunk in transcription_result['chunks']:
                    ts = chunk.get("timestamp", (None, None))
                    if ts[0] is not None and ts[1] is not None:
                        f.write(f"[{format_timestamp(ts[0])} --> {format_timestamp(ts[1])}] {chunk['text']}\n")
                    else:
                        f.write(f"{chunk['text']}\n")
        
        transcription_result["output_files"]["txt"] = str(txt_path)
    
    if save_json:
        json_path = output_dir / f"{base_name}_transcription.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, ensure_ascii=False, indent=2)
        transcription_result["output_files"]["json"] = str(json_path)


def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "zh",
    output_dir: Optional[Union[str, Path]] = None,
    save_txt: bool = True,
    save_json: bool = True
) -> Dict[str, Any]:
    """转录音频文件并保存结果（自动启用时间戳模式）
    
    Args:
        audio_path: 音频文件路径
        language: 语言代码，默认 zh
        output_dir: 输出目录，默认 data/transcriptions/
        save_txt: 是否保存文本文件
        save_json: 是否保存 JSON 文件
    
    Returns:
        包含转录文本、时间戳和元数据的字典
    """
    load_model()
    
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 执行转录（使用句子级别时间戳以保留标点符号）
    # 注意：词级别 (return_timestamps="word") 不会输出标点
    result = _pipe(
        str(audio_path),
        generate_kwargs={
            "language": language,
            "task": "transcribe",
        },
        return_timestamps=True,  # 句子级别时间戳，包含标点
        chunk_length_s=30,
    )
    
    # 处理时间戳分段
    # 句子级别的 chunks 已包含标点，只需按字符数限制重新切分
    raw_chunks = result.get("chunks", [])
    if raw_chunks:
        # 如果有 chunks，进行智能合并（保持数字完整性）
        sentence_chunks = merge_word_chunks_to_sentences(
            raw_chunks, 
            max_sentences=2,
            max_chars=80
        )
    else:
        # 没有 chunks，使用完整文本
        sentence_chunks = []
    
    # 构建结果字典
    transcription_result = _build_transcription_result(result, audio_path, language, sentence_chunks)
    
    # 保存文件
    output_dir = Path(output_dir) if output_dir else Path("data/transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_transcription_files(transcription_result, output_dir, audio_path.stem, save_txt, save_json)
    
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
        print("用法: python whisper_parser.py <音频文件> [语言] [输出目录]")
        print("示例: python whisper_parser.py sample.mp3 zh data/output")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "zh"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        print(f"设备: {device} | 模型: {model_name} | 内存: {psutil.virtual_memory().available / 1024**3:.1f}GB")
        result = transcribe_audio(audio_file, language=lang, output_dir=out_dir)
        
        print(f"\n{'='*60}\n{result['text']}\n{'='*60}")
        if result.get("output_files"):
            for ftype, fpath in result["output_files"].items():
                print(f"{ftype.upper()}: {fpath}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
