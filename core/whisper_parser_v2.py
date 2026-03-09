#!/usr/bin/env python3
"""
Whisper 语音转文字模块 v2
支持长音频自动切割 -> 分段转录 -> 文档合并

流程：
  1. split_audio()      - 分析波形静音段，切割长音频为多个短片段
  2. transcribe_parts() - 对每个片段调用 Whisper 转录，输出独立文件
  3. merge_results()    - 将多段结果合并为一个 txt / json 文件
"""

import os
import json
import re
import inspect
import psutil
import numpy as np
import soundfile as sf
import librosa
import torch
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# --------------------------------------------------------------------------- #
#  配置常量
# --------------------------------------------------------------------------- #

CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/tmp/models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# 切割参数经验值
SILENCE_TOP_DB = 40          # 静音阈值（相对于最大振幅的 dB 值），librosa 标准推荐 40~60
MIN_SILENCE_DURATION = 0.5   # 最短静音段长度（秒），短于此不作为切割候选
MIN_SEGMENT_DURATION = 5.0   # 切割后每段最小时长（秒），短于此向后合并
MAX_SEGMENT_DURATION = 30.0  # 每段最大时长（秒），超过此强制切割（防止 Whisper 超时）

# --------------------------------------------------------------------------- #
#  设备 / 模型选择（与 v1 保持一致）
# --------------------------------------------------------------------------- #

def _get_optimal_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            arch = f"sm_{props.major}{props.minor}"
            if arch in torch.cuda.get_arch_list():
                return "cuda:0", torch.float16
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Whisper 在部分 Apple Silicon + MPS 上使用 float16 会出现空转录结果
        # （text/chunks 全空），此处固定使用 float32 提升稳定性。
        return "mps", torch.float32
    return "cpu", torch.float32


def _select_model() -> Tuple[str, str]:
    available_mem = psutil.virtual_memory().available / (1024 ** 3)
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    total = max(available_mem, gpu_mem)
    if total >= 6:
        return "openai/whisper-large-v3-turbo", "large-v3-turbo"
    elif total >= 4:
        return "openai/whisper-medium", "medium"
    else:
        return "openai/whisper-small", "small"


_DEVICE, _TORCH_DTYPE = _get_optimal_device()
_MODEL_ID, _MODEL_NAME = _select_model()

# 全局延迟加载 pipeline
_pipe = None


def _is_debug_enabled(debug: Optional[bool]) -> bool:
    if debug is not None:
        return debug
    return os.getenv("WHISPER_DEBUG", "0").strip() == "1"


def _is_breakpoint_enabled(debug_breakpoint: Optional[bool]) -> bool:
    if debug_breakpoint is not None:
        return debug_breakpoint
    return os.getenv("WHISPER_BREAKPOINT", "0").strip() == "1"


def _to_json_safe(obj: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return str(type(obj).__name__)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "size": int(obj.size),
        }
    if isinstance(obj, torch.Tensor):
        return {
            "type": "tensor",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v, depth + 1) for v in list(obj)[:20]]
    if hasattr(obj, "__dict__"):
        return {
            "type": type(obj).__name__,
            "attrs": _to_json_safe(vars(obj), depth + 1),
        }
    return str(obj)


def _debug_log(title: str, payload: Optional[Dict[str, Any]] = None, enabled: bool = False) -> None:
    if not enabled:
        return
    print(f"[debug] {title}")
    if payload is not None:
        try:
            print(json.dumps(_to_json_safe(payload), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[debug] payload 序列化失败: {e}")
            print(str(payload))


def _debug_breakpoint(stage: str, enabled: bool) -> None:
    if not enabled:
        return
    print(f"[debug] breakpoint at: {stage}")
    breakpoint()


def _sanitize_whisper_generation_config(model: Any, debug: bool = False) -> None:
    """
    修正部分模型仓库（如部分 ModelScope 镜像）中异常的 generation_config。

    已观察到 forced_decoder_ids 可能包含 [position, None]，这会导致
    解码前缀约束异常，进而出现整段输出为空文本的现象。
    """
    if model is None or not hasattr(model, "generation_config"):
        return

    gen_cfg = model.generation_config
    original_forced = getattr(gen_cfg, "forced_decoder_ids", None)
    sanitized_forced = None

    if isinstance(original_forced, list):
        cleaned: List[List[int]] = []
        invalid_found = False
        for pair in original_forced:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and pair[0] is not None
                and pair[1] is not None
            ):
                cleaned.append([int(pair[0]), int(pair[1])])
            else:
                invalid_found = True

        if invalid_found:
            # 交由 generate_kwargs(language/task) 决定解码前缀，避免错误强制约束
            sanitized_forced = cleaned if cleaned else None
            gen_cfg.forced_decoder_ids = sanitized_forced
            # 兼容部分版本从 model.config 读取该字段
            if hasattr(model, "config"):
                model.config.forced_decoder_ids = sanitized_forced

    _debug_log("generation_config 清洗结果", {
        "forced_decoder_ids_before": original_forced,
        "forced_decoder_ids_after": getattr(gen_cfg, "forced_decoder_ids", None),
        "suppress_tokens": getattr(gen_cfg, "suppress_tokens", None),
    }, debug)


# =========================================================================== #
#  第一部分：长音频切割
# =========================================================================== #

def _find_silence_boundaries(
    y: np.ndarray,
    sr: int,
    top_db: float = SILENCE_TOP_DB,
    min_silence_duration: float = MIN_SILENCE_DURATION,
) -> List[float]:
    """
    利用 librosa 检测静音段，返回每段静音区间的中心时间点（秒）列表。

    Args:
        y:                    音频时域信号（mono float32）
        sr:                   采样率
        top_db:               静音判断阈值（dB），高于此值的帧视为有声
        min_silence_duration: 最短静音持续时间（秒），短于此不做切割候选

    Returns:
        候选切割时间点列表（秒，升序）
    """
    # librosa.effects.split 返回有声段的帧区间 [[start, end], ...]
    # top_db 越大，对静音越宽容（更多片段被视为静音）
    voiced_intervals = librosa.effects.split(y, top_db=top_db)

    if len(voiced_intervals) <= 1:
        return []

    cut_points = []
    min_silence_frames = int(min_silence_duration * sr)

    for i in range(len(voiced_intervals) - 1):
        silence_start = voiced_intervals[i][1]      # 上一段有声的结束帧
        silence_end   = voiced_intervals[i + 1][0]  # 下一段有声的起始帧
        silence_len   = silence_end - silence_start

        if silence_len >= min_silence_frames:
            # 取静音段中心作为切割点
            mid_frame = (silence_start + silence_end) // 2
            cut_points.append(mid_frame / sr)

    return cut_points


def _merge_short_segments(
    boundaries: List[float],
    total_duration: float,
    min_duration: float = MIN_SEGMENT_DURATION,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> List[Tuple[float, float]]:
    """
    根据切割点生成时间区间，并将过短的区间向后合并，过长的区间强制切分。

    合并规则：
      - 如果一段时长 < min_duration，则与下一段合并（不断循环直至满足条件）
      - 如果合并后的段仍超过 max_duration，按 max_duration 强制均匀切分

    Args:
        boundaries:     候选切割时间点（秒，升序），不含 0 和 total_duration
        total_duration: 音频总时长（秒）
        min_duration:   最小段时长（秒）
        max_duration:   最大段时长（秒）

    Returns:
        [(start, end), ...] 形式的时间区间列表（秒）
    """
    # 构造初始段列表
    edges = [0.0] + sorted(boundaries) + [total_duration]
    raw_segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    if not raw_segments:
        return [(0.0, total_duration)]

    # --- 向后合并过短段 ---
    merged: List[Tuple[float, float]] = []
    acc_start = raw_segments[0][0]
    acc_end   = raw_segments[0][1]

    for seg_start, seg_end in raw_segments[1:]:
        acc_duration = acc_end - acc_start
        if acc_duration < min_duration:
            # 当前累积段太短，继续向后合并
            acc_end = seg_end
        else:
            merged.append((acc_start, acc_end))
            acc_start = seg_start
            acc_end   = seg_end

    # 最后一段：如果太短，合并到前一段
    last_duration = acc_end - acc_start
    if last_duration < min_duration and merged:
        prev_start, _ = merged.pop()
        merged.append((prev_start, acc_end))
    else:
        merged.append((acc_start, acc_end))

    # --- 切分过长段 ---
    final_segments: List[Tuple[float, float]] = []
    for seg_start, seg_end in merged:
        duration = seg_end - seg_start
        if duration <= max_duration:
            final_segments.append((seg_start, seg_end))
        else:
            # 均匀切分
            n_parts = int(np.ceil(duration / max_duration))
            part_len = duration / n_parts
            for k in range(n_parts):
                ps = seg_start + k * part_len
                pe = seg_start + (k + 1) * part_len
                final_segments.append((ps, min(pe, seg_end)))

    return final_segments


def split_audio(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    top_db: float = SILENCE_TOP_DB,
    min_silence_duration: float = MIN_SILENCE_DURATION,
    min_segment_duration: float = MIN_SEGMENT_DURATION,
    max_segment_duration: float = MAX_SEGMENT_DURATION,
) -> List[Path]:
    """
    将长音频按静音段切割为多个短片段并保存。

    若没有找到任何静音切割点（音频本身已足够短或全段有声），
    则直接返回原文件路径（不复制，不切割）。

    Args:
        audio_path:           原始音频文件路径
        output_dir:           切割片段的保存目录（默认同目录下 <stem>_parts/）
        top_db:               静音阈值（dB）
        min_silence_duration: 最短静音持续时间（秒）
        min_segment_duration: 每段最小时长（秒），短于此向后合并
        max_segment_duration: 每段最大时长（秒），超过此强制均匀切分

    Returns:
        切割后的音频文件路径列表（按顺序排列），如未切割则返回 [audio_path]
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 加载音频（转为 mono，保持原采样率用于波形分析）
    print(f"[split] 加载音频: {audio_path.name}")
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    total_duration = len(y) / sr
    print(f"[split] 时长: {total_duration:.2f}s | 采样率: {sr}Hz")

    # 如果音频本身已足够短，直接返回原文件
    if total_duration <= max_segment_duration:
        print(f"[split] 音频时长 <= {max_segment_duration}s，无需切割")
        return [audio_path]

    # 检测静音边界
    cut_points = _find_silence_boundaries(y, sr, top_db, min_silence_duration)
    print(f"[split] 检测到 {len(cut_points)} 个候选切割点")

    if not cut_points:
        print("[split] 未找到静音切割点，将按最大时长强制均匀切割")

    # 生成并合并时间区间
    segments = _merge_short_segments(cut_points, total_duration, min_segment_duration, max_segment_duration)
    print(f"[split] 最终切割为 {len(segments)} 段:")
    for i, (s, e) in enumerate(segments):
        print(f"  Part {i+1}: {s:.2f}s -> {e:.2f}s  ({e-s:.2f}s)")

    # 输出目录
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_parts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将整段音频重采样到 16kHz（Whisper 要求的采样率），
    # 在切割前统一做一次，避免对每个片段重复 resample。
    # soundfile 不支持 mp3 写入，统一输出 wav。
    WHISPER_SR = 16000
    out_suffix = ".wav"
    if sr != WHISPER_SR:
        print(f"[split] 重采样: {sr}Hz -> {WHISPER_SR}Hz")
        y_out = librosa.resample(y, orig_sr=sr, target_sr=WHISPER_SR)
        out_sr = WHISPER_SR
    else:
        y_out = y
        out_sr = sr

    part_paths: List[Path] = []
    for i, (seg_start, seg_end) in enumerate(segments):
        frame_start = int(seg_start * out_sr)
        frame_end   = min(int(seg_end * out_sr), len(y_out))
        segment_y   = y_out[frame_start:frame_end]

        part_name = f"{audio_path.stem}_part_{i+1}{out_suffix}"
        part_path = output_dir / part_name
        sf.write(str(part_path), segment_y, out_sr)
        part_paths.append(part_path)
        print(f"[split] 保存: {part_path.name}")

    return part_paths


# =========================================================================== #
#  第二部分：音频转录
# =========================================================================== #

def _load_model(debug: Optional[bool] = None, debug_breakpoint: Optional[bool] = None) -> None:
    """延迟加载 Whisper pipeline（全局单例）"""
    global _pipe
    dbg = _is_debug_enabled(debug)
    dbg_bp = _is_breakpoint_enabled(debug_breakpoint)

    if _pipe is not None:
        _debug_log("pipeline 已加载，跳过重复初始化", {
            "pipeline_class": type(_pipe).__name__,
        }, dbg)
        return

    cache_dir_abs = str(Path(CACHE_DIR).resolve())
    os.environ.setdefault("MODELSCOPE_CACHE", cache_dir_abs)
    os.environ.setdefault("HF_HOME", cache_dir_abs)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir_abs)

    model = None
    processor = None

    # 优先从 ModelScope 加载
    _modelscope_ids = {
        "large-v3-turbo": "AI-ModelScope/whisper-large-v3-turbo",
        "medium":         "AI-ModelScope/whisper-medium",
        "small":          "AI-ModelScope/whisper-small",
    }
    ms_id = _modelscope_ids.get(_MODEL_NAME, "AI-ModelScope/whisper-large-v3-turbo")

    _debug_log("模型加载前参数", {
        "CACHE_DIR": CACHE_DIR,
        "cache_dir_abs": cache_dir_abs,
        "device": _DEVICE,
        "torch_dtype": str(_TORCH_DTYPE),
        "model_id": _MODEL_ID,
        "model_name": _MODEL_NAME,
        "ms_id": ms_id,
    }, dbg)
    _debug_breakpoint("_load_model.before_download", dbg_bp)

    try:
        from modelscope.hub.snapshot_download import snapshot_download
        model_dir = snapshot_download(ms_id, cache_dir=cache_dir_abs)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=_TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(_DEVICE)
        processor = AutoProcessor.from_pretrained(model_dir)
        _debug_log("ModelScope 加载成功", {
            "model_dir": model_dir,
        }, dbg)
    except Exception:
        _debug_log("ModelScope 加载失败，回退 HuggingFace", {
            "fallback_model_id": _MODEL_ID,
        }, dbg)
        # 回退到 HuggingFace Hub
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _MODEL_ID,
            torch_dtype=_TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir_abs,
        )
        model.to(_DEVICE)
        processor = AutoProcessor.from_pretrained(_MODEL_ID, cache_dir=cache_dir_abs)

    _debug_log("模型与处理器信息", {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__ if processor else None,
        "feature_extractor_class": type(processor.feature_extractor).__name__ if processor else None,
        "feature_extractor_sampling_rate": getattr(processor.feature_extractor, "sampling_rate", None) if processor else None,
        "model_config_forced_decoder_ids": getattr(model.generation_config, "forced_decoder_ids", None) if model else None,
        "model_config_suppress_tokens": getattr(model.generation_config, "suppress_tokens", None) if model else None,
    }, dbg)

    _sanitize_whisper_generation_config(model, debug=dbg)
    _debug_breakpoint("_load_model.before_pipeline", dbg_bp)

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=_TORCH_DTYPE,
        device=_DEVICE,
        model_kwargs={"use_cache": True},
    )

    pipe_call_sig = None
    pipe_preprocess_sig = None
    try:
        pipe_call_sig = str(inspect.signature(_pipe.__call__))
    except Exception:
        pass
    try:
        pipe_preprocess_sig = str(inspect.signature(_pipe.preprocess))
    except Exception:
        pass

    _debug_log("pipeline 初始化完成", {
        "pipeline_class": type(_pipe).__name__,
        "pipeline_task": getattr(_pipe, "task", None),
        "pipe_call_signature": pipe_call_sig,
        "pipe_preprocess_signature": pipe_preprocess_sig,
    }, dbg)
    _debug_breakpoint("_load_model.after_pipeline", dbg_bp)


def _format_timestamp(seconds: Optional[float]) -> str:
    """将秒数转为 HH:MM:SS.mmm 字符串"""
    if seconds is None:
        return "00:00:00.000"
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _raw_chunks_to_safe_chunks(raw_chunks: List[Dict]) -> List[Dict]:
    """
    清洗 Whisper 返回的 chunks：
    - 过滤掉 timestamp 含 None 的 chunk（Whisper 预测失败时出现）
    - 将 tuple 统一转为 list 以便 JSON 序列化
    """
    safe = []
    for c in raw_chunks:
        ts = c.get("timestamp", (None, None))
        # ts 可能是 tuple 或 list
        start = ts[0] if ts else None
        end   = ts[1] if ts else None
        safe.append({
            "timestamp": [start, end],
            "text":      c.get("text", ""),
        })
    return safe


def transcribe_part(
    audio_path: Union[str, Path],
    language: str = "zh",
    output_dir: Optional[Union[str, Path]] = None,
    save_txt: bool = True,
    save_json: bool = True,
    time_offset: float = 0.0,
    debug: Optional[bool] = None,
    debug_breakpoint: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    对单个音频片段调用 Whisper 转录，输出 txt / json 文件。

    Args:
        audio_path:   音频文件路径
        language:     语言代码（zh / en / …）
        output_dir:   结果文件输出目录（默认与音频同目录）
        save_txt:     是否输出 txt
        save_json:    是否输出 json
        time_offset:  时间戳偏移量（秒），用于标注在原始音频中的绝对时间

    Returns:
        转录结果字典，包含 text / chunks / output_files 等字段
    """
    dbg = _is_debug_enabled(debug)
    dbg_bp = _is_breakpoint_enabled(debug_breakpoint)

    _load_model(debug=dbg, debug_breakpoint=dbg_bp)

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    print(f"[transcribe] 转录: {audio_path.name}  (offset={time_offset:.2f}s)")

    # 关键：用 librosa 强制重采样到 Whisper 要求的 16000Hz
    # 以字典形式传入 pipeline，绕过 pipeline 内部的文件读取路径，
    # 确保 feature_extractor 拿到的是正确采样率的信号。
    # 若直接传文件路径，pipeline 依赖后端自动重采样，
    # 当原始文件为 44100Hz 时极易产生采样率不匹配，导致输出全为 "!" 幻觉。
    WHISPER_SR = 16000
    y_16k, _ = librosa.load(str(audio_path), sr=WHISPER_SR, mono=True)
    # pipeline 要求字典键名为 "raw"（不是 "array"）
    audio_input = {"raw": y_16k, "sampling_rate": WHISPER_SR}

    _debug_log("transcribe_part 输入结构", {
        "audio_path": str(audio_path),
        "audio_exists": audio_path.exists(),
        "audio_suffix": audio_path.suffix,
        "audio_input_keys": list(audio_input.keys()),
        "raw_dtype": str(y_16k.dtype),
        "raw_shape": list(y_16k.shape),
        "raw_size": int(y_16k.size),
        "raw_min": float(np.min(y_16k)) if y_16k.size else None,
        "raw_max": float(np.max(y_16k)) if y_16k.size else None,
        "raw_mean": float(np.mean(y_16k)) if y_16k.size else None,
        "raw_std": float(np.std(y_16k)) if y_16k.size else None,
        "raw_nan_count": int(np.isnan(y_16k).sum()) if y_16k.size else 0,
        "raw_inf_count": int(np.isinf(y_16k).sum()) if y_16k.size else 0,
        "sampling_rate": WHISPER_SR,
        "duration_sec": float(y_16k.size / WHISPER_SR) if y_16k.size else 0.0,
        "language": language,
        "time_offset": time_offset,
        "device": _DEVICE,
        "dtype": str(_TORCH_DTYPE),
        "feature_extractor_sampling_rate": getattr(getattr(_pipe, "feature_extractor", None), "sampling_rate", None),
    }, dbg)
    _debug_breakpoint("transcribe_part.before_pipe_call", dbg_bp)

    generate_kwargs = {
        "language": language,
        "task":     "transcribe"
    }
    
    _debug_log("_pipe 调用参数", {
        "input_type": type(audio_input).__name__,
        "input_schema": {
            "raw": type(audio_input["raw"]).__name__,
            "sampling_rate": type(audio_input["sampling_rate"]).__name__,
        },
        "generate_kwargs": generate_kwargs,
        "return_timestamps": True,
    }, dbg)

    # 调用 Whisper pipeline
    # - return_timestamps=True：获取句子级时间戳（含标点）
    # - 不传 chunk_length_s：分段已由 split_audio 保证 ≤ MAX_SEGMENT_DURATION，
    #   无需 pipeline 内部再分块（内部分块会导致末尾 timestamp=None 的问题）
    raw_result = _pipe(
        audio_input,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )

    _debug_log("_pipe 返回结构", {
        "raw_result_type": type(raw_result).__name__,
        "raw_result_keys": list(raw_result.keys()) if isinstance(raw_result, dict) else None,
        "text_len": len(raw_result.get("text", "")) if isinstance(raw_result, dict) else None,
        "chunks_count": len(raw_result.get("chunks", [])) if isinstance(raw_result, dict) else None,
        "chunks_preview": raw_result.get("chunks", [])[:3] if isinstance(raw_result, dict) else None,
    }, dbg)
    _debug_breakpoint("transcribe_part.after_pipe_call", dbg_bp)

    # 清洗 chunks，处理 None timestamp
    raw_chunks = raw_result.get("chunks", [])
    chunks = _raw_chunks_to_safe_chunks(raw_chunks)

    # 叠加时间偏移
    if time_offset > 0:
        for c in chunks:
            if c["timestamp"][0] is not None:
                c["timestamp"][0] += time_offset
            if c["timestamp"][1] is not None:
                c["timestamp"][1] += time_offset

    result: Dict[str, Any] = {
        "text":         raw_result.get("text", ""),
        "chunks":       chunks,
        "language":     language,
        "audio_file":   audio_path.name,
        "audio_path":   str(audio_path),
        "time_offset":  time_offset,
        "timestamp":    datetime.now().isoformat(),
        "model":        _MODEL_ID,
        "device":       _DEVICE,
        "output_files": {},
    }

    # 确定输出目录
    out_dir = Path(output_dir) if output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = audio_path.stem  # e.g. test_part_1

    # 写 txt
    if save_txt:
        txt_path = out_dir / f"{base}_transcription.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# 音频: {result['audio_file']} | 语言: {language} | 模型: {_MODEL_NAME}\n")
            f.write(f"# 设备: {_DEVICE} | 时间: {result['timestamp']}\n\n")
            f.write(f"{'='*60}\n完整文本:\n{'='*60}\n{result['text']}\n\n")
            if chunks:
                f.write(f"{'='*60}\n分段文本（带时间戳）:\n{'='*60}\n")
                for c in chunks:
                    ts = c["timestamp"]
                    if ts[0] is not None and ts[1] is not None:
                        f.write(
                            f"[{_format_timestamp(ts[0])} --> {_format_timestamp(ts[1])}] {c['text']}\n"
                        )
                    else:
                        f.write(f"{c['text']}\n")
        result["output_files"]["txt"] = str(txt_path)
        print(f"[transcribe] TXT -> {txt_path.name}")

    # 写 json
    if save_json:
        json_path = out_dir / f"{base}_transcription.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["output_files"]["json"] = str(json_path)
        print(f"[transcribe] JSON -> {json_path.name}")

    return result


def transcribe_parts(
    part_paths: List[Union[str, Path]],
    language: str = "zh",
    output_dir: Optional[Union[str, Path]] = None,
    save_txt: bool = True,
    save_json: bool = True,
    part_durations: Optional[List[float]] = None,
    debug: Optional[bool] = None,
    debug_breakpoint: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    对一组音频片段依次调用 transcribe_part，返回所有转录结果列表。

    Args:
        part_paths:      音频片段路径列表（split_audio 的返回值）
        language:        语言代码
        output_dir:      所有结果文件的输出目录
        save_txt:        是否输出 txt
        save_json:       是否输出 json
        part_durations:  各片段时长列表（秒），用于计算时间偏移量；
                         若为 None，则自动通过 soundfile 读取各片段时长

    Returns:
        各片段转录结果字典的列表
    """
    results = []
    offset = 0.0

    for i, p in enumerate(part_paths):
        p = Path(p)

        # 计算当前片段的时间偏移（在原始音频中的起始秒数）
        result = transcribe_part(
            audio_path=p,
            language=language,
            output_dir=output_dir,
            save_txt=save_txt,
            save_json=save_json,
            time_offset=offset,
            debug=debug,
            debug_breakpoint=debug_breakpoint,
        )
        results.append(result)

        # 更新 offset：使用已知时长或读取文件
        if part_durations and i < len(part_durations):
            offset += part_durations[i]
        else:
            info = sf.info(str(p))
            offset += info.duration

    return results


# =========================================================================== #
#  第三部分：文档合并
# =========================================================================== #

def merge_txt_files(
    txt_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> Path:
    """
    将多个分段 txt 转录文件按顺序拼接为一个 txt 文件。

    合并策略：
    - 第一个文件保留完整头部注释行（# 开头）
    - 后续文件跳过头部注释行
    - 各分段完整文本依次追加
    - 各分段带时间戳的分段文本依次追加

    Args:
        txt_paths:    分段 txt 文件路径列表（按顺序）
        output_path:  合并后输出的 txt 文件路径

    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_full_texts: List[str] = []
    all_segment_lines: List[str] = []

    for i, p in enumerate(txt_paths):
        p = Path(p)
        if not p.exists():
            print(f"[merge_txt] 警告: 文件不存在，跳过: {p}")
            continue

        content = p.read_text(encoding="utf-8")

        # txt 格式固定为（由 transcribe_part 写入）：
        # # 头部注释行 ...
        #
        # ============================================================
        # 完整文本:
        # ============================================================
        # <完整文本内容>
        #
        # ============================================================
        # 分段文本（带时间戳）:
        # ============================================================
        # [ts --> ts] 句子
        # ...
        #
        # 用正则按块切割，避免手写脆弱的状态机
        SEP = "=" * 60

        # 提取"完整文本"块
        full_match = re.search(
            r"(?:^|\n)" + re.escape(SEP) + r"\n完整文本:\n" + re.escape(SEP) + r"\n(.*?)(?:\n\n" + re.escape(SEP) + r"|$)",
            content,
            re.DOTALL,
        )
        full_text = full_match.group(1).strip() if full_match else ""

        # 提取"分段文本"块
        seg_match = re.search(
            r"(?:^|\n)" + re.escape(SEP) + r"\n分段文本（带时间戳）:\n" + re.escape(SEP) + r"\n(.*?)$",
            content,
            re.DOTALL,
        )
        seg_lines = []
        if seg_match:
            seg_lines = [l for l in seg_match.group(1).splitlines() if l.strip()]

        all_full_texts.append(full_text)
        all_segment_lines.extend(seg_lines)

    sep = "=" * 60
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# 合并转录文件 | 共 {len(txt_paths)} 段 | 时间: {datetime.now().isoformat()}\n\n")
        f.write(f"{sep}\n完整文本:\n{sep}\n")
        f.write("\n".join(all_full_texts))
        f.write("\n\n")
        if all_segment_lines:
            f.write(f"{sep}\n分段文本（带时间戳）:\n{sep}\n")
            f.write("\n".join(all_segment_lines))
            f.write("\n")

    print(f"[merge] TXT 合并完成 -> {output_path}")
    return output_path


def merge_json_files(
    json_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> Path:
    """
    将多个分段 json 转录文件合并为一个 json 文件。

    合并策略：
    - text:    各段 text 字符串按顺序拼接（以换行符分隔）
    - chunks:  各段 chunks 列表按顺序合并为一个列表
    - 其余元数据字段（language / model / device 等）取第一段的值
    - 新增 parts 字段，记录各段的 audio_file / audio_path / time_offset

    Args:
        json_paths:   分段 json 文件路径列表（按顺序）
        output_path:  合并后输出的 json 文件路径

    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_text:   List[str]      = []
    merged_chunks: List[Dict]     = []
    parts_meta:    List[Dict]     = []
    base_meta:     Dict[str, Any] = {}

    for i, p in enumerate(json_paths):
        p = Path(p)
        if not p.exists():
            print(f"[merge_json] 警告: 文件不存在，跳过: {p}")
            continue

        with open(p, encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        if i == 0:
            # 保留第一段的元数据
            base_meta = {
                k: v for k, v in data.items()
                if k not in ("text", "chunks", "output_files", "audio_file", "audio_path", "time_offset", "timestamp")
            }

        text = data.get("text", "").strip()
        if text:
            merged_text.append(text)

        merged_chunks.extend(data.get("chunks", []))

        parts_meta.append({
            "audio_file":  data.get("audio_file", ""),
            "audio_path":  data.get("audio_path", ""),
            "time_offset": data.get("time_offset", 0.0),
            "text":        text,
        })

    result: Dict[str, Any] = {
        **base_meta,
        "text":      "\n".join(merged_text),
        "chunks":    merged_chunks,
        "timestamp": datetime.now().isoformat(),
        "parts":     parts_meta,
        "output_files": {"json": str(output_path)},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[merge] JSON 合并完成 -> {output_path}")
    return output_path


def merge_results(
    part_results: List[Dict[str, Any]],
    output_stem: str,
    output_dir: Union[str, Path],
    save_txt: bool = True,
    save_json: bool = True,
) -> Dict[str, str]:
    """
    将 transcribe_parts 返回的多段结果合并为单个文件。

    Args:
        part_results: transcribe_parts 的返回列表
        output_stem:  输出文件名（不含扩展名），如 "test"
        output_dir:   输出目录
        save_txt:     是否合并 txt
        save_json:    是否合并 json

    Returns:
        {"txt": <合并后txt路径>, "json": <合并后json路径>}（按实际输出填充）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_paths: Dict[str, str] = {}

    if save_txt:
        txt_paths = [
            r["output_files"]["txt"]
            for r in part_results
            if "txt" in r.get("output_files", {})
        ]
        if txt_paths:
            out_txt = output_dir / f"{output_stem}_transcription.txt"
            merge_txt_files(txt_paths, out_txt)
            merged_paths["txt"] = str(out_txt)

    if save_json:
        json_paths = [
            r["output_files"]["json"]
            for r in part_results
            if "json" in r.get("output_files", {})
        ]
        if json_paths:
            out_json = output_dir / f"{output_stem}_transcription.json"
            merge_json_files(json_paths, out_json)
            merged_paths["json"] = str(out_json)

    return merged_paths


# =========================================================================== #
#  公共入口：一键完成切割 -> 转录 -> 合并
# =========================================================================== #

def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "zh",
    output_dir: Optional[Union[str, Path]] = None,
    save_txt: bool = True,
    save_json: bool = True,
    keep_parts: bool = False,
    debug: Optional[bool] = None,
    debug_breakpoint: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    对任意时长音频执行完整录转流程：
    切割 -> 分段转录 -> 合并结果。

    Args:
        audio_path:  原始音频文件路径
        language:    语言代码（zh / en / …）
        output_dir:  最终合并文件的输出目录（默认 data/transcriptions/）
        save_txt:    是否输出 txt
        save_json:   是否输出 json
        keep_parts:  是否保留切割片段和分段转录文件（默认删除）

    Returns:
        {
          "text":        合并后全文,
          "parts":       各分段结果列表,
          "output_files":{"txt": ..., "json": ...},
        }
    """
    audio_path = Path(audio_path).resolve()

    # 确定输出目录
    final_out = Path(output_dir) if output_dir else Path("data/transcriptions")
    final_out.mkdir(parents=True, exist_ok=True)

    # 切割片段保存在临时子目录
    parts_dir = final_out / f"{audio_path.stem}_parts"

    # 1. 切割
    part_paths = split_audio(
        audio_path=audio_path,
        output_dir=parts_dir,
    )

    # 2. 逐段转录（分段结果保存在同一子目录，后续合并）
    parts_out_dir = parts_dir / "transcriptions"
    if len(part_paths) == 1 and part_paths[0] == audio_path:
        # 未切割（原文件直接使用），输出到最终目录
        part_results = transcribe_parts(
            part_paths, language=language,
            output_dir=final_out,
            save_txt=save_txt, save_json=save_json,
            debug=debug,
            debug_breakpoint=debug_breakpoint,
        )
        merged_files = {
            k: v
            for r in part_results
            for k, v in r.get("output_files", {}).items()
        }
    else:
        part_results = transcribe_parts(
            part_paths, language=language,
            output_dir=parts_out_dir,
            save_txt=save_txt, save_json=save_json,
            debug=debug,
            debug_breakpoint=debug_breakpoint,
        )

        # 3. 合并
        merged_files = merge_results(
            part_results=part_results,
            output_stem=audio_path.stem,
            output_dir=final_out,
            save_txt=save_txt,
            save_json=save_json,
        )

        # 清理临时文件（可选）
        if not keep_parts:
            import shutil
            shutil.rmtree(parts_dir, ignore_errors=True)
            print(f"[transcribe_audio] 已清理临时目录: {parts_dir}")

    full_text = "\n".join(
        r.get("text", "").strip() for r in part_results if r.get("text", "").strip()
    )

    return {
        "text":         full_text,
        "parts":        part_results,
        "output_files": merged_files,
    }


# =========================================================================== #
#  命令行入口
# =========================================================================== #

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python whisper_parser_v2.py <音频文件> [语言] [输出目录]")
        print("示例: python whisper_parser_v2.py sample.mp3 zh data/output")
        sys.exit(1)
    
    _audio_file = sys.argv[1]
    _lang       = sys.argv[2] if len(sys.argv) > 2 else "zh"
    _out_dir    = sys.argv[3] if len(sys.argv) > 3 else None
    # _debug = os.getenv("WHISPER_DEBUG", "0").strip() == "1"
    # _debug_bp = os.getenv("WHISPER_BREAKPOINT", "0").strip() == "1"
    _debug = True
    _debug_bp = False
    try:
        print(f"设备: {_DEVICE} | 模型: {_MODEL_NAME} | 内存: {psutil.virtual_memory().available / 1024**3:.1f}GB")
        print(f"调试: {_debug} | 断点: {_debug_bp}")
        _result = transcribe_audio(
            _audio_file,
            language=_lang,
            output_dir=_out_dir,
            debug=_debug,
            debug_breakpoint=_debug_bp,
        )
        print(f"\n{'='*60}\n{_result['text'][:500]}\n{'='*60}")
        for ftype, fpath in _result.get("output_files", {}).items():
            print(f"{ftype.upper()}: {fpath}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
