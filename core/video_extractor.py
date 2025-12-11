#!/usr/bin/env python3
"""
视频转文字主程序
整合音频提取和语音转文字功能，完成完整的视频到文字的转换流程

工作流程:
1. 从视频中提取音频 (audio_extractor)
2. 使用 Whisper 将音频转为文字 (whisper_parser)
3. 保存转录结果到指定目录

使用示例:
    python video_extractor.py video.mp4
    python video_extractor.py video.mp4 --language en
    python video_extractor.py video.mp4 --output-dir ./output
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

# 导入自定义模块
from audio_extractor import extract_audio_from_video, AudioExtractor
from whisper_parser import transcribe_audio, load_model


class VideoToTextConverter:
    """视频转文字转换器"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        keep_audio: bool = False,
        verbose: bool = True
    ):
        """
        初始化转换器
        
        Args:
            output_dir: 输出目录，默认为 data/transcriptions/
            keep_audio: 是否保留提取的音频文件
            verbose: 是否显示详细信息
        """
        self.output_dir = Path(output_dir) if output_dir else Path("data/transcriptions")
        self.keep_audio = keep_audio
        self.verbose = verbose
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时音频目录
        self.temp_audio_dir = Path("data/temp_audio")
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.audio_extractor = AudioExtractor()
    
    def convert(
        self,
        video_path: Union[str, Path],
        language: str = "zh",
        audio_format: str = "mp3",
        audio_bitrate: str = "128k"
    ) -> dict:
        """
        将视频转换为文字
        
        Args:
            video_path: 视频文件路径
            language: 语言代码 (zh, en, ja, ko 等)
            audio_format: 音频格式 (mp3, wav 等)
            audio_bitrate: 音频比特率
        
        Returns:
            包含转录结果和元数据的字典
        """
        video_path = Path(video_path).resolve()
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        start_time = datetime.now()
        
        if self.verbose:
            print("="*70)
            print("视频转文字处理流程")
            print("="*70)
            print(f"输入视频: {video_path.name}")
            print(f"语言: {language}")
            print(f"输出目录: {self.output_dir}")
            print("="*70 + "\n")
        
        # 步骤 1: 提取音频
        if self.verbose:
            print("[步骤 1/2] 提取音频...")
            print("-" * 70)
        
        audio_filename = f"{video_path.stem}.{audio_format}"
        
        if self.keep_audio:
            # 保存到输出目录
            audio_path = self.output_dir / audio_filename
        else:
            # 保存到临时目录
            audio_path = self.temp_audio_dir / audio_filename
        
        try:
            extracted_audio = self.audio_extractor.extract_audio_from_video(
                video_path=video_path,
                output_path=audio_path,
                audio_format=audio_format,
                bitrate=audio_bitrate,
                verbose=self.verbose
            )
        except Exception as e:
            raise RuntimeError(f"音频提取失败: {e}")
        
        if self.verbose:
            print()
        
        # 步骤 2: 转录音频
        if self.verbose:
            print("[步骤 2/2] 转录音频为文字...")
            print("-" * 70)
        
        try:
            transcription_result = transcribe_audio(
                audio_path=extracted_audio,
                language=language,
                output_dir=self.output_dir,
                save_txt=True,
                save_json=True,
                verbose=self.verbose
            )
        except Exception as e:
            raise RuntimeError(f"音频转录失败: {e}")
        
        # 清理临时音频文件
        if not self.keep_audio and Path(extracted_audio).exists():
            try:
                Path(extracted_audio).unlink()
                if self.verbose:
                    print(f"✓ 已清理临时音频文件")
            except Exception:
                pass
        
        # 添加视频相关信息
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        result = {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "audio_file": Path(extracted_audio).name if self.keep_audio else None,
            "language": language,
            "transcription": transcription_result["text"],
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": start_time.isoformat(),
            "output_dir": str(self.output_dir),
            "output_files": transcription_result.get("output_files", {})
        }
        
        # 保存完整结果
        result_json_path = self.output_dir / f"{video_path.stem}_result.json"
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        result["result_json"] = str(result_json_path)
        
        if self.verbose:
            print("\n" + "="*70)
            print("处理完成！")
            print("="*70)
            print(f"处理时长: {processing_time:.2f} 秒")
            print(f"文本长度: {len(result['transcription'])} 字符")
            print(f"\n输出文件:")
            for file_type, file_path in result["output_files"].items():
                print(f"  {file_type.upper()}: {file_path}")
            print(f"  RESULT: {result_json_path}")
            print("="*70 + "\n")
        
        return result


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="视频转文字工具 - 提取音频并转录为文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s video.mp4
  %(prog)s video.mp4 --language en
  %(prog)s video.mp4 --output-dir ./output --keep-audio
  %(prog)s video.mp4 -l ja -o ./results -k

支持的语言:
  zh (中文), en (英语), ja (日语), ko (韩语), fr (法语), de (德语), 等
  
支持的视频格式:
  MP4, AVI, MKV, MOV, WMV, FLV, WEBM, MPEG, MPG, 3GP, M4V, TS, VOB, 等
        """
    )
    
    parser.add_argument(
        "video_path",
        help="输入视频文件路径"
    )
    
    parser.add_argument(
        "-l", "--language",
        default="zh",
        help="语言代码 (默认: zh)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出目录 (默认: data/transcriptions/)"
    )
    
    parser.add_argument(
        "-k", "--keep-audio",
        action="store_true",
        help="保留提取的音频文件"
    )
    
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "wav", "aac", "ogg", "flac"],
        help="音频格式 (默认: mp3)"
    )
    
    parser.add_argument(
        "--audio-bitrate",
        default="128k",
        help="音频比特率 (默认: 128k)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式，不显示详细信息"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = VideoToTextConverter(
            output_dir=args.output_dir,
            keep_audio=args.keep_audio,
            verbose=not args.quiet
        )
        
        # 执行转换
        result = converter.convert(
            video_path=args.video_path,
            language=args.language,
            audio_format=args.audio_format,
            audio_bitrate=args.audio_bitrate
        )
        
        # 输出转录文本
        if not args.quiet:
            print("\n" + "="*70)
            print("转录文本:")
            print("="*70)
            print(result["transcription"])
            print("="*70)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
