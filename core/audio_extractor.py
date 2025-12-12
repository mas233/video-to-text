#!/usr/bin/env python3
"""
音频提取模块
从视频文件中提取音频，支持主流视频格式

支持的格式:
- MP4, AVI, MKV, MOV, WMV, FLV, WEBM
- MPEG, MPG, 3GP, M4V, TS, VOB
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union


class AudioExtractor:
    """音频提取器类"""
    
    # 支持的视频格式
    SUPPORTED_FORMATS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
        '.mpeg', '.mpg', '.3gp', '.m4v', '.ts', '.vob', '.ogv',
        '.rm', '.rmvb', '.asf', '.divx'
    }
    
    # 默认音频参数
    DEFAULT_AUDIO_CODEC = 'mp3'
    DEFAULT_AUDIO_BITRATE = '128k'
    DEFAULT_AUDIO_CHANNELS = 2  # 立体声
    DEFAULT_SAMPLE_RATE = 44100  # 44.1kHz
    
    def __init__(self):
        """初始化音频提取器"""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查 ffmpeg 是否已安装"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError("ffmpeg 未正确安装")
        except FileNotFoundError:
            raise RuntimeError(
                "错误: 未找到 ffmpeg。\n"
                "请安装 ffmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  Windows: 从 https://ffmpeg.org/download.html 下载"
            )
    
    def is_supported_format(self, video_path: Union[str, Path]) -> bool:
        """
        检查视频格式是否支持
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            是否支持该格式
        """
        ext = Path(video_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS
    
    def extract_audio_from_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        audio_format: str = 'mp3',
        bitrate: str = '128k',
        sample_rate: int = 44100,
        channels: int = 2,
        overwrite: bool = True,
        verbose: bool = False
    ) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 输入视频文件路径
            output_path: 输出音频文件路径（默认为视频同目录，扩展名为 .mp3）
            audio_format: 音频格式（mp3, wav, aac, ogg, flac 等）
            bitrate: 音频比特率（如 '128k', '192k', '256k'）
            sample_rate: 采样率（如 44100, 48000）
            channels: 声道数（1=单声道, 2=立体声）
            overwrite: 是否覆盖已存在的文件
            verbose: 是否显示详细信息
            
        Returns:
            输出音频文件的路径
            
        Raises:
            FileNotFoundError: 视频文件不存在
            ValueError: 不支持的视频格式
            RuntimeError: 音频提取失败
        """
        video_path = Path(video_path).resolve()
        
        # 检查视频文件是否存在
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 检查格式是否支持
        if not self.is_supported_format(video_path):
            ext = video_path.suffix
            raise ValueError(
                f"不支持的视频格式: {ext}\n"
                f"支持的格式: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        # 确定输出路径
        if output_path is None:
            output_path = video_path.with_suffix(f'.{audio_format}')
        else:
            output_path = Path(output_path)
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查输出文件是否存在
        if output_path.exists() and not overwrite:
            if verbose:
                print(f"音频文件已存在: {output_path}")
            return str(output_path)
        
        if verbose:
            print(f"正在从视频提取音频...")
            print(f"  输入: {video_path.name}")
            print(f"  输出: {output_path.name}")
            print(f"  格式: {audio_format.upper()} | 比特率: {bitrate} | 采样率: {sample_rate}Hz | 声道: {channels}")
        
        # 构建 ffmpeg 命令
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # 不处理视频流
            '-acodec', self._get_audio_codec(audio_format),
            '-ab', bitrate,
            '-ar', str(sample_rate),
            '-ac', str(channels),
        ]
        
        if overwrite:
            cmd.append('-y')  # 覆盖已存在的文件
        
        cmd.append(str(output_path))
        
        try:
            # 执行提取
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                raise RuntimeError(f"音频提取失败:\n{error_msg}")
            
            if verbose:
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ 音频提取成功! 大小: {file_size:.2f} MB")
            
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("音频提取超时（超过5分钟）")
        except Exception as e:
            raise RuntimeError(f"音频提取过程中出错: {str(e)}")
    
    def _get_audio_codec(self, audio_format: str) -> str:
        """
        根据音频格式获取对应的编码器
        
        Args:
            audio_format: 音频格式
            
        Returns:
            ffmpeg 编码器名称
        """
        codec_map = {
            'mp3': 'libmp3lame',
            'aac': 'aac',
            'wav': 'pcm_s16le',
            'ogg': 'libvorbis',
            'flac': 'flac',
            'opus': 'libopus',
            'm4a': 'aac',
        }
        return codec_map.get(audio_format.lower(), 'libmp3lame')
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        获取视频信息（时长、编码等）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            else:
                return {}
        except Exception:
            return {}


# 便捷函数
def extract_audio_from_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> str:
    """
    从视频中提取音频的便捷函数
    
    Args:
        video_path: 输入视频文件路径
        output_path: 输出音频文件路径
        **kwargs: 其他参数传递给 AudioExtractor.extract_audio_from_video
        
    Returns:
        输出音频文件的路径
    """
    extractor = AudioExtractor()
    return extractor.extract_audio_from_video(video_path, output_path, **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python audio_extractor.py <视频文件路径> [输出音频路径]")
        print("\n示例:")
        print("  python audio_extractor.py video.mp4")
        print("  python audio_extractor.py video.mp4 output.mp3")
        print("\n支持的格式:")
        print(f"  {', '.join(sorted(AudioExtractor.SUPPORTED_FORMATS))}")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result_path = extract_audio_from_video(video_file, output_file)
        print(f"\n音频文件已保存到: {result_path}")
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)
