#!/usr/bin/env python3
"""
视频帧分析器 - 提取关键帧并生成文字描述
用于分析纯视频内容（如主播讲解、K线图等场景变化较少的视频）

核心思路:
1. 通过帧间像素差异检测关键帧
2. 使用视觉语言模型 (VLM) 将关键帧转为文字描述
3. 结合音频转录，生成完整的视频理解

推荐的开源模型:
- LLaVA (Large Language and Vision Assistant) - Meta
- BLIP-2 (Bootstrapping Language-Image Pre-training) - Salesforce
- Qwen-VL (通义千问视觉语言模型) - 阿里巴巴
- CogVLM - 智谱AI
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import timedelta
import json


@dataclass
class KeyFrame:
    """关键帧数据类"""
    frame_number: int
    timestamp: float
    timestamp_str: str
    image: np.ndarray
    difference_score: float
    

class VideoFrameAnalyzer:
    """视频帧分析器"""
    
    def __init__(
        self,
        diff_threshold: float = 0.1,
        min_frame_interval: int = 30,
        max_frames: int = 50,
        resize_for_comparison: Tuple[int, int] = (320, 240)
    ):
        """
        初始化帧分析器
        
        Args:
            diff_threshold: 帧差异阈值 (0-1)，超过此值认为是关键帧
            min_frame_interval: 最小帧间隔（避免提取过于相似的连续帧）
            max_frames: 最多提取的关键帧数量
            resize_for_comparison: 比较时缩放的尺寸（提高速度）
        """
        self.diff_threshold = diff_threshold
        self.min_frame_interval = min_frame_interval
        self.max_frames = max_frames
        self.resize_for_comparison = resize_for_comparison
    
    def calculate_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        计算两帧之间的差异度
        
        使用多种方法综合评估:
        1. 结构相似性 (SSIM)
        2. 直方图差异
        3. 像素级 MSE
        
        Args:
            frame1: 第一帧
            frame2: 第二帧
            
        Returns:
            差异分数 (0-1)，越大表示差异越大
        """
        # 缩放到统一尺寸以加速计算
        f1 = cv2.resize(frame1, self.resize_for_comparison)
        f2 = cv2.resize(frame2, self.resize_for_comparison)
        
        # 转为灰度图
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        
        # 方法1: 计算直方图差异
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_diff = 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 方法2: 计算均方误差 (MSE)
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        mse_normalized = min(mse / 10000.0, 1.0)  # 归一化到 0-1
        
        # 综合评分
        diff_score = 0.6 * hist_diff + 0.4 * mse_normalized
        
        return diff_score
    
    def extract_key_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_images: bool = True,
        verbose: bool = True
    ) -> List[KeyFrame]:
        """
        从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录（保存关键帧图像）
            save_images: 是否保存关键帧图像
            verbose: 是否显示详细信息
            
        Returns:
            关键帧列表
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if verbose:
            print(f"视频信息:")
            print(f"  文件: {video_path.name}")
            print(f"  总帧数: {total_frames}")
            print(f"  帧率: {fps:.2f} FPS")
            print(f"  时长: {timedelta(seconds=int(duration))}")
            print(f"\n开始提取关键帧...")
        
        key_frames = []
        prev_frame = None
        frame_count = 0
        last_keyframe_number = -self.min_frame_interval
        
        # 确保第一帧被选为关键帧
        ret, first_frame = cap.read()
        if ret:
            key_frames.append(KeyFrame(
                frame_number=0,
                timestamp=0.0,
                timestamp_str=str(timedelta(seconds=0)),
                image=first_frame.copy(),
                difference_score=1.0
            ))
            prev_frame = first_frame
            last_keyframe_number = 0
            frame_count = 1
        
        # 遍历视频帧
        while cap.isOpened() and len(key_frames) < self.max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # 检查是否满足最小间隔
            if frame_count - last_keyframe_number < self.min_frame_interval:
                prev_frame = frame
                continue
            
            # 计算帧差异
            diff_score = self.calculate_frame_difference(prev_frame, frame)
            
            # 如果差异超过阈值，认为是关键帧
            if diff_score > self.diff_threshold:
                timestamp = frame_count / fps
                timestamp_str = str(timedelta(seconds=int(timestamp)))
                
                key_frames.append(KeyFrame(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    timestamp_str=timestamp_str,
                    image=frame.copy(),
                    difference_score=diff_score
                ))
                
                last_keyframe_number = frame_count
                
                if verbose:
                    print(f"  关键帧 #{len(key_frames)}: "
                          f"帧号={frame_count}, "
                          f"时间={timestamp_str}, "
                          f"差异度={diff_score:.3f}")
            
            prev_frame = frame
        
        cap.release()
        
        # 确保最后一帧也被包含（如果还没达到最大数量）
        if len(key_frames) < self.max_frames and frame_count > last_keyframe_number + self.min_frame_interval:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
            if ret:
                timestamp = (total_frames - 1) / fps
                key_frames.append(KeyFrame(
                    frame_number=total_frames - 1,
                    timestamp=timestamp,
                    timestamp_str=str(timedelta(seconds=int(timestamp))),
                    image=last_frame,
                    difference_score=0.0
                ))
            cap.release()
        
        # 保存关键帧图像
        if save_images and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, kf in enumerate(key_frames):
                img_filename = f"{video_path.stem}_keyframe_{i+1:03d}_t{int(kf.timestamp)}s.jpg"
                img_path = output_dir / img_filename
                cv2.imwrite(str(img_path), kf.image)
            
            if verbose:
                print(f"\n✓ 已保存 {len(key_frames)} 个关键帧到: {output_dir}")
        
        # 保存元数据
        if output_dir:
            metadata = {
                "video_file": video_path.name,
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration,
                "key_frames_count": len(key_frames),
                "diff_threshold": self.diff_threshold,
                "key_frames": [
                    {
                        "index": i + 1,
                        "frame_number": kf.frame_number,
                        "timestamp": kf.timestamp,
                        "timestamp_str": kf.timestamp_str,
                        "difference_score": kf.difference_score
                    }
                    for i, kf in enumerate(key_frames)
                ]
            }
            
            metadata_path = output_dir / f"{video_path.stem}_keyframes_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ 元数据已保存: {metadata_path}")
        
        return key_frames


def main():
    """命令行测试入口"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python frame_analyzer.py <视频文件> [输出目录]")
        print("\n示例:")
        print("  python frame_analyzer.py video.mp4")
        print("  python frame_analyzer.py video.mp4 data/keyframes")
        sys.exit(1)
    
    video_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/keyframes"
    
    analyzer = VideoFrameAnalyzer(
        diff_threshold=0.1,      # 差异阈值
        min_frame_interval=30,   # 最小间隔1秒（假设30fps）
        max_frames=50            # 最多50个关键帧
    )
    
    try:
        key_frames = analyzer.extract_key_frames(
            video_path=video_file,
            output_dir=output_dir,
            save_images=True,
            verbose=True
        )
        
        print(f"\n成功提取 {len(key_frames)} 个关键帧")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
