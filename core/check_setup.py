#!/usr/bin/env python3
"""
项目测试脚本
验证所有组件是否正确安装和工作
"""

import sys
from pathlib import Path

def test_imports():
    """测试所有必要的导入"""
    print("测试依赖包导入...")
    
    required_modules = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'transformers': 'Transformers',
        'ffmpeg': 'FFmpeg-Python',
        'langchain': 'LangChain'
    }
    
    missing = []
    
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - 未安装")
            missing.append(module)
    
    return missing

def test_ffmpeg():
    """测试 ffmpeg 是否已安装"""
    print("\n测试 FFmpeg 安装...")
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ FFmpeg 已安装")
            return True
        else:
            print("  ✗ FFmpeg 未正确安装")
            return False
    except Exception as e:
        print(f"  ✗ FFmpeg 错误: {e}")
        return False

def test_project_structure():
    """测试项目目录结构"""
    print("\n测试项目结构...")
    
    required_files = {
        'python/audio_extractor.py': '音频提取模块',
        'python/whisper_parser.py': 'Whisper 转录模块',
        'python/video_extractor.py': '视频转文字主程序',
        'python/frame_analyzer.py': '帧分析模块',
        'python/vision_content.py': '视觉内容分析模块',
        'data': '数据目录',
        'models': '模型缓存目录'
    }
    
    all_exist = True
    
    for path, description in required_files.items():
        full_path = Path(path)
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {description}")
        if not exists:
            all_exist = False
    
    return all_exist

def print_next_steps():
    """打印下一步指步"""
    print("\n" + "="*70)
    print("快速开始指南")
    print("="*70 + "\n")
    
    print("""
【安装依赖】
  pip install -r requirements.txt

【下载 Whisper 模型】
  python python/download_model.py --model turbo

【完整工作流测试】
  python python/video_extractor.py data/sample.mp4

【单独测试组件】
  
  1. 音频提取:
     python python/audio_extractor.py data/sample.mp4
  
  2. 语音转文字:
     python python/whisper_parser.py data/sample.mp3
  
  3. 关键帧提取:
     python python/frame_analyzer.py data/sample.mp4

【VLM 模型对比】
  python python/vision_content.py --help
""")

def main():
    print("="*70)
    print("项目环境检测")
    print("="*70 + "\n")
    
    # 测试导入
    missing = test_imports()
    
    # 测试 ffmpeg
    ffmpeg_ok = test_ffmpeg()
    
    # 测试项目结构
    structure_ok = test_project_structure()
    
    # 汇总结果
    print("\n" + "="*70)
    print("检测结果汇总")
    print("="*70 + "\n")
    
    if missing:
        print(f"❌ 缺少 {len(missing)} 个依赖包:\n   {', '.join(missing)}")
        print("\n请运行: pip install -r requirements.txt")
    else:
        print("✓ 所有依赖包已安装")
    
    if ffmpeg_ok:
        print("✓ FFmpeg 已正确安装")
    else:
        print("✗ FFmpeg 未找到，请先安装")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt-get install ffmpeg")
    
    if structure_ok:
        print("✓ 项目结构正确")
    else:
        print("✗ 项目结构不完整")
    
    # 打印后续步骤
    if not missing and ffmpeg_ok and structure_ok:
        print_next_steps()
        return 0
    else:
        print("\n请解决上述问题后重试")
        return 1

if __name__ == "__main__":
    sys.exit(main())
