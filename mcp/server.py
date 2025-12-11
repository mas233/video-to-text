"""
MCP (Model Context Protocol) 服务器
为视频处理工具提供标准化的 MCP 接口
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sys
import os
import json
from pathlib import Path
from typing import Optional

# 添加 core 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from audio_extractor import extract_audio_from_video
from whisper_parser import transcribe_audio
from video_extractor import VideoToTextConverter


# 创建 MCP 服务器
app = Server("video-processing-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的工具"""
    return [
        Tool(
            name="extract_audio",
            description="从视频文件中提取音频。支持 MP4, AVI, MKV, MOV, WMV, FLV, WEBM 等格式。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "视频文件的完整路径"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "输出音频文件路径（可选，默认为视频同名.mp3）"
                    },
                    "audio_format": {
                        "type": "string",
                        "description": "音频格式（mp3, wav, aac 等）",
                        "default": "mp3"
                    },
                    "bitrate": {
                        "type": "string",
                        "description": "音频比特率（如 128k, 192k）",
                        "default": "128k"
                    }
                },
                "required": ["video_path"]
            }
        ),
        Tool(
            name="transcribe_audio",
            description="将音频文件转录为文字。支持 99+ 种语言，包括中文、英文、日语、韩语等。输出包含纯文本和 JSON 格式。",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "音频文件的完整路径"
                    },
                    "language": {
                        "type": "string",
                        "description": "语言代码（zh=中文, en=英文, ja=日语, ko=韩语等）",
                        "default": "zh"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "输出目录（可选，默认为 data/transcriptions/）"
                    }
                },
                "required": ["audio_path"]
            }
        ),
        Tool(
            name="video_to_text",
            description="完整的视频转文字流程：自动提取音频并转录为文字。这是一个端到端的工具，会处理整个流程。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "视频文件的完整路径"
                    },
                    "language": {
                        "type": "string",
                        "description": "语言代码（zh=中文, en=英文, ja=日语, ko=韩语等）",
                        "default": "zh"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "输出目录（可选，默认为 data/transcriptions/）"
                    },
                    "keep_audio": {
                        "type": "boolean",
                        "description": "是否保留提取的音频文件",
                        "default": False
                    }
                },
                "required": ["video_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """调用指定的工具"""
    
    try:
        if name == "extract_audio":
            # 提取音频
            video_path = arguments["video_path"]
            output_path = arguments.get("output_path")
            audio_format = arguments.get("audio_format", "mp3")
            bitrate = arguments.get("bitrate", "128k")
            
            # 验证文件存在
            if not Path(video_path).exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"视频文件不存在: {video_path}",
                        "error_type": "FileNotFoundError"
                    }, ensure_ascii=False, indent=2)
                )]
            
            # 执行提取
            result_path = extract_audio_from_video(
                video_path=video_path,
                output_path=output_path,
                audio_format=audio_format,
                bitrate=bitrate,
                verbose=True
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "audio_path": result_path,
                    "message": f"音频已成功提取到: {result_path}"
                }, ensure_ascii=False, indent=2)
            )]
        
        elif name == "transcribe_audio":
            # 转录音频
            audio_path = arguments["audio_path"]
            language = arguments.get("language", "zh")
            output_dir = arguments.get("output_dir")
            
            # 验证文件存在
            if not Path(audio_path).exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"音频文件不存在: {audio_path}",
                        "error_type": "FileNotFoundError"
                    }, ensure_ascii=False, indent=2)
                )]
            
            # 执行转录
            result = transcribe_audio(
                audio_path=audio_path,
                language=language,
                output_dir=output_dir,
                save_txt=True,
                save_json=True,
                verbose=True
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "transcription": result["text"],
                    "language": result["language"],
                    "output_files": result.get("output_files", {}),
                    "audio_file": result["audio_file"],
                    "message": "转录成功"
                }, ensure_ascii=False, indent=2)
            )]
        
        elif name == "video_to_text":
            # 视频转文字（完整流程）
            video_path = arguments["video_path"]
            language = arguments.get("language", "zh")
            output_dir = arguments.get("output_dir")
            keep_audio = arguments.get("keep_audio", False)
            
            # 验证文件存在
            if not Path(video_path).exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"视频文件不存在: {video_path}",
                        "error_type": "FileNotFoundError"
                    }, ensure_ascii=False, indent=2)
                )]
            
            # 创建转换器并执行
            converter = VideoToTextConverter(
                output_dir=output_dir,
                keep_audio=keep_audio,
                verbose=True
            )
            
            result = converter.convert(
                video_path=video_path,
                language=language
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "transcription": result["transcription"],
                    "video_file": result["video_file"],
                    "language": result["language"],
                    "processing_time_seconds": result["processing_time_seconds"],
                    "output_files": result["output_files"],
                    "output_dir": result["output_dir"],
                    "message": "视频转文字成功"
                }, ensure_ascii=False, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"未知工具: {name}",
                    "error_type": "UnknownToolError"
                }, ensure_ascii=False, indent=2)
            )]
    
    except Exception as e:
        # 捕获所有异常并返回详细错误信息
        import traceback
        error_trace = traceback.format_exc()
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_trace,
                "arguments": arguments
            }, ensure_ascii=False, indent=2)
        )]


async def main():
    """启动 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
