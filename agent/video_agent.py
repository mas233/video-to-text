"""
智能视频处理 Agent - 使用 Qwen 原生工具调用
不依赖 LangChain，直接使用 Qwen 的 <tool_call> XML 格式
"""

import os
import sys
import json
import re
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 添加 core 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from audio_extractor import extract_audio_from_video
from whisper_parser import transcribe_audio
from video_extractor import VideoToTextConverter
from file_writer import save_text_file, save_json_file


class VideoProcessingAgent:
    """使用 Qwen 原生工具调用的视频处理 Agent"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: str = "./models",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_iterations: int = 10
    ):
        """
        初始化 Agent
        
        Args:
            model_name: 模型名称
            cache_dir: 模型缓存目录
            device: 设备 (cuda/mps/cpu)
            temperature: 生成温度
            max_iterations: 最大迭代次数
        """
        print(f"正在初始化 Agent...")
        print(f"模型: {model_name}")
        
        # 检测设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        print(f"设备: {device}")
        print(f"缓存目录: {cache_dir}")
        
        # 加载模型
        self.tokenizer, self.model = self._load_model(model_name, cache_dir)
        
        # 定义工具
        self.tools = self._define_tools()
        
        # 对话历史
        self.conversation_history = []
        
        print("✓ Agent 初始化完成!\n")
    
    def _load_model(self, model_name: str, cache_dir: str):
        """加载模型和分词器"""
        print("\n加载模型...")
        
        # 设置缓存目录环境变量
        # 确保 ModelScope 和 HuggingFace 都使用项目的 models 目录
        cache_dir_abs = str(Path(cache_dir).resolve())
        os.environ['MODELSCOPE_CACHE'] = cache_dir_abs
        os.environ['HF_HOME'] = cache_dir_abs
        os.environ['TRANSFORMERS_CACHE'] = cache_dir_abs
        
        # 尝试从 ModelScope 加载
        model_path = None
        try:
            from modelscope import snapshot_download
            print("尝试从 ModelScope 加载...")
            
            # 转换模型名称: Qwen/Qwen2.5-7B-Instruct -> qwen/qwen2.5-7b-instruct
            ms_model_name = model_name.lower().replace("qwen/qwen", "qwen/qwen")
            
            model_path = snapshot_download(
                ms_model_name,
                cache_dir=cache_dir_abs
            )
            print(f"✓ 从 ModelScope 加载: {model_path}")
            
        except Exception as e:
            print(f"ModelScope 加载失败: {e}")
            print("尝试从 HuggingFace 加载...")
            model_path = model_name
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=cache_dir_abs
        )
        
        # 加载模型
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=self.device if self.device == "cuda" else None,
            cache_dir=cache_dir_abs
        )
        
        # 手动移动到设备 (MPS/CPU)
        if self.device in ["mps", "cpu"]:
            model = model.to(self.device)
        
        print(f"✓ 模型加载完成 (dtype={dtype}, device={self.device})")
        
        return tokenizer, model
    
    def _define_tools(self) -> List[Dict]:
        """定义可用工具 (Qwen 格式)"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "列出指定目录下的所有文件和子目录",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "要列出的目录路径，例如 'data' 或 'data/videos'"
                            }
                        },
                        "required": ["directory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_file_exists",
                    "description": "检查文件或目录是否存在",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "文件或目录的路径"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_audio",
                    "description": "从视频文件中提取音频",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "video_path": {
                                "type": "string",
                                "description": "视频文件路径"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "输出音频文件路径 (可选)"
                            }
                        },
                        "required": ["video_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transcribe_audio",
                    "description": "将音频转录为文字",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "音频文件路径"
                            },
                            "language": {
                                "type": "string",
                                "description": "语言代码，例如 'zh' 或 'en'"
                            }
                        },
                        "required": ["audio_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_text",
                    "description": "保存文本内容到文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要保存的文本内容"
                            },
                            "output_path": {
                                "type": "string",
                                "description": "输出文件路径，例如 'output/result.txt'"
                            }
                        },
                        "required": ["content", "output_path"]
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """执行工具调用"""
        try:
            if tool_name == "list_files":
                directory = arguments["directory"]
                path = Path(directory)
                
                if not path.exists():
                    return f"错误: 目录 '{directory}' 不存在"
                
                if not path.is_dir():
                    return f"错误: '{directory}' 不是目录"
                
                items = []
                for item in path.iterdir():
                    if item.is_dir():
                        items.append(f"{item.name}/")
                    else:
                        size = item.stat().st_size
                        items.append(f"{item.name} ({size} bytes)")
                
                return "\n".join(items) if items else "目录为空"
            
            elif tool_name == "check_file_exists":
                path = Path(arguments["path"])
                if path.exists():
                    if path.is_dir():
                        return f"'{arguments['path']}' 是一个目录"
                    else:
                        size = path.stat().st_size
                        return f"'{arguments['path']}' 存在 (大小: {size} bytes)"
                else:
                    return f"'{arguments['path']}' 不存在"
            
            elif tool_name == "extract_audio":
                video_path = arguments["video_path"]
                output_path = arguments.get("output_path")
                
                result = extract_audio_from_video(
                    video_path,
                    output_path=output_path
                )
                return f"音频已提取到: {result}"
            
            elif tool_name == "transcribe_audio":
                audio_path = arguments["audio_path"]
                language = arguments.get("language", "zh")
                
                result = transcribe_audio(audio_path, language=language)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "save_text":
                content = arguments["content"]
                output_path = arguments["output_path"]
                
                result = save_text_file(content, output_path, verbose=False)
                if result["success"]:
                    return f"文件已保存: {result['file_path']} ({result['size_bytes']} bytes)"
                else:
                    return f"保存失败: {result['error']}"
            
            else:
                return f"错误: 未知工具 '{tool_name}'"
        
        except Exception as e:
            return f"工具执行错误: {str(e)}\n{traceback.format_exc()}"
    
    def _generate(self, messages: List[Dict]) -> str:
        """使用 Qwen chat template 生成回复"""
        # 应用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码 (只解码新生成的 tokens)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """解析 <tool_call> XML 标签"""
        tool_calls = []
        
        # 匹配 <tool_call>...</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                print(f"解析工具调用失败: {e}")
                print(f"原始文本: {match}")
        
        return tool_calls
    
    def run(self, task: str, verbose: bool = True) -> str:
        """
        执行任务
        
        Args:
            task: 用户任务描述
            verbose: 是否打印详细过程
        
        Returns:
            最终回答
        """
        # 初始化对话
        messages = [
            {"role": "system", "content": "你是一个专业的视频处理助手，可以帮助用户处理视频文件、提取音频和文字内容。"},
            {"role": "user", "content": task}
        ]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"任务: {task}")
            print(f"{'='*60}\n")
        
        # 迭代执行
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"--- 迭代 {iteration + 1} ---")
            
            # 生成回复
            response = self._generate(messages)
            
            if verbose:
                print(f"模型输出:\n{response}\n")
            
            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用，返回最终答案
                if verbose:
                    print(f"{'='*60}")
                    print("✓ 任务完成")
                    print(f"{'='*60}\n")
                
                # 移除 tool_call 标签后的文本
                final_answer = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL).strip()
                return final_answer if final_answer else response
            
            # 执行工具调用
            tool_responses = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                if verbose:
                    print(f"调用工具: {tool_name}")
                    print(f"参数: {json.dumps(arguments, ensure_ascii=False)}")
                
                # 执行工具
                result = self._execute_tool(tool_name, arguments)
                tool_responses.append(result)
                
                if verbose:
                    print(f"结果:\n{result}\n")
            
            # 将工具响应添加到对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "content": "\n\n".join(tool_responses)})
        
        return "达到最大迭代次数，任务未完成"


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频处理 Agent (Qwen 原生工具调用)")
    parser.add_argument("task", nargs="?", help="要执行的任务")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名称")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="设备")
    parser.add_argument("--cache-dir", default="./models", help="模型缓存目录")
    
    args = parser.parse_args()
    
    # 初始化 Agent
    agent = VideoProcessingAgent(
        model_name=args.model,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    if args.interactive:
        print("进入交互模式（输入 'quit' 或 'exit' 退出）")
        print("-" * 60)
        
        while True:
            try:
                task = input("\n请输入任务: ").strip()
                
                if task.lower() in ["quit", "exit", "退出"]:
                    print("\n再见!")
                    break
                
                if not task:
                    continue
                
                print("\n执行中...")
                result = agent.run(task, verbose=True)
                
                print(f"\n{'='*60}")
                print("最终答案:")
                print(result)
                print(f"{'='*60}")
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                traceback.print_exc()
    
    elif args.task:
        result = agent.run(args.task, verbose=True)
        print(f"\n{'='*60}")
        print("最终答案:")
        print(result)
        print(f"{'='*60}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
