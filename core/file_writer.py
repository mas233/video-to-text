"""
文件写入工具
提供保存文本文件的功能
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def save_text_file(
    content: str,
    output_path: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    保存文本内容到文件
    
    Args:
        content: 要保存的文本内容
        output_path: 输出文件路径
        encoding: 文件编码 (默认: utf-8)
        create_dirs: 是否自动创建目录 (默认: True)
        verbose: 是否打印详细信息
        
    Returns:
        包含保存信息的字典
    """
    try:
        output_file = Path(output_path)
        
        # 创建目录
        if create_dirs and not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"✓ 创建目录: {output_file.parent}")
        
        # 保存文件
        with open(output_file, 'w', encoding=encoding) as f:
            f.write(content)
        
        # 获取文件信息
        file_size = output_file.stat().st_size
        
        if verbose:
            print(f"✓ 文件已保存: {output_path}")
            print(f"  大小: {file_size} bytes ({file_size / 1024:.2f} KB)")
        
        return {
            "success": True,
            "file_path": str(output_file.absolute()),
            "size_bytes": file_size,
            "encoding": encoding,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def save_json_file(
    data: Dict[str, Any],
    output_path: str,
    indent: int = 2,
    ensure_ascii: bool = False,
    create_dirs: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    保存 JSON 数据到文件
    
    Args:
        data: 要保存的数据字典
        output_path: 输出文件路径
        indent: JSON 缩进 (默认: 2)
        ensure_ascii: 是否确保 ASCII (默认: False, 支持中文)
        create_dirs: 是否自动创建目录
        verbose: 是否打印详细信息
        
    Returns:
        包含保存信息的字典
    """
    try:
        output_file = Path(output_path)
        
        # 创建目录
        if create_dirs and not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"✓ 创建目录: {output_file.parent}")
        
        # 保存 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        # 获取文件信息
        file_size = output_file.stat().st_size
        
        if verbose:
            print(f"✓ JSON 已保存: {output_path}")
            print(f"  大小: {file_size} bytes ({file_size / 1024:.2f} KB)")
        
        return {
            "success": True,
            "file_path": str(output_file.absolute()),
            "size_bytes": file_size,
            "format": "json",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
