import os
import json
from typing import List, Dict, Any, Optional, Union

"""
数据处理工具模块

此模块提供了GraphRAG评估系统中常用的数据处理工具函数，主要功能包括：
- JSON数据的保存和加载
- 从不同格式的数据结构中提取问题和答案
- 支持灵活的数据格式处理，适应不同的输入数据源

这些工具函数为评估系统提供了统一的数据处理接口，确保数据的正确读取和格式化。
"""

def save_json(data: Any, file_path: str, ensure_ascii: bool = False, indent: int = 2):
    """
    保存数据到JSON文件
    
    此函数用于将Python数据结构（如字典、列表等）保存为JSON格式文件。
    它会自动创建不存在的目录，并使用UTF-8编码确保中文字符正确保存。
    
    Args:
        data: 要保存的数据，可以是任何可JSON序列化的Python对象
        file_path: 文件保存路径
        ensure_ascii: 是否确保ASCII编码，默认为False允许非ASCII字符（如中文）
        indent: 缩进空格数，默认为2，使JSON文件更易读
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)

def load_json(file_path: str) -> Any:
    """
    从JSON文件加载数据
    
    此函数用于读取JSON文件并将其解析为相应的Python数据结构。
    使用UTF-8编码确保正确读取包含中文等非ASCII字符的数据。
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        Any: 解析后的Python对象（通常是字典或列表）
        
    Raises:
        FileNotFoundError: 当指定文件不存在时
        json.JSONDecodeError: 当文件内容不是有效的JSON格式时
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_questions_from_data(data: Union[List, Dict], field_name: str = "question") -> List[str]:
    """
    从数据中提取问题列表
    
    此函数设计用于处理不同格式的输入数据，自动识别并提取问题文本。
    支持两种主要数据结构：列表和字典，并提供了智能字段检测功能，
    当指定字段不存在时，会尝试查找其他可能的问题字段名称。
    
    Args:
        data: 数据源，可以是列表（包含多个问题项）或字典（单个问题）
        field_name: 问题字段名称，默认为"question"
        
    Returns:
        List[str]: 提取的问题字符串列表
    
    处理逻辑：
    1. 如果输入是列表，遍历每个元素
       - 如果元素是字典且包含指定字段，则提取该字段值
       - 如果元素是字符串，则直接添加到问题列表
    2. 如果输入是字典
       - 如果包含指定字段，则提取该字段值
       - 如果不包含，则尝试寻找其他可能的问题字段名称
    """
    questions = []
    
    # 如果是列表
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and field_name in item:
                questions.append(item[field_name])
            elif isinstance(item, str):
                questions.append(item)
    # 如果是字典
    elif isinstance(data, dict):
        if field_name in data:
            questions.append(data[field_name])
        else:
            # 尝试寻找可能的问题字段
            possible_fields = ["question", "q", "query", "text", "content"]
            for field in possible_fields:
                if field in data:
                    questions.append(data[field])
                    break
    
    return questions

def extract_answers_from_data(data: Union[List, Dict], field_name: str = "answer") -> List[str]:
    """
    从数据中提取答案列表
    
    此函数设计用于处理不同格式的输入数据，自动识别并提取答案文本。
    与问题提取函数类似，支持列表和字典两种主要数据结构，并提供智能字段检测功能。
    当指定字段不存在时，会尝试查找其他可能的答案字段名称。
    
    Args:
        data: 数据源，可以是列表（包含多个答案项）或字典（单个答案）
        field_name: 答案字段名称，默认为"answer"
        
    Returns:
        List[str]: 提取的答案字符串列表
    
    处理逻辑：
    1. 如果输入是列表，遍历每个元素
       - 如果元素是字典且包含指定字段，则提取该字段值
       - 如果元素是字符串，则直接添加到答案列表
    2. 如果输入是字典
       - 如果包含指定字段，则提取该字段值
       - 如果不包含，则尝试寻找其他可能的答案字段名称
    """
    answers = []
    
    # 如果是列表
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and field_name in item:
                answers.append(item[field_name])
            elif isinstance(item, str):
                answers.append(item)
    # 如果是字典
    elif isinstance(data, dict):
        if field_name in data:
            answers.append(data[field_name])
        else:
            # 尝试寻找可能的答案字段
            possible_fields = ["answer", "a", "response", "text", "content"]
            for field in possible_fields:
                if field in data:
                    answers.append(data[field])
                    break
    
    return answers