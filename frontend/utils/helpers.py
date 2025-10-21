"""
Graph-RAG Agent 前端辅助工具模块

此模块提供前端界面所需的各种辅助功能，主要包括：

1. 内容处理工具
   - 从回答文本中提取源文档ID
   - 格式化和美化显示源文档内容
   - 处理AI的思考过程与最终回答

2. 用户界面增强
   - 优化文本展示格式
   - 支持富文本渲染

模块中的函数主要被Streamlit界面组件调用，用于处理和展示从后端获取的数据。
"""

import re
from typing import List
import streamlit as st

def extract_source_ids(answer: str) -> List[str]:
    """
    从回答中提取引用的源文档ID
    
    功能：
    - 使用正则表达式从AI回答中识别并提取引用的文档片段ID
    - 支持多种格式的ID提取（带引号和不带引号的）
    - 去重处理，确保返回的ID列表中没有重复项
    
    参数：
        answer: str - AI生成的回答文本，包含对源文档的引用
    
    返回值：
        List[str] - 提取的唯一源文档ID列表
    
    实现思路：
    - 首先查找包含"Chunks': [...]"格式的文本段
    - 然后分别处理带引号和不带引号的ID格式
    - 使用set数据结构自动去重
    - 最终转换为列表返回
    """
    source_ids = []  # 存储提取的源ID
    
    # 提取Chunks IDs - 使用正则表达式匹配格式
    chunks_pattern = r"Chunks':\s*\[([^\]]*)\]"
    matches = re.findall(chunks_pattern, answer)
    
    # 处理找到的匹配项
    if matches:
        for match in matches:
            # 处理带引号的ID格式
            quoted_ids = re.findall(r"'([^']*)'", match)
            if quoted_ids:
                source_ids.extend(quoted_ids)
            else:
                # 处理不带引号的ID格式
                ids = [id.strip() for id in match.split(',') if id.strip()]
                source_ids.extend(ids)
    
    # 去重并返回唯一ID列表
    return list(set(source_ids))

def display_source_content(content: str):
    """
    美化显示源文档内容
    
    功能：
    - 应用CSS样式增强源文档的可读性
    - 支持长文本的水平和垂直滚动
    - 保持原始文本的格式（如换行）
    - 使用等宽字体提高代码和结构化文本的可读性
    
    参数：
        content: str - 要显示的源文档内容
    
    实现思路：
    - 首先注入CSS样式定义，创建自定义类
    - 将文本中的换行符转换为HTML的<br>标签
    - 使用Streamlit的HTML渲染功能，将格式化内容显示在应用中
    - 设置最大高度和溢出滚动，确保长文本不会破坏布局
    """
    # 注入CSS样式，定义.source-content类的样式
    st.markdown("""
    <style>
    .source-content {
        white-space: pre-wrap;  # 保留空白和换行
        overflow-x: auto;      # 水平滚动条
        font-family: monospace;  # 等宽字体
        line-height: 1.6;      # 行高
        background-color: #f5f5f5;  # 浅灰色背景
        border-radius: 5px;    # 圆角边框
        padding: 15px;         # 内边距
        max-height: 600px;     # 最大高度
        overflow-y: auto;      # 垂直滚动条
        border: 1px solid #e1e4e8;  # 边框
        color: #24292e;        # 文本颜色
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 将换行符转换为HTML换行，确保格式正确显示
    formatted_content = content.replace("\n", "<br>")
    # 使用HTML渲染内容，应用自定义样式
    st.markdown(f'<div class="source-content">{formatted_content}</div>', unsafe_allow_html=True)


def process_thinking_content(content: str, show_thinking: bool = False):
    """
    处理带有思考过程的内容
    
    功能：
    - 从AI生成的内容中提取思考过程和最终答案
    - 将思考过程格式化为Markdown引用样式
    - 支持选择性显示或隐藏思考过程
    - 保留原始内容的完整性
    
    参数：
        content: str - AI生成的原始内容，可能包含思考过程
        show_thinking: bool - 是否显示思考过程（保留参数以便将来扩展）
    
    返回值：
        dict - 包含处理后内容的字典，具有以下字段：
            - processed: 处理后的最终答案
            - has_thinking: 是否包含思考过程
            - thinking: 格式化后的思考过程（如果有）
            - original: 原始内容（如果包含思考过程）
    
    实现思路：
    - 首先检查输入是否为字符串类型
    - 使用正则表达式查找并提取"</think>"标签之间的思考内容
    - 移除思考过程，提取纯答案部分
    - 将思考过程格式化为Markdown引用样式（每行前添加>符号）
    - 返回包含所有相关信息的字典
    """
    # 类型检查，确保输入为字符串
    if not isinstance(content, str):
        return {"processed": content, "has_thinking": False}
        
    # 检查是否包含思考过程标记
    if "</think>" in content and "</think>" in content:
        # 使用正则表达式提取思考过程，re.DOTALL使.匹配换行符
        think_match = re.search(r'</think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            # 提取并清理思考过程
            thinking_process = think_match.group(1).strip()
            # 移除思考过程部分，只保留答案
            answer_only = content.replace(f"</think>{thinking_process}</think>", "").strip()
            
            # 将思考过程格式化为Markdown引用格式（每行前添加>）
            thinking_lines = thinking_process.split('\n')
            quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
            
            # 返回处理后的内容信息
            return {
                "processed": answer_only,    # 纯答案内容
                "has_thinking": True,       # 标记包含思考过程
                "thinking": quoted_thinking,  # 格式化后的思考过程
                "original": content         # 原始完整内容
            }
    
    # 如果没有思考过程或提取失败，返回原内容
    return {"processed": content, "has_thinking": False}