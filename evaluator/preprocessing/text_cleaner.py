import re

"""
文本清理模块

此模块提供了用于清理和预处理AI生成回答的工具函数，主要用于评估过程中的数据准备。

在GraphRAG评估系统中，为了准确评估回答质量，需要从AI生成的原始输出中：
1. 移除引用数据部分，专注于实际内容的评估
2. 清理深度研究Agent的思考过程，将重点放在最终答案上

这些清理操作确保了评估过程只关注核心内容，不受元数据或中间思考步骤的干扰。
"""

def clean_references(answer: str) -> str:
    """
    清理AI回答中的引用数据部分
    
    从AI生成的回答中移除引用数据部分，保留核心回答内容。这在评估回答质量时非常重要，
    因为引用数据通常包含元信息，而不是回答的主体内容。
    
    支持处理多种引用数据格式，包括不同级别的Markdown标题格式。
    清理后的文本将用于答案质量评估指标的计算，如精确匹配、F1分数等。
    
    Args:
        answer: AI生成的回答，可能包含引用数据部分
        
    Returns:
        str: 清理后的回答，仅包含核心内容，不包含引用数据
    """
    # 移除引用数据部分
    cleaned = re.sub(r'###\s*引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 如果没有引用数据部分，尝试其他格式
    if cleaned == answer:
        cleaned = re.sub(r'#### 引用数据[\s\S]*?(\{\s*[\'"]data[\'"][\s\S]*?\}\s*)', '', answer)
    
    # 移除任何尾部空行
    cleaned = cleaned.rstrip()
    
    return cleaned

def clean_thinking_process(answer: str) -> str:
    """
    清理深度研究Agent回答中的思考过程
    
    从深度研究Agent生成的回答中移除中间思考过程部分，只保留最终答案。
    这对于公平评估深度研究Agent的回答质量至关重要，因为思考过程通常包含草稿性质的内容，
    而评估应该聚焦于最终呈现给用户的答案。
    
    该函数特别适用于使用特定标记（如</think>）包裹思考过程的深度研究Agent输出格式。
    清理后还会规范化换行符，确保文本格式一致，便于后续评估处理。
    
    Args:
        answer: 深度研究Agent生成的回答，可能包含思考过程
        
    Returns:
        str: 清理后的回答，只包含最终答案部分，没有思考过程
    """
    # 移除思考过程部分
    cleaned = re.sub(r'<think>[\s\S]*?</think>\s*', '', answer)
    
    # 移除任何多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned