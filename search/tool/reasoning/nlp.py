import re
from typing import List, Optional

def extract_between(text: str, start_marker: str, end_marker: str) -> List[str]:
    """
    提取起始和结束标记之间的内容
    
    该函数使用正则表达式从文本中提取位于指定起始标记和结束标记之间的所有内容片段，
    是Graph-RAG系统中处理结构化文本的基础工具函数。它能够高效地从复杂文本中提取
    特定格式的信息，如从LLM回复中提取搜索查询、从文档中提取特定段落等。
    
    参数:
        text: 要搜索的文本，原始输入字符串
        start_marker: 起始标记，标识内容开始位置的字符串
        end_marker: 结束标记，标识内容结束位置的字符串
        
    返回:
        List[str]: 提取的内容字符串列表，包含所有匹配的中间内容
    
    实现思路：
    1. 构建正则表达式模式，包含转义后的起始标记和结束标记
    2. 使用非贪婪匹配(.*?)捕获标记之间的所有内容
    3. 应用DOTALL标志，使点号可以匹配换行符
    4. 使用findall函数提取所有匹配的内容
    
    技术特点：
    - 使用正则表达式实现高效的文本模式匹配
    - 转义特殊字符，确保标记可以包含正则表达式特殊字符
    - 非贪婪匹配，确保每个提取片段尽可能小
    - 支持跨行内容提取，适应各种文本格式
    - 返回所有匹配结果，支持多段内容提取
    
    业务意义：
    - 从LLM回复中提取搜索查询和思考内容
    - 从文档中提取特定格式的信息块
    - 处理结构化文本的标准化解析
    - 为后续推理和分析提供精确的信息提取
    - 支持复杂模板中的内容提取和处理
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    return re.findall(pattern, text, flags=re.DOTALL)

def extract_from_templates(text: str, templates: List[str], regex: bool = False) -> List[str]:
    """
    基于带占位符的模板提取内容
    
    该函数提供了一种灵活的文本提取方法，通过使用带占位符的模板从文本中提取结构化信息。
    它支持两种工作模式：简单占位符模式和正则表达式模式，能够适应不同复杂度的文本提取需求。
    这是Graph-RAG系统中处理LLM输出格式和结构化文本的重要工具函数。
    
    参数:
        text: 要搜索的文本，原始输入字符串
        templates: 带{}占位符的模板字符串列表，或正则表达式模式列表
        regex: 是否将模板作为正则表达式处理，默认为False（使用简单占位符模式）
        
    返回:
        List[str]: 提取的内容字符串列表，包含所有模板匹配的提取结果
    
    实现思路：
    1. 初始化空的结果列表
    2. 遍历每个输入模板
    3. 如果设置为正则表达式模式，则直接使用模板作为正则表达式进行匹配
    4. 如果是简单占位符模式，则将模板中的{}替换为正则表达式捕获组(.*?)
    5. 对所有匹配结果进行提取并添加到结果列表
    6. 返回完整的提取结果列表
    
    技术特点：
    - 双重模式支持，兼顾易用性和灵活性
    - 智能转义处理，确保特殊字符不会干扰正则表达式匹配
    - 批量处理多模板，提高提取效率
    - 支持复杂的文本结构识别和内容提取
    - 可扩展的设计，适应各种文本处理需求
    
    业务意义：
    - 从LLM输出中提取结构化信息，如搜索查询、思考过程等
    - 解析格式化的提示模板和响应结构
    - 提取特定句式或模式的文本内容
    - 支持系统组件间的标准化文本交换
    - 为复杂推理过程提供文本解析支持
    """
    results = []
    
    for template in templates:
        if regex:
            # 直接使用模板作为正则表达式
            matches = re.findall(template, text, re.DOTALL)
            results.extend(matches)
        else:
            # 将模板转换为正则表达式（通过转义和替换占位符）
            pattern = template.replace("{}", "(.*?)")
            pattern = re.escape(pattern).replace("\\(\\*\\*\\?\\)", "(.*?)")
            matches = re.findall(pattern, text, re.DOTALL)
            results.extend(matches)
    
    return results

def extract_sentences(text: str, max_sentences: Optional[int] = None) -> List[str]:
    """
    从文本中提取句子
    
    该函数负责将连续文本分割成单独的句子，是Graph-RAG系统中自然语言处理的基础功能之一。
    它使用基于标点符号的简单句子分割算法，可以高效地处理大多数文本格式，并支持限制最大
    提取句子数量，便于控制后续处理的数据量。
    
    参数:
        text: 要提取句子的文本，原始输入字符串
        max_sentences: 最大提取句子数，可选参数，用于限制返回结果数量
        
    返回:
        List[str]: 句子列表，每个元素是一个独立的句子字符串
    
    实现思路：
    1. 检查输入文本是否为空，为空则直接返回空列表
    2. 定义句子结束模式：在句号、问号或感叹号后接空格，且下一个字符是大写字母
    3. 使用正则表达式分割文本成句子列表
    4. 清理句子，移除首尾空白字符，并过滤空字符串
    5. 如果指定了最大句子数，则截取列表的前N个元素
    6. 返回处理后的句子列表
    
    技术特点：
    - 基于正则表达式的高效文本分割
    - 简单有效的句子边界检测算法
    - 智能清理和过滤，确保返回高质量的句子
    - 灵活的结果数量控制，支持处理长文本
    - 良好的空值处理，增强系统鲁棒性
    
    业务意义：
    - 为文本分析提供基础的句子级处理
    - 支持关键语句提取和摘要生成
    - 便于对文本进行片段式处理和分析
    - 为引用生成系统提供句子级的文本单元
    - 支持基于句子的相似度计算和匹配
    """
    if not text:
        return []
    
    # 简单的句子分割（可以使用NLP库进行改进）
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)
    
    # 移除空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if max_sentences:
        return sentences[:max_sentences]
    return sentences