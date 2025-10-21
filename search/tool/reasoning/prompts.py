from typing import List, Dict, Any
from collections import defaultdict
import logging

def num_tokens_from_string(text: str) -> int:
    """
    估算文本字符串中的token数量
    
    该函数负责估算给定文本字符串在LLM处理时占用的token数量，是Graph-RAG系统中管理
    上下文窗口和控制API调用成本的重要工具。它优先使用模型接口提供的准确计数方法，
    同时提供备用的简单估算方法，确保系统在各种环境下都能正常工作。
    
    参数:
        text: 文本字符串，需要估算token数量的输入文本
        
    返回:
        int: 估计的token数，用于上下文窗口管理和成本控制
    
    实现思路：
    1. 尝试导入和使用模型接口中的count_tokens函数获取准确计数
    2. 如果导入失败或函数调用异常，则使用简单的备用方法：文本长度除以4
    3. 返回估算的token数量
    
    技术特点：
    - 优先使用模型专用的token计数方法，保证准确性
    - 实现优雅的降级机制，确保在各种环境下都能正常运行
    - 简单高效的备用估算方法，适用于大多数文本
    - 异常处理机制，增强系统稳定性
    - 轻量级实现，不增加系统负担
    
    业务意义：
    - 帮助控制API调用的token使用量，优化成本
    - 确保提示内容不会超出模型上下文窗口限制
    - 为大型知识库内容的分段处理提供依据
    - 支持智能提示压缩和优化
    - 为系统资源分配提供参考
    """
    try:
        from model.get_models import count_tokens
        return count_tokens(text)
    except:
        # 简单备用
        return len(text) // 4

def kb_prompt(kbinfos: Dict[str, List[Dict[str, Any]]], max_tokens: int = 4096) -> List[str]:
    """
    将知识库信息格式化为结构化提示
    
    该函数负责将从知识库检索到的信息组织和格式化，转换为适合LLM处理的结构化提示。
    它能够智能地按文档分组信息，管理token数量，并确保提示的结构化和可读性，是
    Graph-RAG系统中知识呈现和LLM交互的关键组件。
    
    参数:
        kbinfos: 包含chunks和文档聚合信息的字典，通常来自知识库检索结果
        max_tokens: 结果提示的最大token数限制，默认为4096，用于控制上下文窗口大小
        
    返回:
        List[str]: 格式化的信息块列表，每个元素包含一个文档的相关信息，
                  或无结果时的提示信息
    
    实现思路：
    1. 从知识库检索结果中提取内容信息
    2. 进行token数量估算和限制，确保不超过最大token限制
    3. 获取并整理文档级别的聚合信息
    4. 使用defaultdict按文档将内容块分组
    5. 为每个文档构建包含元数据和相关片段的格式化字符串
    6. 返回完整的格式化知识块列表，如果为空则返回无结果提示
    
    技术特点：
    - 智能token管理，避免超出模型上下文限制
    - 文档级分组，提高信息的组织性和相关性
    - 丰富的元数据呈现，增强信息的可追溯性
    - 优雅的异常处理和降级策略
    - 结构化的信息格式，优化LLM的理解和处理
    
    业务意义：
    - 将原始知识库检索结果转换为LLM友好的格式
    - 确保知识呈现的结构化和可读性
    - 优化token使用，控制成本并避免上下文溢出
    - 提供文档级别的信息组织，增强答案的可解释性
    - 支持有效的知识整合和推理过程
    """
    # 从chunks中提取content_with_weight
    knowledges = []
    for ck in kbinfos.get("chunks", []):
        content = ck.get("content_with_weight", ck.get("text", ""))
        if content:
            knowledges.append(content)
    
    # 限制总token数
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"未将所有检索结果放入提示: {i+1}/{len(knowledges)}")
            break
    
    # 获取文档信息
    doc_aggs = kbinfos.get("doc_aggs", [])
    docs = {d.get("doc_id", ""): d for d in doc_aggs}
    
    # 按文档分组chunks
    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": {}})
    for i, ck in enumerate(kbinfos.get("chunks", [])[:chunks_num]):
        # 获取文档名称或ID
        doc_id = ck.get("doc_id", ck.get("chunk_id", "unknown").split("_")[0] if "_" in ck.get("chunk_id", "") else "unknown")
        doc_name = doc_id
        
        # 如果有URL则添加
        url_prefix = f"URL: {ck['url']}\n" if "url" in ck else ""
        
        # 获取内容
        content = ck.get("content_with_weight", ck.get("text", ""))
        
        # 将chunk添加到文档组
        doc2chunks[doc_name]["chunks"].append(f"{url_prefix}ID: {i}\n{content}")
        
        # 如果有元数据则添加
        if doc_id in docs:
            doc2chunks[doc_name]["meta"] = {
                "title": docs[doc_id].get("title", doc_id),
                "type": docs[doc_id].get("type", "unknown")
            }
    
    # 格式化最终知识块
    formatted_knowledges = []
    for doc_name, cks_meta in doc2chunks.items():
        txt = f"\nDocument: {doc_name} \n"
        
        # 添加元数据
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
            
        txt += "Relevant fragments as following:\n"
        
        # 添加chunk内容
        for chunk in cks_meta["chunks"]:
            txt += f"{chunk}\n"
            
        formatted_knowledges.append(txt)
    
    # 如果没有找到chunks
    if not formatted_knowledges:
        return ["在知识库中未找到相关信息。"]
        
    return formatted_knowledges