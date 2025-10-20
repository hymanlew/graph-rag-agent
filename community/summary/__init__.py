from langchain_community.graphs import Neo4jGraph
from typing import Union
from .base import BaseSummarizer
from .leiden import LeidenSummarizer
from .sllpa import SLLPASummarizer

"""
社区摘要模块

本模块负责为检测到的社区生成摘要，从社区中提取关键信息和主题。
社区摘要是RAG系统中的重要环节，可以帮助用户快速理解社区的核心内容，
并为图查询提供上下文感知能力。

模块结构：
- base: 定义摘要生成器的抽象基类，规范接口
- leiden: 为Leiden算法检测的社区实现摘要生成
- sllpa: 为SLLPA算法检测的社区实现摘要生成

主要功能：
1. 提供统一的社区摘要生成接口
2. 支持针对不同社区检测算法的特定摘要策略
3. 通过工厂模式简化摘要生成器的创建和使用
4. 从社区节点中提取关键实体和关系

在整个RAG系统中的作用：
- 生成社区级别的主题和内容概述
- 提供更丰富的上下文信息给检索和生成模块
- 帮助理解图的高层结构和语义组织
- 支持基于社区的问答和信息聚合
"""

class CommunitySummarizerFactory:
    """社区摘要生成器工厂类
    
    实现工厂设计模式，负责创建和配置不同类型的社区摘要生成器。
    这个类隐藏了具体摘要生成器实现的细节，提供一个简单的接口来根据算法类型创建摘要生成器。
    
    支持的摘要生成器：
    - leiden: 为Leiden算法检测的社区生成摘要
    - sllpa: 为SLLPA算法检测的社区生成摘要
    
    工厂模式的优势：
    - 解耦客户端代码和具体实现
    - 简化摘要生成器的创建流程
    - 方便扩展新的摘要生成策略
    - 统一管理和配置不同类型的摘要生成器
    """
    
    # 摘要生成器映射字典，将算法名称映射到对应的摘要生成器类
    SUMMARIZERS = {
        'leiden': LeidenSummarizer,
        'sllpa': SLLPASummarizer
    }
    
    @classmethod
    def create_summarizer(cls, 
                         algorithm: str,
                         graph: Neo4jGraph) -> BaseSummarizer:
        """
        创建社区摘要生成器实例
        
        参数:
            algorithm: 算法类型 ('leiden' 或 'sllpa')，指定使用哪种摘要生成策略
            graph: Neo4j图实例，提供对图数据库的访问能力
            
        返回:
            BaseSummarizer: 摘要生成器实例，用于生成社区摘要
            
        异常:
            ValueError: 如果指定了不支持的算法类型
        """
        # 转换为小写，确保不区分大小写
        algorithm = algorithm.lower()
        
        # 验证算法类型是否支持
        if algorithm not in cls.SUMMARIZERS:
            raise ValueError(f"不支持的算法类型: {algorithm}")
            
        # 获取对应的摘要生成器类并创建实例
        summarizer_class = cls.SUMMARIZERS[algorithm]
        return summarizer_class(graph)

# 导出的公共接口，定义模块的公开API
__all__ = ['CommunitySummarizerFactory', 'BaseSummarizer', 
           'LeidenSummarizer', 'SLLPASummarizer']