"""
搜索模块初始化文件

该模块是Graph-RAG系统的核心检索组件，提供了多种搜索策略和工具，支持在图结构中高效检索信息。
模块采用分层设计，将基础搜索逻辑与工具接口分离，实现了灵活的检索功能。

主要组件说明：
1. 基础搜索类 - 实现核心搜索算法和逻辑
2. 工具类 - 为外部调用提供标准化接口

搜索策略包括：
- 局部搜索：在图的局部区域内查找相关信息
- 全局搜索：在整个图结构中进行广泛检索
- 混合搜索：结合多种策略的优点
- 深度搜索：对特定主题进行深入探索
- 朴素搜索：简单直接的搜索方法（通常用作基线比较）

该模块与其他组件的关系：
- 依赖于向量计算工具（VectorUtils）进行相似度判断
- 为问答系统提供检索支持
- 处理文档与图结构之间的信息关联
"""

# 导出主要搜索类
# LocalSearch: 实现图的局部区域搜索，专注于相关度高的临近节点
from search.local_search import LocalSearch
# GlobalSearch: 实现图的全局搜索，可跨多个区域检索信息
from search.global_search import GlobalSearch

# 导出工具类（为外部调用提供标准化接口）
# LocalSearchTool: 封装LocalSearch功能，提供工具化接口
from search.tool.local_search_tool import LocalSearchTool
# GlobalSearchTool: 封装GlobalSearch功能，提供工具化接口
from search.tool.global_search_tool import GlobalSearchTool
# HybridSearchTool: 混合搜索工具，结合多种搜索策略
from search.tool.hybrid_tool import HybridSearchTool
# NaiveSearchTool: 简单搜索工具，通常作为基线或快速检索使用
from search.tool.naive_search_tool import NaiveSearchTool
# DeepResearchTool: 深度研究工具，专注于特定主题的深入探索
from search.tool.deep_research_tool import DeepResearchTool

# 定义模块公开接口，控制from search import *时导入的内容
# 这样可以避免导入模块内部的私有组件
__all__ = [
    "LocalSearch",      # 局部搜索核心类
    "GlobalSearch",     # 全局搜索核心类
    "LocalSearchTool",  # 局部搜索工具类
    "GlobalSearchTool", # 全局搜索工具类
    "HybridSearchTool", # 混合搜索工具类
    "NaiveSearchTool",  # 朴素搜索工具类
    "DeepResearchTool"  # 深度研究工具类
]