"""
Graph-RAG Agent 实体关系提取模块

此模块负责从文本中提取实体、关系和属性信息，是构建知识图谱的核心组件。

主要功能包括：
1. 实体提取与关系识别：从原始文本中识别出实体及其之间的关系
2. 属性提取：识别实体的属性信息
3. 文本批处理：高效处理大量文本内容
4. 图数据写入：将提取的信息写入图数据库

模块组件：
- EntityRelationExtractor: 实体关系提取器，使用LLM模型从文本中提取实体和关系
- GraphWriter: 图数据库写入器，负责将提取的实体和关系数据写入Neo4j图数据库
"""

# 导入实体关系提取器类，负责从文本中提取实体和关系
from .entity_extractor import EntityRelationExtractor

# 导入图数据库写入器类，负责将提取的实体和关系数据写入Neo4j
from .graph_writer import GraphWriter

# 定义模块导出列表，控制公共API
__all__ = [
    # 实体关系提取器，用于从文本中提取实体和关系信息
    'EntityRelationExtractor',
    # 图数据库写入器，用于将提取的数据写入图数据库
    'GraphWriter'
]