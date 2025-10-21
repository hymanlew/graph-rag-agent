"""
Graph-RAG Agent 索引管理模块

此模块提供了知识图谱系统中的索引管理功能，负责创建和维护两种关键索引：

1. 文本块索引 (Chunk Index) - 用于高效检索和管理文档的文本片段
2. 实体索引 (Entity Index) - 用于管理从文本中提取的实体及其相关信息

索引管理是Graph-RAG系统的重要组成部分，它确保了系统能够快速访问和检索
存储在知识图谱中的各类信息，提高了查询效率和响应速度。
"""

# 导入文本块索引管理器 - 负责文档文本块的索引创建和管理
from .chunk_indexer import ChunkIndexManager
# 导入实体索引管理器 - 负责实体的索引创建和管理
from .entity_indexer import EntityIndexManager

# 定义模块的公共API
__all__ = [
    'ChunkIndexManager',  # 文本块索引管理器
    'EntityIndexManager'  # 实体索引管理器
]