"""
图结构构建模块

该模块提供了构建和管理Neo4j图数据库中文档和文本块结构的核心功能。

主要组件：
- GraphStructureBuilder: 负责创建和管理文档节点、文本块节点以及它们之间的关系

功能说明：
- 支持文档和文本块的创建与关联
- 提供并行处理大量文本块的能力
- 维护文档结构的完整性和一致性
- 优化的Cypher查询实现，提高性能
"""
from .struct_builder import GraphStructureBuilder

__all__ = [
    'GraphStructureBuilder'
]