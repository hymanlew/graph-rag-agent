"""
Graph-RAG Agent 知识图谱模块

此模块是Graph-RAG Agent的核心组件之一，负责知识图谱的构建、管理和查询功能。
该模块提供了一套完整的工具集，用于从文档中提取实体和关系，构建结构化知识图谱，并支持高效的图数据查询。

主要功能包括：

1. 核心连接管理 - 提供与图数据库的连接管理和基础工具函数
2. 索引管理 - 负责文本块和实体的索引创建和维护
3. 图结构构建 - 将提取的实体和关系构建为完整的图结构
4. 实体关系提取 - 从文本中识别和提取实体及其之间的关系
5. 实体处理 - 管理实体合并、相似实体检测等高级功能

该模块的设计遵循模块化原则，各子模块之间通过清晰的接口进行交互，支持系统的可扩展性和可维护性。
"""

# 导入核心组件 - 提供基础功能和连接管理
from graph.core import (
    GraphConnectionManager,  # 图数据库连接管理器
    connection_manager,      # 全局连接管理实例
    BaseIndexer,             # 索引器基类
    timer,                   # 计时装饰器
    generate_hash,           # 生成唯一哈希值的工具函数
    batch_process,           # 批量处理数据的工具函数
    retry,                   # 重试装饰器
    get_performance_stats,   # 获取性能统计信息
    print_performance_stats  # 打印性能统计信息
)

# 导入索引相关组件
from graph.indexing import (
    ChunkIndexManager,   # 文本块索引管理器 - 负责文档块的索引创建和查询
    EntityIndexManager   # 实体索引管理器 - 负责实体的索引创建和查询
)

# 导入图结构相关组件
from graph.structure import (
    GraphStructureBuilder  # 图结构构建器 - 负责构建完整的图结构
)

# 导入实体关系提取组件
from graph.extraction import (
    EntityRelationExtractor,  # 实体关系提取器 - 从文本中提取实体和关系
    GraphWriter               # 图数据写入器 - 将提取的数据写入图数据库
)

# 导入实体处理组件
from graph.processing import (
    EntityMerger,           # 实体合并器 - 合并相似实体
    SimilarEntityDetector,  # 相似实体检测器 - 检测语义相似的实体
    GDSConfig               # 图数据科学配置 - 配置图分析功能
)

# 定义模块的公共API，控制模块导入行为
__all__ = [
    # 核心组件
    'GraphConnectionManager',
    'connection_manager',
    'BaseIndexer',
    'timer',
    'generate_hash',
    'batch_process',
    'retry',
    'get_performance_stats',
    'print_performance_stats',
    
    # 索引管理组件
    'ChunkIndexManager',
    'EntityIndexManager',
    
    # 图结构组件
    'GraphStructureBuilder',
    
    # 提取组件
    'EntityRelationExtractor',
    'GraphWriter',
    
    # 处理组件
    'EntityMerger',
    'SimilarEntityDetector',
    'GDSConfig'
]