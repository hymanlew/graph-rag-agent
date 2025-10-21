"""
搜索工具模块初始化

本模块提供了Graph-RAG系统中各种搜索工具的统一入口和管理。
每个搜索工具针对不同的搜索场景和需求进行了专门设计，
形成了一个完整的搜索能力体系。

工具层次结构：
- BaseSearchTool: 所有搜索工具的基类，定义了通用接口和基础功能
- LocalSearchTool: 本地知识库搜索工具，专注于文档检索
- GlobalSearchTool: 全局搜索工具，用于知识图谱查询和社区信息获取
- HybridSearchTool: 混合搜索工具，结合多种搜索策略和关键词提取能力
- NaiveSearchTool: 简单搜索工具，提供基础的搜索功能
- DeepResearchTool: 深度研究工具，实现复杂问题的多步思考和搜索
- DeeperResearchTool: 深度增强研究工具，在DeepResearchTool基础上提供更强大的功能

使用方法：
1. 导入需要的搜索工具类
2. 实例化工具对象
3. 调用search方法执行搜索操作
4. 处理和分析搜索结果

示例：
```python
from search.tool import DeepResearchTool

search_tool = DeepResearchTool()
result = search_tool.search("什么是Graph-RAG技术？")
```

业务意义：
- 为Graph-RAG系统提供多样化的搜索能力
- 支持从简单查询到复杂研究的全场景应用
- 确保搜索结果的准确性、相关性和全面性
- 为知识推理和答案生成提供高质量的信息来源

扩展建议：
- 新的搜索工具应继承BaseSearchTool并实现必要的接口
- 确保搜索结果格式的一致性，便于后续处理
- 考虑性能优化，特别是对于高频调用的场景
"""

# 导入各种搜索工具类
from search.tool.base import BaseSearchTool
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from search.tool.hybrid_tool import HybridSearchTool
from search.tool.naive_search_tool import NaiveSearchTool
from search.tool.deep_research_tool import DeepResearchTool
from search.tool.deeper_research_tool import DeeperResearchTool

# 定义模块公开接口，便于外部导入和使用
__all__ = [
    "BaseSearchTool",
    "LocalSearchTool",
    "GlobalSearchTool",
    "HybridSearchTool",
    "NaiveSearchTool",
    "DeepResearchTool",
    "DeeperResearchTool",
]