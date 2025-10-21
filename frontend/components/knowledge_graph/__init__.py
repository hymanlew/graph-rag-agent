"""
知识图谱组件模块

该模块提供了知识图谱可视化和交互的核心功能，集成了多个子模块的关键组件，
为Streamlit应用提供完整的知识图谱展示和管理能力。

主要组件包括：
1. visualize_knowledge_graph - 知识图谱可视化函数
2. display_knowledge_graph_tab - 知识图谱标签页显示函数
"""

# 导入知识图谱可视化函数 - 负责将图谱数据转换为交互式图形
from .visualization import visualize_knowledge_graph

# 导入知识图谱标签页显示函数 - 负责创建完整的知识图谱交互界面
from .display import display_knowledge_graph_tab

# 定义公共API列表 - 明确指定该模块暴露给外部的组件
__all__ = ['visualize_knowledge_graph', 'display_knowledge_graph_tab']