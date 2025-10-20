"""
社区发现与摘要模块

本模块是graph-rag-agent项目的重要组成部分，主要负责图数据中的社区发现与社区摘要生成。
社区是图中紧密相连的节点集合，发现这些社区有助于理解图的结构和组织模式，
并能有效地进行知识聚合和信息摘要。

模块结构：
- detector: 实现各种社区发现算法，包括Leiden和SLLPA等
- summary: 为检测到的社区生成摘要，提取关键信息

主要功能：
1. 提供统一的社区发现接口，支持多种算法
2. 实现社区摘要生成，从社区中提取有意义的内容
3. 通过工厂模式简化算法选择和使用
4. 支持图数据的社区分析和可视化

在整个RAG系统中的作用：
- 提高信息检索的准确性，通过社区边界控制检索范围
- 实现知识的层次化组织，便于大规模图数据的管理
- 为图查询提供上下文感知能力，理解相关节点的集合
- 支持生成更有针对性的回答摘要
"""

from .detector import CommunityDetectorFactory
from .summary import CommunitySummarizerFactory

# 导出的公共接口，定义模块的公开API
__all__ = ['CommunityDetectorFactory', 'CommunitySummarizerFactory']