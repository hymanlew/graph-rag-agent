"""
社区检测器模块

本模块实现了多种社区发现算法，用于从图数据中识别紧密相连的节点集合。
采用了工厂模式，提供统一的接口来创建和使用不同的社区检测算法，
使得算法的选择和替换更加灵活。

核心组件：
- BaseCommunityDetector: 所有社区检测器的抽象基类，定义统一接口
- LeidenDetector: 基于Leiden算法的社区检测器
- SLLPADetector: 基于SLLPA算法的社区检测器
- CommunityDetectorFactory: 工厂类，负责创建适当的检测器实例

工厂设计的优势：
- 解耦算法实现与使用代码
- 支持运行时动态切换算法
- 简化客户端代码，隐藏算法细节
- 便于扩展新的社区检测算法

社区检测在RAG中的应用：
- 确定相关信息的边界，提高检索精度
- 发现语义相关的知识集群
- 支持层次化的信息组织
- 优化图查询的执行效率
"""

from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from .base import BaseCommunityDetector
from .leiden import LeidenDetector
from .sllpa import SLLPADetector

class CommunityDetectorFactory:
    """社区检测器工厂类
    
    实现工厂设计模式，负责创建和配置不同类型的社区检测器。
    这个类隐藏了具体检测器实现的细节，提供一个简单的接口来根据算法名称创建检测器。
    
    支持的算法：
    - leiden: Leiden算法，一种改进的模块化社区发现算法，优化了社区质量和执行效率
    - sllpa: 结构化标签传播算法，适合大规模网络的社区检测
    """
    
    # 算法映射字典，将算法名称映射到对应的检测器类
    ALGORITHMS = {
        'leiden': LeidenDetector,
        'sllpa': SLLPADetector
    }
    
    @classmethod
    def create(cls, algorithm: str, gds: GraphDataScience, graph: Neo4jGraph) -> BaseCommunityDetector:
        """
        创建指定类型的社区检测器
        
        参数:
            algorithm: 算法名称，支持 'leiden' 或 'sllpa'
            gds: GraphDataScience实例，用于执行图算法
            graph: Neo4jGraph实例，表示要分析的图
            
        返回:
            对应算法的社区检测器实例
            
        异常:
            ValueError: 如果指定了不支持的算法
            
        实现步骤:
        1. 算法名称转换为小写，确保不区分大小写
        2. 检查算法是否在支持的列表中
        3. 创建并返回对应的检测器实例，传入必要的参数
        """
        algorithm = algorithm.lower()
        if algorithm not in cls.ALGORITHMS:
            raise ValueError(f"不支持的算法: {algorithm}")
        return cls.ALGORITHMS[algorithm](gds, graph)

# 定义模块的公共API，导出主要类供外部使用
__all__ = ['CommunityDetectorFactory', 'BaseCommunityDetector', 
           'LeidenDetector', 'SLLPADetector']