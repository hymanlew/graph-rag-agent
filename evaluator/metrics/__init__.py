"""
评估指标模块

此模块集中管理和导出GraphRAG评估系统中所有可用的评估指标，
提供指标分类、注册和实例化的统一接口。

评估指标按照不同的评估维度和方法进行分类，包括：
- 答案评估指标：用于评估系统回答的质量和准确性
- 检索评估指标：用于评估检索模块的性能和效率
- 图评估指标：用于评估知识图谱在回答生成中的有效利用
- LLM评估指标：利用大型语言模型进行深度语义评估
- 深度研究指标：用于评估系统的推理能力和迭代优化效果

这些指标共同构成了一个全面的评估体系，支持从多个维度对GraphRAG系统进行综合评价。
"""

# 答案评估指标 - 用于评估系统回答的准确性和匹配度
# 这些指标主要比较系统回答与标准答案的相似程度
from evaluator.metrics.answer_metrics import (
    ExactMatch,  # 精确匹配指标，评估回答与标准答案的完全匹配程度
    F1Score      # F1分数，结合精确率和召回率的综合指标
)

# 检索评估指标 - 用于评估检索模块的性能和效率
# 这些指标关注检索结果的质量、相关性和利用效率
from evaluator.metrics.retrieval_metrics import (
    RetrievalPrecision,      # 检索精确率，评估检索结果的相关性
    RetrievalUtilization,    # 检索利用率，评估系统对检索结果的有效利用
    RetrievalLatency,        # 检索延迟，评估检索的速度和效率
    ChunkUtilization         # 分块利用率，评估文档分块的有效使用
)

# 图评估指标 - 用于评估知识图谱在回答生成中的有效利用
# 这些指标关注图结构信息的覆盖和利用效果
from evaluator.metrics.graph_metrics import (
    EntityCoverageMetric,          # 实体覆盖度，评估实体识别的全面性
    GraphCoverageMetric,           # 图覆盖度，评估知识图谱利用的广度
    RelationshipUtilizationMetric, # 关系利用率，评估知识关系的有效使用
    CommunityRelevanceMetric,      # 社区相关性，评估检索社区的相关性
    SubgraphQualityMetric          # 子图质量，评估提取子图的完整性和相关性
)

# LLM评估指标 - 利用大型语言模型进行深度语义评估
# 这些指标使用LLM的语义理解能力进行质量评估
from evaluator.metrics.llm_metrics import (
    ResponseCoherence,          # 回答连贯性，评估回答的逻辑流畅性
    FactualConsistency,         # 事实一致性，评估回答的准确性
    ComprehensiveAnswerMetric,  # 回答全面性，评估回答的完整程度
    LLMGraphRagEvaluator        # 综合LLM评估器，使用LLM对图RAG进行全面评价
)

# 深度研究指标 - 用于评估系统的推理能力和迭代优化效果
# 这些指标关注复杂问题解决和知识融合能力
from evaluator.metrics.deep_search_metrics import (
    ReasoningCoherence,                # 推理连贯性，评估推理过程的逻辑一致性
    ReasoningDepth,                    # 推理深度，评估问题分析的深度
    IterativeImprovementMetric,        # 迭代改进度，评估系统优化自身回答的能力
    KnowledgeGraphUtilizationMetric    # 知识图谱利用率，评估知识图谱在推理中的应用效果
)

# 定义所有可用的指标
# 此字典将指标名称映射到对应的类路径，用于动态加载和实例化指标
__all_metrics__ = {
    # 答案评估指标
    'em': 'evaluator.metrics.answer_metrics.ExactMatch',
    'f1': 'evaluator.metrics.answer_metrics.F1Score',
    
    # 检索评估指标
    'retrieval_precision': 'evaluator.metrics.retrieval_metrics.RetrievalPrecision',
    'retrieval_utilization': 'evaluator.metrics.retrieval_metrics.RetrievalUtilization',
    'retrieval_latency': 'evaluator.metrics.retrieval_metrics.RetrievalLatency',
    'chunk_utilization': 'evaluator.metrics.retrieval_metrics.ChunkUtilization',
    
    # 图评估指标
    'entity_coverage': 'evaluator.metrics.graph_metrics.EntityCoverageMetric',
    'graph_coverage': 'evaluator.metrics.graph_metrics.GraphCoverageMetric',
    'relationship_utilization': 'evaluator.metrics.graph_metrics.RelationshipUtilizationMetric',
    'community_relevance': 'evaluator.metrics.graph_metrics.CommunityRelevanceMetric',
    'subgraph_quality': 'evaluator.metrics.graph_metrics.SubgraphQualityMetric',
    
    # LLM评估指标
    'response_coherence': 'evaluator.metrics.llm_metrics.ResponseCoherence',
    'factual_consistency': 'evaluator.metrics.llm_metrics.FactualConsistency',
    'answer_comprehensiveness': 'evaluator.metrics.llm_metrics.ComprehensiveAnswerMetric',
    'llm_evaluation': 'evaluator.metrics.llm_metrics.LLMGraphRagEvaluator',
    
    # 深度研究指标
    'reasoning_coherence': 'evaluator.metrics.deep_search_metrics.ReasoningCoherence',
    'reasoning_depth': 'evaluator.metrics.deep_search_metrics.ReasoningDepth',
    'iterative_improvement': 'evaluator.metrics.deep_search_metrics.IterativeImprovementMetric',
    'knowledge_graph_utilization': 'evaluator.metrics.deep_search_metrics.KnowledgeGraphUtilizationMetric'
}

def list_available_metrics():
    """
    列出所有可用的指标
    
    返回评估系统中注册的所有指标名称列表，便于用户了解系统支持的评估维度。
    
    Returns:
        List[str]: 指标名称列表，包含所有支持的评估指标标识
    """
    return list(__all_metrics__.keys())

def get_metric_class(metric_name: str):
    """
    获取指标类
    
    根据指标名称动态加载并返回对应的指标类。使用Python的动态导入机制，
    实现指标的懒加载，提高系统效率。
    
    Args:
        metric_name: 指标名称，对应__all_metrics__字典中的键
        
    Returns:
        指标类：如果指标存在，则返回对应的类对象；否则返回None
        
    Raises:
        ImportError: 如果模块导入失败
        AttributeError: 如果类在模块中不存在
    """
    metric_name = metric_name.lower()
    if metric_name not in __all_metrics__:
        return None

    import importlib
    # 从类路径中分离模块路径和类名
    module_path, class_name = __all_metrics__[metric_name].rsplit('.', 1)
    # 动态导入模块
    module = importlib.import_module(module_path)
    # 获取并返回类对象
    return getattr(module, class_name)

def get_metric_instance(metric_name: str, config):
    """
    获取指标实例
    
    根据指标名称和配置，创建并返回对应的指标实例。
    此函数是获取可用指标实例的主要入口点，简化了指标的实例化过程。
    
    Args:
        metric_name: 指标名称，对应__all_metrics__字典中的键
        config: 配置对象，包含指标所需的参数设置
        
    Returns:
        指标实例：如果指标存在，则返回初始化的指标实例；否则返回None
        
    Raises:
        TypeError: 如果指标类初始化失败
    """
    metric_cls = get_metric_class(metric_name)
    if metric_cls:
        return metric_cls(config)
    return None