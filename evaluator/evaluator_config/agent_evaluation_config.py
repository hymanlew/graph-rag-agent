from typing import List
from evaluator.metrics import list_available_metrics

"""
Agent评估配置模块

此模块定义了不同类型GraphRAG Agent的默认评估指标配置，根据每种Agent的特性
和功能选择最适合的评估指标集合。模块支持以下Agent类型：
- graph: 图检索Agent
- hybrid: 混合检索Agent
- fusion: 融合推理Agent
- naive: 基础RAG Agent
- deep: 深度研究Agent
"""

# 获取所有可用的评估指标
available_metrics = list_available_metrics()

# 不同Agent类型的默认评估指标配置
# 根据每种Agent的特性和功能选择最适合的评估指标集合
AGENT_EVALUATION_CONFIG = {
    "graph": {
        # 图检索Agent的答案质量评估指标
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        # 图检索Agent的检索性能评估指标
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ]
    },
    
    "hybrid": {
        # 混合检索Agent的答案质量评估指标
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        # 混合检索Agent的检索性能评估指标
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ]
    },
    
    "fusion": {
        # 融合推理Agent的答案质量评估指标
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        # 融合推理Agent的检索性能评估指标
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization',
                        'community_relevance', 'subgraph_quality']
            if m in available_metrics
        ],
        # 融合推理Agent的推理能力评估指标
        "reasoning_metrics": [
            m for m in ['reasoning_coherence', 'reasoning_depth', 'iterative_improvement']
            if m in available_metrics
        ]
    },
    
    "naive": {
        # 基础RAG Agent的答案质量评估指标
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        # 基础RAG Agent的检索性能评估指标（仅包含与传统向量检索相关的指标）
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'chunk_utilization']  # 只使用与传统向量检索相关的指标
            if m in available_metrics
        ]
    },
    
    "deep": {
        # 深度研究Agent的答案质量评估指标
        "answer_metrics": [
            m for m in ['em', 'f1', 'response_coherence', 'factual_consistency', 
                         'answer_comprehensiveness', 'llm_evaluation']
            if m in available_metrics
        ],
        # 深度研究Agent的检索性能评估指标
        "retrieval_metrics": [
            m for m in ['retrieval_precision', 'retrieval_utilization', 'retrieval_latency',
                        'entity_coverage', 'graph_coverage', 'relationship_utilization']
            if m in available_metrics
        ],
        # 深度研究Agent的推理能力评估指标
        "reasoning_metrics": [
            m for m in ['reasoning_coherence', 'reasoning_depth', 'iterative_improvement']
            if m in available_metrics
        ],
        # 深度研究Agent的深度研究能力评估指标
        "deeper_metrics": [
            m for m in ['knowledge_graph_utilization']  # 仅适用于DeeperResearchTool
            if m in available_metrics
        ]
    }
}

def get_agent_metrics(agent_type: str, metric_type: str = None) -> List[str]:
    """
    获取特定Agent类型的评估指标列表
    
    根据Agent类型和可选的指标类型，返回相应的评估指标列表。支持获取特定类型的指标
    或所有类型的指标。
    
    Args:
        agent_type: Agent类型，可选值为 graph, hybrid, naive, deep, fusion
        metric_type: 指标类型，可选值为 answer, retrieval, reasoning, deeper, 
                    None表示返回所有类型的指标
        
    Returns:
        List[str]: 请求的评估指标名称列表
        
    Raises:
        ValueError: 当提供了不支持的Agent类型时
    """
    # 验证Agent类型是否有效
    if agent_type not in AGENT_EVALUATION_CONFIG:
        raise ValueError(f"不支持的Agent类型: {agent_type}")
    
    # 根据指标类型返回相应的指标列表
    if metric_type == "answer":
        # 返回答案质量评估指标
        return AGENT_EVALUATION_CONFIG[agent_type]["answer_metrics"]
    elif metric_type == "retrieval":
        # 返回检索性能评估指标
        return AGENT_EVALUATION_CONFIG[agent_type]["retrieval_metrics"]
    elif metric_type == "reasoning" and "reasoning_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
        # 返回推理能力评估指标（如果Agent支持）
        return AGENT_EVALUATION_CONFIG[agent_type]["reasoning_metrics"]
    elif metric_type == "deeper" and "deeper_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
        # 返回深度研究能力评估指标（如果Agent支持）
        return AGENT_EVALUATION_CONFIG[agent_type]["deeper_metrics"]
    else:
        # 返回所有可用的指标
        metrics = []
        # 添加答案质量指标
        metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["answer_metrics"])
        # 添加检索性能指标
        metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["retrieval_metrics"])
        
        # 添加推理能力指标（如果支持）
        if "reasoning_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
            metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["reasoning_metrics"])
        
        # 添加深度研究能力指标（如果支持）
        if "deeper_metrics" in AGENT_EVALUATION_CONFIG[agent_type]:
            metrics.extend(AGENT_EVALUATION_CONFIG[agent_type]["deeper_metrics"])
            
        return metrics