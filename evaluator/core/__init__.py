"""
评估器核心模块

此模块导出GraphRAG评估系统的基础组件，作为整个评估框架的核心接口。
这些组件共同构成了一个灵活、可扩展的评估系统架构，支持从检索到答案的全流程评估。

主要组件包括：
- 评估器基类：定义所有评估器必须实现的接口和通用功能
- 指标基类：提供评估指标的抽象接口和注册机制
- 评估数据结构：用于规范化存储和管理各类评估数据

此模块的设计采用了面向对象的抽象和多态原则，允许系统通过继承和组合方式
灵活扩展新的评估器和指标类型，同时保持接口的一致性。
"""

# 评估指标基类 - 所有具体评估指标类的抽象基类
# 提供指标注册、计算和结果格式化的通用接口
from evaluator.core.base_metric import BaseMetric

# 评估器基类 - 所有具体评估器的抽象基类
# 提供评估流程控制、指标管理、数据保存等通用功能
from evaluator.core.base_evaluator import BaseEvaluator

# 评估数据结构 - 用于规范化存储和管理评估过程中的各类数据
# 包括：
# - AnswerEvaluationSample: 单个答案评估样本
# - AnswerEvaluationData: 管理多个答案评估样本的集合
# - RetrievalEvaluationSample: 单个检索评估样本
# - RetrievalEvaluationData: 管理多个检索评估样本的集合
from evaluator.core.evaluation_data import (
    AnswerEvaluationSample, AnswerEvaluationData,
    RetrievalEvaluationSample, RetrievalEvaluationData
)

# 导出列表，定义该模块的公共API
export = [
    "BaseMetric", "BaseEvaluator",
    "AnswerEvaluationSample", "AnswerEvaluationData",
    "RetrievalEvaluationSample", "RetrievalEvaluationData"
]