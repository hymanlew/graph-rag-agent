"""
评估工具模块初始化

此模块作为评估系统工具包的入口点，导出了评估过程中常用的核心工具函数，
主要包括：

1. 文本处理工具：用于文本标准化和评估指标计算
2. 日志工具：用于配置和管理系统日志

通过在__init__.py中导出这些函数，可以简化其他模块的导入语句，
使代码更加清晰和易于维护。
"""

# 从text_utils模块导出文本处理和评估指标计算函数
from evaluator.utils.text_utils import normalize_answer, compute_precision_recall_f1
# 从logging_utils模块导出日志配置和管理函数
from evaluator.utils.logging_utils import setup_logger, get_logger
