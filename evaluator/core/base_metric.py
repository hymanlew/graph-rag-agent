from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

"""
评估指标基类模块

此模块定义了所有评估指标的通用接口和基础功能，采用抽象基类设计模式，
确保所有具体评估指标实现统一的接口。核心功能包括：
- 评估指标的基础配置和初始化
- 定义指标计算的标准接口
- 提供调试日志功能
- 实现LLM回退评分机制，处理规则难以评估的情况
"""

class BaseMetric(ABC):
    """
    评估指标基类，所有具体评估指标都必须继承此类
    
    提供评估指标的通用功能和标准化接口，采用模板方法设计模式，
    子类只需实现具体的指标计算逻辑。
    """
    
    # 指标名称，子类必须重写此属性，作为指标的唯一标识
    metric_name = "base"
    
    def __init__(self, config):
        """
        初始化评估指标基类
        
        Args:
            config: 评估配置，可以是字典或EvaluatorConfig对象
            
        初始化流程：
        1. 处理配置对象
        2. 提取数据集名称
        3. 设置调试模式
        4. 获取LLM模型用于回退评分
        """
        # 支持灵活的配置传入方式
        if isinstance(config, dict):
            from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
            self.config = EvaluatorConfig(config)
        else:
            self.config = config
            
        # 初始化基本配置属性
        self.dataset_name = self.config.get('dataset_name', 'default')
        self.debug = self.config.get('debug', False)
        # 获取LLM模型，用于在规则评分难以处理的情况下进行回退评分
        self.llm = self.config.get('llm', None)
    
    @abstractmethod
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List]:
        """
        计算评估指标的抽象方法，所有子类必须实现
        
        定义了评估指标计算的标准接口，确保所有指标输出格式一致。
        
        Args:
            data: 评估数据对象，类型根据具体评估任务而定
            
        Returns:
            Tuple[Dict[str, float], List]: 
                - 第一个元素：评估结果字典，包含指标名称和总体得分
                - 第二个元素：每个样本的评分列表
        """
        return {}, []
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        提供条件日志输出功能，仅在debug模式下输出日志，
        并自动添加类名前缀，便于追踪日志来源。
        
        Args:
            message: 日志消息
            *args, **kwargs: 格式化消息的额外参数
        """
        from evaluator import debug_print
        if self.debug:
            debug_print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)
            
    def get_llm_fallback_score(self, prompt: str, default_score: float = 0.5) -> float:
        """
        使用LLM进行回退评分
        
        提供规则评分的补充机制，处理难以通过规则客观评估的质量问题。
        当规则评分不足或需要更主观的质量判断时，可以调用LLM进行回退评分。
        
        Args:
            prompt: 提示文本，包含需要评估的内容和评分要求
            default_score: 默认分数，当LLM评分失败时返回
            
        Returns:
            float: LLM评分结果或默认分数，范围为0-1
        """
        # 如果没有配置LLM，直接返回默认分数
        if not self.llm:
            self.log(f"  LLM不可用，使用默认分数: {default_score:.4f}")
            return default_score
            
        try:
            self.log("  正在使用LLM进行回退评分...")
            # 调用LLM模型获取评分
            response = self.llm.invoke(prompt)
            # 处理不同格式的响应
            score_text = response.content if hasattr(response, 'content') else response
            
            self.log(f"  LLM响应: {score_text}")
            
            # 使用正则表达式从响应中提取数字分数
            import re
            score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            if score_match:
                extracted_score = float(score_match.group(1))
                # 确保分数在0-1范围内，这是评估系统的标准范围
                score = max(0.0, min(1.0, extracted_score))
                self.log(f"  LLM评分结果: {score:.4f}")
                return score
            else:
                self.log(f"  无法从LLM响应中提取分数，使用默认分数: {default_score:.4f}")
                return default_score
        except Exception as e:
            # 捕获所有异常，确保评估过程不会中断
            self.log(f"  LLM评分出错: {e}")
            return default_score