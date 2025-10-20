import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Type

from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
from evaluator.core.base_metric import BaseMetric

"""
评估器基类模块

此模块定义了所有评估器的基础接口和通用功能，采用抽象基类设计模式，确保所有具体评估器实现统一的接口。
核心功能包括：
- 评估指标的动态收集和初始化
- 评估结果的保存和格式化
- 中间数据的序列化和存储
- 调试日志记录
"""

class BaseEvaluator(ABC):
    """
    评估器基类，定义所有评估器必须实现的通用功能和接口
    
    采用模板方法设计模式，提供评估框架和通用功能，
    具体评估逻辑由子类实现。作为整个评估系统的核心抽象，
    为各种具体评估器提供统一的接口和基础服务。
    """
    
    def __init__(self, config):
        """
        初始化评估器
        
        配置评估环境、加载评估指标、准备保存路径，为评估过程做准备。
        支持灵活的配置传入方式，自动发现和初始化评估指标。
        
        Args:
            config: 评估配置，可以是字典或EvaluatorConfig对象，
                   包含评估指标、保存路径、调试模式等配置信息
            
        初始化流程：
        1. 处理配置对象，支持字典或专用配置类
        2. 从配置中提取并设置保存路径和功能标志
        3. 收集并初始化所有需要的评估指标
        4. 创建结果保存目录，确保文件操作环境就绪
        
        Raises:
            NotImplementedError: 如果指定的评估指标未实现
        """
        # 支持字典或EvaluatorConfig对象，实现配置的灵活传入
        if isinstance(config, dict):
            self.config = EvaluatorConfig(config)
        else:
            self.config = config
            
        # 从配置中提取评估参数
        self.save_dir = self.config.get('save_dir', './evaluation_results')
        self.save_metric_flag = self.config.get('save_metric_score', True)
        self.save_data_flag = self.config.get('save_intermediate_data', True)
        self.metrics = self.config.get_metrics()
        self.debug = self.config.get('debug', False)
        
        # 确保保存目录存在，避免文件操作异常
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 自动发现并收集所有可用的评估指标类
        self.available_metrics = self._collect_metrics()
        
        # 初始化指定的评估指标实例
        self.metric_class = {}
        for metric in self.metrics:
            if metric in self.available_metrics:
                # 为每个指标创建实例并传入配置
                self.metric_class[metric] = self.available_metrics[metric](self.config.to_dict())
            else:
                # 处理未实现的指标，提示并抛出异常
                print(f"{metric} 评估指标未实现!")
                raise NotImplementedError(f"评估指标 {metric} 未实现")
    
    def _collect_metrics(self) -> Dict[str, Type[BaseMetric]]:
        """
        收集所有继承自BaseMetric的评估指标类
        
        利用Python的反射机制，动态发现所有BaseMetric的子类，
        实现评估指标的自动注册和管理，提高系统的可扩展性。
        这种动态发现机制使得新添加的指标类无需手动注册即可被识别。
        
        Returns:
            Dict[str, Type[BaseMetric]]: 指标名称到指标类的映射字典，
                                         键为指标名称(metric_name)，值为对应的指标类
        """
        # 递归查找所有子类的辅助函数
        def find_descendants(base_class, subclasses=None):
            """
            递归查找基类的所有子类
            
            实现深度优先搜索，找出所有继承自指定基类的子类，包括间接子类。
            使用集合确保不会重复添加同一子类。
            
            Args:
                base_class: 要查找子类的基类
                subclasses: 已发现的子类集合，用于递归传递
                
            Returns:
                set: 包含所有子类的集合
            """
            if subclasses is None:
                subclasses = set()
            
            # 获取直接子类
            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    # 递归查找子类的子类，支持多层继承
                    find_descendants(subclass, subclasses)
            return subclasses
        
        # 创建指标名称到指标类的映射
        available_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            available_metrics[metric_name] = cls
        
        return available_metrics
    
    @abstractmethod
    def evaluate(self, data) -> Dict[str, float]:
        """
        执行评估的抽象方法，必须由子类实现
        
        定义了评估器的核心接口，确保所有评估器遵循统一的调用方式。
        子类需要根据自身功能实现具体的评估逻辑，处理特定类型的数据。
        
        Args:
            data: 评估数据，可以是不同类型的评估数据对象，具体类型由子类定义
            
        Returns:
            Dict[str, float]: 评估结果字典，键为指标名称，值为对应的得分
                             通常得分范围为0-1，数值越大表示性能越好
            
        Raises:
            ValueError: 如果数据格式不正确或缺少必要信息
        """
        pass
    
    def save_metric_score(self, result_dict: Dict[str, float]):
        """
        保存评估指标结果到文本文件
        
        将评估结果以简单的键值对形式保存到文本文件中，便于人工阅读和检查。
        使用UTF-8编码确保中文等非ASCII字符正确保存。
        
        Args:
            result_dict: 评估结果字典，包含各指标名称和对应得分
        """
        file_name = "metric_score.txt"
        save_path = os.path.join(self.save_dir, file_name)
        
        # 写入评估结果，确保使用UTF-8编码以支持中文
        with open(save_path, "w", encoding='utf-8') as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")
    
    def save_data(self, data):
        """
        保存评估中间数据到JSON文件
        
        提供灵活的数据保存机制，支持自定义保存方法和自动序列化。
        可以保存评估过程中生成的中间数据，便于后续分析和调试。
        
        Args:
            data: 评估数据对象，可以是自定义数据类或基本数据结构
                  如果对象有save方法，将调用其自定义保存逻辑
        """
        file_name = "intermediate_data.json"
        save_path = os.path.join(self.save_dir, file_name)
        
        # 支持自定义保存方法的对象
        if hasattr(data, 'save'):
            # 如果对象有save方法，调用其自定义保存逻辑
            data.save(save_path)
        else:
            # 对于没有自定义保存方法的对象，尝试自动序列化
            try:
                # 将复杂对象转换为可JSON序列化的格式
                serializable_data = self._convert_to_serializable(data)
                with open(save_path, "w", encoding='utf-8') as f:
                    # 确保中文正常显示，美化JSON格式
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                # 处理序列化异常，记录错误信息但不中断流程
                print(f"保存数据时出错: {e}")
    
    def _convert_to_serializable(self, data):
        """
        递归地将复杂数据结构转换为可JSON序列化的格式
        
        处理常见的Python数据结构和自定义对象，确保可以正确序列化。
        采用递归方法处理嵌套数据结构，实现通用的序列化转换。
        
        Args:
            data: 要转换的数据，可以是字典、列表或自定义对象
            
        Returns:
            可JSON序列化的数据结构，通常是字典或列表的组合
        """
        # 处理字典类型
        if isinstance(data, dict):
            # 递归转换字典的每个值
            return {k: self._convert_to_serializable(v) for k, v in data.items()}
        # 处理列表类型
        elif isinstance(data, list):
            # 递归转换列表中的每个元素
            return [self._convert_to_serializable(item) for item in data]
        # 处理自定义对象
        elif hasattr(data, '__dict__'):
            # 将对象的属性字典递归转换
            return self._convert_to_serializable(data.__dict__)
        # 基本类型直接返回
        else:
            return data
    
    def format_results_table(self, results: Dict[str, float]) -> str:
        """
        将评估结果格式化为Markdown表格形式
        
        生成易于阅读的评估结果表格，便于在控制台或文档中展示。
        对浮点数结果进行格式化，保持一致的显示精度。
        
        Args:
            results: 评估结果字典，键为指标名称，值为对应的得分
            
        Returns:
            str: 格式化的Markdown表格字符串，包含表头和数据行
        """
        # 表格头部和分隔行
        header = "| 指标 | 得分 |"
        separator = "| --- | --- |"
        
        # 生成表格行
        rows = []
        for metric, score in results.items():
            # 格式化得分，浮点数保留4位小数
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)
            rows.append(f"| {metric} | {score_str} |")
        
        # 组合完整表格
        table = "\n".join([header, separator] + rows)
        return table
    
    def log(self, message, *args, **kwargs):
        """
        输出调试日志
        
        提供条件日志输出功能，仅在debug模式下输出日志，
        并自动添加类名前缀，便于追踪日志来源。
        使用evaluator模块的debug_print函数实现统一的日志输出格式。
        
        Args:
            message: 日志消息内容
            *args, **kwargs: 格式化消息的额外参数，用于字符串格式化
        """
        from evaluator import debug_print
        # 仅在debug模式下输出日志，避免在生产环境产生过多输出
        if self.debug:
            # 自动添加类名前缀，便于识别日志来源
            debug_print(f"[{self.__class__.__name__}] {message}", *args, **kwargs)