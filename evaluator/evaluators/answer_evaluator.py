import os
from typing import Dict

from evaluator.core.base_evaluator import BaseEvaluator
from evaluator.core.evaluation_data import AnswerEvaluationData

"""
答案评估器模块

此模块实现了答案评估器(AnswerEvaluator)，用于评估GraphRAG系统生成答案的质量。
评估过程包括多个指标的计算，如精确度、召回率、一致性、连贯性等，
并支持结果保存和中间数据记录功能。
"""
class AnswerEvaluator(BaseEvaluator):
    """
    答案评估器类
    
    继承自BaseEvaluator，专门用于评估GraphRAG系统生成答案的质量。
    支持多种评估指标，能够计算并记录各类答案质量指标的得分，
    并保存评估结果和中间数据供后续分析。
    """
    
    def __init__(self, config):
        """
        初始化答案评估器
        
        Args:
            config: 评估配置对象，包含评估指标、保存路径等设置
            
        初始化过程中会调用父类的初始化方法，设置评估指标、保存路径等参数。
        """
        super().__init__(config)
    
    def evaluate(self, data: AnswerEvaluationData) -> Dict[str, float]:
        """
        执行答案质量评估
        
        遍历所有配置的评估指标，计算每个指标的得分，更新样本的评估结果，
        并根据配置保存评估结果和中间数据。
        
        Args:
            data: AnswerEvaluationData对象，包含待评估的答案样本集合
            
        Returns:
            Dict[str, float]: 评估结果字典，键为指标名称，值为得分
            
        评估流程:
        1. 遍历每个配置的评估指标
        2. 实例化对应的指标计算类
        3. 计算指标得分并更新结果字典
        4. 记录指标统计信息(最小值、最大值、平均值)
        5. 更新每个样本的指标得分
        6. 保存评估结果和中间数据(如果配置启用)
        """
        # 记录评估开始
        self.log("\n======== 开始评估答案质量 ========")
        self.log(f"样本总数: {len(data.samples)}")
        self.log(f"使用的评估指标: {', '.join(self.metrics)}")
        
        # 存储评估结果的字典
        result_dict = {}
        
        # 遍历配置的评估指标
        for metric_name in self.metrics:
            try:
                self.log(f"\n开始计算指标: {metric_name}")
                # 获取指标计算类的名称
                metric_class_name = self.metric_class[metric_name].__class__.__name__
                self.log(f"使用评估类: {metric_class_name}")
                
                # 计算指标得分
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                # 更新结果字典
                result_dict.update(metric_result)
                
                # 统计基本信息 - 处理不同类型的评分
                if metric_scores and not isinstance(metric_scores[0], dict):
                    min_score = min(metric_scores)
                    max_score = max(metric_scores)
                    avg_score = sum(metric_scores) / len(metric_scores)
                    self.log(f"指标统计: 最小值={min_score:.4f}, 最大值={max_score:.4f}, 平均值={avg_score:.4f}")
                
                # 更新每个样本的评分
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
                    
                self.log(f"完成指标 {metric_name} 计算，总体得分: {list(metric_result.values())[0]:.4f}")
            except Exception as e:
                import traceback
                self.log(f'评估 {metric_name} 时出错: {e}')
                self.log(traceback.format_exc())
                # 发生异常时继续处理下一个指标，避免整个评估过程中断
                continue
        
        # 输出所有指标的计算结果
        self.log("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            self.log(f"  {metric}: {score:.4f}")
        
        # 记录评估结束
        self.log("======== 答案质量评估结束 ========\n")
        
        # 根据配置保存评估结果
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
            self.log(f"评估结果已保存至: {os.path.join(self.save_dir, 'metric_score.txt')}")
        
        # 根据配置保存评估中间数据
        if self.save_data_flag:
            self.save_data(data)
            self.log(f"评估中间数据已保存至: {os.path.join(self.save_dir, 'intermediate_data.json')}")
        
        # 返回评估结果字典
        return result_dict