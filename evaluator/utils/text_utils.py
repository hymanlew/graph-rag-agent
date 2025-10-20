import re
import string
from typing import List, Dict

"""
文本处理工具模块

此模块提供了评估系统中用于文本处理和评估指标计算的核心工具函数。
主要功能包括：
- 文本标准化处理，用于消除评估过程中的文本差异干扰
- 评估指标计算，包括精确率、召回率和F1分数

这些函数在Agent评估过程中起着关键作用，特别是在对比不同Agent的输出质量时，
通过标准化文本并计算精确的评估指标，确保评估结果的准确性和一致性。
"""

def normalize_answer(s: str) -> str:
    """
    标准化答案文本，消除文本差异以提高评估准确性
    
    此函数实现了完整的文本标准化流程，用于在评估过程中消除不影响语义的文本差异，
    包括：标点符号、大小写、冠词和多余空格等。标准化处理是保证评估公平性的关键步骤，
    可以避免因格式差异而非内容差异导致的评估偏差。
    
    函数采用了函数式编程风格，通过一系列内部辅助函数组合实现文本转换：
    1. 转为小写字母（统一大小写）
    2. 移除标点符号（包括中英文标点）
    3. 移除冠词（包括英文和中文冠词）
    4. 修复空格（移除多余空格，统一为单空格分隔）
    
    Args:
        s (str): 需要标准化的原始文本
        
    Returns:
        str: 标准化处理后的文本，格式统一，便于比较
    """
    def remove_articles(text: str) -> str:
        # 移除英文和中文冠词，这些冠词通常不影响语义理解
        return re.sub(r"\b(a|an|the|一个|一种|这个|那个)\b", " ", text)
    
    def white_space_fix(text: str) -> str:
        # 移除多余空格，统一为单空格分隔
        return " ".join(text.split())
    
    def remove_punc(text: str) -> str:
        # 移除所有标点符号，包括英文和中文标点
        exclude = set(string.punctuation + "，。！？《》【】""''：；（）、")
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text: str) -> str:
        # 转换为小写字母
        return text.lower()
    
    # 按顺序应用标准化步骤：小写化 -> 移除标点 -> 移除冠词 -> 修复空格
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_precision_recall_f1(pred: List[str], truth: List[str]) -> Dict[str, float]:
    """
    计算信息检索评估的核心指标：精确率、召回率和F1分数
    
    此函数实现了信息检索和文本评估中的三个关键指标计算：
    1. 精确率(Precision)：预测结果中正确结果的比例
    2. 召回率(Recall)：正确结果被预测出来的比例
    3. F1分数：精确率和召回率的调和平均，综合考虑了两者
    
    函数的计算流程包括：
    1. 处理边界情况（空列表）
    2. 对预测和真实结果进行标准化处理
    3. 计算两者的交集作为真阳性(TP)结果
    4. 根据公式计算精确率、召回率和F1分数
    
    这些指标在评估Agent的检索和生成质量时至关重要，可以全面反映Agent输出的准确性和完整性。
    
    Args:
        pred (List[str]): 预测结果列表，如Agent生成的答案片段
        truth (List[str]): 真实结果列表，如标准答案片段
        
    Returns:
        Dict[str, float]: 包含precision、recall和f1三个指标的字典
            precision: 精确率，范围[0,1]
            recall: 召回率，范围[0,1]
            f1: F1分数，范围[0,1]
    """
    # 处理边界情况：如果任一列表为空，返回全0指标
    if not pred or not truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 对所有文本进行标准化处理，消除格式差异
    pred_norm = [normalize_answer(p) for p in pred]
    truth_norm = [normalize_answer(t) for t in truth]
    
    # 计算交集大小作为真阳性(TP)计数
    tp = len(set(pred_norm).intersection(set(truth_norm)))
    
    # 计算精确率：正确预测的比例
    precision = tp / len(pred_norm) if pred_norm else 0.0
    # 计算召回率：正确结果被预测的比例
    recall = tp / len(truth_norm) if truth_norm else 0.0
    
    # 计算F1分数：精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }