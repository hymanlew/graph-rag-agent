import re
from typing import Dict, List, Tuple
from evaluator.core.base_metric import BaseMetric
from evaluator.core.evaluation_data import AnswerEvaluationData
from evaluator.utils.text_utils import normalize_answer

"""
答案评估指标模块

此模块实现了用于评估GraphRAG系统回答质量的核心指标，包括：
- ExactMatch（精确匹配）：评估系统回答与标准答案的匹配程度
- F1Score（F1分数）：评估系统回答的精确率和召回率的综合表现

这些指标采用了混合评分策略，结合规则匹配和LLM评估，以提高评估的准确性和鲁棒性。
"""

class ExactMatch(BaseMetric):
    """
    精确匹配评估指标
    
    评估系统回答与标准答案的匹配程度，采用混合评分策略：
    1. 首先进行严格的标准化匹配
    2. 对于不匹配的情况，计算内容相似度
    3. 根据相似度高低决定直接评分或使用LLM进行深度评估
    
    这种混合方法既保证了评估的客观性，又能处理语言表达的多样性。
    """
    
    # 指标名称，用于在评估系统中唯一标识此指标
    metric_name = "em"

    def __init__(self, config):
        """
        初始化精确匹配评估器
        
        Args:
            config: 评估配置，包含LLM和其他参数设置
        """
        super().__init__(config)
        # 获取可选的LLM实例，用于深度语义评估
        self.llm = config.get("llm", None)
    
    def calculate_em(self, prediction: str, golden_answer: str) -> float:
        """
        计算单个预测的精确匹配得分
        
        基本的精确匹配计算方法，通过标准化文本后进行严格匹配。
        
        Args:
            prediction: 预测答案，系统生成的回答
            golden_answer: 标准答案，期望的正确回答
            
        Returns:
            float: 得分（1.0表示完全匹配，0.0表示不匹配）
        """
        if not prediction or not golden_answer:
            return 0.0
            
        normalized_prediction = normalize_answer(prediction)
        normalized_golden = normalize_answer(golden_answer)
        
        # 完全匹配
        if normalized_prediction == normalized_golden:
            return 1.0
        return 0.0
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算精确匹配指标 - 使用规则匹配和LLM回退混合评分
        
        实现了混合评分策略，结合文本标准化、内容相似度计算和LLM语义评估，
        提高评估的准确性和鲁棒性，特别适合处理自然语言表达多样性的情况。
        
        Args:
            data: 评估数据，包含系统答案和标准答案
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 
                - 第一个元素：总体平均得分，格式为{"em": 得分}
                - 第二个元素：每个样本的得分列表
        """
        # 记录评估过程日志
        self.log("======== ExactMatch 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        # 存储每个样本的得分
        metric_score_list = []
        
        # 遍历所有样本
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 标准化答案，去除无关字符和空格，统一大小写等
            normalized_pred = normalize_answer(cleaned_pred)
            normalized_golden = normalize_answer(golden)
            
            # 记录详细的处理过程
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            self.log(f"  清理后的系统答案(前30字符): {cleaned_pred[:30]}...")
            self.log(f"  标准化后的标准答案(前30字符): {normalized_golden[:30]}...")
            self.log(f"  标准化后的系统答案(前30字符): {normalized_pred[:30]}...")
            
            # 完全匹配检查
            if normalized_pred == normalized_golden:
                score = 1.0
                self.log(f"  完全匹配 ✓")
            else:
                # 规则匹配失败，尝试内容相似性评估
                similarity_score = self._calculate_content_similarity(cleaned_pred, golden)
                self.log(f"  基本内容相似度: {similarity_score:.4f}")
                
                # 如果内容相似度较高，给予较高分数
                if similarity_score >= 0.7:
                    # 线性映射相似度到0.7-1.0范围
                    score = 0.7 + (similarity_score - 0.7) * 3/3
                    self.log(f"  内容高度相似，给予分数: {score:.4f}")
                # 如果内容相似度一般，回退到LLM评分
                elif self.llm:
                    self.log(f"  内容相似度一般，回退到LLM评分")
                    
                    # 构建LLM评估提示
                    prompt = f"""
                    请比较下面两个答案，评估它们在内容上的等价性，给出0到1之间的分数。
                    0表示完全不同，1表示内容上完全等价。
                    请只考虑实质内容，忽略格式、表达方式和顺序的差异。
                    
                    标准答案:
                    {golden}
                    
                    系统答案:
                    {cleaned_pred}
                    
                    只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                    """
                    
                    # 使用基类的LLM回退评分方法
                    score = self.get_llm_fallback_score(prompt, default_score=similarity_score)
                    self.log(f"  LLM评估的匹配度分数: {score:.4f}")
                else:
                    # 没有LLM，直接使用内容相似度作为分数
                    score = similarity_score
                    self.log(f"  使用内容相似度作为分数: {score:.4f}")
            
            # 保存样本得分
            metric_score_list.append(score)
        
        # 计算总体平均得分
        em_score = sum(metric_score_list) / len(metric_score_list) if metric_score_list else 0.0
        
        # 记录评估结果统计信息
        self.log(f"\n样本总数: {len(metric_score_list)}")
        self.log(f"匹配样本数: {sum(1 for s in metric_score_list if s > 0.8)}")
        self.log(f"精确匹配平均得分: {em_score:.4f}")
        self.log("======== ExactMatch 计算结束 ========\n")
        
        return {"em": em_score}, metric_score_list
    
    def _calculate_content_similarity(self, pred: str, golden: str) -> float:
        """
        计算两个文本的内容相似度
        
        使用Jaccard相似度和词覆盖率的加权组合来评估文本内容的相似性，
        特别适合处理自然语言答案中关键词匹配的情况。
        
        Args:
            pred: 预测答案，系统生成的回答
            golden: 标准答案，期望的正确回答
            
        Returns:
            float: 内容相似度分数 (0-1)，值越大表示相似度越高
        """
        # 标准化处理，移除标点符号、停用词等
        pred_norm = normalize_answer(pred).split()
        golden_norm = normalize_answer(golden).split()
        
        # 处理空文本情况
        if not pred_norm or not golden_norm:
            return 0.0
            
        # 计算共有词的数量
        common_words = set(pred_norm) & set(golden_norm)
        
        # 计算Jaccard相似度
        union_words = set(pred_norm) | set(golden_norm)
        if union_words:
            jaccard = len(common_words) / len(union_words)
        else:
            jaccard = 0.0
            
        # 计算词覆盖率：评估预测文本覆盖标准答案关键词的程度
        # 和标准答案覆盖预测文本关键词的程度
        pred_coverage = len(common_words) / len(set(pred_norm)) if pred_norm else 0
        golden_coverage = len(common_words) / len(set(golden_norm)) if golden_norm else 0
        
        # 综合得分 - Jaccard占40%，两个覆盖率各占30%
        similarity = 0.4 * jaccard + 0.3 * pred_coverage + 0.3 * golden_coverage
        
        return similarity

class F1Score(BaseMetric):
    """
    F1分数评估指标
    
    评估系统回答的精确率和召回率的综合表现，采用混合评分策略：
    1. 首先使用jieba分词器进行中文分词
    2. 计算标准F1分数（2*精确率*召回率/(精确率+召回率)）
    3. 如果有LLM，进行深度语义评估，选择最高分作为最终结果
    
    这种方法平衡了关键词匹配的精确性和语义理解的深度，
    特别适合评估中文自然语言答案的质量。
    """
    
    # 指标名称，用于在评估系统中唯一标识此指标
    metric_name = "f1"

    def __init__(self, config):
        """
        初始化F1分数评估器
        
        Args:
            config: 评估配置，包含LLM和其他参数设置
        """
        super().__init__(config)
        # 获取可选的LLM实例，用于深度语义评估
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data: AnswerEvaluationData) -> Tuple[Dict[str, float], List[float]]:
        """
        计算F1分数 - 使用规则匹配和LLM回退混合评分
        
        实现了基于分词的F1计算和LLM语义评估的混合方法，
        特别优化了中文文本处理，包括分词和停用词过滤，以提高评估准确性。
        
        Args:
            data: 评估数据，包含系统答案和标准答案
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 
                - 第一个元素：总体平均得分，格式为{"f1": 得分}
                - 第二个元素：每个样本的得分列表
        """
        # 记录评估过程日志
        self.log("\n======== F1Score 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        golden_answers = data.golden_answers
        system_answers = data.system_answers
        
        # 存储每个样本的F1得分
        f1_scores = []
        
        # 遍历所有样本
        for idx, (pred, golden) in enumerate(zip(system_answers, golden_answers)):
            # 预处理系统答案 - 移除Markdown标题和多余空行
            cleaned_pred = re.sub(r'^###.*?\n+', '', pred, flags=re.MULTILINE)
            cleaned_pred = re.sub(r'\n\s*\n', '\n', cleaned_pred)
            cleaned_pred = cleaned_pred.strip()
            
            # 将文本标准化，去除无关字符
            pred_text = normalize_answer(cleaned_pred)
            golden_text = normalize_answer(golden)
            
            # 记录处理过程
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  标准答案(前30字符): {golden[:30]}...")
            self.log(f"  系统答案(前30字符): {pred[:30]}...")
            
            # 尝试使用传统F1计算
            try:
                # 进行中文分词
                import jieba
                pred_tokens = list(jieba.cut(pred_text))
                golden_tokens = list(jieba.cut(golden_text))
                
                # 过滤停用词和过短的词，提高评估准确性
                stopwords = {'的', '了', '和', '在', '是', '为', '以', '与', '或', '且'}
                pred_tokens = [token for token in pred_tokens if len(token) > 1 and token not in stopwords]
                golden_tokens = [token for token in golden_tokens if len(token) > 1 and token not in stopwords]
                
                # 记录分词信息
                self.log(f"  标准答案分词数: {len(golden_tokens)}")
                self.log(f"  系统答案分词数: {len(pred_tokens)}")
                
                # 处理空文本情况
                if not pred_tokens or not golden_tokens:
                    # 空文本处理
                    if not pred_tokens and not golden_tokens:
                        rule_f1 = 1.0  # 两者都为空，视为完全匹配
                        self.log(f"  两者都为空，视为完全匹配，F1=1.0")
                    else:
                        rule_f1 = 0.0  # 一个为空一个不为空
                        self.log(f"  一个为空一个不为空，规则F1=0.0")
                else:
                    # 计算标准F1分数
                    # 找出共同词
                    common_tokens = set(pred_tokens) & set(golden_tokens)
                    # 计算精确率：正确识别的词占预测词的比例
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                    # 计算召回率：正确识别的词占标准答案词的比例
                    recall = len(common_tokens) / len(golden_tokens) if golden_tokens else 0
                    
                    # 计算F1分数：精确率和召回率的调和平均
                    if precision + recall > 0:
                        rule_f1 = 2 * precision * recall / (precision + recall)
                    else:
                        rule_f1 = 0.0
                    
                    # 记录详细计算信息
                    self.log(f"  共有词汇: {len(common_tokens)}/{len(set(pred_tokens) | set(golden_tokens))}")
                    self.log(f"  精确率: {precision:.4f}")
                    self.log(f"  召回率: {recall:.4f}")
                    self.log(f"  规则F1分数: {rule_f1:.4f}")
            except Exception as e:
                # 处理可能的异常，如分词失败
                self.log(f"  规则F1计算出错: {e}")
                rule_f1 = 0.0
            
            # 无论规则F1分数如何，如果有LLM都尝试使用LLM评估
            if self.llm:
                self.log(f"  尝试使用LLM评估内容相似度")
                
                # 构建内容相似度评估提示
                prompt = f"""
                请比较下面两个答案的内容相似度，评估它们包含的信息重叠程度，并给出0到1之间的分数。
                0表示完全不同信息，1表示信息完全重叠。
                请考虑实质内容的相似性，而不仅是表面文字的匹配。在评估时，请特别关注关键信息点是否一致。
                
                标准答案:
                {golden}
                
                系统答案:
                {cleaned_pred}
                
                只返回一个0到1之间的数字表示分数，不要有任何其他文字。
                """
                
                # 使用基类的LLM回退评分方法
                llm_f1 = self.get_llm_fallback_score(prompt, default_score=0.5)
                self.log(f"  LLM评估的F1分数: {llm_f1:.4f}")
                
                # 选择规则评分和LLM评分中的较高者作为最终得分
                if llm_f1 > rule_f1:
                    self.log(f"  LLM分数更高，采用LLM评估")
                    f1 = llm_f1
                else:
                    self.log(f"  规则F1分数更高，保留规则评估")
                    f1 = rule_f1
            else:
                # 没有LLM可用，使用规则F1分数
                f1 = rule_f1
            
            # 保存样本得分
            f1_scores.append(f1)
        
        # 计算总体平均得分
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # 记录评估结果统计信息
        self.log(f"\n样本总数: {len(f1_scores)}")
        self.log(f"F1得分大于0.5的样本数: {sum(1 for s in f1_scores if s > 0.5)}")
        self.log(f"F1平均得分: {avg_f1:.4f}")
        self.log("======== F1Score 计算结束 ========\n")
        
        return {"f1": avg_f1}, f1_scores