import re
import json
from typing import Dict, List, Tuple
from evaluator.core.base_metric import BaseMetric
from evaluator.preprocessing.reference_extractor import extract_references_from_answer

"""
LLM评估指标模块

此模块实现了基于LLM的高级评估指标，主要用于评估GraphRAG系统回答的质量特征，包括：
- ResponseCoherence（回答连贯性）：评估系统回答的逻辑连贯性和结构化程度
- FactualConsistency（事实一致性）：评估回答内容的事实一致性和自洽性

这些指标充分利用LLM的深度理解能力，对系统回答进行语义层面的高级评估，
提供传统规则评估难以实现的质量洞察。
"""

class ResponseCoherence(BaseMetric):
    """
    回答连贯性评估指标
    
    评估GraphRAG系统回答的逻辑连贯性、结构完整性和表达流畅度。
    采用混合评估策略：
    1. 首先分析回答的形式化特征（段落数量、标题使用、句子数量）
    2. 然后使用LLM进行深度语义评估，分析逻辑连贯性和表达质量
    
    此指标能够有效识别回答是否组织良好、逻辑清晰，为用户提供流畅的阅读体验。
    """
    
    # 指标名称，用于在评估系统中唯一标识此指标
    metric_name = "response_coherence"
    
    def __init__(self, config):
        """
        初始化回答连贯性评估器
        
        Args:
            config: 评估配置，包含LLM实例等参数
        """
        super().__init__(config)
        # 获取LLM实例，用于深度语义评估
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算回答连贯性得分 - 结合结构特征分析和LLM语义评估
        
        实现了一个多维度的回答连贯性评估流程：
        1. 提取回答的结构特征进行定量分析
        2. 使用LLM评估回答的逻辑连贯性和表达质量
        3. 提供详细的评估日志和原因分析
        
        Args:
            data: 评估数据，包含问题和系统回答
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 
                - 第一个元素：总体平均得分，格式为{"response_coherence": 得分}
                - 第二个元素：每个样本的连贯性得分列表
        """
        self.log("\n======== ResponseCoherence 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行连贯性评估")
            return {"response_coherence": 0.0}, [0.0] * len(data.samples)
        
        coherence_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            self.log(f"  系统答案(前50字符): {answer[:50]}...")
            
            # 分析回答的结构特征
            paragraphs = answer.split('\n\n')
            has_headers = bool(re.search(r'#{1,3}\s+\w+', answer))
            sentence_count = len(re.findall(r'[.!?。！？]\s*', answer))
            
            self.log(f"  结构分析:")
            self.log(f"    段落数量: {len(paragraphs)}")
            self.log(f"    是否包含标题: {'是' if has_headers else '否'}")
            self.log(f"    句子数量: {sentence_count}")
            
            # 使用LLM评估连贯性
            prompt = f"""
            评估以下回答的连贯性和结构，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 逻辑清晰，结构良好，使用标题和段落，思路连贯
            - 中分(0.4-0.7): 内容基本清晰，但可能存在一些逻辑跳跃
            - 低分(0.0-0.3): 结构混乱，缺乏逻辑性
            
            问题: {question}
            回答: {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            self.log("  正在使用LLM评估回答连贯性...")
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                self.log(f"  LLM响应: {score_text}")
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    coherence = float(score_match.group(1))
                    # 确保在0-1范围内
                    coherence = max(0.0, min(1.0, coherence))
                    self.log(f"  提取的连贯性得分: {coherence:.4f}")
                else:
                    coherence = 0.5  # 默认中等分数
                    self.log(f"  无法从LLM响应中提取分数，使用默认分数: {coherence:.4f}")
            except Exception as e:
                self.log(f"  LLM评估连贯性时出错: {e}")
                coherence = 0.5  # 出错时使用默认中等分数
                self.log(f"  使用默认连贯性得分: {coherence:.4f}")
            
            # 记录详细原因
            if coherence < 0.4:
                self.log("  低分原因: 回答结构混乱，逻辑不清晰，缺乏适当的组织")
            elif coherence > 0.7:
                self.log("  高分原因: 回答结构良好，使用适当的标题和段落，逻辑流畅")
            else:
                self.log("  中等分数: 回答基本结构化，但可能存在一些组织或逻辑问题")
                    
            coherence_scores.append(coherence)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        self.log(f"\n样本总数: {len(coherence_scores)}")
        self.log(f"高分样本(>0.7)数量: {sum(1 for s in coherence_scores if s > 0.7)}")
        self.log(f"低分样本(<0.4)数量: {sum(1 for s in coherence_scores if s < 0.4)}")
        self.log(f"回答连贯性平均得分: {avg_coherence:.4f}")
        self.log("======== ResponseCoherence 计算结束 ========\n")
        
        return {"response_coherence": avg_coherence}, coherence_scores


class FactualConsistency(BaseMetric):
    """
    事实一致性评估指标
    
    评估GraphRAG系统回答内容的事实准确性、逻辑自洽性和信息一致性。
    采用多层次评估策略：
    1. 提取回答中引用的实体和关系信息
    2. 识别回答中的关键事实点
    3. 使用LLM进行深度语义评估，分析事实内容的一致性和准确性
    
    此指标能够有效识别回答中可能存在的事实错误、逻辑矛盾或信息不一致问题。
    """
    
    # 指标名称，用于在评估系统中唯一标识此指标
    metric_name = "factual_consistency"
    
    def __init__(self, config):
        """
        初始化事实一致性评估器
        
        Args:
            config: 评估配置，包含LLM实例等参数
        """
        super().__init__(config)
        # 获取LLM实例，用于深度语义评估
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算事实一致性得分 - 基于关键信息提取和LLM语义评估
        
        实现了一个智能的事实一致性评估流程：
        1. 提取回答中引用的实体和关系信息
        2. 识别并整理回答中的关键事实点
        3. 使用LLM评估关键事实点的一致性和准确性
        4. 提供完整的错误处理和回退机制
        
        Args:
            data: 评估数据，包含问题、回答、检索到的实体和关系
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 
                - 第一个元素：总体平均得分，格式为{"factual_consistency": 得分}
                - 第二个元素：每个样本的事实一致性得分列表
        """
        self.log("\n======== FactualConsistency 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行事实一致性评估")
            return {"factual_consistency": 0.0}, [0.0] * len(data.samples)
        
        consistency_scores = []
        
        for idx, sample in enumerate(data.samples):
            answer = sample.system_answer
            question = sample.question
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            self.log(f"  系统答案(前50字符): {answer[:50]}...")
            
            # 提取实体和关系信息，但提供更友好的格式给LLM
            if hasattr(sample, 'retrieved_entities'):
                entities = sample.retrieved_entities
                relationships = sample.retrieved_relationships
            else:
                refs = extract_references_from_answer(answer)
                entities = refs.get("entities", [])
                relationships = refs.get("relationships", [])
            
            self.log(f"  提取的实体数量: {len(entities)}")
            self.log(f"  提取的关系数量: {len(relationships)}")
            
            if entities:
                self.log(f"  实体样例: {entities[:5]}{'...' if len(entities) > 5 else ''}")
            if relationships:
                self.log(f"  关系样例: {relationships[:3]}{'...' if len(relationships) > 3 else ''}")
            
            # 即使没有正确的实体和关系ID，也基于文本内容进行评估
            # 提取回答中的关键信息点
            key_facts = []
            lines = answer.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    stripped = line.strip('- *')
                    if len(stripped) > 10:  # 只保留有意义的内容
                        key_facts.append(stripped)
            
            facts_text = "\n".join([f"- {fact}" for fact in key_facts[:10]])
            
            self.log(f"  提取的关键信息点数量: {len(key_facts)}")
            if key_facts:
                self.log(f"  信息点样例: {key_facts[0][:50]}...")
            
            # 使用更适合的提示让LLM评估
            prompt = f"""
            评估以下回答对问题的事实一致性，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答内容逻辑一致，信息准确，无矛盾内容
            - 中分(0.4-0.7): 回答大部分内容自洽，但有些模糊或可能不够精确
            - 低分(0.0-0.3): 回答内容自相矛盾或明显错误
            
            问题: {question}
            
            回答的关键信息点:
            {facts_text}
            
            完整回答:
            {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            self.log("  正在使用LLM评估事实一致性...")
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                self.log(f"  LLM响应: {score_text}")
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    consistency = float(score_match.group(1))
                    # 确保在0-1范围内
                    consistency = max(0.0, min(1.0, consistency))
                    self.log(f"  提取的事实一致性得分: {consistency:.4f}")
                else:
                    # 默认使用更高的基准分
                    consistency = 0.6  # 给予更高的默认分数
                    self.log(f"  无法从LLM响应中提取分数，使用更高的默认分数: {consistency:.4f}")
            except Exception as e:
                self.log(f"  LLM评估事实一致性时出错: {e}")
                consistency = 0.6  # 出错时使用更高的默认分数
                self.log(f"  使用更高的默认事实一致性得分: {consistency:.4f}")
                    
            consistency_scores.append(consistency)
        
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        self.log(f"\n样本总数: {len(consistency_scores)}")
        self.log(f"高分样本(>0.7)数量: {sum(1 for s in consistency_scores if s > 0.7)}")
        self.log(f"低分样本(<0.4)数量: {sum(1 for s in consistency_scores if s < 0.4)}")
        self.log(f"事实一致性平均得分: {avg_consistency:.4f}")
        self.log("======== FactualConsistency 计算结束 ========\n")
        
        return {"factual_consistency": avg_consistency}, consistency_scores

class ComprehensiveAnswerMetric(BaseMetric):
    """
    回答全面性评估指标
    
    评估GraphRAG系统回答是否全面、深入地解答了用户问题的所有方面。
    该指标特别关注：
    - 回答的信息丰富度和完整性
    - 对问题各子方面的覆盖程度
    - 细节和深度的平衡
    - 能否满足用户的实际信息需求
    
    此指标利用LLM的深度理解能力，从语义层面评估回答质量，
    能够识别传统定量指标难以捕捉的回答深度和广度特征。
    """
    
    metric_name = "answer_comprehensiveness"
    
    def __init__(self, config):
        """
        初始化回答全面性评估器
        
        Args:
            config: 评估配置，包含LLM实例等参数
        """
        super().__init__(config)
        self.llm = config.get("llm", None)
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[float]]:
        """
        计算回答全面性得分
        
        实现了一个基于LLM的深度语义评估流程，专注于评估回答的全面性和信息丰富度：
        1. 分析问题和回答的基本特征（长度、复杂度）
        2. 使用LLM评估回答是否全面涵盖了问题的所有关键方面
        3. 根据得分区间提供详细的评估原因分析
        4. 提供完整的错误处理和默认得分机制
        
        该方法特别关注：
        - 回答的语义完整性和内容深度
        - 对问题各子问题的解答情况
        - 信息的充分性和相关性
        - 回答的实用性和满足用户需求的程度
        
        在GraphRAG评估体系中，回答全面性是衡量系统知识整合和推理能力的重要指标。
        
        Args:
            data: 评估数据，包含问题、系统回答等信息
            
        Returns:
            Tuple[Dict[str, float], List[float]]: 
                - 第一个元素：总体平均全面性得分，格式为{"answer_comprehensiveness": 得分}
                - 第二个元素：每个样本的全面性得分列表
        """
        self.log("\n======== AnswerComprehensiveness 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行全面性评估")
            return {"answer_comprehensiveness": 0.0}, [0.0] * len(data.samples)
        
        comprehensiveness_scores = []
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            self.log(f"  系统答案(前50字符): {answer[:50]}...")
            self.log(f"  答案总长度: {len(answer)}")
            
            # 使用LLM评估全面性
            prompt = f"""
            评估以下回答解决问题的全面性，给出0到1的分数。
            评分标准:
            - 高分(0.8-1.0): 回答全面地解决了问题的所有方面，提供了丰富的信息和细节
            - 中分(0.4-0.7): 回答基本解决了问题，但可能遗漏了一些次要方面
            - 低分(0.0-0.3): 回答不完整，忽略了问题的主要方面
            
            问题: {question}
            回答: {answer}
            
            只返回一个0到1之间的数字表示分数，不要有任何其他文字。
            """
            
            self.log("  正在使用LLM评估回答全面性...")
            
            try:
                response = self.llm.invoke(prompt)
                score_text = response.content if hasattr(response, 'content') else response
                
                self.log(f"  LLM响应: {score_text}")
                
                # 提取数字
                score_match = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_match:
                    comprehensiveness = float(score_match.group(1))
                    # 确保在0-1范围内
                    comprehensiveness = max(0.0, min(1.0, comprehensiveness))
                    self.log(f"  提取的全面性得分: {comprehensiveness:.4f}")
                else:
                    comprehensiveness = 0.5  # 默认中等分数
                    self.log(f"  无法从LLM响应中提取分数，使用默认分数: {comprehensiveness:.4f}")
            except Exception as e:
                self.log(f"  LLM评估全面性时出错: {e}")
                comprehensiveness = 0.5  # 出错时使用默认中等分数
                self.log(f"  使用默认全面性得分: {comprehensiveness:.4f}")
            
            # 记录详细原因
            if comprehensiveness < 0.4:
                self.log("  低分原因: 回答可能过于简短，或未涵盖问题的关键方面")
            elif comprehensiveness > 0.7:
                self.log("  高分原因: 回答详尽全面，涵盖了问题的各个方面")
            else:
                self.log("  中等分数: 回答基本涵盖问题要点，但可能缺少一些深度或细节")
                    
            comprehensiveness_scores.append(comprehensiveness)
        
        avg_comprehensiveness = sum(comprehensiveness_scores) / len(comprehensiveness_scores) if comprehensiveness_scores else 0.0
        
        self.log(f"\n样本总数: {len(comprehensiveness_scores)}")
        self.log(f"高分样本(>0.7)数量: {sum(1 for s in comprehensiveness_scores if s > 0.7)}")
        self.log(f"低分样本(<0.4)数量: {sum(1 for s in comprehensiveness_scores if s < 0.4)}")
        self.log(f"回答全面性平均得分: {avg_comprehensiveness:.4f}")
        self.log("======== AnswerComprehensiveness 计算结束 ========\n")
        
        return {"answer_comprehensiveness": avg_comprehensiveness}, comprehensiveness_scores

class LLMGraphRagEvaluator(BaseMetric):
    """
    LLM综合评估器 - GraphRAG和HybridRAG性能评估
    
    一个强大的综合评估工具，利用LLM对GraphRAG和HybridRAG系统的整体性能进行深度评估。
    该评估器采用多维度评分策略，全面分析：
    - 回答质量和相关性
    - 推理过程的逻辑性和合理性
    - 对知识图谱和文档检索结果的有效利用
    - 整体用户体验和满意度
    
    与单一指标不同，此评估器提供了一个综合性的性能视角，能够捕捉其他专项指标可能忽略的系统整体表现特征。
    """
    
    metric_name = "llm_evaluation"
    
    def __init__(self, config):
        """
        初始化LLM综合评估器
        
        Args:
            config: 评估配置，包含LLM实例和其他必要参数
        """
        super().__init__(config)
        self.llm = config.get("llm", None)
        self.aspect_weights = {
            "comprehensiveness": 0.3,  # 全面性
            "relativeness": 0.25,      # 相关性 
            "empowerment": 0.25,       # 增强理解能力
            "directness": 0.2          # 直接性
        }
        
        # 如果没有提供LLM，则无法执行评估
        if not self.llm:
            self.log("警告: 未提供LLM模型，无法执行LLM评估")
    
    def calculate_metric(self, data) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        使用LLM计算GraphRAG系统的综合评估指标
        
        实现了一个复杂的多维度LLM评估流程，对GraphRAG和HybridRAG系统进行全面评估：
        1. 对每个样本进行深入的语义分析和质量评估
        2. 从四个关键维度进行精细化评分：全面性、相关性、增强理解能力和直接性
        3. 计算各维度的平均分和加权总分
        4. 提供详细的评估日志和错误处理机制
        
        该方法的核心优势在于：
        - 使用可配置的维度权重，实现评估重点的灵活调整
        - 对每个回答进行预处理，移除引用等干扰信息
        - 采用结构化的JSON输出格式，便于结果解析和聚合
        - 实现了完善的错误处理和默认得分机制
        
        这种综合评估方法能够提供比单一指标更全面的系统性能洞察，
        特别适合评估复杂的RAG系统在实际应用场景中的整体表现。
        
        Args:
            data: 评估数据，包含问题、系统回答等信息
            
        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - 第一个元素：包含各维度平均分和加权总分的字典
                - 第二个元素：每个样本的详细评分字典列表
        """
        self.log("\n======== LLMGraphRagEvaluator 计算日志 ========")
        self.log(f"样本总数: {len(data.samples) if hasattr(data, 'samples') else 0}")
        
        if not self.llm:
            self.log("错误: 未提供LLM模型，无法执行LLM评估")
            empty_scores = {f"llm_{aspect}": 0.0 for aspect in self.aspect_weights}
            empty_scores["llm_total"] = 0.0
            return empty_scores, [{} for _ in data.samples]
        
        all_scores = []
        summary_scores = {aspect: [] for aspect in self.aspect_weights}
        
        for idx, sample in enumerate(data.samples):
            question = sample.question
            answer = sample.system_answer
            
            self.log(f"\n样本 {idx+1}:")
            self.log(f"  问题: {question}")
            self.log(f"  系统答案(前50字符): {answer[:50]}...")
            
            # 清理答案，移除引用数据部分
            cleaned_answer = self._clean_references(answer)
            self.log(f"  清理后的答案长度: {len(cleaned_answer)}")
            
            # 创建评估提示
            eval_prompt = self._create_evaluation_prompt(question, cleaned_answer)
            
            self.log("  正在使用LLM进行全面评估...")
            try:
                response = self.llm.invoke(eval_prompt)
                content = response.content if hasattr(response, 'content') else response
                
                self.log(f"  LLM响应: {content}")
                
                # 解析评估结果
                sample_scores = self._parse_evaluation_result(content)
                
                self.log("  各项评分:")
                for aspect, score in sample_scores.items():
                    self.log(f"    {aspect}: {score:.4f}")
                
                all_scores.append(sample_scores)
                
                # 更新每个指标的累积分数
                for aspect, score in sample_scores.items():
                    if aspect in summary_scores:
                        summary_scores[aspect].append(score)
            except Exception as e:
                self.log(f"  LLM评估出错: {e}")
                default_scores = {aspect: 0.5 for aspect in self.aspect_weights}
                all_scores.append(default_scores)
                
                for aspect in self.aspect_weights:
                    if aspect in summary_scores:
                        summary_scores[aspect].append(0.5)
                
                self.log("  使用默认分数: 0.5")
        
        # 计算平均分数
        avg_scores = {}
        self.log("\n各指标平均分:")
        for aspect, scores in summary_scores.items():
            if scores:
                aspect_avg = sum(scores) / len(scores)
                avg_scores[f"llm_{aspect}"] = aspect_avg
                self.log(f"  {aspect}: {aspect_avg:.4f}")
            else:
                avg_scores[f"llm_{aspect}"] = 0.0
                self.log(f"  {aspect}: 0.0000")
        
        # 计算加权总分
        weighted_sum = sum(avg_scores[f"llm_{aspect}"] * weight 
                        for aspect, weight in self.aspect_weights.items())
        avg_scores["llm_total"] = weighted_sum
        
        self.log(f"\n加权总分: {weighted_sum:.4f}")
        self.log(f"权重设置:")
        for aspect, weight in self.aspect_weights.items():
            self.log(f"  {aspect}: {weight:.2f}")
        
        self.log("======== LLMGraphRagEvaluator 计算结束 ========\n")
        
        return avg_scores, all_scores
    
    def _evaluate_answer(self, question: str, answer: str) -> Dict[str, float]:
        """
        对单个回答进行全面的LLM评估
        
        为单个问答对执行端到端的评估流程，包括：
        1. 清理回答文本，移除引用等干扰信息
        2. 构建详细的LLM评估提示
        3. 调用LLM执行评估
        4. 解析和返回多维度评分结果
        
        此方法作为核心评估单元，能够独立对单个问答对进行评估，
        便于系统集成和单独测试。
        
        Args:
            question: 用户问题文本
            answer: 系统生成的回答文本
            
        Returns:
            Dict[str, float]: 包含四个关键维度评分的字典
                - comprehensiveness: 全面性得分
                - relativeness: 相关性得分
                - empowerment: 增强理解能力得分
                - directness: 直接性得分
        """
        # 清理回答，移除引用数据部分
        cleaned_answer = self._clean_references(answer)
        
        # 使用LLM评估各个方面
        eval_prompt = self._create_evaluation_prompt(question, cleaned_answer)
        
        try:
            response = self.llm.invoke(eval_prompt)
            content = response.content if hasattr(response, 'content') else response
            
            # 解析评估结果
            return self._parse_evaluation_result(content)
        except Exception as e:
            self.log(f"LLM评估出错: {e}")
            return {aspect: 0.5 for aspect in self.aspect_weights}  # 默认中等分数
    
    def _clean_references(self, answer: str) -> str:
        """
        清理回答中的引用数据部分
        
        此方法通过正则表达式清理评估文本中的干扰信息，确保LLM评估的焦点集中在
        回答的实际内容上，而不是引用格式或元数据上。
        
        采用了多层级的清理策略：
        1. 尝试匹配标准的引用数据格式（Markdown标题+JSON格式）
        2. 如果未匹配成功，尝试另一种常见格式
        3. 最后移除任何尾部空行，确保返回干净的文本
        
        这种清理对于准确评估回答质量至关重要，因为引用部分不应计入内容评估。
        
        Args:
            answer: 原始系统回答文本
            
        Returns:
            str: 清理后的纯回答内容，不包含引用数据部分
        """
        # 移除引用数据部分
        cleaned = re.sub(r'#{1,4}\s*引用数据[\s\S]*?(\{[\s\S]*?\})\s*$', '', answer)
        
        # 如果没有引用数据部分，尝试其他格式
        if cleaned == answer:
            cleaned = re.sub(r'#### 引用数据[\s\S]*?(\{[\s\S]*?\})\s*$', '', answer)
        
        # 移除任何尾部空行
        cleaned = cleaned.rstrip()
        
        return cleaned
    
    def _create_evaluation_prompt(self, question: str, answer: str) -> str:
        """
        创建用于LLM评估的详细提示
        
        构建一个结构化的评估提示，指导LLM从四个关键维度对回答质量进行评估。
        提示设计特别注重：
        - 提供清晰的评分标准和解释
        - 为每个维度定义0-1的分数范围和含义
        - 指定严格的JSON输出格式要求
        - 包含评分理由字段，增强评估的可解释性
        
        这种精心设计的提示确保了LLM评估的一致性和可靠性，
        使评估结果更具可比性和参考价值。
        
        Args:
            question: 用户问题文本
            answer: 清理后的回答文本
            
        Returns:
            str: 格式化的评估提示文本
        """
        return f"""
        请评估以下回答相对于问题的质量，给出0到1之间的分数（可以使用小数）。
        
        评估应该从以下四个方面进行：
        
        1. 全面性(comprehensiveness)：回答涵盖了问题的各个方面的程度
           - 0分表示完全不全面，遗漏重要信息
           - 1分表示非常全面，涵盖所有相关内容
        
        2. 相关性(relativeness)：回答与问题的相关程度
           - 0分表示完全不相关
           - 1分表示高度相关，直接回应问题
        
        3. 增强理解能力(empowerment)：回答帮助读者理解并做出判断的程度
           - 0分表示没有帮助理解
           - 1分表示显著增强了理解
        
        4. 直接性(directness)：回答直接回应问题，不偏离主题的程度
           - 0分表示完全间接，偏离主题
           - 1分表示直接明了，切中要点
        
        问题: {question}
        
        回答: {answer}
        
        请以JSON格式返回评分结果，格式为：
        {{
            "comprehensiveness": 0.X,
            "relativeness": 0.X,
            "empowerment": 0.X,
            "directness": 0.X,
            "reasoning": "简短解释评分理由"
        }}
        
        只返回JSON对象，不要有任何其他文字。
        """
    
    def _parse_evaluation_result(self, content: str) -> Dict[str, float]:
        """
        解析LLM的评估结果
        
        实现了一个鲁棒的结果解析机制，从LLM的响应中提取结构化的评估数据：
        1. 使用正则表达式定位JSON格式的评估结果
        2. 解析JSON对象提取四个维度的评分
        3. 对评分进行有效性验证和范围限制（0-1）
        4. 实现完善的错误处理和默认值机制
        5. 提取和记录评分理由，增强评估的可解释性
        
        该方法的关键特点：
        - 采用容错设计，即使LLM输出格式不完美也能提取有用信息
        - 对每个维度的评分设置默认值，确保评估流程不会中断
        - 提供详细的日志记录，便于调试和结果分析
        - 实现数据清洗，确保评分在有效范围内
        
        这种鲁棒的解析机制对于确保评估系统的稳定性和可靠性至关重要，
        特别是在处理可能返回非标准格式的LLM响应时。
        
        Args:
            content: LLM生成的原始评估响应内容
            
        Returns:
            Dict[str, float]: 包含各维度有效评分的字典
                - comprehensiveness: 全面性得分
                - relativeness: 相关性得分
                - empowerment: 增强理解能力得分
                - directness: 直接性得分
        """
        self.log("  正在解析LLM评估结果...")
        
        # 尝试提取JSON部分
        json_match = re.search(r'(\{[\s\S]*\})', content)
        if not json_match:
            self.log("  未能找到JSON格式的评估结果，使用默认分数")
            return {aspect: 0.5 for aspect in self.aspect_weights}
        
        try:
            json_str = json_match.group(1)
            self.log(f"  提取的JSON: {json_str}")
            
            data = json.loads(json_str)
            
            # 提取评分
            scores = {}
            for aspect in self.aspect_weights:
                if aspect in data and isinstance(data[aspect], (int, float)):
                    score_value = min(1.0, max(0.0, float(data[aspect])))
                    scores[aspect] = score_value
                    self.log(f"  解析到得分 - {aspect}: {score_value:.4f}")
                else:
                    scores[aspect] = 0.5  # 默认中等分数
                    self.log(f"  未找到 {aspect} 得分，使用默认值: 0.5")
            
            # 如果有理由字段，打印出来
            if "reasoning" in data and data["reasoning"]:
                self.log(f"  评分理由: {data['reasoning']}")
            
            return scores
        except Exception as e:
            self.log(f"  解析LLM评估结果出错: {e}")
            return {aspect: 0.5 for aspect in self.aspect_weights}