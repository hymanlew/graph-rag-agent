"""
答案验证模块

本模块提供了用于验证答案质量和估计查询复杂度的工具，是Graph-RAG系统中
确保回答质量和选择合适搜索策略的重要组件。

主要功能：
1. 答案质量验证 - 确保生成的答案满足基本质量要求
2. 查询复杂度评估 - 估计用户问题的复杂程度，用于选择合适的搜索策略

使用场景：
- 验证深度研究工具生成的答案是否符合质量标准
- 在搜索前评估问题复杂度，选择适当的搜索工具和参数
- 监控系统生成内容的质量，及时发现问题
- 为系统优化提供质量反馈数据

设计理念：
- 轻量级实现，易于集成
- 多维度评估，确保全面性
- 提供详细的验证日志，便于调试和优化
- 健壮的异常处理，提高系统稳定性
"""

from typing import Dict, List

class AnswerValidator:
    """
    答案验证器：评估生成答案的质量，确保满足基本要求
    
    该类实现了对生成答案的多维度验证，包括长度检查、错误模式检测和关键词相关性分析，
    确保返回给用户的答案满足基本质量标准，避免低质量或不相关的回答。
    
    核心验证维度：
    - 长度验证：确保答案有足够的内容
    - 错误模式检测：识别常见的错误提示和拒绝回答模式
    - 关键词相关性：确保答案与原始查询相关，包含重要关键词
    
    设计特点：
    - 模块化设计，易于扩展新的验证规则
    - 详细的验证日志，便于调试和质量分析
    - 与关键词提取系统集成，提供更准确的相关性评估
    - 灵活性强，可配置验证规则和阈值
    """
    
    def __init__(self, keyword_extractor=None):
        """
        初始化验证器
        
        参数:
            keyword_extractor: 用于提取关键词的函数或对象，用于相关性分析
        
        实现思路：
        1. 保存关键词提取器引用，用于后续相关性分析
        2. 定义常见的错误模式列表，用于检测低质量回答
        3. 初始化验证规则和阈值
        
        设计考量：
        - 关键词提取器可选，确保在无关键词提取功能时也能工作
        - 错误模式列表可扩展，根据实际情况添加新的模式
        - 保持初始化过程轻量级，便于快速实例化
        """
        self.keyword_extractor = keyword_extractor
        self.error_patterns = [
            "抱歉，处理您的问题时遇到了错误",
            "技术原因:",
            "无法获取",
            "无法回答这个问题",
            "没有找到相关信息",
            "对不起，我不能"
        ]
    
    def validate(self, query: str, answer: str) -> Dict[str, bool]:
        """
        验证生成答案的质量
        
        参数:
            query: 原始查询，用户提出的问题
            answer: 生成的答案，需要验证的内容
            
        返回:
            Dict[str, bool]: 各项验证的结果，包括：
            - length: 答案长度是否满足要求
            - no_error_patterns: 答案是否不包含错误模式
            - keyword_relevance: 答案与查询的关键词相关性
            - passed: 所有验证是否都通过
        
        实现思路：
        1. 创建结果字典，用于存储各项验证结果
        2. 执行长度验证，确保答案有足够内容
        3. 检查是否包含错误模式，识别低质量回答
        4. 执行关键词相关性检查，确保答案与问题相关
        5. 计算总体验证结果，只有所有检查都通过才算验证成功
        6. 记录详细的验证日志，便于调试和分析
        
        技术特点：
        - 多维度验证策略，全面评估答案质量
        - 详细的日志记录，便于问题排查
        - 灵活的验证规则，可根据需求调整
        - 高效的错误模式检测算法
        
        业务意义：
        - 确保返回给用户的答案满足基本质量标准
        - 过滤低质量或不相关的回答
        - 为系统优化提供质量反馈
        - 提高用户满意度和系统可信度
        """
        results = {}
        
        # 检查最小长度，确保答案有足够内容
        results["length"] = len(answer) >= 50
        if not results["length"]:
            print(f"[验证] 答案太短: {len(answer)}字符")
        
        # 检查是否包含错误模式，识别常见的失败回答
        results["no_error_patterns"] = not any(pattern in answer for pattern in self.error_patterns)
        if not results["no_error_patterns"]:
            for pattern in self.error_patterns:
                if pattern in answer:
                    print(f"[验证] 答案包含错误模式: {pattern}")
                    break
        
        # 关键词相关性检查，确保答案与查询相关
        results["keyword_relevance"] = self._check_keyword_relevance(query, answer)
        
        # 总体通过验证，只有所有检查都通过才算成功
        results["passed"] = all(results.values())
        
        return results
    
    def _check_keyword_relevance(self, query: str, answer: str) -> bool:
        """
        检查答案是否包含查询的关键词，评估相关性
        
        参数:
            query: 查询字符串，原始用户问题
            answer: 生成的答案，需要验证相关性的内容
            
        返回:
            bool: 是否满足关键词相关性要求
        
        实现思路：
        1. 检查是否有关键词提取器，如果没有则默认通过
        2. 从查询中提取高级和低级关键词
        3. 检查答案是否包含至少一个高级关键词（重要概念）
        4. 检查答案是否包含至少一半的低级关键词（具体细节）
        5. 记录详细的验证日志，包括通过情况和缺失的关键词
        6. 返回总体相关性评估结果
        
        技术特点：
        - 区分高级和低级关键词，优先检查重要概念
        - 基于比例的相关性评估，避免过度严格
        - 不区分大小写的匹配，提高匹配灵活性
        - 详细的日志记录，便于分析和优化
        
        业务意义：
        - 确保答案与用户问题直接相关
        - 避免生成不相关或偏离主题的内容
        - 保证回答的针对性和有用性
        - 提高用户满意度和系统质量
        """
        # 如果没有关键词提取器，则默认通过
        if not self.keyword_extractor:
            return True
            
        # 提取关键词，获取查询的核心概念
        keywords = self.keyword_extractor(query)
        if not keywords:
            return True
            
        # 分离高级和低级关键词
        high_level_keywords = keywords.get("high_level", [])
        low_level_keywords = keywords.get("low_level", [])
        
        # 至少有一个高级关键词应该在答案中出现（核心概念必须覆盖）
        if high_level_keywords:
            keyword_found = any(keyword.lower() in answer.lower() for keyword in high_level_keywords)
            if not keyword_found:
                print(f"[验证] 答案未包含任何高级关键词: {high_level_keywords}")
                return False
                
        # 至少有一半的低级关键词应该在答案中出现（细节覆盖率要求）
        if low_level_keywords and len(low_level_keywords) > 1:
            matches = sum(1 for keyword in low_level_keywords if keyword.lower() in answer.lower())
            if matches < len(low_level_keywords) / 2:
                print(f"[验证] 答案未包含足够的低级关键词: {matches}/{len(low_level_keywords)}")
                return False
        
        # 记录通过信息
        print("[验证] 答案通过关键词相关性检查")
        return True

def complexity_estimate(query: str) -> float:
    """
    估计查询复杂度，用于选择合适的搜索策略
    
    参数:
        query: 查询字符串，用户提出的问题
        
    返回:
        float: 复杂度评分(0.0-1.0)，数值越大表示问题越复杂
    
    实现思路：
    1. 进行输入验证，处理空值和非字符串输入
    2. 分析查询长度，较长的查询通常更复杂
    3. 统计问号数量，多问题通常更复杂
    4. 识别复杂问题关键词（如"为什么"、"如何"等）
    5. 综合计算复杂度评分，考虑长度、问题数量和关键词
    6. 确保评分在0-1范围内
    7. 异常处理，确保函数健壮性
    
    技术特点：
    - 基于启发式规则的复杂度评估
    - 多因素综合考虑，提高评估准确性
    - 完善的输入验证和异常处理
    - 评分范围标准化，便于后续处理
    
    业务意义：
    - 为系统提供问题复杂度的客观评估
    - 支持根据复杂度选择适当的搜索工具
    - 复杂问题使用深度研究工具，简单问题使用基础搜索
    - 优化资源分配，提高系统效率
    - 改善用户体验，针对不同复杂度问题提供合适的回答方式
    
    设计考量：
    - 保持轻量级实现，确保快速评估
    - 包含完善的边界条件处理
    - 提供合理的默认值，增强系统健壮性
    - 详细的日志记录，便于调试和优化
    """
    # 添加None检查和类型验证，确保函数健壮性
    if query is None:
        print(f'complexity_estimate: 返回0，因为query:{query}为空\n')
        return 0.0
    
    # 确保query是字符串类型
    if not isinstance(query, str):
        query = str(query) if query is not None else ""
    
    # 如果查询为空，返回最低复杂度
    if not query.strip():
        print(f'complexity_estimate: 返回0，因为query:{query}为空\n')
        return 0.0
    
    try:
        # 基于查询长度的复杂度因素，最长100个字符
        length_factor = min(1.0, len(query) / 100)
        
        # 统计问号数量，考虑中英文问号
        question_marks = query.count("?") + query.count("？")
        question_factor = min(1.0, question_marks * 0.2)  # 最多5个问号
        
        # 识别复杂问题的关键词列表
        complexity_indicators = [
            "为什么", "如何", "机制", "原因", "关系", "比较", "区别",
            "影响", "分析", "评估", "预测", "如果", "假设", "还是",
            "多少", "怎样", "多大", "是否", "哪些", "优缺点"
        ]
        
        # 检查复杂问题关键词的出现次数
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query)
        indicator_factor = min(1.0, indicator_count * 0.15)  # 最多约7个关键词
        
        # 综合评分，加权计算总体复杂度
        if all(factor is not None for factor in [length_factor, question_factor, indicator_factor]):
            # 权重分配：长度30%，问题数量30%，复杂关键词40%
            complexity = (length_factor * 0.3 + question_factor * 0.3 + indicator_factor * 0.4)
            return min(1.0, max(0.0, complexity))  # 确保在0-1范围内
        else:
            return 0.5  # 默认中等复杂度
            
    except Exception as e:
        # 异常处理，确保函数不会崩溃
        print(f"计算查询复杂度时出错: {e}")
        return 0.5  # 出错时返回默认值