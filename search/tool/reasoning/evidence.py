"""
证据追踪模块

本模块实现了Graph-RAG系统中的证据链追踪功能，是确保深度研究过程可解释性和可靠性的核心组件。
该模块负责记录、组织和分析整个推理过程中的证据来源、推理步骤以及它们之间的关系，
为最终答案提供可追溯的证据支持，并能检测证据间的矛盾，提高系统输出的可信度。

主要功能：
1. 证据收集与管理：记录和组织各类证据来源
2. 推理链构建：追踪和维护完整的推理步骤序列
3. 可信度评估：为证据分配可信度评分
4. 矛盾检测：识别不同证据之间的冲突
5. 引用生成：自动为答案添加证据引用标记
6. 统计分析：提供推理过程的各种统计信息

使用场景：
- 深度研究工具中的证据管理
- 复杂问题解答的可解释性提供
- 多源信息整合与矛盾调和
- 推理过程的质量监控与评估

技术特点：
- 轻量级设计，高效的内存管理
- 灵活的证据索引机制，支持快速检索
- 强大的矛盾检测功能，结合规则和LLM
- 完善的统计和分析功能
- 支持多种类型的证据源
"""

from typing import Dict, List
import time
import hashlib

from model.get_models import get_llm_model

class EvidenceChainTracker:
    """
    证据链收集和推理跟踪器
    
    该类是Graph-RAG系统中的证据管理核心组件，负责收集、组织和分析深度研究过程中的
    所有证据和推理步骤，构建完整的可追溯证据链，确保推理过程的可解释性和结果的可信度。
    
    核心功能：
    - 记录完整的推理步骤序列
    - 管理各类证据来源及其元数据
    - 跟踪证据与推理步骤的关联关系
    - 评估证据的可信度
    - 检测证据间的矛盾和不一致
    - 为生成的答案提供引用支持
    - 提供推理过程的统计和分析
    
    设计理念：
    - 采用图状数据结构，支持复杂的证据关联
    - 轻量级实现，确保高效率运行
    - 提供完整的API支持证据链的构建和查询
    - 结合规则和LLM进行智能分析
    - 支持推理过程的可视化和解释
    """
    
    def __init__(self):
        """
        初始化证据链跟踪器
        
        实现思路：
        1. 初始化LLM模型，用于高级分析任务（如矛盾检测）
        2. 创建数据结构存储推理步骤、证据项、查询上下文等
        3. 初始化计数器和索引结构，优化数据访问
        4. 设置默认值和初始状态
        
        数据结构设计：
        - reasoning_steps: 列表存储所有推理步骤，按时间顺序排列
        - evidence_items: 字典存储所有证据项，按ID快速访问
        - query_contexts: 字典存储查询上下文信息，支持多查询并行处理
        - confidence_scores: 存储证据的可信度评分
        - contradictions: 记录检测到的矛盾信息
        - citation_index: 建立引用索引，支持快速查找相关证据
        
        业务意义：
        - 为深度研究工具提供完整的证据管理基础设施
        - 确保推理过程的可追溯性和可解释性
        - 支持对推理质量的评估和优化
        - 为最终答案提供可靠的证据支持
        """
        # 初始化LLM模型，用于高级分析任务
        self.llm = get_llm_model()
        
        # 核心数据结构初始化
        self.reasoning_steps = []  # 推理步骤列表，存储完整推理历史
        self.evidence_items = {}   # 证据项字典，ID为键
        self.query_contexts = {}   # 查询上下文，支持多查询管理
        self.step_counter = 0      # 步骤计数器，用于生成唯一ID
        self.confidence_scores = {}  # 证据的可信度评分
        self.contradictions = {}     # 记录相互矛盾的证据
        self.citation_index = {}     # 引用索引，优化证据检索
        
    def start_new_query(self, query: str, keywords: Dict[str, List[str]]) -> str:
        """
        开始新的查询跟踪
        
        参数:
            query: 用户查询，原始问题文本
            keywords: 查询关键词，包含高级和低级关键词的字典
            
        返回:
            str: 唯一的查询ID，用于后续引用和跟踪
        
        实现思路：
        1. 生成唯一的查询ID，使用MD5哈希+时间戳确保唯一性
        2. 创建查询上下文对象，存储原始查询、关键词和时间信息
        3. 初始化步骤ID列表，用于后续关联推理步骤
        4. 将查询上下文存储在跟踪器中
        5. 返回生成的查询ID
        
        技术特点：
        - 使用MD5哈希算法生成短ID，确保唯一性同时保持可读性
        - 结合时间戳防止重复，支持高并发场景
        - 保留前10个字符，平衡唯一性和简洁性
        - 完整记录查询元数据，便于后续分析
        
        业务意义：
        - 为每个查询创建独立的跟踪空间，支持多查询并行处理
        - 保存原始查询信息，作为推理过程的起点
        - 记录关键词，用于后续相关性分析
        - 提供时间戳，支持性能监控和分析
        - 建立查询与推理步骤的关联关系
        """
        # 生成查询ID，使用MD5哈希和时间戳确保唯一性
        query_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()[:10]
        
        # 存储查询上下文，包含完整的查询元数据
        self.query_contexts[query_id] = {
            "query": query,             # 原始查询文本
            "keywords": keywords,       # 查询相关的关键词
            "start_time": time.time(),  # 开始时间，用于性能分析
            "step_ids": []              # 关联的推理步骤ID列表
        }
        
        return query_id
    
    def add_reasoning_step(self, 
                         query_id: str, 
                         search_query: str, 
                         reasoning: str) -> str:
        """
        添加推理步骤
        
        参数:
            query_id: 查询ID，关联的查询标识符
            search_query: 搜索查询，该步骤使用的具体搜索语句
            reasoning: 推理过程，该步骤的思考和分析内容
            
        返回:
            str: 步骤ID，新创建的推理步骤的唯一标识符
        
        实现思路：
        1. 生成唯一的步骤ID，使用计数器确保唯一性
        2. 创建步骤记录对象，包含步骤ID、查询ID、搜索查询、推理内容等
        3. 初始化证据ID列表，用于后续关联证据
        4. 记录时间戳，用于步骤排序和性能分析
        5. 将步骤添加到全局推理步骤列表
        6. 在查询上下文中关联该步骤
        7. 返回生成的步骤ID
        
        技术特点：
        - 使用简单计数器生成连续步骤ID，易于跟踪和理解
        - 完整记录步骤元数据，支持后续分析
        - 建立查询与步骤的双向关联，优化数据访问
        - 时间戳记录，支持按时间顺序处理
        - 证据ID列表预初始化，提高代码健壮性
        
        业务意义：
        - 构建完整的推理链，记录解决问题的思考过程
        - 保存搜索查询，便于后续分析搜索策略效果
        - 记录推理内容，提供可解释性支持
        - 为证据关联提供容器
        - 支持多步骤协同解决复杂问题
        """
        # 生成步骤ID，使用计数器确保唯一性和顺序性
        step_id = f"step_{self.step_counter}"
        self.step_counter += 1
        
        # 创建步骤记录，包含完整的步骤信息
        step = {
            "step_id": step_id,           # 步骤唯一标识符
            "query_id": query_id,         # 关联的查询ID
            "search_query": search_query, # 该步骤使用的搜索查询
            "reasoning": reasoning,       # 该步骤的推理内容
            "evidence_ids": [],           # 关联的证据ID列表
            "timestamp": time.time()      # 创建时间戳
        }
        
        # 添加步骤到全局列表并关联到查询上下文
        self.reasoning_steps.append(step)
        if query_id in self.query_contexts:
            self.query_contexts[query_id]["step_ids"].append(step_id)
        
        return step_id
    
    def add_evidence(self, 
                   step_id: str, 
                   source_id: str, 
                   content: str, 
                   source_type: str) -> str:
        """
        添加证据项
        
        参数:
            step_id: 步骤ID，关联的推理步骤标识符
            source_id: 来源ID，如文档块ID、URL或其他唯一标识符
            content: 证据内容，原始文本数据
            source_type: 来源类型，如"chunk"、"web"、"graph"等
            
        返回:
            str: 证据ID，新创建的证据项的唯一标识符
        
        实现思路：
        1. 生成唯一的证据ID，使用来源ID和内容前缀的哈希值
        2. 创建证据记录对象，包含ID、来源、内容和类型信息
        3. 记录时间戳，用于证据时序分析
        4. 将证据存储在全局证据字典中
        5. 查找对应的推理步骤并关联该证据
        6. 返回生成的证据ID
        
        技术特点：
        - 使用来源ID和内容前缀生成证据ID，平衡唯一性和稳定性
        - 完整记录证据元数据，支持后续分析
        - 建立步骤与证据的双向关联，优化数据访问
        - 防止重复证据关联，提高数据一致性
        - 支持多种类型的证据源
        
        业务意义：
        - 记录和管理各类证据来源
        - 建立证据与推理步骤的关联关系
        - 为最终答案提供可靠的依据
        - 支持证据溯源和验证
        - 便于后续的证据分析和矛盾检测
        """
        # 生成证据ID，使用来源ID和内容前缀确保唯一性
        evidence_id = hashlib.md5(f"{source_id}:{content[:50]}".encode()).hexdigest()[:10]
        
        # 创建证据记录，包含完整的证据信息
        evidence = {
            "evidence_id": evidence_id,  # 证据唯一标识符
            "source_id": source_id,      # 原始来源ID
            "content": content,          # 证据文本内容
            "source_type": source_type,  # 来源类型标识
            "timestamp": time.time()     # 创建时间戳
        }
        
        # 存储证据到全局证据字典
        self.evidence_items[evidence_id] = evidence
        
        # 查找关联的推理步骤并添加证据引用
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                if evidence_id not in step["evidence_ids"]:
                    step["evidence_ids"].append(evidence_id)
                break
        
        return evidence_id

    def add_evidence_with_confidence(
        self, 
        step_id: str, 
        source_id: str, 
        content: str, 
        source_type: str, 
        confidence=0.5,
        metadata=None
    ):
        """
        添加带可信度得分的证据
        
        参数:
            step_id: 步骤ID，关联的推理步骤标识符
            source_id: 来源ID，如文档块ID、URL或其他唯一标识符
            content: 证据内容，原始文本数据
            source_type: 来源类型，如"chunk"、"web"、"graph"等
            confidence: 可信度评分(0-1)，默认为0.5（中等可信度）
            metadata: 元数据字典，包含额外的证据属性信息
            
        返回:
            str: 证据ID，新创建的证据项的唯一标识符
        
        实现思路：
        1. 调用基础add_evidence方法添加证据基本信息
        2. 存储证据的可信度评分，用于后续证据排序和优先级评估
        3. 添加可选的元数据，增强证据描述
        4. 更新引用索引，提取关键短语用于快速检索
        5. 返回生成的证据ID
        
        技术特点：
        - 复用基础证据添加逻辑，保持代码一致性
        - 支持可信度量化评估，便于证据排序和筛选
        - 灵活的元数据支持，适应各种证据类型
        - 自动更新引用索引，优化搜索性能
        - 默认可信度值，简化调用
        
        业务意义：
        - 支持证据质量的量化评估
        - 为后续推理提供证据优先级参考
        - 支持基于可信度的证据筛选
        - 丰富的元数据支持，便于证据分类和分析
        - 优化的引用索引，提高查询效率
        """
        # 调用基础方法添加证据基本信息
        evidence_id = self.add_evidence(step_id, source_id, content, source_type)
        
        # 保存可信度评分，用于证据质量评估和排序
        self.confidence_scores[evidence_id] = confidence
        
        # 添加元数据（如果提供），增强证据描述信息
        if metadata:
            if evidence_id in self.evidence_items:
                self.evidence_items[evidence_id]["metadata"] = metadata
        
        # 更新引用索引，便于后续快速检索和引用生成
        self._update_citation_index(evidence_id, content)
        
        return evidence_id
    
    def _update_citation_index(self, evidence_id: str, content: str):
        """
        更新引用索引
        
        这是一个私有方法，负责为新添加的证据建立关键词索引，支持后续高效的证据检索和引用生成。
        
        参数:
            evidence_id: 证据ID，要索引的证据唯一标识符
            content: 证据内容，要提取关键词的文本数据
        
        实现思路：
        1. 调用_extract_key_phrases方法从证据内容中提取关键短语
        2. 对每个提取的关键短语，在索引中建立与证据ID的映射关系
        3. 确保索引中不会重复存储同一证据ID
        4. 维护倒排索引结构，支持从关键词快速定位相关证据
        
        技术特点：
        - 基于倒排索引原理设计，优化关键词到证据的映射
        - 避免重复索引，保持索引数据的一致性
        - 高效的数据结构设计，支持快速查询
        - 与关键词提取系统集成，提供高质量索引
        - 私有方法设计，确保只有授权途径可以更新索引
        
        业务意义：
        - 支持高效的证据检索，提高查询速度
        - 为引用生成提供基础支持，确保答案引用的准确性
        - 优化关键词匹配，提高相关证据的发现率
        - 支持多证据关联，便于综合分析
        - 为矛盾检测提供数据基础
        """
        # 从证据内容中提取关键短语，作为索引项
        key_phrases = self._extract_key_phrases(content)
        
        # 更新倒排索引，建立关键词到证据ID的映射
        for phrase in key_phrases:
            # 如果关键词不在索引中，创建新条目
            if phrase not in self.citation_index:
                self.citation_index[phrase] = []
            # 避免重复添加同一证据ID到索引中
            if evidence_id not in self.citation_index[phrase]:
                self.citation_index[phrase].append(evidence_id)
    
    def _extract_key_phrases(self, content):
        """
        从文本中提取关键短语
        
        这是一个私有方法，负责从证据内容中提取有意义的关键词和短语，用于建立引用索引。
        实现了中英文混合的关键词提取，支持数值提取、英文名词短语和中文短语识别。
        
        参数:
            content: 文本内容，要提取关键词的证据文本
            
        返回:
            list: 关键短语列表，包含提取的数值、名词短语和中文短语
        
        实现思路：
        1. 预处理文本，将内容按句子边界分割
        2. 为每种类型的关键信息定义正则表达式模式
        3. 从每个句子中提取数值信息（如百分比、货币等）
        4. 提取符合规则的英文名词短语（首字母大写的多词短语）
        5. 使用滑动窗口方法提取中文短语
        6. 合并所有提取的短语，并去重和过滤
        
        技术特点：
        - 支持中英文混合文本处理
        - 使用正则表达式提取结构化信息
        - 针对中文采用滑动窗口策略，提高短语提取质量
        - 过滤过于简短的无意义短语
        - 去重处理，避免冗余索引
        
        业务意义：
        - 提取文本中的核心概念、数值信息和专业术语
        - 为证据检索提供高质量的索引项
        - 支持答案与证据的精确关联
        - 提高引用生成的准确性和相关性
        - 为矛盾检测提供关键词基础
        """
        # 导入正则表达式模块
        import re
        
        # 第一步：将文本按句子边界分割
        sentences = re.split(r'[.!?。！？]', content)
        
        # 初始化关键短语列表
        key_phrases = []
        
        # 定义数值模式，用于提取百分比、金额等数值信息
        number_pattern = r'\d+(?:[.,]\d+)?(?:\s*%|\s*元|\s*美元|\s*人民币)?'
        
        # 定义英文名词短语模式（匹配首字母大写的多词短语）
        noun_phrase_pattern = r'[A-Z][a-z]+\s+(?:[a-z]+\s+){0,2}[a-z]+'
        
        # 遍历每个句子，提取不同类型的关键短语
        for sentence in sentences:
            # 提取数值信息（如数量、百分比、金额等）
            numbers = re.findall(number_pattern, sentence)
            key_phrases.extend(numbers)
            
            # 提取英文名词短语（通常是专业术语或重要概念）
            noun_phrases = re.findall(noun_phrase_pattern, sentence)
            key_phrases.extend(noun_phrases)
            
            # 针对中文文本，使用滑动窗口方法提取短语
            if len(sentence) > 3:
                # 使用4字符窗口滑动提取中文短语
                for i in range(len(sentence) - 3):
                    phrase = sentence[i:i+4]
                    # 过滤掉空白过多的短语，确保提取的短语有实际意义
                    if len(phrase.strip()) >= 2:
                        key_phrases.append(phrase.strip())
        
        # 最后处理：去重并过滤过于简短的短语，保留最有意义的关键信息
        return list(set([p for p in key_phrases if len(p) > 1]))
    
    def detect_contradictions(self, evidence_ids):
        """
        检测证据之间的矛盾
        
        该方法负责识别指定证据列表中的不一致和冲突，是确保答案可靠性的核心机制。
        实现了两种矛盾检测方式：数值矛盾检测和语义矛盾检测，通过双重验证提高检测准确性。
        
        参数:
            evidence_ids: 证据ID列表，包含要分析的证据标识符
            
        返回:
            list: 矛盾信息列表，每个元素包含矛盾类型、相关证据和详细描述
            
        实现思路：
        1. 验证输入参数有效性，至少需要两个证据才能检测矛盾
        2. 获取所有有效证据对象，过滤掉不存在的证据ID
        3. 第一阶段：通过正则表达式检测数值类矛盾
           a. 提取每个证据中的数值及其上下文
           b. 比较不同证据中相同上下文的数值
           c. 如果发现显著差异，记录数值矛盾
        4. 第二阶段：使用LLM检测语义矛盾
           a. 跳过已发现数值矛盾的证据对，避免重复分析
           b. 调用LLM模型分析证据内容的语义冲突
           c. 记录语义矛盾信息
        5. 将检测到的矛盾保存到全局矛盾记录中
        6. 返回完整的矛盾信息列表
        
        技术特点：
        - 采用多策略检测方法，覆盖数值和语义两类矛盾
        - 使用上下文相似度判断确保数值比较的准确性
        - 利用LLM处理复杂语义冲突，提高检测能力
        - 避免重复分析，优化处理效率
        - 结构化记录矛盾信息，便于后续分析和展示
        
        业务意义：
        - 自动发现多源信息间的不一致，提高答案准确性
        - 为证据质量评估提供重要依据
        - 支持对推理过程的质量监控
        - 增强系统输出的可信度和可靠性
        - 帮助识别潜在的信息冲突，便于人工介入和审核
        """
        # 验证输入参数：至少需要两个证据才能检测矛盾
        if len(evidence_ids) < 2:
            return []
            
        # 初始化矛盾列表，存储检测结果
        contradictions = []
        
        # 获取有效证据对象，过滤掉不存在的证据ID
        evidences = [self.evidence_items[eid] for eid in evidence_ids if eid in self.evidence_items]
        
        # 第一阶段：通过正则表达式检测数值类矛盾
        import re
        for i in range(len(evidences)):
            for j in range(i+1, len(evidences)):
                # 提取第一个证据中的数值及其上下文
                content1 = evidences[i]["content"]
                numbers1 = self._extract_numbers_with_context(content1)
                
                # 提取第二个证据中的数值及其上下文
                content2 = evidences[j]["content"]
                numbers2 = self._extract_numbers_with_context(content2)
                
                # 比较数值：查找相同上下文中的数值差异
                for num1_info in numbers1:
                    for num2_info in numbers2:
                        # 检查上下文是否相似（相似度阈值设为0.7）
                        if self._context_similarity(num1_info["context"], num2_info["context"]) > 0.7:
                            # 检查数值是否存在显著差异
                            # 使用相对误差判断，避免绝对值比较造成的误判
                            if abs(num1_info["value"] - num2_info["value"]) > 0.001 * max(num1_info["value"], num2_info["value"]):
                                # 记录数值矛盾信息
                                contradictions.append({
                                    "type": "numerical",
                                    "evidence1": evidence_ids[i],
                                    "evidence2": evidence_ids[j],
                                    "context": num1_info["context"],
                                    "value1": num1_info["value"],
                                    "value2": num2_info["value"]
                                })
        
        # 第二阶段：使用LLM检测语义矛盾
        if hasattr(self, 'llm') and self.llm:
            for i in range(len(evidences)):
                for j in range(i+1, len(evidences)):
                    # 检查是否已经发现数值矛盾，避免重复分析
                    if any(c["evidence1"] == evidence_ids[i] and c["evidence2"] == evidence_ids[j] for c in contradictions):
                        continue
                    
                    # 提取两个证据的内容
                    content1 = evidences[i]["content"]
                    content2 = evidences[j]["content"]
                    
                    # 使用LLM检测语义层面的矛盾
                    contradiction = self._detect_semantic_contradiction(content1, content2, evidence_ids[i], evidence_ids[j])
                    if contradiction:
                        contradictions.append(contradiction)
        
        # 将检测到的矛盾保存到全局矛盾记录中
        for contradiction in contradictions:
            contradiction_id = f"contradiction_{len(self.contradictions)}"
            self.contradictions[contradiction_id] = contradiction
            
        # 返回完整的矛盾信息列表
        return contradictions
    
    def _extract_numbers_with_context(self, text):
        """
        从文本中提取数值和上下文
        
        这是一个私有方法，负责从文本中提取数值信息并保留其上下文，用于矛盾检测。
        该方法能够识别各种数值格式，包括带单位的数值（如百分比、货币等）。
        
        参数:
            text: 文本内容，要提取数值的证据文本
            
        返回:
            list: 包含数值和上下文的对象列表，每个元素包含解析后的数值、原始字符串和上下文
        
        实现思路：
        1. 定义正则表达式模式，匹配各种格式的数值（包括带单位的）
        2. 使用finditer查找文本中所有匹配的数值
        3. 对每个匹配进行处理：
           a. 提取原始数值字符串
           b. 清理非数字字符，转换为浮点数
           c. 提取数值前后的上下文（各20个字符）
           d. 构建包含数值、原始字符串和上下文的对象
        4. 收集所有提取的数值信息并返回
        
        技术特点：
        - 使用正则表达式精确匹配各种数值格式
        - 支持带单位的数值识别（百分比、货币等）
        - 错误处理机制，确保转换失败不会中断整个过程
        - 保留数值上下文，便于后续比较和分析
        - 提取的数值标准化为浮点数，便于计算和比较
        
        业务意义：
        - 为数值矛盾检测提供结构化的数值数据
        - 保留上下文信息，确保比较的相关性和准确性
        - 支持多种数值格式，适应不同来源的证据
        - 提高矛盾检测的精度，特别是对定量信息的检测
        - 为后续的统计分析提供基础数据
        """
        import re
        
        # 定义数值模式，支持整数、小数以及带单位的数值（百分比、货币等）
        number_pattern = r'(\d+(?:[.,]\d+)?(?:\s*%|\s*元|\s*美元|\s*人民币)?)'
        
        # 查找文本中所有匹配的数值
        matches = list(re.finditer(number_pattern, text))
        results = []
        
        # 处理每个匹配的数值
        for match in matches:
            # 获取原始数值字符串
            value_str = match.group(1)
            
            # 清理非数字字符并转换为浮点数
            clean_value = re.sub(r'[^\d.,]', '', value_str).replace(',', '.')
            try:
                value = float(clean_value)
            except:
                # 如果转换失败，跳过该数值
                continue
            
            # 提取数值前后的上下文（各20个字符）
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            
            # 构建结果对象，包含数值、原始字符串和上下文
            results.append({
                "value": value,       # 解析后的浮点数
                "original": value_str, # 原始数值字符串
                "context": context    # 数值的上下文
            })
            
        return results
    
    def _context_similarity(self, context1, context2):
        """
        计算两个上下文的相似度
        
        这是一个私有方法，用于计算两个文本片段的相似度，主要用于确定数值是否在相同上下文中被提及，
        从而进行有效的矛盾检测。采用了Jaccard相似度算法，简单高效地衡量文本相似度。
        
        参数:
            context1: 第一个上下文，文本字符串
            context2: 第二个上下文，文本字符串
            
        返回:
            float: 相似度得分(0-1)，0表示完全不相似，1表示完全相似
        
        实现思路：
        1. 将两个上下文文本转换为小写并分词
        2. 将分词结果转换为集合，去除重复词
        3. 处理空集合边界情况，避免除以零错误
        4. 计算两个词集合的交集大小（共同词数）
        5. 计算两个词集合的并集大小（总不同词数）
        6. 计算Jaccard相似度：交集大小 / 并集大小
        7. 返回相似度得分
        
        技术特点：
        - 采用Jaccard相似度算法，简单高效
        - 忽略单词顺序，关注词汇重叠度
        - 转换为小写，实现大小写不敏感的比较
        - 处理边界情况，提高鲁棒性
        - 计算效率高，适合实时处理
        
        业务意义：
        - 准确识别同一概念在不同证据中的提及
        - 确保数值比较的相关性和准确性
        - 减少误判，提高矛盾检测的精确度
        - 为数值矛盾检测提供关键的上下文关联
        - 支持跨文档的信息一致性验证
        """
        # 实现简单高效的基于单词重叠的相似度计算
        # 转换为小写并分词，忽略大小写差异
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        # 处理边界情况：如果任一上下文为空，返回相似度为0
        if not words1 or not words2:
            return 0
            
        # 计算Jaccard相似度
        # 交集大小：两个文本共有的不同单词数量
        intersection = len(words1.intersection(words2))
        # 并集大小：两个文本中所有不同单词的总数
        union = len(words1.union(words2))
        
        # 计算并返回相似度得分，避免除以零错误
        return intersection / union if union > 0 else 0
    
    def _detect_semantic_contradiction(self, content1, content2, evidence_id1, evidence_id2):
        """
        使用LLM检测语义矛盾
        
        这是一个私有方法，利用LLM的自然语言理解能力来检测两个证据内容之间的语义层面矛盾，
        弥补了基于规则的数值矛盾检测的不足。该方法能够识别复杂的语义不一致，包括逻辑冲突、
        事实陈述矛盾、观点对立等非数值类矛盾。
        
        参数:
            content1: 第一个内容，证据的文本内容
            content2: 第二个内容，证据的文本内容
            evidence_id1: 第一个证据ID，用于引用和关联
            evidence_id2: 第二个证据ID，用于引用和关联
            
        返回:
            Dict或None: 矛盾信息字典或None（如果没有检测到矛盾）
        
        实现思路：
        1. 构建专用提示模板，引导LLM分析两段内容的一致性
        2. 将两个证据的内容填入提示模板
        3. 调用LLM模型进行分析
        4. 解析LLM响应，提取矛盾信息
        5. 判断是否存在矛盾（基于特定标记词）
        6. 如果存在矛盾，处理分析结果并格式化返回
        7. 如果不存在矛盾，返回None
        
        技术特点：
        - 利用LLM的高级自然语言理解能力
        - 结构化提示模板，提高分析准确性
        - 灵活处理不同格式的LLM响应
        - 内容长度限制，避免过长输出
        - 返回结构化的矛盾信息，便于后续处理
        
        业务意义：
        - 识别复杂的语义层面矛盾，超越简单规则检测
        - 提高矛盾检测的全面性和准确性
        - 为用户提供详细的矛盾原因说明
        - 增强系统对不一致信息的敏感度
        - 为推理过程中的证据质量评估提供支持
        """
        # 构建专用提示模板，引导LLM进行矛盾分析
        prompt = f"""
        分析以下两段内容，判断它们之间是否存在矛盾或不一致：
        
        内容1:
        {content1}
        
        内容2:
        {content2}
        
        如果存在矛盾，请具体说明矛盾点。如果不存在矛盾，请回答"没有矛盾"。
        """
        
        # 调用LLM模型进行分析
        response = self.llm.invoke(prompt)
        # 处理不同格式的响应，确保正确提取内容
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        # 判断是否发现矛盾（基于特定标记词）
        if "没有矛盾" in analysis:
            return None
        
        # 提取并处理矛盾点分析结果
        contradiction_point = analysis.replace("矛盾点：", "").strip()
        # 限制内容长度，避免过长输出
        if len(contradiction_point) > 300:
            contradiction_point = contradiction_point[:300] + "..."
            
        # 返回结构化的矛盾信息
        return {
            "type": "semantic",      # 矛盾类型：语义矛盾
            "evidence1": evidence_id1, # 第一个相关证据ID
            "evidence2": evidence_id2, # 第二个相关证据ID
            "analysis": contradiction_point # 详细的矛盾分析
        }
    
    def generate_citations(self, answer):
        """
        在答案中生成引用标记
        
        该方法负责自动为生成的答案添加证据引用标记，增强回答的可信度和可追溯性。
        它通过分析答案内容，查找匹配的证据来源，并在适当位置添加引用标记，最后生成
        完整的引用列表。这是实现可解释AI的关键功能之一。
        
        参数:
            answer: 答案文本，需要添加引用的原始回答
            
        返回:
            Dict: 包含带引用的答案和引用信息的字典，结构为：
                  {
                      "cited_answer": 带引用标记的答案文本,
                      "citations": 引用信息列表
                  }
        
        实现思路：
        1. 初始化引用列表
        2. 从答案中提取关键语句，作为引用定位点
        3. 对每个关键语句，查找最匹配的证据
        4. 构建引用信息，包括语句、证据ID、来源ID和可信度
        5. 将引用信息添加到引用列表
        6. 调用_add_citations_to_answer方法在答案中插入引用标记
        7. 返回带引用的答案和引用信息
        
        技术特点：
        - 自动识别关键语句作为引用目标
        - 基于关键词索引快速匹配相关证据
        - 考虑证据可信度，优先使用高质量证据
        - 结构化的引用信息，便于后续处理
        - 与引用索引系统紧密集成
        
        业务意义：
        - 提高生成答案的可信度和权威性
        - 支持用户验证答案的信息来源
        - 实现推理过程的可解释性
        - 帮助用户了解答案的依据
        - 增强系统输出的专业性和规范性
        """
        # 初始化引用列表
        citations = []
        
        # 从答案中提取关键语句，作为引用的定位点
        key_statements = self._extract_key_statements(answer)
        
        # 为每个关键语句查找最匹配的证据
        for statement in key_statements:
            matching_evidence = self._find_matching_evidence(statement)
            if matching_evidence:
                # 构建引用信息
                citation = {
                    "statement": statement,        # 被引用的语句
                    "evidence_id": matching_evidence["evidence_id"], # 匹配的证据ID
                    "source_id": matching_evidence["source_id"],    # 原始来源ID
                    "confidence": self.confidence_scores.get(matching_evidence["evidence_id"], 0.5) # 证据可信度
                }
                citations.append(citation)
        
        # 生成带引用标记的最终答案
        cited_answer = self._add_citations_to_answer(answer, citations)
        
        # 返回带引用的答案和引用信息
        return {
            "cited_answer": cited_answer,  # 带引用标记的答案
            "citations": citations         # 完整的引用信息列表
        }
    
    def _extract_key_statements(self, text):
        """
        从文本中提取关键语句
        
        这是一个私有方法，负责从答案文本中提取有意义的关键语句，作为引用生成的基础。
        该方法通过句子分割、合并和长度过滤，确保只选择有实质性内容的语句。
        
        参数:
            text: 文本内容，要提取关键语句的答案文本
            
        返回:
            list: 关键语句列表，包含筛选后的有意义语句
        
        实现思路：
        1. 使用正则表达式按标点符号（.!?。！？）分割文本为句子
        2. 设计巧妙的合并算法，将分割的句子与标点符号重新组合
        3. 过滤掉过短的句子，保留长度大于10个字符的有意义语句
        4. 去除句子前后空白，返回处理后的关键语句列表
        
        技术特点：
        - 支持中英文混合文本的句子分割
        - 保留句子的标点符号，维持原始表达
        - 使用正则表达式实现高效分割
        - 智能合并算法，确保句子完整性
        - 长度过滤，去除无意义的简短语句
        
        业务意义：
        - 自动识别答案中的核心陈述内容
        - 为引用生成提供准确的定位点
        - 避免对无意义内容添加引用
        - 确保引用的质量和相关性
        - 提高引用生成的效率和准确性
        """
        import re
        
        # 按句子边界分割文本，同时保留分隔符
        sentences = re.split(r'([.!?。！？]\s*)', text)
        
        # 智能合并句子和分隔符，确保句子完整性
        merged_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                # 合并句子和其后的标点符号
                merged_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                # 处理最后一个句子（可能没有分隔符）
                merged_sentences.append(sentences[i])
                i += 1
        
        # 筛选出有意义的句子（长度大于10个字符）
        # 去除空白并过滤，确保只保留有实质性内容的语句
        key_statements = [s.strip() for s in merged_sentences if len(s.strip()) > 10]
        
        return key_statements
    
    def _find_matching_evidence(self, statement):
        """
        为语句查找最匹配的证据
        
        这是一个私有方法，负责为答案中的关键语句查找最相关的证据，是引用生成系统的核心组件。
        该方法通过关键词匹配、频率统计和可信度加权，计算每个候选证据的相关性得分，
        并返回得分最高的证据。
        
        参数:
            statement: 语句，答案中的关键陈述内容
            
        返回:
            Dict或None: 匹配的证据对象或None（如果没有找到匹配的证据）
        
        实现思路：
        1. 从语句中提取关键短语
        2. 使用引用索引查找包含这些关键短语的证据ID
        3. 收集所有候选证据ID
        4. 如果没有候选证据，返回None
        5. 对每个候选证据计算匹配得分：
           a. 基础得分基于关键短语的出现频率
           b. 使用证据可信度进行加权
           c. 计算最终得分（基础得分 × 可信度）
        6. 选择得分最高的证据返回
        
        技术特点：
        - 基于倒排索引的高效检索
        - 多因素评分机制，综合考虑关键词匹配和证据质量
        - 使用可信度加权，优先选择高质量证据
        - 频率统计，识别更相关的证据
        - 去重处理，避免重复计算
        
        业务意义：
        - 为答案中的每个关键陈述找到最相关的证据支持
        - 确保引用的准确性和相关性
        - 考虑证据质量，优先使用高可信度证据
        - 支持引用生成的自动化和准确性
        - 增强答案的可解释性和说服力
        """
        # 从语句中提取关键短语，作为检索依据
        key_phrases = self._extract_key_phrases(statement)
        
        # 使用引用索引查找可能匹配的证据
        candidate_evidence_ids = []
        for phrase in key_phrases:
            if phrase in self.citation_index:
                # 添加包含该短语的所有证据ID
                candidate_evidence_ids.extend(self.citation_index[phrase])
        
        # 如果没有找到候选证据，返回None
        if not candidate_evidence_ids:
            return None
            
        # 计算每个候选证据的匹配得分
        evidence_scores = {}
        # 去重处理，避免重复计算
        for evidence_id in set(candidate_evidence_ids):
            if evidence_id in self.evidence_items:
                # 计算得分：关键词出现频率 × 证据可信度
                # 基础得分基于证据ID在候选列表中的出现频率
                base_score = candidate_evidence_ids.count(evidence_id)
                # 使用证据可信度进行加权
                confidence = self.confidence_scores.get(evidence_id, 0.5)
                # 计算最终得分
                evidence_scores[evidence_id] = base_score * confidence
        
        # 找出得分最高的证据
        if evidence_scores:
            best_evidence_id = max(evidence_scores, key=evidence_scores.get)
            return self.evidence_items[best_evidence_id]
        
        return None
    
    def _add_citations_to_answer(self, answer, citations):
        """
        在答案中添加引用标记
        
        这是一个私有方法，负责将引用标记插入到答案文本中的适当位置，并在答案末尾添加完整的
        引用列表。该方法采用了巧妙的排序策略，确保引用标记的正确插入，避免替换冲突。
        
        参数:
            answer: 原始答案，未经引用标记的文本
            citations: 引用信息列表，包含语句、证据ID和来源ID等信息
            
        返回:
            str: 带引用的答案，包含引用标记和引用列表
        
        实现思路：
        1. 复制原始答案，避免修改原始数据
        2. 对引用列表按语句长度从长到短排序，优先处理长语句，避免替换冲突
        3. 遍历排序后的引用列表，为每个引用生成引用标记（如[1], [2]等）
        4. 在答案中查找对应的语句，并在其后添加引用标记
        5. 如果存在引用，在答案末尾添加引用列表
        6. 返回带引用标记的完整答案
        
        技术特点：
        - 采用长度排序策略，有效避免替换冲突
        - 精确的字符串替换，确保引用标记的准确位置
        - 格式良好的引用列表，符合学术规范
        - 智能处理空引用情况，避免格式错误
        - 保留原始答案内容，只添加必要的引用标记
        
        业务意义：
        - 将抽象的引用信息转化为用户可读的格式
        - 增强答案的可信度和可追溯性
        - 提供清晰的信息来源说明
        - 使答案符合学术和专业标准
        - 方便用户验证和深入了解答案依据
        """
        # 复制原始答案，避免直接修改原始数据
        cited_answer = answer
        
        # 按语句长度从长到短排序，优先处理长语句，避免替换冲突
        sorted_citations = sorted(citations, key=lambda x: len(x["statement"]), reverse=True)
        
        # 遍历排序后的引用列表，添加引用标记
        for i, citation in enumerate(sorted_citations):
            statement = citation["statement"]
            source_id = citation["source_id"]
            # 生成引用标记，如 [1], [2] 等
            citation_mark = f"[{i+1}]"
            
            # 在答案中查找对应的语句，并在其后添加引用标记
            if statement in cited_answer:
                cited_answer = cited_answer.replace(statement, f"{statement}{citation_mark}")
        
        # 添加引用列表，格式化为Markdown样式
        if citations:
            cited_answer += "\n\n#### 引用\n"
            for i, citation in enumerate(citations):
                cited_answer += f"[{i+1}] {citation['source_id']}\n"
            
        return cited_answer
    
    def get_reasoning_chain(self, query_id: str) -> Dict:
        """
        获取完整的推理链
        
        该方法是推理链追踪系统的核心功能，负责组装和返回与特定查询关联的完整推理过程信息。
        它将推理步骤、关联证据、可信度评分等数据整合为结构化的推理链对象，为最终答案
        生成和推理过程可视化提供基础。
        
        参数:
            query_id: 查询ID，要获取推理链的查询唯一标识符
            
        返回:
            Dict: 完整的推理链信息字典，包含查询内容、关键词、时间信息、推理步骤和矛盾统计
                 每个推理步骤包含关联的详细证据和可信度评分
        
        实现思路：
        1. 验证查询ID是否存在于查询上下文字典中
        2. 获取与查询关联的所有步骤ID
        3. 遍历步骤ID列表，查找对应的步骤详情
        4. 对每个步骤，收集关联的完整证据信息并添加可信度评分
        5. 按时间戳对收集到的步骤进行排序
        6. 统计与当前推理链相关的矛盾数量
        7. 构建并返回完整的推理链字典
        
        技术特点：
        - 结构化数据组装，将分散的信息整合为完整视图
        - 深度嵌套数据处理，确保证据与步骤的正确关联
        - 时间序列排序，保证推理过程的时序一致性
        - 矛盾信息交叉引用，提供质量控制指标
        - 优雅的空值处理，增强系统鲁棒性
        
        业务意义：
        - 提供完整的推理过程追溯，增强系统透明度
        - 为答案生成提供结构化的推理依据
        - 支持推理过程可视化，帮助用户理解系统思考路径
        - 为质量评估和问题诊断提供数据基础
        - 实现推理过程的可解释性，增强用户信任
        """
        if query_id not in self.query_contexts:
            return {}
        
        # 获取查询相关的步骤ID
        step_ids = self.query_contexts[query_id]["step_ids"]
        
        # 按时间顺序收集步骤
        steps = []
        for step_id in step_ids:
            for step in self.reasoning_steps:
                if step["step_id"] == step_id:
                    # 复制步骤并添加完整证据
                    step_copy = step.copy()
                    step_copy["evidence"] = []
                    
                    # 添加证据详情
                    for evidence_id in step["evidence_ids"]:
                        if evidence_id in self.evidence_items:
                            evidence_copy = self.evidence_items[evidence_id].copy()
                            # 添加可信度评分
                            evidence_copy["confidence"] = self.confidence_scores.get(evidence_id, 0.5)
                            step_copy["evidence"].append(evidence_copy)
                    
                    steps.append(step_copy)
                    break
        
        # 按时间戳排序
        steps.sort(key=lambda x: x["timestamp"])
        
        # 构建完整推理链
        reasoning_chain = {
            "query": self.query_contexts[query_id]["query"],
            "keywords": self.query_contexts[query_id]["keywords"],
            "start_time": self.query_contexts[query_id]["start_time"],
            "end_time": time.time(),
            "steps": steps,
            "contradiction_count": len([c for c in self.contradictions.values() 
                                     if any(c.get("evidence1", "") == e_id or c.get("evidence2", "") == e_id 
                                         for e_id in 
                                         [e for s in steps for e in s["evidence_ids"]])
                                    ])
        }
        
        return reasoning_chain
    
    def get_step_evidence(self, step_id: str) -> List[Dict]:
        """
        获取特定步骤的证据
        
        该方法负责检索与特定推理步骤相关联的所有证据项，是构建可解释推理链的重要组件。
        通过这个方法，系统能够追踪每个推理步骤所依赖的具体证据来源，为用户提供透明的
        推理过程视图。
        
        参数:
            step_id: 步骤ID，要检索证据的推理步骤唯一标识符
            
        返回:
            List[Dict]: 证据列表，包含与该步骤关联的所有有效证据对象
            每个证据对象包含证据ID、来源ID、内容、类型等完整信息
        
        实现思路：
        1. 遍历全局推理步骤列表，查找与指定步骤ID匹配的步骤
        2. 如果找到匹配步骤，初始化空的证据列表
        3. 遍历步骤中关联的所有证据ID
        4. 检查每个证据ID是否存在于全局证据字典中
        5. 如果证据存在，将其添加到证据列表中
        6. 返回收集到的证据列表；如果未找到步骤，返回空列表
        
        技术特点：
        - 线性查找算法，适用于中等规模的步骤数据
        - 直接返回证据对象引用，避免不必要的数据复制
        - 空值处理，确保即使步骤不存在也能返回有效响应
        - 类型标注，增强代码可读性和IDE支持
        - 结构化返回格式，便于后续处理和展示
        
        业务意义：
        - 提供推理步骤与证据的关联查询
        - 支持针对特定推理环节的证据审查
        - 为步骤级推理质量评估提供数据支持
        - 增强推理过程的可解释性
        - 便于用户验证每个推理步骤的依据
        """
        # 查找步骤
        for step in self.reasoning_steps:
            if step["step_id"] == step_id:
                # 收集证据
                evidence_list = []
                for evidence_id in step["evidence_ids"]:
                    if evidence_id in self.evidence_items:
                        evidence_list.append(
                            self.evidence_items[evidence_id]
                        )
                return evidence_list
        
        return []
    
    def summarize_reasoning(self, query_id: str) -> Dict:
        """
        总结推理过程
        
        该方法负责生成整个推理过程的摘要信息，提供对复杂推理链的高层次概览。
        它通过分析推理步骤、证据数量、处理时间等关键指标，生成结构化的推理
        摘要，帮助用户快速了解系统的思考过程和主要发现。
        
        参数:
            query_id: 查询ID，要总结的推理过程的唯一标识符
            
        返回:
            Dict: 推理摘要字典，包含查询内容、步骤数量、证据数量、处理时间、
                 矛盾数量和关键步骤等统计信息
        
        实现思路：
        1. 调用get_reasoning_chain获取完整推理链数据
        2. 如果未找到推理链，返回提示信息
        3. 计算推理步骤总数和证据总数的统计信息
        4. 识别关键步骤（按证据数量排序，取前3个）
        5. 计算整个推理过程的处理时间
        6. 构建并返回包含所有统计信息的摘要字典
        
        技术特点：
        - 复用推理链数据，避免重复计算
        - 智能识别关键步骤，突出重要推理环节
        - 多维度统计，提供全面的推理过程视图
        - 结构化数据返回，便于前端展示和分析
        - 异常处理，确保即使在数据不完整的情况下也能正常运行
        
        业务意义：
        - 提供推理过程的高效概览
        - 帮助用户快速理解系统的思考路径
        - 突出显示关键推理步骤和证据
        - 为推理质量评估提供量化指标
        - 支持系统性能监控和优化
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"summary": "没有找到相关推理链"}
        
        # 计算统计信息
        steps_count = len(chain.get("steps", []))
        evidence_count = sum(len(step.get("evidence", [])) 
                           for step in chain.get("steps", []))
        
        # 识别关键步骤（有最多证据的步骤）
        key_steps = []
        if steps_count > 0:
            # 按证据数量排序
            sorted_steps = sorted(
                chain.get("steps", []),
                key=lambda x: len(x.get("evidence", [])),
                reverse=True
            )
            
            # 取前3个关键步骤
            key_steps = sorted_steps[:min(3, len(sorted_steps))]
        
        # 计算处理时间
        duration = chain.get("end_time", time.time()) - chain.get("start_time", time.time())
        
        # 生成摘要
        summary = {
            "query": chain.get("query", ""),
            "steps_count": steps_count,
            "evidence_count": evidence_count,
            "duration_seconds": duration,
            "contradiction_count": chain.get("contradiction_count", 0),
            "key_steps": [
                {
                    "step_id": step.get("step_id"),
                    "search_query": step.get("search_query"),
                    "evidence_count": len(step.get("evidence", []))
                }
                for step in key_steps
            ]
        }
        
        return summary
    
    def get_evidence_source_stats(self, query_id: str) -> Dict:
        """
        获取证据来源统计
        
        该方法负责统计特定查询的推理过程中各类证据来源的分布情况，帮助分析信息来源的多样性
        和可靠性。通过对证据来源类型的统计分析，可以评估推理过程的信息广度和深度，为质量
        控制提供重要参考。
        
        参数:
            query_id: 查询ID，要统计的推理过程的唯一标识符
            
        返回:
            Dict: 证据来源统计字典，格式为{"sources": {"来源类型1": 数量, "来源类型2": 数量, ...}}
            其中键为来源类型（如"chunk"、"web"、"graph"等），值为该类型的证据数量
        
        实现思路：
        1. 调用get_reasoning_chain获取完整推理链数据
        2. 如果未找到推理链，返回空统计信息
        3. 初始化空的证据集合，用于收集所有步骤的证据
        4. 遍历所有推理步骤，收集步骤中的证据
        5. 按证据来源类型对收集到的证据进行分组统计
        6. 返回包含来源统计信息的字典
        
        技术特点：
        - 高效的数据聚合和统计分析
        - 支持多种证据来源类型的自动识别
        - 结构化的数据返回格式
        - 基于推理链数据的二次分析，避免重复数据收集
        - 空值处理，确保在各种情况下都能正常运行
        
        业务意义：
        - 评估推理过程中信息来源的多样性和丰富度
        - 帮助识别信息获取的偏好和盲点
        - 为系统改进提供数据支持，如平衡不同来源的证据
        - 支持对信息可靠性的间接评估（不同来源类型可能有不同的可信度特征）
        - 为用户提供证据构成的透明视图
        """
        chain = self.get_reasoning_chain(query_id)
        if not chain:
            return {"sources": {}}
        
        # 收集所有证据
        all_evidence = []
        for step in chain.get("steps", []):
            all_evidence.extend(step.get("evidence", []))
        
        # 按来源类型分组
        sources = {}
        for evidence in all_evidence:
            source_type = evidence.get("source_type", "unknown")
            if source_type not in sources:
                sources[source_type] = 0
            sources[source_type] += 1
        
        return {"sources": sources}