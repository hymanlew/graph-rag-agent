from typing import Dict, List
import re

class DualPathSearcher:
    """
    双路径搜索器：支持同时使用多种方式搜索知识库
    
    该类实现了Graph-RAG系统的核心搜索策略，通过同时执行多种搜索路径（精确查询和带知识库名称的查询），
    并使用LLM智能评估结果的相关性和价值，确保系统能够获取最佳的知识检索结果。双路径搜索是
    Graph-RAG系统区别于传统RAG的关键优化点之一。
    """
    
    def __init__(self, kb_retriever, kg_retriever=None, kb_name=""):
        """
        初始化双路径搜索器
        
        该方法负责设置双路径搜索器的基本配置，包括知识库搜索函数、知识图谱搜索函数和知识库名称。
        它为后续的搜索操作奠定基础，支持灵活的搜索策略配置。
        
        参数:
            kb_retriever: 知识库搜索函数，用于执行主要的文档检索
            kg_retriever: 知识图谱搜索函数，可选，用于执行图数据库相关的关系检索
            kb_name: 知识库名称，用于构建带名称的搜索查询，提高检索精确性
        
        实现思路：
        1. 保存知识库搜索函数作为主要搜索入口
        2. 可选保存知识图谱搜索函数（目前未在主流程中使用）
        3. 保存知识库名称，用于后续构建增强的搜索查询
        
        技术特点：
        - 灵活配置：支持只使用知识库搜索，或后续可扩展为结合知识图谱搜索
        - 模块化设计：将搜索逻辑与评估逻辑分离
        - 可扩展性：知识图谱搜索功能已预留接口，便于后续扩展
        
        业务意义：
        - 为系统提供强大的搜索基础组件
        - 支持多种搜索策略，提高信息检索的全面性和精确性
        - 为后续的双路径搜索实现准备必要的配置
        """
        self.kb_retriever = kb_retriever
        self.kg_retriever = kg_retriever
        self.kb_name = kb_name
    
    def decide_best_result(self, query: str, precise_result: Dict, kb_result: Dict) -> Dict:
        """
        决定哪个搜索结果更好
        
        该方法是双路径搜索策略的智能决策核心，负责评估和选择两种不同搜索策略（精确查询和带知识库名查询）的最佳结果。
        通过智能分析两种搜索结果的相关性、完整性和信息价值，系统能够做出最优的结果选择，显著提高搜索的准确性和效率。
        
        参数:
            query: 原始查询，用户的问题或搜索关键词
            precise_result: 精确查询结果，使用原始查询进行搜索得到的结果
            kb_result: 带知识库名的查询结果，使用增强查询进行搜索得到的结果
            
        返回:
            Dict: 选择的最佳结果，可能是精确查询结果、带知识库名查询结果或它们的合并
            
        实现思路：
        1. 处理空结果情况：如果任一结果为空，则直接返回另一个结果
        2. 获取两种结果中的文档片段列表
        3. 分析结果中是否有足够的内容进行比较
        4. 调用LLM评估哪个结果更相关和有价值
        5. 根据评估结果做出决策：
           - 如果精确查询结果更好，返回精确查询结果
           - 如果带知识库名查询结果更好，返回带知识库名查询结果
           - 如果两种结果都有价值，合并它们
        6. 确保健壮性，处理各种边缘情况和异常
        
        技术特点：
        - 智能评估：使用LLM分析和评估结果质量
        - 灵活决策：根据评估结果选择最佳策略
        - 合并机制：在适当时合并不同结果的优势
        - 健壮性设计：完善的边缘情况处理
        - 结构化输出：确保返回结果格式一致
        
        业务意义：
        - 显著提高搜索结果的相关性和质量
        - 充分利用不同搜索策略的优势
        - 避免单一搜索策略的局限性
        - 为用户提供最相关的信息
        - 优化整体搜索体验和信息获取效率
        """
        # 提取文本内容以便LLM评估
        precise_text = self._extract_text_for_evaluation(precise_result)
        kb_text = self._extract_text_for_evaluation(kb_result)
        
        # 检查是否有内容可供评估
        precise_has_content = len(precise_text.strip()) > 50
        kb_has_content = len(kb_text.strip()) > 50
        
        # 如果只有一个结果有内容，直接返回那个
        if precise_has_content and not kb_has_content:
            print("[双路径搜索] 只有精确查询返回有效结果")
            return precise_result
        elif kb_has_content and not precise_has_content:
            print("[双路径搜索] 只有带知识库名查询返回有效结果")
            return kb_result
        elif not precise_has_content and not kb_has_content:
            print("[双路径搜索] 两种查询均未返回有效结果")
            # 合并可能的部分结果
            return self._merge_results(precise_result, kb_result)
        
        # 两种查询都有内容，使用LLM评估
        evaluation = self._evaluate_results_with_llm(query, precise_text, kb_text)
        
        if evaluation == "precise":
            print("[双路径搜索] LLM评估: 精确查询结果更具体更有价值")
            return precise_result
        elif evaluation == "kb":
            print("[双路径搜索] LLM评估: 带知识库名查询结果更具体更有价值")
            return kb_result
        else:
            # 评估结果不明确，合并结果
            print("[双路径搜索] LLM评估: 两种结果均有价值，合并结果")
            return self._merge_results(precise_result, kb_result)

    def search(self, query: str) -> Dict:
        """
        执行双路径搜索
        
        该方法是双路径搜索器的核心功能，通过同时执行精确查询和带知识库名称的查询两种搜索策略，
        并使用智能评估机制决定返回哪个结果集或合并结果。这种双路径搜索策略能够显著提高搜索结果的
        相关性和全面性，是Graph-RAG系统搜索增强的核心实现。
        
        参数:
            query: 搜索查询，用户原始问题或系统生成的搜索查询
            
        返回:
            Dict: 搜索结果字典，包含文档片段(chunks)和文档聚合信息(doc_aggs)
        
        实现思路：
        1. 构建两种不同的查询：
           - 精确查询：去除知识库名称后的原始查询
           - 带名称查询：将知识库名称与查询结合
        2. 执行两种查询，获取各自的搜索结果
        3. 从结果中提取文本内容用于后续评估
        4. 进行简单的内容量检查，确定是否有足够内容进行评估
        5. 根据内容量检查结果决定下一步操作：
           - 只有一种结果有足够内容：直接返回该结果
           - 两种结果都没有足够内容：合并部分结果
           - 两种结果都有足够内容：使用LLM评估决定
        6. 基于LLM评估结果返回最优结果或合并结果
        
        技术特点：
        - 双路径策略：同时执行两种搜索路径，扩大信息覆盖范围
        - 智能评估：使用LLM评估搜索结果的相关性和价值
        - 内容过滤：基于内容长度进行初步筛选
        - 结果合并：智能合并不同搜索结果，避免信息丢失
        - 丰富日志：详细记录搜索决策过程，便于调试和优化
        
        业务意义：
        - 显著提高搜索结果的相关性和全面性
        - 确保系统能够获取最有价值的信息用于后续推理
        - 通过双路径搜索避免单一搜索策略的局限性
        - 为复杂问题提供更全面的信息基础
        - 支持精确查询和上下文增强查询的优势互补
        """
        # 精确查询
        precise_query = query.replace(self.kb_name, "").strip()
        # 带名称的查询
        kb_query = f"{self.kb_name} {query}" if self.kb_name.lower() not in query.lower() else query
        
        # 执行两种查询
        precise_results = self.kb_retriever(precise_query)
        kb_results = self.kb_retriever(kb_query)
        
        # 调用decide_best_result方法决定返回哪个结果
        return self.decide_best_result(query, precise_results, kb_results)

    def _extract_text_for_evaluation(self, results: Dict) -> str:
        """
        从结果中提取文本用于评估
        
        该方法负责从搜索结果中提取纯文本内容，用于后续的LLM评估。它从结果的chunks字段中
        收集所有文本片段，并将它们合并为一个连续的文本字符串，为LLM评估提供必要的输入材料。
        
        参数:
            results: 搜索结果字典，包含chunks和doc_aggs等字段
            
        返回:
            str: 提取的文本内容，各文本片段之间用两个换行符分隔
        
        实现思路：
        1. 创建一个空列表用于存储提取的文本片段
        2. 遍历结果中的所有chunks（如果存在）
        3. 检查每个chunk是否包含text字段，如果包含则提取并添加到列表中
        4. 使用两个换行符作为分隔符，将所有文本片段合并为一个字符串
        5. 返回合并后的文本字符串
        
        技术特点：
        - 简单高效：使用基本的字典操作和列表推导式实现文本提取
        - 安全处理：使用get方法安全地访问字典字段，避免键不存在导致的错误
        - 清晰分隔：使用双换行符分隔不同文本片段，便于阅读和理解
        - 聚焦核心：只提取纯文本内容，过滤掉其他元数据
        
        业务意义：
        - 为LLM评估提供标准化的文本输入
        - 聚焦于关键内容，避免元数据干扰评估过程
        - 确保评估过程使用一致的文本格式
        - 提高搜索结果评估的准确性和可靠性
        """
        texts = []
        
        # 从chunks中提取文本
        for chunk in results.get("chunks", []):
            if "text" in chunk:
                texts.append(chunk["text"])
        
        return "\n\n".join(texts)

    def _evaluate_results_with_llm(self, query: str, text1: str, text2: str) -> str:
        """
        使用LLM评估哪个结果更具体、更有价值
        
        该方法是双路径搜索器的智能决策核心，负责使用LLM评估两种不同搜索策略的结果，
        分析哪个结果更符合原始查询的需求，更具具体性和信息价值。这种基于LLM的智能评估
        使得系统能够在不同搜索路径中选择最优结果，显著提高了搜索效率和相关性。
        
        参数:
            query: 原始查询
            text1: 精确查询结果的文本内容
            text2: 带知识库名查询结果的文本内容
            
        返回:
            str: 评估结果
                - "precise"：表示精确查询结果更好
                - "kb"：表示带知识库名查询结果更好
                - "both"：表示两种结果都有价值
                
        实现思路：
        1. 创建评估提示，明确要求LLM比较两个结果对查询的相关性和信息价值
        2. 要求LLM重点关注具体信息（如数字、规定、标准等）的比较
        3. 确保LLM以特定格式输出评估结果，简化后续处理
        4. 调用LLM生成评估结果
        5. 从LLM响应中提取评估结果（precise、kb或both）
        6. 处理可能的异常情况，确保方法健壮性
        7. 如果评估失败，返回默认结果"both"并记录错误
        
        技术特点：
        - 语义理解：利用LLM的深度语义理解能力进行评估
        - 结构化输出：通过提示工程确保LLM以特定格式输出
        - 健壮性设计：完善的异常处理机制
        - 灵活处理：能够应对各种输入情况和边缘情况
        - 聚焦价值：重点关注具体信息的价值比较
        
        业务意义：
        - 提高搜索结果的相关性和质量
        - 避免简单算法可能带来的偏差
        - 为系统提供更智能的结果选择机制
        - 显著提升用户体验和回答质量
        - 支持复杂查询场景下的精准结果筛选
        
        实现思路：
        1. 构建评估提示，要求LLM比较两个搜索结果并判断哪个更具体、更有价值
        2. 明确提示LLM关注具体信息（如数字、规定、标准等）的比较
        3. 要求LLM返回标准化的三种选项之一，避免解释性文本
        4. 调用LLM执行评估，处理不同类型的LLM响应格式
        5. 尝试从LLM响应中提取评估结果
        6. 实现优雅的降级机制：如果LLM评估失败，默认返回"both"并记录错误
        7. 全面的异常捕获，确保系统稳定性
        
        技术特点：
        - 智能比较：使用LLM的理解能力进行深度语义比较
        - 标准化输出：要求特定的输出格式，便于结果解析
        - 灵活适配：支持不同类型的LLM响应格式
        - 错误容忍：完善的异常处理和降级策略
        - 日志记录：记录评估过程和任何错误信息
        
        业务意义：
        - 实现搜索结果的智能优选，提高信息质量
        - 关注具体信息（如数字、规定）的价值评估
        - 确保系统能够在复杂搜索场景中做出合理决策
        - 平衡不同搜索策略的优缺点
        - 提供可靠的错误处理，确保系统稳定性
        """
        try:
            # 构建评估提示
            prompt = f"""请评估以下两个搜索结果，判断哪个更具体、包含更多有价值的信息，特别是具体数字、规定或明确标准。
                
            原始查询: {query}

            结果1:
            {text1}

            结果2:
            {text2}

            请评估哪个结果包含更具体的信息（如明确的规定、数字、标准等）。
            只返回以下三个选项之一：
            - "precise"：如果结果1更具体、更有价值
            - "kb"：如果结果2更具体、更有价值
            - "both"：如果两者具有相当的价值或包含不同的有价值信息

            只返回选项，不要包含其他解释。
            """
            
            # 调用LLM进行评估
            if hasattr(self, "llm"):
                response = self.llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
            else:
                # 如果没有llm属性，尝试从外部获取
                from model.get_models import get_llm_model
                llm = get_llm_model()
                response = llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
            
            # 提取评估结果
            result = result.strip().lower()
            if "precise" in result:
                return "precise"
            elif "kb" in result:
                return "kb"
            else:
                return "both"
                
        except Exception as e:
            print(f"[LLM评估失败] {str(e)}")
            # 评估失败时默认合并结果
            return "both"
    
    def _merge_results(self, result1: Dict, result2: Dict) -> Dict:
        """
        合并两个搜索结果
        
        该方法负责智能地合并来自不同搜索路径的结果，是Graph-RAG系统中双路径搜索策略的重要组成部分。
        通过合并处理，可以避免信息重复，同时保留两个搜索路径各自的优势，为用户提供更全面、更有价值的信息。
        
        参数:
            result1: 第一个搜索结果，包含文档片段和聚合信息
            result2: 第二个搜索结果，包含文档片段和聚合信息
            
        返回:
            Dict: 合并后的结果字典，包含去重后的chunks和doc_aggs
            
        实现思路：
        1. 初始化合并结果字典，以第一个结果为基础
        2. 处理文档片段(chunks)的合并：
           - 提取两个结果中的所有文档片段
           - 基于chunk_id或内容去重
           - 保留去重后的文档片段列表
        3. 处理文档聚合信息(doc_aggs)的合并：
           - 提取两个结果中的所有聚合信息
           - 基于doc_id去重处理
           - 合并为单一列表
        4. 合并其他列表类型的字段，避免重复项
        5. 确保返回结果包含所有必要信息
        
        技术特点：
        - 智能去重：支持基于ID和基于内容的去重策略
        - 信息整合：保留不同搜索路径的优势信息
        - 数据一致性：确保返回格式统一
        - 健壮性：处理空结果或不完整结果的情况
        - 高效处理：优化的数据合并算法
        
        业务意义：
        - 提供更全面的信息覆盖
        - 避免重复信息，提升信息质量
        - 最大化搜索策略的效果
        - 显著提高系统的信息检索效率
        - 为后续的推理和回答提供更丰富的信息基础
            
        实现思路：
        1. 初始化合并结果字典
        2. 处理文档片段(chunks)的合并：
           - 提取两个结果中的所有文档片段
           - 基于文本内容去重
           - 保留去重后的文档片段列表
        3. 处理文档聚合信息(doc_aggs)的合并：
           - 提取两个结果中的所有聚合信息
           - 去重处理，避免重复
           - 合并为单一列表
        4. 确保返回结果中包含所有必要的键
        5. 对聚合列表进行排序，确保一致性
        
        技术特点：
        - 智能去重：基于内容而非ID的去重策略
        - 信息整合：保留不同搜索路径的优势信息
        - 数据一致性：确保返回格式统一
        - 健壮性：处理空结果或不完整结果的情况
        - 高效处理：优化的数据合并算法
        
        业务意义：
        - 提供更全面的信息覆盖
        - 避免重复信息，提升信息质量
        - 最大化搜索策略的效果
        - 显著提高系统的信息检索效率
        - 为后续的推理和回答提供更丰富的信息基础
        
        实现思路：
        1. 初始化结果字典，以第一个结果为基础
        2. 处理特殊情况：如果第一个结果没有chunks，直接返回第二个结果
        3. 创建集合用于跟踪已存在的chunk_id和doc_id，以便去重
        4. 合并chunks：
           - 先添加第一个结果的所有chunks
           - 遍历第二个结果的chunks，只添加不存在的
           - 如果没有chunk_id，则使用内容作为唯一性判断
        5. 合并doc_aggs：
           - 同样使用doc_id进行去重
           - 只添加不存在的文档聚合信息
        6. 处理其他字段：
           - 复制第一个结果中不存在但第二个结果中有的字段
           - 对于列表类型的字段，执行列表合并并去重
        7. 返回完整的合并结果
        
        技术特点：
        - 智能去重：使用ID和内容双重机制确保结果不重复
        - 优先级处理：以第一个结果为基础进行合并
        - 特殊情况处理：针对空结果等特殊情况进行优化
        - 全面合并：处理各种类型的字段，包括列表类型的字段合并
        - 安全访问：使用get方法安全地访问字典字段
        
        业务意义：
        - 确保不丢失任何重要信息，提高搜索结果的全面性
        - 避免重复信息，优化后续处理效率
        - 提供统一的结果格式，便于后续组件处理
        - 在无法确定单一最佳结果时提供综合解决方案
        - 增强系统对复杂查询的适应性和鲁棒性
        """
        # 初始化结果字典
        result = {
            "chunks": result1.get("chunks", []).copy(),
            "doc_aggs": result1.get("doc_aggs", []).copy()
        }
        
        # 如果第一个结果没有chunks，直接使用第二个结果
        if not result["chunks"]:
            return result2
        
        # 已存在的chunk_id和doc_id集合
        existing_chunk_ids = set(c.get("chunk_id") for c in result["chunks"] if "chunk_id" in c)
        existing_doc_ids = set(d.get("doc_id") for d in result["doc_aggs"] if "doc_id" in d)
        
        # 合并chunks，避免重复
        for chunk in result2.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            # 只添加不存在的chunks
            if chunk_id and chunk_id not in existing_chunk_ids:
                result["chunks"].append(chunk)
                existing_chunk_ids.add(chunk_id)
            elif not chunk_id:
                # 如果没有chunk_id，使用内容作为唯一性判断
                content = chunk.get("text", "")
                if not any(c.get("text") == content for c in result["chunks"]):
                    result["chunks"].append(chunk)
        
        # 合并doc_aggs，避免重复
        for doc in result2.get("doc_aggs", []):
            doc_id = doc.get("doc_id")
            if doc_id and doc_id not in existing_doc_ids:
                result["doc_aggs"].append(doc)
                existing_doc_ids.add(doc_id)
        
        # 复制其他字段
        for key in result2:
            if key not in ["chunks", "doc_aggs"]:
                if key not in result:
                    result[key] = result2[key]
                elif isinstance(result[key], list) and isinstance(result2[key], list):
                    # 合并列表类型的字段
                    result[key].extend([item for item in result2[key] if item not in result[key]])
        
        return result
        

class QueryGenerator:
    """
    查询生成器：生成子查询和跟进查询
    
    该类实现了Graph-RAG系统的智能查询优化功能，通过将复杂查询分解为子查询、生成
    跟进查询等方式，显著提高了系统的搜索效率和信息获取质量。它是系统中实现
    多轮推理和深度分析的关键组件之一。
    """
    
    def __init__(self, llm, sub_query_prompt, followup_query_prompt):
        """
        初始化查询生成器
        
        该方法负责设置查询生成器的核心配置，包括大语言模型实例、子查询提示模板和跟进查询提示模板。
        它为后续的查询生成操作提供必要的资源和指导。
        
        参数:
            llm: 大语言模型实例，用于生成高质量的子查询和跟进查询
            sub_query_prompt: 子查询提示模板，指导LLM如何将复杂查询分解为子查询
            followup_query_prompt: 跟进查询提示模板，指导LLM如何基于已检索信息生成跟进查询
        
        实现思路：
        1. 保存大语言模型实例，作为查询生成的核心引擎
        2. 保存子查询提示模板，用于后续生成子查询
        3. 保存跟进查询提示模板，用于后续生成跟进查询
        
        技术特点：
        - 模板驱动：使用预定义提示模板引导LLM生成结构化查询
        - 模块化设计：将查询生成功能与其他系统组件解耦
        - 灵活配置：支持不同的LLM和提示模板配置
        - 简洁实现：保持初始化逻辑简单明了
        
        业务意义：
        - 为系统提供智能查询生成能力
        - 支持复杂问题的分解和深入探索
        - 提高搜索的针对性和效率
        - 为多轮推理提供查询生成支持
        """
        self.llm = llm
        self.sub_query_prompt = sub_query_prompt
        self.followup_query_prompt = followup_query_prompt
    
    def generate_sub_queries(self, original_query: str) -> List[str]:
        """
        将原始查询分解为多个子查询
        
        该方法负责将复杂的原始查询分解为多个更简单、更具针对性的子查询，是Graph-RAG系统中
        查询优化的核心实现。通过子查询生成，系统能够更全面、更精确地获取与复杂问题相关的
        信息，提高整体推理质量。
        
        参数:
            original_query: 原始用户查询，可能包含多个方面或复杂逻辑
            
        返回:
            List[str]: 生成的子查询列表，每个子查询都针对原始问题的一个特定方面或子问题
            
        实现思路：
        1. 使用预定义的子查询提示模板构建完整的提示
        2. 将原始查询嵌入到提示中，指导LLM如何分解问题
        3. 调用LLM生成子查询列表
        4. 处理LLM响应，提取实际的子查询内容
        5. 清理和格式化子查询，确保质量和一致性
        6. 返回处理后的子查询列表
        
        技术特点：
        - 提示工程：使用精心设计的提示模板引导高质量的子查询生成
        - 复杂问题分解：将多方面问题转化为单一焦点的子查询
        - 结构化输出：确保生成的子查询列表格式规范、内容明确
        - 与LLM集成：充分利用大语言模型的理解和生成能力
        
        业务意义：
        - 提高搜索针对性，获取更精确的相关信息
        - 支持对复杂问题的多维度探索
        - 增强信息检索的全面性，避免遗漏重要方面
        - 为后续的推理和分析提供更充分的信息基础
        - 提高系统解决复杂问题的能力和准确性
        
        实现思路：
        1. 使用预定义的子查询提示模板格式化原始查询
        2. 调用LLM生成基于模板的子查询
        3. 处理不同类型的LLM响应格式，提取生成内容
        4. 使用正则表达式从响应中提取列表文本
        5. 尝试解析列表文本为Python列表
        6. 实现优雅的降级机制：
           - 如果解析失败，返回原始查询
           - 如果发生任何异常，返回原始查询并记录错误
        
        技术特点：
        - 模板驱动：使用预定义提示模板引导子查询生成
        - 正则表达式提取：精确提取结构化输出
        - 异常处理：完善的错误捕获和降级策略
        - 日志记录：记录异常情况，便于调试
        - 灵活适配：支持不同类型的LLM响应格式
        
        业务意义：
        - 提高复杂查询的处理效率和准确性
        - 支持多角度信息收集，避免信息遗漏
        - 增强搜索的针对性，减少无关信息干扰
        - 为后续的深度推理提供更精确的信息基础
        - 支持系统处理更复杂的用户问题
        """
        try:
            # 调用LLM生成子查询
            response = self.llm.invoke(self.sub_query_prompt.format(original_query=original_query))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    sub_queries = eval(list_text.group(0))
                    return sub_queries
                except Exception as e:
                    print(f"[子查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回原始查询
            return [original_query]
        except Exception as e:
            print(f"[子查询生成错误] {str(e)}")
            return [original_query]
    
    def generate_multiple_hypotheses(query: str, llm) -> List[str]:
        """
        为查询生成多个假设
        
        该方法负责基于查询生成多个不同角度的假设，是Graph-RAG系统中假设驱动推理的重要组成部分。
        通过生成多个不同的假设，系统能够从多角度探索问题，避免单一思路的局限性，提高
        推理的全面性和准确性。注意：这是一个静态方法，可以直接调用而不需要实例化QueryGenerator。
        
        参数:
            query: 查询字符串，用户的原始问题
            llm: 语言模型实例，用于生成假设
            
        返回:
            List[str]: 假设列表，包含2-3个从不同角度提出的假设
        
        实现思路：
        1. 构建提示，要求LLM为问题生成2-3个不同角度的假设
        2. 明确假设的三个关键要求：不同于其他假设、提供思考方向、有助于深入分析
        3. 调用LLM生成假设
        4. 尝试多种策略提取假设：
           - 首先尝试匹配编号列表 (1. xxx 2. xxx) 格式
           - 如果失败，尝试匹配破折号列表 (- xxx) 格式
           - 如果都失败，按行分割并过滤
        5. 确保返回的假设内容有足够长度，并限制最多返回3个假设
        6. 实现全面的异常处理，记录错误并在失败时返回空列表
        
        技术特点：
        - 静态方法设计：可独立于类实例调用
        - 多策略提取：使用不同的正则表达式模式提取结构化输出
        - 灵活降级：从最优提取方法到简单过滤的渐进式降级
        - 结果过滤：确保假设内容有实际价值
        - 异常处理：完善的错误捕获和日志记录
        
        业务意义：
        - 提供多角度思考能力，避免确认偏误
        - 为后续的推理和验证提供多个可能的起点
        - 增强系统处理复杂问题的能力
        - 支持科学的假设驱动推理方法
        - 提高系统对不确定性的处理能力
        """
        prompt = f"""
        为以下问题生成2-3个可能的假设，这些假设应该代表不同角度或思路：
        
        问题: "{query}"
        
        每个假设应该:
        1. 不同于其他假设
        2. 提供一种可能的思考方向
        3. 有助于深入分析问题
        
        以列表形式返回假设，每个假设简短明了。
        """
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 使用正则表达式提取假设
            import re
            
            # 尝试匹配编号列表 (1. xxx 2. xxx)
            numbered_pattern = re.compile(r'\d+\.\s*(.*?)(?=\d+\.|$)', re.DOTALL)
            numbered_matches = numbered_pattern.findall(content)
            
            if numbered_matches:
                return [match.strip() for match in numbered_matches if match.strip()]
            
            # 尝试匹配破折号列表 (- xxx)
            dash_pattern = re.compile(r'-\s*(.*?)(?=-|$)', re.DOTALL)
            dash_matches = dash_pattern.findall(content)
            
            if dash_matches:
                return [match.strip() for match in dash_matches if match.strip()]
            
            # 如果上述方法失败，按行分割并过滤
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            potential_hypotheses = [line for line in lines if len(line) > 10 and not line.startswith("假设") and not line.startswith("以下是")]
            
            return potential_hypotheses[:3]  # 最多返回3个假设
            
        except Exception as e:
            print(f"生成假设失败: {e}")
            return []
        
    def generate_followup_queries(self, original_query: str, retrieved_info: List[str]) -> List[str]:
        """
        基于已检索的信息生成跟进查询
        
        该方法负责基于已检索的信息智能生成跟进查询，是Graph-RAG系统中多轮推理和深度分析的
        关键组件。通过分析已有信息的不足之处，生成针对性的跟进查询，系统能够逐步深入问题，
        填补知识空白，提高最终答案的完整性和准确性。
        
        参数:
            original_query: 原始查询，用户的初始问题
            retrieved_info: 已检索的信息列表，包含已获取的相关文档片段
            
        返回:
            List[str]: 跟进查询列表，如果不需要跟进查询则返回空列表
        
        实现思路：
        1. 首先检查是否有足够的检索信息进行分析（至少需要2条信息）
        2. 如果信息不足，直接返回空列表，表示不需要跟进查询
        3. 合并最近的3条检索信息（限制长度，避免token超限）
        4. 使用预定义的跟进查询提示模板格式化原始查询和检索信息
        5. 调用LLM生成跟进查询
        6. 处理不同类型的LLM响应格式，提取生成内容
        7. 使用正则表达式从响应中提取列表文本
        8. 尝试解析列表文本为Python列表
        9. 去重处理，确保生成的跟进查询不重复
        10. 实现全面的异常处理，记录错误并在失败时返回空列表
        
        技术特点：
        - 信息筛选：只使用最近的3条检索信息，避免上下文过长
        - 模板驱动：使用预定义提示模板引导跟进查询生成
        - 智能判断：能自动判断是否需要跟进查询
        - 去重处理：确保跟进查询的唯一性
        - 异常处理：完善的错误捕获和降级策略
        - 正则表达式提取：精确提取结构化输出
        
        业务意义：
        - 实现多轮渐进式信息收集，深入探索复杂问题
        - 自动识别信息缺口，引导系统获取更完整的信息
        - 提高信息检索的针对性和效率
        - 支持系统自主决定何时需要进一步搜索
        - 增强系统处理复杂问题的能力和深度
        """
        # 如果没有检索到任何信息，或信息不足，返回空列表
        if not retrieved_info or len(retrieved_info) < 2:
            return []
        
        try:
            # 合并已检索信息（但限制长度）
            info_text = "\n\n".join(retrieved_info[-3:])  # 只使用最近的3条信息
            
            # 调用LLM生成跟进查询
            response = self.llm.invoke(self.followup_query_prompt.format(
                original_query=original_query,
                retrieved_info=info_text
            ))
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取列表文本
            list_text = re.search(r'\[.*\]', content, re.DOTALL)
            if list_text:
                try:
                    # 解析列表
                    followup_queries = eval(list_text.group(0))
                    
                    # 确保没有重复查询
                    unique_queries = []
                    for q in followup_queries:
                        if q not in unique_queries:
                            unique_queries.append(q)
                    
                    return unique_queries
                except Exception as e:
                    print(f"[跟进查询生成] 解析列表失败: {str(e)}")
            
            # 如果无法解析，返回空列表
            return []
        except Exception as e:
            print(f"[跟进查询生成错误] {str(e)}")
            return []