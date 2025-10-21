import time
import json
from typing import List, Dict, Any

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import gl_description
from search.tool.base import BaseSearchTool


class GlobalSearchTool(BaseSearchTool):
    """
    全局搜索工具
    
    该类实现了基于知识图谱和Map-Reduce分布式计算模式的全局搜索功能，能够跨多个社区进行广泛查询。
    通过关键词提取、社区检索、批量处理和结果合并等步骤，它提供了一种高效的全局信息检索机制，
    特别适合处理需要跨多个知识领域的复杂查询。
    
    核心功能：
    - 关键词提取与分析
    - 社区层级数据检索
    - Map-Reduce模式实现
    - 批量处理优化性能
    - 结果缓存机制
    
    设计特点：
    - 分布式处理：使用Map-Reduce模式处理大规模数据
    - 社区层级控制：支持不同层级的社区搜索
    - 批量处理：优化性能，减少计算资源消耗
    - 缓存机制：提高重复查询的响应速度
    - 容错处理：完善的错误捕获和降级策略
    
    业务意义：
    - 支持跨领域知识检索
    - 提供全局视角的信息分析
    - 处理复杂的多维度查询
    - 为Graph-RAG系统提供高级检索能力
    """

    def __init__(self, level: int = 0):
        """
        初始化全局搜索工具
        
        参数:
            level: 社区层级，默认为0
        """
        # 设置社区层级
        self.level = level
        
        # 调用父类构造函数
        super().__init__(cache_dir="./cache/global_search")

        # 设置处理链
        self._setup_chains()
    
    def _setup_chains(self):
        """
        设置Map-Reduce处理链
        
        该方法是GlobalSearchTool的核心配置方法，负责初始化和设置Map-Reduce分布式计算模式所需的各个处理链。
        通过配置不同的提示模板和LLM组合，它为全局搜索提供了三个关键的处理组件：Map处理链、Reduce处理链和关键词提取链。
        
        实现思路：
        1. 创建Map阶段的提示模板，用于处理社区数据并生成初步分析
        2. 构建Map处理链，连接提示模板、LLM和输出解析器
        3. 创建Reduce阶段的提示模板，用于整合Map结果并生成最终答案
        4. 构建Reduce处理链，连接提示模板、LLM和输出解析器
        5. 创建关键词提取提示模板，专门用于从查询中提取关键概念
        6. 构建关键词提取处理链
        
        技术特点：
        - 链式处理：使用LangChain的管道操作构建处理流
        - 模板化提示：使用ChatPromptTemplate配置不同阶段的提示
        - 输出标准化：使用StrOutputParser确保输出格式一致
        - 专业提示设计：针对不同处理阶段优化提示内容
        - 语义提取：专门的关键词提取提示设计
        
        业务意义：
        - 提供分布式计算的核心组件
        - 确保Map和Reduce阶段的正确执行
        - 支持高效的关键词提取和分析
        - 为全局搜索提供结构化的处理流程
        """
        # 设置Map阶段的处理链
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        self.map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 设置Reduce阶段的处理链
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 关键词提取链
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。提取最相关的关键词，这些关键词将用于在知识库中查找信息。
                
                请返回一个关键词列表，格式为JSON数组：
                ["关键词1", "关键词2", ...]
                
                注意：
                - 提取5-8个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON数组
                - 关键词应该是名词短语、概念或专有名词
                """),
            ("human", "{query}")
        ])
        
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        该方法是GlobalSearchTool的关键预处理方法，负责从用户查询中提取最具代表性的关键词和概念。
        这些关键词将用于过滤和检索相关的社区数据，是实现精准搜索的基础。通过使用LLM进行语义分析，
        它能够识别查询中的核心概念，而不仅仅是简单的关键词匹配。
        
        参数:
            query: 查询字符串，用户的原始问题或搜索关键词
            
        返回:
            Dict[str, List[str]]: 包含不同类型关键词的字典，包括通用关键词、低层次关键词和高层次关键词
            
        实现思路：
        1. 首先检查缓存中是否已有该查询的关键词结果
        2. 若缓存未命中，记录LLM处理开始时间
        3. 调用关键词提取链，获取LLM生成的关键词列表
        4. 解析LLM返回的JSON格式关键词数组
        5. 记录LLM处理时间，更新性能指标
        6. 将关键词数组转换为标准格式的字典
        7. 缓存处理结果，优化后续查询
        8. 实现完善的异常处理，确保方法健壮性
        
        技术特点：
        - 缓存机制：优先从缓存获取关键词结果
        - LLM驱动：使用语言模型进行语义关键词提取
        - 性能监控：记录LLM处理时间
        - 标准化输出：确保返回格式统一
        - 异常处理：处理各种可能的错误情况
        
        业务意义：
        - 提高搜索精度：通过关键概念识别优化检索结果
        - 降低计算负担：避免处理无关的社区数据
        - 加速搜索过程：缓存机制减少重复处理
        - 支持多层次搜索：提供不同抽象层次的关键词
        """
        # 检查缓存
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            llm_start = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            # 解析JSON结果
            keywords = json.loads(result)
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] = time.time() - llm_start
            
            # 将关键词数组转换为标准格式
            if isinstance(keywords, list):
                formatted_keywords = {
                    "keywords": keywords,
                    "low_level": [],
                    "high_level": keywords  # 全局搜索主要关注高级概念
                }
            else:
                # 默认空结构
                formatted_keywords = {
                    "keywords": [],
                    "low_level": [],
                    "high_level": []
                }
                
            # 缓存结果
            self.cache_manager.set(f"keywords:{query}", formatted_keywords)
            
            return formatted_keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 返回空字典作为默认值
            return {"keywords": [], "low_level": [], "high_level": []}
    
    def _get_community_data(self, keywords: List[str] = None) -> List[dict]:
        """
        使用关键词检索社区数据
        
        该方法是GlobalSearchTool的核心数据检索方法，负责从知识图谱中查询与给定关键词相关的社区数据。
        通过构建优化的Cypher查询，它能够高效地过滤和排序社区数据，为后续的Map-Reduce处理提供输入。
        
        参数:
            keywords: 关键词列表，用于过滤社区数据，可选择性提供
            
        返回:
            List[dict]: 包含社区ID和完整内容的社区数据列表，按社区排名和权重降序排列
            
        实现思路：
        1. 构建基础Cypher查询，匹配指定层级的社区节点
        2. 设置层级参数，确保查询正确的社区层级
        3. 如果提供了关键词，动态构建关键词过滤条件
        4. 为每个关键词创建参数化查询部分，避免SQL注入风险
        5. 添加排序条件，优先返回排名高和权重大的社区
        6. 限制返回结果数量，提高查询效率
        7. 执行Cypher查询并返回结果
        
        技术特点：
        - 参数化查询：使用参数化Cypher查询避免注入风险
        - 动态过滤：根据关键词动态构建查询条件
        - 多条件排序：按社区排名和权重排序
        - 结果限制：限制返回数量优化性能
        - 灵活查询：支持有无关键词两种查询模式
        
        业务意义：
        - 高效过滤：快速筛选出相关社区数据
        - 精准定位：基于关键词找到最相关的知识社区
        - 优化排序：确保重要社区优先处理
        - 性能优化：限制结果数量提高处理效率
        - 为Map阶段提供高质量输入数据
        """
        # 构建基础查询
        cypher_query = """
        MATCH (c:__Community__)
        WHERE c.level = $level
        """
        
        params = {"level": self.level}
        
        # 如果提供了关键词，使用它们过滤社区
        if keywords and len(keywords) > 0:
            keywords_condition = []
            for i, keyword in enumerate(keywords):
                keyword_param = f"keyword{i}"
                keywords_condition.append(f"c.full_content CONTAINS ${keyword_param}")
                params[keyword_param] = keyword
            
            if keywords_condition:
                cypher_query += " AND (" + " OR ".join(keywords_condition) + ")"
        
        # 添加排序和返回语句
        cypher_query += """
        WITH c
        ORDER BY c.community_rank DESC, c.weight DESC
        LIMIT 20
        RETURN {communityId: c.id, full_content: c.full_content} AS output
        """
        
        # 执行查询
        return self.graph.query(cypher_query, params=params)
    
    def _process_community_batch(self, query: str, batch: List[dict]) -> str:
        """
        处理社区批次，提高效率
        
        该方法是GlobalSearchTool的性能优化方法，负责批量处理多个社区数据，而不是逐个处理。
        通过一次调用LLM处理多个社区的组合数据，它显著减少了LLM调用次数，提高了整体处理效率，
        同时保持了处理结果的质量。
        
        参数:
            query: 查询字符串，用户的原始问题或搜索关键词
            batch: 社区数据批次，包含多个社区的ID和内容信息
            
        返回:
            str: 批次处理结果，LLM对该批次所有社区数据的分析输出
            
        实现思路：
        1. 创建一个列表用于存储合并后的社区数据
        2. 遍历批次中的每个社区，格式化其ID和内容信息
        3. 使用分隔符将多个社区的数据合并为一个完整的上下文
        4. 调用Map处理链，将查询和合并后的上下文作为输入
        5. 返回LLM生成的批次处理结果
        
        技术特点：
        - 批量处理：一次处理多个社区数据
        - 数据合并：将多个社区信息整合为单一上下文
        - 效率优化：减少LLM调用次数
        - 上下文组织：结构化组织社区数据便于LLM理解
        - 一次分析：对多个相关社区进行综合分析
        
        业务意义：
        - 显著提高处理速度：减少LLM调用开销
        - 优化资源使用：降低计算资源消耗
        - 保持处理质量：同时处理相关社区提高上下文关联性
        - 支持大规模处理：使系统能够高效处理大量社区数据
        - 为Map阶段提供高效处理机制
        """
        # 合并批次内的社区数据
        combined_data = []
        for item in batch:
            combined_data.append(f"社区ID: {item['output']['communityId']}\n内容: {item['output']['full_content']}")
        
        batch_context = "\n---\n".join(combined_data)
        
        # 一次性处理整个批次
        return self.map_chain.invoke({
            "question": query, 
            "context_data": batch_context
        })
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段）
        
        该方法是GlobalSearchTool中Map-Reduce模式的Map阶段实现，负责将大量社区数据分批次处理，
        并生成每个批次的中间分析结果。通过批处理机制，它在保持处理质量的同时显著提高了处理效率，
        是全局搜索中数据并行处理的核心环节。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            communities: 社区数据列表，包含多个社区的ID和内容信息
            
        返回:
            List[str]: 中间结果列表，包含每个批次处理后的分析报告
            
        实现思路：
        1. 设置批处理大小，控制每批处理的社区数量
        2. 创建结果列表存储中间分析结果
        3. 使用循环分批处理社区数据
        4. 对每个批次调用_process_community_batch方法进行处理
        5. 过滤空结果，确保只保留有效内容
        6. 实现异常处理，确保单个批次失败不影响整体处理
        7. 返回所有批次的处理结果列表
        
        技术特点：
        - 批处理机制：将大数据集分割成小批次处理
        - 并行思想：采用Map阶段的并行处理思路
        - 容错处理：单个批次失败不影响整体流程
        - 结果过滤：自动排除空或无效结果
        - 效率优化：通过批量处理减少资源消耗
        
        业务意义：
        - 实现分布式处理：Map阶段的核心实现
        - 提高处理效率：批量处理大量社区数据
        - 增强系统可靠性：完善的异常处理机制
        - 为Reduce阶段准备高质量输入：生成结构化分析结果
        - 支持大规模数据处理：使系统能够处理复杂的全局搜索请求
        """
        batch_size = 5  # 每批处理5个社区，提高效率
        
        results = []
        
        # 使用批处理提高效率
        for i in range(0, len(communities), batch_size):
            batch = communities[i:i+batch_size]
            try:
                batch_result = self._process_community_batch(query, batch)
                if batch_result and len(batch_result.strip()) > 0:
                    results.append(batch_result)
            except Exception as e:
                print(f"批处理失败: {e}")
        
        return results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        
        该方法是GlobalSearchTool中Map-Reduce模式的Reduce阶段实现，负责将Map阶段生成的多个中间分析结果
        整合为一个综合性的最终答案。通过调用专门的Reduce处理链，它能够识别各个中间结果中的关键信息，
        去除冗余内容，并生成一个连贯、全面且针对原始查询的最终回答。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            intermediate_results: 中间结果列表，Map阶段处理后生成的各个批次的分析报告
            
        返回:
            str: 最终生成的答案，基于所有中间结果整合而成的综合性回应
            
        实现思路：
        1. 接收Map阶段生成的所有中间分析结果
        2. 调用预先配置的Reduce处理链
        3. 将中间结果列表和原始查询作为输入传递给处理链
        4. 设置响应类型为"多个段落"，确保生成结构化的详细回答
        5. 返回Reduce链生成的最终整合答案
        
        技术特点：
        - 结果整合：将多个中间结果合并为单一最终答案
        - 信息去重：自动识别和去除冗余信息
        - 一致性确保：生成与原始查询一致的综合回答
        - 结构化输出：生成格式良好的多个段落回答
        - 语义理解：基于上下文理解整合不同来源的信息
        
        业务意义：
        - 完成Map-Reduce流程：Reduce阶段的核心实现
        - 生成全面答案：整合多个社区的相关信息
        - 确保信息一致性：协调不同来源的信息
        - 优化用户体验：提供连贯、全面的最终回答
        - 实现分布式计算的价值：结合并行处理和结果整合
        """
        # 调用Reduce链生成最终答案
        return self.reduce_chain.invoke({
            "report_data": intermediate_results,
            "question": query,
            "response_type": "多个段落",
        })
    
    def search(self, query_input: Any) -> List[str]:
        """
        执行全局搜索，实现Map-Reduce模式
        
        该方法是GlobalSearchTool的核心执行方法，负责协调整个全局搜索流程，从输入解析、关键词提取、
        社区检索到Map-Reduce处理的完整实现。它支持多种输入格式，实现了缓存机制优化性能，并提供了完善
        的异常处理，确保搜索过程的健壮性和高效性。
        
        参数:
            query_input: 查询输入，可以是字符串或包含查询和关键词的字典，提供灵活的输入方式
            
        返回:
            List[str]: 中间结果列表，包含Map阶段生成的分析结果，供GraphAgent的reduce阶段使用
            
        实现思路：
        1. 记录搜索开始时间，用于性能监控
        2. 解析输入参数，支持字符串和字典两种格式
        3. 处理关键词：如果输入中没有提供关键词，则自动提取
        4. 构建缓存键，支持基于查询和关键词的复合缓存键
        5. 检查缓存，避免重复计算
        6. 获取社区数据，使用关键词进行过滤
        7. 处理空结果情况，确保系统稳定性
        8. 执行Map阶段处理，批量处理社区数据
        9. 缓存处理结果，优化后续查询
        10. 记录总处理时间，更新性能指标
        11. 实现完善的异常处理，确保系统稳定性
        
        技术特点：
        - 灵活输入：支持多种查询输入格式
        - 缓存优化：基于查询和关键词的复合缓存
        - 性能监控：记录总处理时间
        - Map-Reduce实现：完整实现分布式计算模式
        - 容错设计：完善的异常处理机制
        
        业务意义：
        - 提供全局搜索入口：系统的主要搜索执行方法
        - 实现高效检索：结合缓存和批处理优化性能
        - 支持复杂查询：处理需要跨社区分析的复杂问题
        - 为GraphAgent提供数据支持：生成中间结果供后续处理
        - 确保系统稳定性：完善的错误处理和降级机制
        """
        overall_start = time.time()
        
        # 解析输入
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            keywords = extracted_keywords.get("keywords", [])
        
        # 检查缓存
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 获取社区数据
            community_data = self._get_community_data(keywords)
            
            # 如果没有找到相关社区，返回空结果
            if not community_data:
                return []
            
            # 处理社区数据，生成中间结果
            intermediate_results = self._process_communities(query, community_data)
            
            # 缓存结果
            self.cache_manager.set(cache_key, intermediate_results)
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            
            return intermediate_results
        
        except Exception as e:
            print(f"全局搜索失败: {e}")
            return [f"搜索过程中出现错误: {str(e)}"]
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具实例
        
        该方法是GlobalSearchTool与LangChain工具系统集成的关键方法，负责创建一个符合LangChain BaseTool接口的工具对象，
        使得全局搜索功能可以被LangChain Agent或其他组件以标准方式调用。通过定义内部工具类，它实现了自定义搜索工具
        到LangChain工具的适配。
        
        返回:
            BaseTool: 一个GlobalRetrievalTool实例，实现了BaseTool接口，可以被LangChain组件调用
            
        实现思路：
        1. 在方法内部定义GlobalRetrievalTool类，继承自BaseTool
        2. 设置工具名称为"global_retriever"
        3. 设置工具描述，使用全局配置的描述文本
        4. 实现_run方法，将调用委托给外部self.search方法
        5. 声明_arun方法（异步执行）为未实现
        6. 创建并返回GlobalRetrievalTool的实例
        
        技术特点：
        - 内部类定义：在运行时动态定义工具类
        - 委托模式：将工具执行委托给原始搜索工具
        - 接口适配：将自定义工具适配到LangChain接口
        - 静态配置：使用预定义的工具名称和描述
        - 同步执行：只实现同步执行方法
        
        业务意义：
        - 实现工具集成：将全局搜索工具与LangChain生态集成
        - 支持Agent调用：使GraphAgent能够使用全局搜索功能
        - 标准化接口：提供统一的工具调用方式
        - 简化集成：避免创建额外的工具类文件
        - 为系统提供标准的工具交互机制
        """
        class GlobalRetrievalTool(BaseTool):
            name : str= "global_retriever"
            description : str = gl_description
            
            def _run(self_tool, query: Any) -> List[str]:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> List[str]:
                raise NotImplementedError("异步执行未实现")
        
        return GlobalRetrievalTool()
    
    def close(self):
        """关闭资源"""
        # 调用父类方法关闭资源
        super().close()