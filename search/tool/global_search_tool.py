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
    通过关键词提取与分析、社区层级数据检索、Map-Reduce模式实现、批量处理、结果合并和结果缓存机制等步骤，提供了一种高效的全局信息检索机制，
    特别适合处理需要跨多个知识领域的复杂查询。
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
        通过配置不同的提示模板和LLM组合，为全局搜索提供了三个关键的处理组件：Map处理链、Reduce处理链和关键词提取链。

        业务意义：
        - 提供分布式计算的核心组件
        - 确保Map和Reduce阶段的正确执行
        - 支持高效的关键词提取和分析
        - 为全局搜索提供结构化的处理流程
        """
        # 创建Map阶段的提示模板，生成一个回答用户问题所需的要点列表，用于从图社区中提取关键信息
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
        
        # 设置Reduce阶段的处理链，用于整合Map结果并生成最终答案
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
        
        # 关键词提取链，专门用于从查询中提取关键概念，将用于在知识库中查找信息
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
        从查询中提取关键词，这些关键词将用于过滤和检索相关的社区数据，是实现精准搜索的基础。
        通过使用LLM进行语义分析，它能够识别查询中的核心概念，而不仅仅是简单的关键词匹配。
        
        参数:
            query: 查询字符串，用户的原始问题或搜索关键词
            
        返回:
            Dict[str, List[str]]: 包含不同类型关键词的字典，包括通用关键词、低层次关键词和高层次关键词
        
        业务意义：
        - 提高搜索精度：通过关键概念识别优化检索结果
        - 降低计算负担：避免处理无关的社区数据
        - 加速搜索过程：缓存机制减少重复处理
        - 支持多层次搜索：提供不同抽象层次的关键词
        """
        # 检查缓存中是否已有该查询的关键词结果
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            # 调用关键词提取链，获取LLM生成的关键词列表
            llm_start = time.time()
            result = self.keyword_chain.invoke({"query": query})
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
        使用关键词检索社区数据。构建优化的Cypher查询，高效地过滤和排序社区数据，为后续的Map-Reduce处理提供输入。
        
        参数:
            keywords: 关键词列表，用于过滤社区数据，可选择性提供
            
        返回:
            List[dict]: 包含社区ID和完整内容的社区数据列表，按社区排名和权重降序排列
        
        业务意义：
        - 高效过滤：快速筛选出相关社区数据
        - 精准定位：基于关键词找到最相关的知识社区
        - 优化排序：确保重要社区优先处理
        - 性能优化：限制结果数量提高处理效率
        - 为Map阶段提供高质量输入数据
        """
        # 构建基础查询，匹配指定层级的社区节点
        cypher_query = """
        MATCH (c:__Community__)
        WHERE c.level = $level
        """
        
        params = {"level": self.level}

        # 参数化查询：使用参数化Cypher查询避免注入风险
        # 如果提供了关键词，动态构建关键词过滤条件，过滤社区
        if keywords and len(keywords) > 0:
            keywords_condition = []
            for i, keyword in enumerate(keywords):
                keyword_param = f"keyword{i}"
                params[keyword_param] = keyword
                keywords_condition.append(f"c.full_content CONTAINS ${keyword_param}")

            if keywords_condition:
                cypher_query += " AND (" + " OR ".join(keywords_condition) + ")"
        
        # 添加排序和返回语句，优先返回排名高和权重大的社区
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
        批量处理多个社区数据，而不是逐个处理，提高效率
        通过一次调用LLM处理多个社区的组合数据，显著减少了LLM调用次数，提高了整体处理效率
        
        参数:
            query: 查询字符串，用户的原始问题或搜索关键词
            batch: 社区数据批次，包含多个社区的ID和内容信息
            
        返回:
            str: 批次处理结果，LLM对该批次所有社区数据的分析输出
        """
        # 合并批次内的社区数据
        combined_data = []
        for item in batch:
            combined_data.append(f"社区ID: {item['output']['communityId']}\n内容: {item['output']['full_content']}")

        # 使用分隔符将多个社区的数据合并为一个完整的上下文
        batch_context = "\n---\n".join(combined_data)
        
        # 一次性处理整个批次
        return self.map_chain.invoke({
            "question": query, 
            "context_data": batch_context
        })
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段实现），负责将大量社区数据分批次处理，并生成每个批次的中间分析结果。
        通过批处理机制，它在保持处理质量的同时显著提高了处理效率，是全局搜索中数据并行处理的核心环节。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            communities: 社区数据列表，包含多个社区的ID和内容信息
            
        返回:
            List[str]: 中间结果列表，包含每个批次处理后的分析报告

        业务意义：
        - 实现分布式处理：Map阶段的核心实现
        - 提高处理效率：批量处理大量社区数据
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
                # 过滤空结果，确保只保留有效内容
                if batch_result and len(batch_result.strip()) > 0:
                    results.append(batch_result)
            except Exception as e:
                print(f"批处理失败: {e}")
        
        return results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        负责将Map阶段生成的多个中间分析结果，整合为一个综合性的最终答案。去除冗余内容，并生成一个连贯、全面且针对原始查询的最终回答。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            intermediate_results: 中间结果列表，Map阶段处理后生成的各个批次的分析报告
            
        返回:
            str: 最终生成的答案，基于所有中间结果整合而成的综合性回应
        """
        # 调用Reduce链生成最终答案，设置响应类型为"多个段落"，确保生成结构化的详细回答
        return self.reduce_chain.invoke({
            "report_data": intermediate_results,
            "question": query,
            "response_type": "多个段落",
        })
    
    def search(self, query_input: Any) -> List[str]:
        """
        执行全局搜索，实现Map-Reduce模式
        协调整个全局搜索流程，从输入解析、关键词提取、社区检索到Map-Reduce处理的完整实现。支持多种输入格式，实现了缓存机制优化性能，
        并提供了完善的异常处理，确保搜索过程的健壮性和高效性。
        
        参数:
            query_input: 查询输入，可以是字符串或包含查询和关键词的字典，提供灵活的输入方式
            
        返回:
            List[str]: 中间结果列表，包含Map阶段生成的分析结果，供GraphAgent的reduce阶段使用
        """
        overall_start = time.time()
        
        # 解析输入参数，支持字符串和字典两种格式
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            # 提取关键词
            extracted_keywords = self.extract_keywords(query)
            keywords = extracted_keywords.get("keywords", [])
        
        # 检查缓存，支持基于查询和关键词的复合缓存键
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 获取社区数据
            community_data = self._get_community_data(keywords)
            if not community_data:
                return []
            
            # 执行Map阶段处理，批量处理社区数据，生成中间结果
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

        返回:
            BaseTool: 一个GlobalRetrievalTool实例，实现了BaseTool接口，可以被LangChain组件调用
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