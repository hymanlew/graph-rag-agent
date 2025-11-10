from typing import List, Dict, Any
import time
import json
from langsmith import traceable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser

from config.prompt import LC_SYSTEM_PROMPT, contextualize_q_system_prompt
from config.settings import lc_description
from search.tool.base import BaseSearchTool
from search.local_search import LocalSearch


class LocalSearchTool(BaseSearchTool):
    """
    本地搜索工具
    
    该工具实现了基于向量检索的社区内部精确查询功能（graphrag的局部检索），继承自BaseSearchTool基类，
    并增强了历史感知检索、对话上下文管理和结果缓存等特性。

    核心功能：
    1. 历史感知的查询处理
    2. 关键词提取与分类
    3. 文档相关性过滤
    4. 搜索结果缓存
    5. LangSmith性能监控
    6. 异常处理和错误恢复
    """

    def __init__(self):
        """
        初始化本地搜索工具
        
        设计特点:
        - 继承基类的共享功能（数据库连接、缓存等）
        - 专注于本地（社区内）搜索功能
        - 支持对话历史，实现上下文感知
        - 预配置处理链，提高初始化效率
        """
        # 调用父类构造函数，指定本地搜索的缓存目录
        super().__init__(cache_dir="./cache/local_search")
        
        # 初始化聊天历史列表，用于连续对话的上下文管理
        self.chat_history = []
                
        # 创建本地搜索器实例，传入语言模型和嵌入模型
        self.local_searcher = LocalSearch(self.llm, self.embeddings)
        # 获取检索器接口，用于向量检索
        self.retriever = self.local_searcher.as_retriever()
        # 设置各种处理链，包括历史感知检索、问答和关键词提取
        self._setup_chains()

    def _setup_chains(self):
        """设置处理链"""
        # 创建上下文理解提示模板，用于优化处理带历史的查询
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 创建历史感知检索器，结合对话历史优化检索效果
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_q_prompt,
        )

        # 创建带历史的本地查询提示模板，控制输出格式
        lc_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            {context}
            
            用户的问题是：
            {input}

            请使用三级标题(###)标记主题
            """),
        ])

        # 创建问答链，用于生成回答
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            lc_prompt_with_history,
        )

        # 创建完整的RAG链，组合检索和生成功能
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain,
        )
        
        # 创建关键词提取链，用于查询分析和分类
        self.keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专门从用户查询中提取搜索关键词的助手。你需要将关键词分为两类：
                1. 低级关键词：具体实体名称、人物、地点、具体事件等
                2. 高级关键词：主题、概念、关系类型等
                
                返回格式必须是JSON格式：
                {{
                    "low_level": ["关键词1", "关键词2", ...], 
                    "high_level": ["关键词1", "关键词2", ...]
                }}
                
                注意：
                - 每类提取3-5个关键词即可
                - 不要添加任何解释或其他文本，只返回JSON
                - 如果某类无关键词，则返回空列表
                """),
            ("human", "{query}")
        ])
        
        # 构建关键词提取链：提示模板 -> LLM -> 输出解析器
        self.keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词并分类
        
        参数:
            query: 用户查询字符串
            
        返回:
            Dict[str, List[str]]: 包含低级和高级关键词的字典

        关键词分类说明:
        - 低级关键词: 具体实体、人物、地点、事件等具体概念
        - 高级关键词: 主题、概念、关系类型等抽象概念
        """
        # 检查缓存，避免重复计算
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            # 记录开始时间
            llm_start = time.time()
            
            # 调用LLM提取关键词
            result = self.keyword_chain.invoke({"query": query})
            
            # 解析JSON结果
            keywords = json.loads(result)
            
            # 记录LLM处理时间
            self.performance_metrics["llm_time"] = time.time() - llm_start
            
            # 确保结果格式正确，包含必要的键
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []
                
            # 缓存提取结果
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 降级处理：返回空关键词字典
            return {"low_level": [], "high_level": []}

    def _filter_documents_by_relevance(self, docs, query: str) -> List:
        """
        基于向量相似度进行排序，过滤文档
        
        参数:
            docs: 待过滤的文档列表
            query: 查询字符串
            
        返回:
            List: 按相关性降序排序的文档列表
        """
        # 调用基类的标准方法，设置返回前5个最相关的文档
        return self.filter_by_relevance(query, docs, top_k=5)

    @traceable
    def search(self, query_input: Any) -> str:
        """
        执行本地搜索的核心方法，LangSmith跟踪支持（@traceable装饰器）
        
        参数:
            query_input: 查询输入，可以是字符串或包含query和keywords的字典
            
        返回:
            str: 格式化的搜索结果
        """
        # 记录开始时间，用于性能监控
        overall_start = time.time()
        
        # 解析输入参数，支持字符串和字典两种格式
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
            keywords = query_input.get("keywords", [])
        else:
            query = str(query_input)
            keywords = []
        
        # 构建缓存键，考虑查询和关键词
        cache_key = query
        if keywords:
            cache_key = f"{query}||{','.join(sorted(keywords))}"
        
        # 检查缓存，避免重复搜索
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # 使用RAG链执行搜索
        try:
            ai_msg = self.rag_chain.invoke({
                "input": query,
                "response_type": "多个段落",
                "chat_history": self.chat_history,
            })
            result = ai_msg.get("answer", "抱歉，我无法回答这个问题。")
            
            # 缓存搜索结果
            self.cache_manager.set(cache_key, result)
            
            # 记录总处理时间
            self.performance_metrics["total_time"] = time.time() - overall_start

            # 处理空结果
            if not result:
                return "未找到相关信息"
            return result
        except Exception as e:
            # 异常处理
            print(f"本地搜索失败: {e}")
            error_msg = f"搜索过程中出现问题: {str(e)}"
            
            # 记录性能指标
            self.performance_metrics["total_time"] = time.time() - overall_start
            return error_msg

    def get_tool(self):
        """
        获取LangChain兼容的检索工具
        
        返回:
            BaseTool: 可用于LangChain工具调用的检索工具实例
        """
        # 创建并返回LangChain兼容的检索工具
        return create_retriever_tool(
            self.retriever,       # 本地检索器
            "lc_search_tool",     # 工具名称
            lc_description,       # 工具描述
        )

    def close(self):
        """
        关闭资源连接

        设计特点:
        - 分层资源管理
        - 防御性编程，避免属性不存在错误，防止资源泄漏
        - 确保所有资源正确释放，遵循上下文管理器协议
        """
        # 先调用父类方法关闭基础资源（数据库连接等）
        super().close()
        
        # 关闭本地搜索器资源
        if hasattr(self, 'local_searcher'):
            self.local_searcher.close()