from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time

from langchain_core.tools import BaseTool

from model.get_models import get_llm_model, get_embeddings_model
from CacheManage.manager import CacheManager, ContextAndKeywordAwareCacheKeyStrategy, MemoryCacheBackend
from config.neo4jdb import get_db_manager
from search.utils import VectorUtils


class BaseSearchTool(ABC):
    """
    搜索工具基础类
    
    这是一个抽象基类，为Graph-RAG系统中的所有搜索工具提供通用功能和基础设施。
    它实现了共享的搜索逻辑、数据库连接、缓存机制和性能监控等功能，
    同时定义了子类必须实现的核心抽象方法。
    
    设计思路：
    - 采用抽象基类模式，定义统一接口
    - 实现共享功能，减少代码重复
    - 提供灵活的配置选项
    - 集成缓存和性能监控
    - 支持资源生命周期管理
    
    主要功能：
    1. 统一的数据库连接管理
    2. 向量搜索和文本搜索实现
    3. 缓存机制优化性能
    4. 性能指标收集
    5. 工具类接口适配
    6. 上下文管理器支持
    """
    
    def __init__(self, cache_dir: str = "./cache/search"):
        """
        初始化搜索工具
        
        参数:
            cache_dir: 缓存目录，用于存储搜索结果
        """
        # 初始化大语言模型和嵌入模型
        self.llm = get_llm_model()
        self.embeddings = get_embeddings_model()
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager(
            key_strategy=ContextAndKeywordAwareCacheKeyStrategy(),
            storage_backend=MemoryCacheBackend(max_size=200),
            cache_dir=cache_dir
        )
        
        # 性能监控指标
        self.performance_metrics = {
            "query_time": 0,  # 数据库查询时间
            "llm_time": 0,    # 大语言模型处理时间
            "total_time": 0   # 总处理时间
        }
        
        # 初始化Neo4j连接
        self._setup_neo4j()
    
    def _setup_neo4j(self):
        """设置Neo4j连接"""
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        
        # 获取图数据库实例
        self.graph = db_manager.get_graph()
        
        # 获取驱动（用于直接执行查询）
        self.driver = db_manager.get_driver()
    
    def db_query(self, cypher: str, params: Dict[str, Any] = {}):
        """
        执行Cypher查询
        
        该方法是Graph-RAG系统中数据库交互的核心入口，负责执行针对图数据库的Cypher查询语句。
        通过封装数据库操作，它提供了统一的查询接口，同时支持参数化查询以防止注入攻击，
        为所有搜索工具提供了安全、高效的数据访问机制。
        
        参数:
            cypher: Cypher查询语句，用于查询Neo4j图数据库
            params: 查询参数字典，用于参数化查询，提高安全性
            
        返回:
            查询结果，通常是一个包含节点和关系数据的DataFrame或字典列表
            
        实现思路：
        1. 调用数据库管理器执行参数化Cypher查询
        2. 自动处理查询执行和结果返回
        3. 隐藏底层数据库连接的复杂性
        4. 提供统一的错误处理机制
        
        技术特点：
        - 参数化查询：防止SQL注入攻击
        - 统一接口：所有搜索工具共用同一数据库访问方法
        - 依赖注入：通过数据库管理器获取连接
        - 简洁设计：简化数据库操作流程
        
        业务意义：
        - 确保数据库查询的安全性
        - 提供统一的数据访问层
        - 简化搜索工具的数据库交互
        - 支持Graph-RAG系统的核心功能实现
        - 为知识图谱操作提供基础支持
        """
        # 使用连接管理器执行查询
        return get_db_manager().execute_query(cypher, params)
        
    @abstractmethod
    def _setup_chains(self):
        """
        设置处理链，子类必须实现
        用于配置各种LLM处理链和提示模板
        """
        pass
    
    @abstractmethod
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词
        """
        pass
    
    @abstractmethod
    def search(self, query: Any) -> str:
        """
        执行搜索
        
        参数:
            query: 查询内容，可以是字符串或包含更多信息的字典
            
        返回:
            str: 搜索结果
        """
        pass

    def vector_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于向量相似度的搜索方法
        
        该方法是Graph-RAG系统中向量检索的核心实现，负责将文本查询转换为向量表示，
        然后在向量索引中查找语义相似度最高的实体。这种基于语义的搜索方式能够超越
        传统关键词匹配的限制，找到概念相似但表达方式不同的内容。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            limit: 最大返回结果数，控制搜索结果的数量，默认为5
            
        返回:
            List[str]: 匹配实体ID列表，按照相似度降序排列
            
        实现思路：
        1. 将查询文本转换为向量表示（嵌入）
        2. 构建Neo4j向量搜索查询，指定向量索引、返回结果数量和查询向量
        3. 执行向量搜索查询，获取节点和相似度分数
        4. 从结果中提取实体ID并返回
        5. 实现全面的异常处理，当向量搜索失败时自动降级到文本搜索
        
        技术特点：
        - 向量嵌入：使用嵌入模型将文本转换为向量表示
        - 语义搜索：基于向量相似度进行语义匹配
        - 自动降级：在向量搜索失败时自动使用文本搜索作为备选
        - 结果排序：按相似度分数降序返回结果
        - Neo4j向量索引：利用数据库内置的向量索引加速搜索
        
        业务意义：
        - 提供语义级别的搜索能力，超越关键词匹配
        - 发现概念相似但表达方式不同的内容
        - 提高搜索结果的相关性和准确性
        - 为知识图谱查询提供向量检索支持
        - 支持Graph-RAG系统的核心语义理解功能
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 构建Neo4j向量搜索查询
            cypher = """
            CALL db.index.vector.queryNodes('vector', $limit, $embedding)
            YIELD node, score
            RETURN node.id AS id, score
            ORDER BY score DESC
            """
            
            # 执行搜索
            results = self.db_query(cypher, {
                "embedding": query_embedding,
                "limit": limit
            })
            
            # 提取实体ID
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"向量搜索失败: {e}")
            # 如果向量搜索失败，尝试使用文本搜索作为备用
            return self.text_search(query, limit)
    
    def text_search(self, query: str, limit: int = 5) -> List[str]:
        """
        基于文本匹配的搜索方法（作为向量搜索的备选）
        
        该方法是Graph-RAG系统中的文本检索备用机制，负责在向量搜索失败时提供基于关键词匹配的搜索能力。
        通过在实体ID和描述字段中查找包含查询关键词的内容，它提供了一种简单但可靠的搜索方式，
        确保即使在复杂场景下系统也能获取相关信息。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            limit: 最大返回结果数，控制搜索结果的数量，默认为5
            
        返回:
            List[str]: 匹配实体ID列表，按照匹配顺序返回
            
        实现思路：
        1. 构建基于CONTAINS操作符的Cypher查询，在实体ID和描述字段中查找关键词
        2. 设置查询限制，控制返回结果数量
        3. 执行文本搜索查询
        4. 从结果中提取实体ID并返回
        5. 实现异常处理，确保即使搜索失败也能返回空列表而不是中断
        
        技术特点：
        - 关键词匹配：基于简单的字符串包含关系进行匹配
        - 多字段搜索：同时在实体ID和描述字段中搜索
        - 限制结果：控制返回的结果数量，避免过多信息
        - 异常处理：完善的错误捕获机制
        - 作为降级方案：与向量搜索配合使用，提高系统可靠性
        
        业务意义：
        - 提供向量搜索的降级方案，增强系统稳定性
        - 在特定场景下可能更准确（如精确ID匹配）
        - 确保系统在任何情况下都能提供基本搜索功能
        - 为Graph-RAG系统提供双重保障的检索机制
        - 适应不同类型的查询需求
        """
        try:
            # 构建全文搜索查询
            cypher = """
            MATCH (e:__Entity__)
            WHERE e.id CONTAINS $query OR e.description CONTAINS $query
            RETURN e.id AS id
            LIMIT $limit
            """
            
            results = self.db_query(cypher, {
                "query": query,
                "limit": limit
            })
            
            if not results.empty:
                return results['id'].tolist()
            else:
                return []
                
        except Exception as e:
            print(f"文本搜索失败: {e}")
            return []
            
    def semantic_search(self, query: str, entities: List[Dict], 
                        embedding_field: str = "embedding", 
                        top_k: int = 5) -> List[Dict]:
        """
        对一组实体进行语义相似度搜索
        
        该方法是Graph-RAG系统中高级语义搜索的核心实现，负责计算查询与一组预检索实体之间的语义相似度，
        并根据相似度对实体进行排序。与基础向量搜索不同，它针对已经检索到的特定实体集合进行精细化排序，
        显著提高了搜索结果的针对性和相关性。
        
        参数:
            query: 搜索查询字符串，用户的原始问题或搜索关键词
            entities: 实体列表，包含预检索的实体及其嵌入向量
            embedding_field: 嵌入向量的字段名，指定实体中存储嵌入向量的字段，默认为"embedding"
            top_k: 返回的最大结果数，控制返回的相似度最高的实体数量，默认为5
            
        返回:
            List[Dict]: 按相似度排序的实体列表，每项增加"score"字段表示相似度得分
            
        实现思路：
        1. 将查询文本转换为向量表示（嵌入）
        2. 调用VectorUtils.rank_by_similarity方法计算查询向量与每个实体向量的相似度
        3. 根据相似度得分对实体列表进行排序
        4. 限制返回的实体数量为top_k个
        5. 实现异常处理，当语义搜索失败时返回原始实体的前top_k个
        
        技术特点：
        - 语义相似度计算：计算查询与实体之间的向量相似度
        - 精细排序：对预检索实体进行二次排序，提高相关性
        - 工具类复用：利用VectorUtils工具类实现相似度计算
        - 自动降级：在搜索失败时提供基本结果
        - 结果增强：为实体添加相似度得分信息
        
        业务意义：
        - 提供精细化的语义排序能力
        - 显著提高搜索结果的相关性
        - 支持复杂查询场景下的精准实体筛选
        - 为知识图谱实体提供语义层面的排序机制
        - 增强Graph-RAG系统的语义理解和推理能力
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 使用工具类进行排序
            return VectorUtils.rank_by_similarity(
                query_embedding, 
                entities, 
                embedding_field, 
                top_k
            )
        except Exception as e:
            print(f"语义搜索失败: {e}")
            return entities[:top_k] if top_k else entities
    
    def filter_by_relevance(self, query: str, docs: List, top_k: int = 5) -> List:
        """
        根据相关性过滤文档
        
        该方法是Graph-RAG系统中文档相关性过滤的核心实现，负责根据查询与文档的语义相似度对文档列表进行排序和筛选。
        通过计算语义相关性，它能够从大量文档中找出最相关的内容，显著提高信息检索的精准度和效率。
        
        参数:
            query: 查询字符串，用户的原始问题或搜索关键词
            docs: 文档列表，包含需要进行相关性过滤的文档集合
            top_k: 返回的最大结果数，控制返回的相关性最高的文档数量，默认为5
            
        返回:
            List: 按相关性排序的文档列表，仅包含相关性最高的top_k个文档
            
        实现思路：
        1. 将查询文本转换为向量表示（嵌入）
        2. 调用VectorUtils.filter_documents_by_relevance方法基于向量相似度过滤文档
        3. 按照相关性得分对文档进行排序
        4. 限制返回的文档数量为top_k个
        5. 实现异常处理，当过滤失败时返回原始文档的前top_k个
        
        技术特点：
        - 语义相关性计算：基于向量相似度评估文档相关性
        - 高效过滤：快速从大量文档中筛选出最相关内容
        - 工具类复用：利用VectorUtils工具类实现文档过滤
        - 自动降级：在过滤失败时提供基本结果
        - 结果排序：确保最相关文档排在前面
        
        业务意义：
        - 提高信息检索的精准度和效率
        - 减少无关文档的干扰
        - 为后续的回答生成提供高质量的输入
        - 优化用户的搜索体验
        - 支持Graph-RAG系统的文档语义理解功能
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            return VectorUtils.filter_documents_by_relevance(
                query_embedding,
                docs,
                top_k=top_k
            )
        except Exception as e:
            print(f"文档过滤失败: {e}")
            return docs[:top_k] if top_k else docs
    
    def get_tool(self) -> BaseTool:
        """
        获取搜索工具实例
        
        该方法是Graph-RAG系统中工具集成的核心实现，负责将BaseSearchTool的子类实例转换为LangChain的BaseTool对象。
        通过创建动态的工具类，它实现了自定义搜索工具与LangChain工具系统的无缝集成，使得搜索功能可以被Agent
        或其他组件以标准方式调用。
        
        返回:
            BaseTool: 一个基于当前搜索工具类的LangChain BaseTool实例，包含名称、描述和执行方法
            
        实现思路：
        1. 在方法内部定义一个动态的工具类DynamicSearchTool，继承自BaseTool
        2. 设置工具名称为当前类名的小写形式
        3. 设置工具描述为通用的高级搜索工具描述
        4. 实现_run方法，将调用委托给当前实例的search方法
        5. 声明_arun方法（异步执行）为未实现
        6. 创建并返回DynamicSearchTool的实例
        
        技术特点：
        - 动态类创建：在运行时动态定义工具类
        - 委托模式：将工具执行委托给原始搜索工具实例
        - 接口适配：将自定义工具适配到LangChain工具接口
        - 同步执行：只实现同步执行方法
        - 命名约定：使用类名小写作为工具名称
        
        业务意义：
        - 实现搜索工具与LangChain生态的集成
        - 支持Agent和工具链调用搜索功能
        - 提供统一的工具调用接口
        - 简化工具注册和使用过程
        - 为Graph-RAG系统提供标准的工具交互方式
        """
        # 创建动态工具类
        class DynamicSearchTool(BaseTool):
            name : str= f"{self.__class__.__name__.lower()}"
            description : str = "高级搜索工具，用于在知识库中查找信息"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DynamicSearchTool()
    
    def _log_performance(self, operation: str, start_time: float):
        """
        记录性能指标
        
        参数:
            operation: 操作名称
            start_time: 开始时间
        """
        duration = time.time() - start_time
        self.performance_metrics[operation] = duration
        print(f"性能指标 - {operation}: {duration:.4f}s")
    
    def close(self):
        """关闭资源连接"""
        # 关闭Neo4j连接
        if hasattr(self, 'graph'):
            # 如果Neo4jGraph有close方法，调用它
            if hasattr(self.graph, 'close'):
                self.graph.close()
    
    def __enter__(self):
        """
        上下文管理器入口方法
        
        返回:
            self: 当前工具实例
            
        设计目的:
        - 支持Python的上下文管理器协议
        - 允许使用with语句管理工具生命周期
        - 提供简洁的API使用方式
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口方法
        
        参数:
            exc_type: 异常类型（如果有）
            exc_val: 异常值（如果有）
            exc_tb: 异常回溯信息（如果有）
            
        实现思路:
        - 调用close方法释放资源
        - 支持异常处理流程
        - 确保无论是否发生异常，资源都能正确释放
        
        业务意义:
        - 提供安全可靠的资源管理
        - 避免资源泄漏
        - 支持优雅的异常处理
        - 简化API使用方式
        """
        self.close()