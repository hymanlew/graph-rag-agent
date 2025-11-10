from typing import Dict, Any
import pandas as pd
from neo4j import Result
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.prompt import LC_SYSTEM_PROMPT
from config.neo4jdb import get_db_manager

class LocalSearch:
    """
    局部搜索类，其核心特点是聚焦于图中与查询最相关的局部区域，检索相关实体、关系、文本块和社区信息，
    为用户提供深入而精确的知识检索结果。
    该类实现了在知识图谱中的局部区域搜索，通过向量相似度匹配和图结构遍历，
    依次检索与查询相关的实体、关系、文本块和社区信息，并利用大语言模型生成结构化回答。
    
    设计思路：
    - 基于向量相似度的文本检索
    - 采用分层检索策略，先定位相关实体，再扩展获取相关文本和关系
    - 支持多种信息类型的整合（文本、实体、关系、社区）
    - 图结构的遍历与信息提取
    - 使用LLM生成结构化最终答案
    - 支持上下文管理（with语句）
    - 提供灵活的检索参数控制
    """
    
    def __init__(self, llm, embeddings, response_type: str = "多个段落"):
        """
        初始化本地搜索类
        
        参数:
            llm: 大语言模型实例，用于生成最终回答
            embeddings: 向量嵌入模型，用于计算文本相似度
            response_type: 响应类型格式，默认为"多个段落"

        配置说明:
        - 多种检索限制参数，控制返回结果的数量和质量
        - 支持自定义向量索引名称
        - 社区权重初始化，优化社区排序
        """
        # 保存模型实例和配置
        self.llm = llm
        self.embeddings = embeddings
        self.response_type = response_type
        
        # 获取数据库连接管理器
        db_manager = get_db_manager()
        self.driver = db_manager.get_driver()
        
        # 设置检索参数
        self.top_chunks = 3         # 最多返回的文本块数
        self.top_communities = 3    # 最多返回的社区数
        self.top_outside_rels = 10  # 最多返回的外部关系数
        self.top_inside_rels = 10   # 最多返回的内部关系数
        self.top_entities = 10      # 最多返回的实体数
        self.index_name = 'vector'  # 向量索引名称
        
        # 初始化社区节点权重
        self._init_community_weights()
        
        # 配置Neo4j连接信息
        self.neo4j_uri = db_manager.neo4j_uri
        self.neo4j_username = db_manager.neo4j_username
        self.neo4j_password = db_manager.neo4j_password
        
    def _init_community_weights(self):
        """
        初始化Neo4j中社区节点的权重，计算每个社区包含的不同文本块数量作为权重
        
        业务意义:
        - 通过文本块数量衡量社区的信息量和重要性
        - 权重值用于社区排序，优先返回内容丰富的社区
        - 优化搜索结果的相关性排序
        """
        self.db_query("""
        MATCH (n:`__Community__`)<-[:IN_COMMUNITY]-()<-[:MENTIONS]-(c)
        WITH n, count(distinct c) AS chunkCount
        SET n.weight = chunkCount
        """)
        
    def db_query(self, cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        执行Cypher查询并返回结果
        
        参数:
            cypher: Cypher查询语句
            params: 查询参数，以字典形式提供
            
        返回:
            pandas.DataFrame: 查询结果，转换为DataFrame格式方便处理
        """
        # 结果转换为DataFrame，方便后续数据处理和分析
        return self.driver.execute_query(
            cypher,
            parameters_=params,
            result_transformer_=Result.to_df
        )
        
    @property
    def retrieval_query(self) -> str:
        """
        获取Neo4j检索查询语句
        
        返回:
            str: Cypher查询语句，用于检索相关内容
            
        实现思路:
        1. 首先收集所有相关节点 - collect
        2. 使用 collect + UNWIND 子查询分别提取不同类型的信息：
           - 相关文本块（Chunk）：匹配与节点相关的文本内容，按每个文本块关联的节点数量排序
           - 社区报告（Community）：获取节点所属社区的摘要
           - 外部关系：获取节点与外部实体的关系描述
           - 内部关系：获取节点集合内部实体之间的关系
           - 实体描述：收集所有相关实体的描述信息
        3. 将所有信息整合为统一的返回结构
        
        关键查询策略:
        - 使用节点频率（freq）对文本块排序
        - 基于社区排名和权重排序社区信息
        - 基于关系权重排序关系描述
        - 限制返回数量，避免信息过载
        """
        return """
        WITH collect(node) as nodes
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WITH distinct c, c.community_rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT $topCommunities
        } AS report_mapping,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {
            Chunks: text_mapping, 
            Reports: report_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities
        } AS text, 1.0 AS score, {} AS metadata
        """
    
    def as_retriever(self, **kwargs):
        """
        返回检索器实例，用于链式调用
        
        参数:
            **kwargs: 额外的检索参数
            
        返回:
            检索器实例，可用于LangChain的链式调用
        """
        # 生成包含所有检索参数的查询，将参数化的Cypher查询中的占位符替换为实际值
        final_query = self.retrieval_query.replace("$topChunks", str(self.top_chunks))\
            .replace("$topCommunities", str(self.top_communities))\
            .replace("$topOutsideRels", str(self.top_outside_rels))\
            .replace("$topInsideRels", str(self.top_inside_rels))

        # 基于现有索引初始化Neo4jVector向量存储
        db_manager = get_db_manager()
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=db_manager.neo4j_uri,
            username=db_manager.neo4j_username,
            password=db_manager.neo4j_password,
            index_name=self.index_name,
            retrieval_query=final_query
        )
        
        # 返回检索器
        return vector_store.as_retriever(
            search_kwargs={"k": self.top_entities}
        )
        
    def search(self, query: str) -> str:
        """
        执行局部搜索的核心方法
        
        参数:
            query: 用户的搜索查询字符串
            
        返回:
            str: 生成的结构化最终答案
        """
        # 初始化对话提示模板，定义系统指令和输入输出格式
        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                {context}

                用户的问题是：
                {input}

                请按以下格式输出回答：
                1. 使用三级标题(###)标记主题
                2. 主要内容用清晰的段落展示
                3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
                """
             )
        ])
        
        # 创建搜索链：提示模板 -> LLM -> 输出解析器
        chain = prompt | self.llm | StrOutputParser()
        
        # 初始化向量存储，配置数据库连接和检索查询
        vector_store = Neo4jVector.from_existing_index(
            self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name=self.index_name,
            retrieval_query=self.retrieval_query
        )
        
        # 执行相似度搜索，获取相关文档
        # Neo4jVector 类内部会自动处理 query 的向量转换
        # # 1. 向量搜索阶段（自动执行）
        # CALL db.index.vector.queryNodes("index_name", $top_entities, $query_vector)
        # YIELD node, score
        #
        # # 2. 你的 final_query 阶段（处理搜索结果）
        # WITH collect(node) as nodes  # nodes 已经是向量搜索的结果
        # // 这里的所有查询都是基于已经找到的节点进行图遍历，只返回文本和元数据
        docs = vector_store.similarity_search(
            query,
            k=self.top_entities,
            params={
                "topChunks": self.top_chunks,
                "topCommunities": self.top_communities,
                "topOutsideRels": self.top_outside_rels,
                "topInsideRels": self.top_inside_rels,
            }
        )
        
        # 使用LLM生成响应，传入上下文、查询和响应类型
        response = chain.invoke({
            "context": docs[0].page_content if docs else "",  # 安全处理空结果
            "input": query,
            "response_type": self.response_type
        })
        return response
        
    def close(self):
        """
        关闭Neo4j驱动连接
        - 预留的资源释放方法
        - 当前版本为空实现，可在需要时添加驱动关闭逻辑
        """
        pass
        
    def __enter__(self):
        """
        上下文管理器入口方法
        
        返回:
            self: 当前实例，支持with语句使用
            
        设计目的:
        - 支持Python的上下文管理器协议
        - 允许使用with语句管理资源生命周期
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口方法
        
        参数:
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常回溯信息
            
        实现思路:
        - 调用close方法释放资源
        - 支持异常处理流程
        
        业务意义:
        - 确保资源正确释放
        - 提供简洁的API使用方式
        """
        self.close()