"""
全局搜索模块

该模块实现了基于Neo4j图数据库的全局搜索功能，采用Map-Reduce模式在整个知识图谱范围内进行搜索。

全局搜索的核心特点是对知识图谱中的所有社区数据进行系统性检索和处理，通过分布式计算的思想，
将复杂查询分解为多个子任务并行处理，然后整合结果生成全面的回答。
"""
from typing import List
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompt import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.neo4jdb import get_db_manager

class GlobalSearch:
    """
    全局搜索类
    
    该类实现了在整个知识图谱范围内的搜索功能，基于Map-Reduce分布式计算模式，
    通过将大型搜索任务分解为多个并行子任务，然后整合结果生成全面的回答。
    
    设计思路：
    - 采用分布式计算模式处理大规模图数据
    - 将搜索过程分为Map和Reduce两个主要阶段
    - 支持按社区层级进行定向搜索
    - 利用LLM进行语义理解和结果生成
    
    搜索流程：
    1. 获取指定层级的所有社区数据
    2. Map阶段：为每个社区生成中间结果
    3. Reduce阶段：整合所有中间结果生成最终答案
    
    业务价值：
    - 能够处理超大规模知识图谱的搜索需求
    - 通过并行处理提高搜索效率
    - 提供全面而系统的知识检索结果
    - 支持多维度、多层次的信息整合
    """
    
    def __init__(self, llm, response_type: str = "多个段落"):
        """
        初始化全局搜索类
        
        参数:
            llm: 大语言模型实例，用于处理查询和生成答案
            response_type: 响应类型格式，默认为"多个段落"
            
        实现思路:
        1. 保存模型实例和配置参数
        2. 获取数据库连接管理器
        3. 初始化Neo4j图实例，用于后续图数据查询
        
        设计特点:
        - 使用依赖注入模式管理LLM和数据库连接
        - 支持配置不同的响应类型格式
        - 通过数据库连接管理器获取图实例，实现资源管理解耦
        """
        # 保存模型实例和配置
        self.llm = llm
        self.response_type = response_type
        
        # 使用数据库连接管理
        db_manager = get_db_manager()
        
        # 初始化Neo4j图实例
        self.graph = db_manager.get_graph()
        
    def _get_community_data(self, level: int) -> List[dict]:
        """
        获取指定层级的社区数据
        
        参数:
            level: 社区层级，用于指定搜索范围
            
        返回:
            List[dict]: 社区数据字典列表，每个字典包含社区ID和完整内容
            
        实现思路:
        1. 执行Cypher查询，匹配指定层级的所有社区节点
        2. 提取每个社区的ID和完整内容
        3. 将结果格式化为字典列表返回
        
        业务意义:
        - 实现按层级搜索，支持多层次知识结构
        - 为Map阶段提供数据输入
        - 通过层级控制搜索范围，平衡全面性和性能
        """
        return self.graph.query(
            """
            MATCH (c:__Community__)
            WHERE c.level = $level
            RETURN {communityId:c.id, full_content:c.full_content} AS output
            """,
            params={"level": level},
        )
    
    def _process_communities(self, query: str, communities: List[dict]) -> List[str]:
        """
        处理社区数据生成中间结果（Map阶段）
        
        参数:
            query: 搜索查询字符串
            communities: 社区数据列表
            
        返回:
            List[str]: 中间结果列表，每个结果对应一个社区的处理结果
            
        实现思路:
        1. 创建Map阶段的提示模板，配置系统指令和输入格式
        2. 构建处理链：提示模板 -> LLM -> 输出解析器
        3. 遍历每个社区数据，使用tqdm提供进度显示
        4. 对每个社区数据调用LLM，生成与查询相关的中间结果
        5. 收集所有中间结果并返回
        
        Map阶段的关键技术点:
        - 使用LangChain的提示模板控制输出格式
        - 采用进度条显示处理状态
        - 每个社区的处理是独立的，支持并行优化
        - 通过系统提示指导LLM关注与查询相关的信息
        
        业务价值:
        - 将复杂查询分解为多个可并行的子任务
        - 确保每个社区的信息都被充分考虑
        - 为后续Reduce阶段准备结构化的中间结果
        """
        # 设置Map阶段的提示模板
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM_PROMPT),
            ("human", """
                ---数据表格--- 
                {context_data}
                
                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Map阶段的处理链
        map_chain = map_prompt | self.llm | StrOutputParser()
        
        # 处理每个社区
        results = []
        for community in tqdm(communities, desc="正在处理社区数据"):
            response = map_chain.invoke({
                "question": query,
                "context_data": community["output"]
            })
            results.append(response)
            print(response)  # 输出处理进度
            
        return results
    
    def _reduce_results(self, query: str, intermediate_results: List[str]) -> str:
        """
        整合中间结果生成最终答案（Reduce阶段）
        
        参数:
            query: 搜索查询字符串
            intermediate_results: 中间结果列表，来自Map阶段的处理结果
            
        返回:
            str: 最终生成的综合答案
            
        实现思路:
        1. 创建Reduce阶段的提示模板，配置系统指令和输入格式
        2. 构建处理链：提示模板 -> LLM -> 输出解析器
        3. 将所有中间结果作为报告数据传递给LLM
        4. 调用LLM整合所有信息，生成最终答案
        
        Reduce阶段的关键技术点:
        - 使用专门的系统提示指导LLM如何整合多源信息
        - 保持原始查询上下文，确保回答的相关性
        - 控制输出格式，生成结构化的最终答案
        - 支持不同的响应类型配置
        
        业务价值:
        - 整合多个社区的分散信息
        - 消除冗余，提取共同主题和关键信息
        - 生成连贯、全面、结构化的最终答案
        - 确保答案的一致性和完整性
        """
        # 设置Reduce阶段的提示模板
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        
        # 创建Reduce阶段的处理链
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        
        # 生成最终答案
        return reduce_chain.invoke({
            "report_data": intermediate_results,
            "question": query,
            "response_type": self.response_type,
        })
    
    def search(self, query: str, level: int) -> str:
        """
        执行全局搜索的核心方法
        
        参数:
            query: 用户的搜索查询字符串
            level: 要搜索的社区层级，控制搜索范围和粒度
            
        返回:
            str: 生成的最终结构化答案
            
        实现思路:
        1. 获取指定层级的所有社区数据
        2. 执行Map阶段处理，为每个社区生成中间结果
        3. 执行Reduce阶段处理，整合所有中间结果生成最终答案
        4. 返回生成的答案
        
        全局搜索工作流程:
        - 首先确定搜索范围（通过level参数指定社区层级）
        - 获取该层级的所有社区数据
        - 对每个社区并行处理，生成与查询相关的中间分析结果
        - 整合所有中间结果，消除冗余，提取共同主题
        - 生成连贯、全面、结构化的最终答案
        
        业务意义:
        - 实现跨多个社区的全局知识检索
        - 通过层级控制实现灵活的搜索粒度
        - 提供系统化、全面的知识整合能力
        - 适用于需要广泛信息源的复杂查询
        """
        # 获取社区数据
        communities = self._get_community_data(level)
        
        # 处理社区数据（Map阶段）
        intermediate_results = self._process_communities(query, communities)
        
        # 生成最终答案（Reduce阶段）
        return self._reduce_results(query, intermediate_results)
        
    def close(self):
        """
        关闭资源连接
        
        实现思路:
        - 预留的资源释放方法
        - 当前版本为空实现，可在需要时添加资源释放逻辑
        
        设计考虑:
        - 提供统一的资源管理接口
        - 支持未来扩展，如数据库连接关闭、会话终止等
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
        - 提供优雅的API使用方式
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
        - 支持优雅的错误处理和资源管理
        """
        self.close()