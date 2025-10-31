from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from model.get_models import get_llm_model
import concurrent.futures
import time

from config.settings import MAX_WORKERS

"""
社区摘要基础模块
提供了社区摘要生成的通用流程，同时允许不同算法的摘要生成器定制特定的信息收集策略。

主要组件：
- BaseCommunityDescriber: 社区信息格式化工具，将结构化数据转换为LLM友好的格式
- BaseCommunityRanker: 社区权重计算工具，基于社区中的文档和实体数量评估社区重要性
- BaseCommunityStorer: 社区摘要存储工具，负责将生成的摘要保存到图数据库
- BaseSummarizer: 摘要生成器抽象基类，定义摘要生成的标准流程

设计特点：
- 组件化架构：将功能划分为多个独立的组件，便于维护和扩展
- 模板方法模式：定义算法骨架，子类实现具体步骤
- 并行处理：支持多线程并行生成社区摘要
- 性能监控：详细记录各个阶段的执行时间
- 错误处理：完善的异常捕获和回退机制
"""

class BaseCommunityDescriber:
    """社区信息格式化工具
    
    负责将社区的结构化数据转换为LLM可处理的自然语言格式。
    此工具创建节点和关系的可读表示，使LLM能够更好地理解社区的结构和内容。
    """
    
    @staticmethod
    def prepare_string(data: Dict) -> str:
        """转换社区信息为可读字符串
        
        将社区中的节点和关系数据转换为结构化的文本格式，便于LLM理解和处理。
        生成的文本包含两部分：节点信息和关系信息。
        
        参数：
            data: 包含社区节点和关系信息的字典
            
        返回：
            格式化后的可读字符串
            
        异常处理：
            如果格式化过程出错，返回错误信息和原始数据的字符串表示
        """
        try:
            # 构建节点信息字符串
            nodes_str = "Nodes are:\n"
            for node in data.get('nodes', []):
                node_id = node.get('id', 'unknown_id')
                node_type = node.get('type', 'unknown_type')
                # 处理可选的节点描述
                node_description = (
                    f", description: {node['description']}"
                    if 'description' in node and node['description']
                    else ""
                )
                nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

            # 构建关系信息字符串
            rels_str = "Relationships are:\n"
            for rel in data.get('rels', []):
                start = rel.get('start', 'unknown_start')
                end = rel.get('end', 'unknown_end')
                rel_type = rel.get('type', 'unknown_type')
                # 处理可选的关系描述
                description = (
                    f", description: {rel['description']}"
                    if 'description' in rel and rel['description']
                    else ""
                )
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            # 合并节点和关系信息
            return nodes_str + "\n" + rels_str
        except Exception as e:
            print(f"格式化社区信息时出错: {e}")
            # 返回错误信息和原始数据，便于调试
            return f"Error: {str(e)}\nData: {str(data)}"

class BaseCommunityRanker:
    """社区权重计算工具
    
    负责为社区计算权重，用于评估社区的重要性。
    实现了基于文档和实体数量的权重计算策略，并提供备用计算方法。
    """
    
    def __init__(self, graph: Neo4jGraph):
        """初始化社区权重计算器
        
        参数：
            graph: Neo4j图实例，用于执行图查询
        """
        self.graph = graph
    
    def calculate_ranks(self) -> None:
        """计算社区权重
        
        基于社区中提到的文档数量计算社区权重。权重值保存在社区节点的community_rank属性中。
        如果主计算方法失败，自动切换到备用方法。
        
        实现步骤：
        1. 记录开始时间并打印日志
        2. 执行主计算查询，通过关系链统计每个社区中的不同文档数
        3. 将计算结果保存到社区节点的community_rank属性
        4. 处理异常并切换到备用计算方法
        """
        start_time = time.time()
        print("计算社区权重...")
        
        try:
            # 主权重计算：通过关系链统计每个社区中不同文档的数量
            # 社区 <- 实体 <- 文档引用
            result = self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank
            RETURN count(c) AS processed_count
            """)
            
            processed_count = result[0]['processed_count'] if result else 0
            print(f"社区权重计算完成，处理了 {processed_count} 个社区，"
                  f"耗时: {time.time() - start_time:.2f}秒")
        except Exception as e:
            print(f"计算社区权重时出错: {e}")
            # 执行备用计算方法
            self._calculate_ranks_fallback()
    
    def _calculate_ranks_fallback(self):
        """备用的权重计算方法
        
        当主计算方法失败时，使用简化的权重计算策略，基于社区中的实体数量计算权重。
        这种方法虽然精度较低，但计算速度更快，且对图结构的要求更简单。
        """
        try:
            # 备用权重计算：统计每个社区中的实体数量
            self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:`__Entity__`)
            WITH c, count(e) AS entity_count
            SET c.community_rank = entity_count
            """)
            print("使用实体计数作为社区权重")
        except Exception as e:
            print(f"备用权重计算也失败: {e}")

class BaseCommunityStorer:
    """社区信息存储工具
    
    负责将生成的社区摘要存储到图数据库中。
    实现了批量存储和单条存储两种模式，支持异常处理和回退机制。
    """
    
    def __init__(self, graph: Neo4jGraph):
        """初始化社区摘要存储工具
        
        参数：
            graph: Neo4j图实例，用于执行图查询和数据存储
        """
        self.graph = graph
    
    def store_summaries(self, summaries: List[Dict]) -> None:
        """存储社区摘要
        
        将生成的社区摘要批量存储到图数据库中。采用批处理策略提高效率，
        当批量存储失败时，自动切换到逐条存储的回退策略。
        
        参数：
            summaries: 社区摘要列表，每个摘要包含community、summary和full_content字段
            
        实现步骤：
        1. 检查摘要列表是否为空
        2. 计算合适的批处理大小
        3. 分批执行存储操作
        4. 处理异常并切换到逐条存储
        """
        # 处理空摘要列表
        if not summaries:
            print("没有社区摘要需要存储")
            return
            
        start_time = time.time()
        print(f"开始存储 {len(summaries)} 个社区摘要...")
        
        # 计算合适的批处理大小：最小10，最大100，或摘要总数的1/5
        batch_size = min(100, max(10, len(summaries) // 5))
        total_batches = (len(summaries) + batch_size - 1) // batch_size
        
        # 批处理摘要存储
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_start = time.time()
            
            try:
                # 使用UNWIND进行批量处理，提高性能
                self.graph.query("""
                UNWIND $data AS row
                MERGE (c:__Community__ {id:row.community})
                SET c.summary = row.summary, 
                    c.full_content = row.full_content,
                    c.summary_created_at = datetime()
                """, params={"data": batch})
                
                print(f"已存储批次 {i//batch_size + 1}/{total_batches}, "
                      f"耗时: {time.time() - batch_start:.2f}秒")
                
            except Exception as e:
                print(f"存储社区摘要批次时出错: {e}")
                # 批次处理失败，尝试逐条存储
                self._store_summaries_one_by_one(batch)
    
    def _store_summaries_one_by_one(self, summaries: List[Dict]):
        """逐个存储社区摘要
        
        当批量存储失败时的回退策略，逐条存储社区摘要。
        虽然效率较低，但可以避免因个别摘要问题影响整个批次。
        
        参数：
            summaries: 需要逐条存储的社区摘要列表
        """
        for summary in summaries:
            try:
                # 单条存储社区摘要
                self.graph.query("""
                MERGE (c:__Community__ {id:$community})
                SET c.summary = $summary, 
                    c.full_content = $full_content,
                    c.summary_created_at = datetime()
                """, params=summary)
            except Exception as e:
                print(f"存储单个社区摘要时出错: {e}")

class BaseSummarizer(ABC):
    """社区摘要生成器基类
    
    定义社区摘要生成的抽象基类，实现了模板方法模式。提供摘要生成的通用流程，
    具体的社区信息收集逻辑由子类实现。
    
    核心功能：
    - 社区权重计算
    - 社区信息收集（抽象方法）
    - 并行摘要生成
    - 摘要存储
    - 性能监控和统计
    """
    
    def __init__(self, graph: Neo4jGraph):
        """初始化社区摘要生成器基类
        
        参数：
            graph: Neo4j图实例，提供图数据访问能力
        """
        self.graph = graph
        # 获取LLM模型用于生成摘要
        self.llm = get_llm_model()
        # 初始化各功能组件
        self.describer = BaseCommunityDescriber()
        self.ranker = BaseCommunityRanker(graph)
        self.storer = BaseCommunityStorer(graph)
        # 设置LLM处理链
        self._setup_llm_chain()
        
        # 性能监控变量
        self.llm_time = 0
        self.query_time = 0
        self.store_time = 0
        
        # 并行处理配置
        self.max_workers = MAX_WORKERS
        print(f"社区摘要生成器初始化，并行线程数: {self.max_workers}")

    def _setup_llm_chain(self) -> None:
        """设置LLM处理链
        
        配置用于生成社区摘要的LLM处理链，包括提示模板、模型和输出解析器。
        
        异常：
            如果设置失败，抛出异常
        """
        try:
            # 定义摘要生成的提示模板
            community_prompt = ChatPromptTemplate.from_messages([
                ("system", "给定一个输入三元组，生成信息摘要。没有序言。"),
                ("human", "{community_info}"),
            ])
            # 构建LLM处理链：提示模板 -> LLM -> 输出解析器
            self.community_chain = community_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"设置LLM处理链时出错: {e}")
            raise

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """收集社区信息的抽象方法
        
        由子类实现，负责从图数据库中收集社区的详细信息，包括节点和关系。
        不同的社区检测算法可能需要不同的信息收集策略。
        
        返回：
            社区信息列表，每个元素是包含社区信息的字典
        """
        pass

    def process_communities(self) -> List[Dict]:
        """处理所有社区
        
        实现社区摘要生成的主流程，执行权重计算、信息收集、摘要生成和存储。
        这是模板方法，定义了算法骨架，具体步骤由子类实现。
        
        返回：
            生成的社区摘要列表
            
        异常：
            如果处理过程出错，抛出异常
        """
        total_start_time = time.time()
        print("开始处理社区摘要...")
        
        try:
            # 计算社区权重
            rank_start = time.time()
            self.ranker.calculate_ranks()
            rank_time = time.time() - rank_start
            
            # 收集社区信息（由子类实现）
            query_start = time.time()
            community_info = self.collect_community_info()
            self.query_time = time.time() - query_start
            
            # 处理空结果
            if not community_info:
                print("没有找到需要处理的社区")
                return []
            
            # 并行生成摘要
            llm_start = time.time()
            # 计算最佳线程数：不超过最大线程数，不少于1，或社区数量的一半
            optimal_workers = min(self.max_workers, max(1, len(community_info) // 2))
            print(f"开始并行生成 {len(community_info)} 个社区摘要，"
                  f"使用 {optimal_workers} 个线程...")
            
            summaries = self._process_communities_parallel(
                community_info, 
                optimal_workers
            )
            
            self.llm_time = time.time() - llm_start
            
            # 保存摘要
            store_start = time.time()
            self.storer.store_summaries(summaries)
            self.store_time = time.time() - store_start
            
            # 输出性能统计
            total_time = time.time() - total_start_time
            self._print_performance_stats(
                total_time, rank_time, 
                self.query_time, self.llm_time, 
                self.store_time
            )
            
            return summaries
            
        except Exception as e:
            print(f"处理社区摘要时出错: {str(e)}")
            raise
    
    def _process_communities_parallel(
        self, 
        community_info: List[Dict], 
        workers: int
    ) -> List[Dict]:
        """并行处理社区摘要
        
        使用线程池并行处理多个社区的摘要生成，提高处理效率。
        
        参数：
            community_info: 社区信息列表
            workers: 并行线程数
            
        返回：
            生成的社区摘要列表
        """
        summaries = []
        # 创建线程池执行器
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务并记录future
            future_to_community = {
                executor.submit(self._process_single_community, info): i 
                for i, info in enumerate(community_info)
            }
            
            # 处理完成的任务
            for i, future in enumerate(concurrent.futures.as_completed(future_to_community)):
                try:
                    result = future.result()
                    summaries.append(result)
                    
                    # 定期输出进度
                    if (i+1) % 10 == 0 or (i+1) == len(community_info):
                        print(f"已处理 {i+1}/{len(community_info)} "
                              f"({(i+1)/len(community_info)*100:.1f}%)")
                        
                except Exception as e:
                    print(f"处理社区摘要时出错: {e}")
        
        return summaries
    
    def _process_single_community(self, community: Dict) -> Dict:
        """处理单个社区摘要
        
        为单个社区生成摘要，包括信息格式化、LLM调用和结果处理。
        
        参数：
            community: 社区信息字典
            
        返回：
            包含社区ID、摘要和完整内容的字典
        """
        community_id = community.get('communityId', 'unknown')
        
        try:
            # 格式化社区信息为LLM可处理的字符串
            stringify_info = self.describer.prepare_string(community)
            
            # 检查信息是否足够生成摘要
            if len(stringify_info) < 10:
                print(f"社区 {community_id} 的信息太少，跳过摘要生成")
                return {
                    "community": community_id,
                    "summary": "此社区没有足够的信息生成摘要。",
                    "full_content": stringify_info
                }
            
            # 调用LLM生成摘要
            summary = self.community_chain.invoke({'community_info': stringify_info})
            
            # 返回格式化的结果
            return {
                "community": community_id,
                "summary": summary,
                "full_content": stringify_info
            }
        except Exception as e:
            print(f"处理社区 {community_id} 摘要时出错: {e}")
            # 异常情况下返回错误信息
            return {
                "community": community_id,
                "summary": f"生成摘要时出错: {str(e)}",
                "full_content": str(community)
            }
    
    def _print_performance_stats(
        self, 
        total_time: float,
        rank_time: float,
        query_time: float,
        llm_time: float,
        store_time: float
    ) -> None:
        """打印性能统计信息
        
        输出摘要生成过程中各个阶段的耗时和比例，用于性能分析和优化。
        
        参数：
            total_time: 总耗时
            rank_time: 权重计算耗时
            query_time: 信息查询耗时
            llm_time: LLM摘要生成耗时
            store_time: 结果存储耗时
        """
        print(f"\n社区摘要处理完成，总耗时: {total_time:.2f}秒")
        print(f"  社区权重计算: {rank_time:.2f}秒 ({rank_time/total_time*100:.1f}%)")
        print(f"  社区信息查询: {query_time:.2f}秒 ({query_time/total_time*100:.1f}%)")
        print(f"  摘要生成(LLM): {llm_time:.2f}秒 ({llm_time/total_time*100:.1f}%)")
        print(f"  结果存储: {store_time:.2f}秒 ({store_time/total_time*100:.1f}%)")