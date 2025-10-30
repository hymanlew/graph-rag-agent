import re
import concurrent.futures
from typing import List, Set
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graph.core import connection_manager
from config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class GraphWriter:
    """
    图数据库写入器
    
    功能：
    - 将提取的实体和关系数据写入Neo4j图数据库
    - 高效处理实体和关系的解析与转换，转换为 GraphDocument 对象
    - 实现节点缓存和批量处理优化
        - 并行处理多批次数据
        - 动态调整批次大小
        - 错误恢复和降级处理
    - 管理 Chunk 与 Document 节点的关系合并
    - 批量写入图文档
    
    实现思路：
    - 使用正则表达式解析实体关系文本
    - 采用节点缓存减少重复创建
    - 实现并行处理和动态批处理策略
    - 提供错误处理和降级机制确保数据写入稳定性
    """
    def __init__(self, graph: Neo4jGraph = None, batch_size: int = 50, max_workers: int = 4):
        """
        初始化图写入器
        
        参数：
            graph: Neo4j图数据库对象，如果为None则使用连接管理器获取
            batch_size: 批处理大小，控制每次写入的文档数量
            max_workers: 并行工作线程数，控制并发处理能力
        """
        # 初始化图数据库连接，如果没有提供则使用连接管理器获取
        self.graph = graph or connection_manager.get_connection()
        
        # 设置批处理大小和工作线程数，优先使用配置文件中的值
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 节点缓存，用于减少重复节点的创建，提高写入效率
        self.node_cache = {}
        
        # 已处理节点集合，用于跟踪已经处理的节点，减少重复操作
        self.processed_nodes: Set[str] = set()
        
    def convert_to_graph_document(self, chunk_id: str, input_text: str, result: str) -> GraphDocument:
        """
        将提取的实体关系文本转换为GraphDocument对象
        
        参数：
            chunk_id: 文本块ID，用于标识来源文本块
            input_text: 原始输入文本
            result: 实体关系提取结果文本
            
        返回：
            GraphDocument: 转换后的图文档对象，包含节点和关系信息
        """
        # 定义正则表达式模式，用于匹配实体和关系信息
        node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
        relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

        # 存储解析出的节点和关系，实现节点缓存机制，避免重复创建相同节点
        nodes = {}
        relationships = []

        try:
            # 解析节点 - 使用缓存提高效率
            for match in node_pattern.findall(result):
                node_id, node_type, description = match
                # 检查节点缓存，优先使用缓存中的节点
                if node_id in self.node_cache:
                    nodes[node_id] = self.node_cache[node_id]
                elif node_id not in nodes:
                    # 创建新节点并加入缓存
                    new_node = Node(
                        id=node_id,
                        type=node_type,
                        properties={'description': description}
                    )
                    nodes[node_id] = new_node
                    self.node_cache[node_id] = new_node

            # 解析关系
            for match in relationship_pattern.findall(result):
                source_id, target_id, rel_type, description, weight = match
                
                # 确保源节点存在，先检查缓存
                if source_id not in nodes:
                    if source_id in self.node_cache:
                        nodes[source_id] = self.node_cache[source_id]
                    else:
                        # 创建未知类型的源节点
                        new_node = Node(
                            id=source_id,
                            type="未知",
                            properties={'description': 'No additional data'}
                        )
                        nodes[source_id] = new_node
                        self.node_cache[source_id] = new_node
                        
                # 确保目标节点存在，先检查缓存
                if target_id not in nodes:
                    if target_id in self.node_cache:
                        nodes[target_id] = self.node_cache[target_id]
                    else:
                        # 创建未知类型的目标节点
                        new_node = Node(
                            id=target_id,
                            type="未知",
                            properties={'description': 'No additional data'}
                        )
                        nodes[target_id] = new_node
                        self.node_cache[target_id] = new_node
                    
                # 创建关系对象
                relationships.append(
                    Relationship(
                        source=nodes[source_id],
                        target=nodes[target_id],
                        type=rel_type,
                        properties={
                            "description": description,  # 关系描述
                            "weight": float(weight)      # 关系权重
                        }
                    )
                )
        except Exception as e:
            print(f"解析文本时出错: {e}")
            # 返回空的GraphDocument而不是引发异常，确保程序继续运行
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=Document(
                    page_content=input_text,
                    metadata={"chunk_id": chunk_id, "error": str(e)}
                )
            )

        # 创建并返回GraphDocument对象
        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(
                page_content=input_text,
                metadata={"chunk_id": chunk_id}
            )
        )
        
    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """
        处理并写入所有文件的GraphDocument对象 - 使用并行处理和批处理优化
        
        参数：
            file_contents: 文件内容列表，每个元素包含文件路径、内容和提取结果
        
        实现流程：
        1. 计算需要处理的总文本块数量并预分配内存
        2. 使用线程池并行处理文本块，转换为GraphDocument对象
        3. 收集处理结果，过滤掉无效文档
        4. 批量写入有效的GraphDocument对象
        5. 合并Chunk节点与Document节点的关系
        
        优化策略：
        - 预分配内存减少动态扩容开销
        - 并行处理提高CPU利用率
        - 错误隔离确保单个文本块失败不影响整体处理
        - 只处理和存储包含有效节点或关系的文档
        """
        # 预分配列表，避免动态扩容带来的性能损失
        total_chunks = sum(len(file_content[3]) for file_content in file_contents)
        all_graph_documents = [None] * total_chunks
        all_chunk_ids = [None] * total_chunks
        
        chunk_index = 0
        error_count = 0
        
        print(f"开始处理 {total_chunks} 个chunks的GraphDocument")
        
        # 使用线程池并行处理文本块转换
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}
            
            # 提交所有转换任务
            for file_content in file_contents:
                chunks = file_content[3]  # chunks_with_hash在索引3的位置
                results = file_content[4]  # 提取结果在索引4的位置
                
                for i, (chunk, result) in enumerate(zip(chunks, results)):
                    # 提交异步转换任务
                    future = executor.submit(
                        self.convert_to_graph_document,
                        chunk["chunk_id"],
                        chunk["chunk_doc"].page_content,
                        result
                    )
                    future_to_index[future] = chunk_index  # 记录任务与索引的映射关系
                    chunk_index += 1
            
            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    graph_document = future.result()
                    
                    # 只保留有效的图文档（包含节点或关系）
                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        all_graph_documents[idx] = graph_document
                        all_chunk_ids[idx] = graph_document.source.metadata.get("chunk_id")
                    else:
                        # 过滤掉空文档
                        all_graph_documents[idx] = None
                        all_chunk_ids[idx] = None
                        
                except Exception as e:
                    error_count += 1
                    print(f"处理chunk时出错 (已有{error_count}个错误): {e}")
                    # 发生错误时设置为None，后续会被过滤掉
                    all_graph_documents[idx] = None
                    all_chunk_ids[idx] = None
        
        # 过滤掉None值，只保留有效的文档和ID
        all_graph_documents = [doc for doc in all_graph_documents if doc is not None]
        all_chunk_ids = [id for id in all_chunk_ids if id is not None]
        
        print(f"共处理 {total_chunks} 个chunks, 有效文档 {len(all_graph_documents)}, 错误 {error_count}")
        
        # 批量写入有效的GraphDocument对象到图数据库
        self._batch_write_graph_documents(all_graph_documents)
        
        # 批量合并Chunk节点与Document节点的关系
        if all_chunk_ids:
            self.merge_chunk_relationships(all_chunk_ids)
    
    def _batch_write_graph_documents(self, documents: List[GraphDocument]) -> None:
        """
        批量写入图文档到Neo4j数据库，所有文件的GraphDocument对象（所有文本块数据）
        
        参数：
            documents: 待写入的图文档列表
        
        实现特点：
        - 动态调整批次大小，根据文档总数自动计算最优批次
        - 实现两级降级策略：批量写入 -> 单个写入
        - 详细的进度显示和错误处理
        - 空文档检查避免无效操作
        """
        # 检查是否有文档需要写入
        if not documents:
            return
            
        # 动态调整批处理大小，根据文档数量计算最优批次大小
        # 确保批次大小在合理范围内，至少10个，最多不超过配置的批处理大小
        optimal_batch_size = min(self.batch_size, max(10, len(documents) // 10))
        total_batches = (len(documents) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"开始批量写入 {len(documents)} 个文档，批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 批量写入图文档
        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i+optimal_batch_size]
            if batch:
                try:
                    # 批量添加图文档到数据库
                    self.graph.add_graph_documents(
                        batch,
                        baseEntityLabel=True,  # 使用基础实体标签
                        include_source=True    # 包含源文档信息
                    )
                    print(f"已写入批次 {i//optimal_batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"写入图文档批次时出错: {e}")
                    # 降级策略：如果批次写入失败，尝试逐个写入以避免整批失败
                    for doc in batch:
                        try:
                            self.graph.add_graph_documents(
                                [doc],
                                baseEntityLabel=True,
                                include_source=True
                            )
                        except Exception as e2:
                            print(f"单个文档写入失败: {e2}")
    
    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """
        合并Chunk节点与Document节点的关系
        
        参数：
            chunk_ids: 需要合并关系的Chunk ID列表
        
        功能说明：
        - 将Document节点的MENTIONS关系转移到对应的Chunk节点
        - 保留关系的所有属性
        - 转移完成后删除原Document节点，避免数据冗余
        """
        # 检查是否有Chunk ID需要处理
        if not chunk_ids:
            return
        
        # 去除重复的chunk_id以减少操作数量
        unique_chunk_ids = list(set(chunk_ids))
        print(f"开始合并 {len(unique_chunk_ids)} 个唯一chunk关系")
            
        # 动态计算最优批处理大小，确保在合理范围内
        optimal_batch_size = min(self.batch_size, max(20, len(unique_chunk_ids) // 5))
        total_batches = (len(unique_chunk_ids) + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"合并关系批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 分批处理，避免一次性处理过多数据导致性能问题
        for i in range(0, len(unique_chunk_ids), optimal_batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i+optimal_batch_size]
            # 准备批处理数据格式
            batch_data = [{"chunk_id": chunk_id} for chunk_id in batch_chunk_ids]
            
            try:
                # 不需要保留，因为 d 不是自己创建的文档，标签不同（__Document__）
                # graph.add_graph_documents 创建 Document 标签，因为标签是区分大小写且需要完全匹配的
                # e是通过关系r从Document节点d连接到的任意节点（可能是各种实体节点），r是d和e之间的MENTIONS关系
                merge_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document {chunk_id:data.chunk_id})
                    WITH c, d
                    MATCH (d)-[r:MENTIONS]->(e)
                    MERGE (c)-[newR:MENTIONS]->(e)
                    ON CREATE SET newR += properties(r)
                    DETACH DELETE d
                """
                
                self.graph.query(merge_query, params={"batch_data": batch_data})
                print(f"已处理合并关系批次 {i//optimal_batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"合并关系批次时出错: {e}")
                # 降级策略：如果批处理失败，尝试逐个处理
                for chunk_id in batch_chunk_ids:
                    try:
                        # 单个Chunk的关系合并
                        single_query = """
                            MATCH (c:`__Chunk__` {id: $chunk_id}), (d:Document{chunk_id:$chunk_id})
                            WITH c, d
                            MATCH (d)-[r:MENTIONS]->(e)
                            MERGE (c)-[newR:MENTIONS]->(e)
                            ON CREATE SET newR += properties(r)
                            DETACH DELETE d
                        """
                        self.graph.query(single_query, params={"chunk_id": chunk_id})
                    except Exception as e2:
                        print(f"处理单个chunk关系时出错: {e2}")