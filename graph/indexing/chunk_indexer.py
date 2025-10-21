"""
Graph-RAG Agent 文本块索引管理模块

此模块实现了文本块(Chunk)索引的创建和管理功能，是Graph-RAG系统中负责
向量检索的核心组件之一。主要功能包括：

1. 文本块向量嵌入计算与存储
2. Neo4j图数据库索引优化
3. 批量处理机制，提高大规模文本处理效率
4. 并行计算优化，充分利用多核处理能力
5. 错误处理和降级策略，确保系统稳定性

该模块通过计算文本块的向量表示并创建索引，为系统提供了高效的语义相似度搜索能力，
是实现RAG(检索增强生成)功能的基础组件。
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector

from model.get_models import get_embeddings_model
from graph.core import BaseIndexer, connection_manager
from config.settings import CHUNK_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class ChunkIndexManager(BaseIndexer):
    """
    文本块索引管理器
    
    功能：
    - 在Neo4j图数据库中创建和管理文本块的向量索引
    - 计算文本块的向量嵌入(embedding)并存储
    - 提供向量存储接口，支持后续的相似度检索
    - 优化数据库查询性能，提高索引效率
    
    实现思路：
    - 继承BaseIndexer基类，复用批量处理框架
    - 使用嵌入模型计算文本的向量表示
    - 采用批量处理和并行计算策略提高效率
    - 实现自动错误恢复和降级机制
    - 跟踪和报告处理性能指标
    """
    
    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        初始化Chunk索引管理器
        
        参数：
            refresh_schema: 是否刷新Neo4j图数据库的schema
            batch_size: 批处理大小，控制每次处理的文本块数量
            max_workers: 并行工作线程数，控制并发计算能力
        """
        # 使用配置文件中的默认值或传入的参数
        batch_size = batch_size or CHUNK_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 调用父类构造函数
        super().__init__(batch_size, max_workers)
        
        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()
        
        # 初始化嵌入模型 - 用于将文本转换为向量表示
        self.embeddings = get_embeddings_model()
        
        # 创建必要的数据库索引
        self._create_indexes()
    
    def _create_indexes(self) -> None:
        """
        创建必要的数据库索引以优化查询性能
        
        实现细节：
        - 为文本块节点的关键属性创建索引
        - 包括id索引、文件名索引和位置索引
        - 这些索引显著提高了后续查询和过滤操作的性能
        
        索引说明：
        - id索引：用于快速查找特定文本块节点
        - fileName索引：用于按文件名批量检索文本块
        - position索引：用于按顺序获取文本块，重建文档结构
        """
        # 定义索引创建查询列表
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.id)",           # 基于id的索引 - 用于快速查找特定文本块
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.fileName)",   # 基于文件名的索引 - 用于按文档检索
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.position)"     # 基于位置的索引 - 用于顺序重建
        ]
        
        # 批量执行索引创建 - 使用连接管理器处理
        connection_manager.create_multiple_indexes(index_queries)
        
    def clear_existing_index(self) -> None:
        """
        清除已存在的普通索引
        
        注意事项：
        - 不尝试删除向量索引，仅清理普通索引
        - 主要用于索引重建或维护场景
        - 清除操作需要在系统维护窗口执行，避免影响正常查询
        """
        # 使用连接管理器删除指定的索引
        connection_manager.drop_index("chunk_embedding")

    def create_chunk_index(self, 
                         node_label: str = '__Chunk__',
                         text_property: str = 'text',
                         embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        为文本块节点生成embeddings并创建向量存储接口
        
        参数：
            node_label: 文本块节点的标签，默认为'__Chunk__'
            text_property: 用于计算embedding的文本属性名
            embedding_property: 存储embedding的属性名
            
        返回：
            Neo4jVector: 创建的向量存储对象，用于向量相似度检索
        
        实现流程：
        1. 查找尚未计算embedding的文本块节点
        2. 如果没有需要处理的节点，尝试连接到现有向量存储
        3. 对需要处理的节点进行批量embedding计算和存储
        4. 创建并返回向量存储接口
        
        性能考量：
        - 增量处理：只处理缺少embedding的节点，避免重复计算
        - 批量优化：使用批处理策略提高计算和存储效率
        - 并行处理：利用多线程加速embedding计算
        """
        # 开始计时，用于性能统计
        start_time = time.time()
        
        # 获取所有需要处理的文本块节点 - 只处理没有embedding的节点
        # 这种增量处理策略避免重复计算，提高效率
        chunks = self.graph.query(
            f"""
            MATCH (c:`{node_label}`)
            WHERE c.{text_property} IS NOT NULL AND c.{embedding_property} IS NULL
            RETURN id(c) AS neo4j_id, c.id AS chunk_id
            """
        )
        
        # 处理没有需要计算embedding的情况
        if not chunks:
            print("没有找到需要处理的文本块节点")
            # 即使没有需要处理的节点，也尝试连接到现有向量存储
            try:
                # 尝试连接到已存在的向量索引
                vector_store = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    node_label=node_label,
                    text_node_properties=[text_property],
                    embedding_node_property=embedding_property
                )
                
                print("成功连接到现有向量索引")
                return vector_store
            except Exception as e:
                print(f"连接到向量存储时出错: {e}")
                return None
            
        # 开始处理文本块
        print(f"开始为 {len(chunks)} 个文本块生成embeddings")
        
        # 批量处理所有文本块 - 计算embedding并更新数据库
        self._process_embeddings_in_batches(chunks, node_label, text_property, embedding_property)
        
        # 连接到向量存储，而不尝试创建新的向量索引
        try:
            # 创建向量存储对象 - 用于后续的向量相似度检索
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=[text_property],
                embedding_node_property=embedding_property
            )
            
            # 计算和报告性能指标
            end_time = time.time()
            print(f"索引创建成功，总耗时: {end_time - start_time:.2f}秒")
            print(f"其中: embedding计算: {self.embedding_time:.2f}秒, 数据库操作: {self.db_time:.2f}秒")
            
            return vector_store
        except Exception as e:
            print(f"创建向量存储时出错: {e}")
            return None
    
    def _process_embeddings_in_batches(self, chunks: List[Dict[str, Any]], 
                                      node_label: str, text_property: str, 
                                      embedding_property: str) -> None:
        """
        批量处理文本块embedding的生成
        
        参数：
            chunks: 文本块列表，包含需要处理的节点信息
            node_label: 节点标签
            text_property: 文本属性名
            embedding_property: embedding属性名
            
        实现策略：
        - 使用动态批处理大小，根据数据量自动优化
        - 对每个批次分别执行：获取文本 -> 计算embedding -> 更新数据库
        - 记录处理性能指标，便于分析优化
        - 采用并行处理策略提高计算效率
        
        性能优化：
        - 三级批处理策略：整体批处理 + 子批处理 + 并行计算
        - 精确计时：分别记录embedding计算和数据库操作时间
        - 内存优化：避免一次性加载全部文本内容
        """
        # 获取最优批处理大小 - 根据数据量动态调整，避免内存溢出
        chunk_count = len(chunks)
        optimal_batch_size = self.get_optimal_batch_size(chunk_count)
        
        # 定义批次处理函数 - 每个批次独立处理
        def process_batch(batch, batch_index):
            # 步骤1: 获取批次内所有文本块的文本内容
            chunk_texts = self._get_chunk_texts_batch(batch, text_property)
            
            # 步骤2: 计算embeddings并记录时间 - 最耗时的步骤
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(chunk_texts)
            embedding_end = time.time()
            # 累加embedding计算时间，用于性能分析
            self.embedding_time += (embedding_end - embedding_start)
            
            # 步骤3: 更新数据库并记录时间
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            # 累加数据库操作时间，用于性能分析
            self.db_time += (db_end - db_start)
        
        # 使用通用批处理方法，带进度显示
        self.batch_process_with_progress(
            chunks, 
            process_batch, 
            optimal_batch_size, 
            "处理文本块embedding"
        )
    
    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        计算一批文本的embedding向量
        
        参数：
            texts: 待处理的文本列表
            
        返回：
            List[List[float]]: 对应的embedding向量列表
            
        实现特点：
        - 并行计算提高效率
        - 内置健壮性处理，确保空文本也能正确处理
        - 实现多级降级策略，保证系统稳定性
        - 错误处理机制，为失败的计算提供备用值
        
        降级策略详解：
        1. 首选：批量嵌入方法（如果模型支持）
        2. 降级1：使用线程池并行处理单个嵌入
        3. 降级2：完全串行处理单个嵌入，确保稳定性
        """
        # 初始化结果列表
        embeddings = []
        
        # 使用线程池实现并行计算，充分利用多核CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 预创建嵌入任务，进行文本预处理
            embedding_tasks = []
            for text in texts:
                # 添加强健性处理，确保文本不为空
                # 防止空文本导致embedding计算失败
                safe_text = text if text and text.strip() else "empty chunk"
                embedding_tasks.append(safe_text)
            
            # 分析批处理的最佳大小 - 避免单次处理过大批次导致内存压力
            # 32是一个安全的默认值，适合大多数embedding模型
            embed_batch_size = min(32, len(embedding_tasks))
            
            # 二级批处理：将整个批次进一步分割成较小的子批次
            # 这种二级批处理策略既保证了并行效率，又避免了内存溢出
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # 第一级策略：尝试使用批量嵌入方法 - 最有效的方式
                    if hasattr(self.embeddings, 'embed_documents'):
                        sub_batch_embeddings = self.embeddings.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # 第二级降级：回退到单个嵌入，使用线程池并行
                        futures = [executor.submit(self.embeddings.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                print(f"嵌入计算失败: {e}")
                                # 添加零向量作为备用，确保结果列表完整性
                                if hasattr(self.embeddings, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings.embedding_size)
                                else:
                                    # 假设使用通用嵌入大小（如OpenAI的1536维）
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    print(f"批量嵌入处理失败: {e}")
                    # 第三级降级：对每个文本单独处理 - 最稳定的方式
                    # 当批量和并行处理都失败时，使用这种最保守的策略
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings.embed_query(text))
                        except Exception as e2:
                            print(f"单个嵌入计算失败: {e2}")
                            # 添加零向量作为备用，保证结果列表与输入列表长度一致
                            if hasattr(self.embeddings, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings.embedding_size)
                            else:
                                # 假设使用OpenAI模型的默认嵌入维度1536
                                embeddings.append([0.0] * 1536)
        

        
        return embeddings
    
    def _get_chunk_texts_batch(self, chunks: List[Dict[str, Any]], text_property: str) -> List[str]:
        """
        获取批量文本块的文本内容
        
        参数：
            chunks: 文本块列表，包含节点ID信息
            text_property: 文本属性名
            
        返回：
            List[str]: 文本块的文本内容列表
            
        实现优化：
        - 使用UNWIND语句批量处理多个ID
        - 一次查询获取多个文本块内容，减少数据库访问次数
        - 处理空文本情况，提供默认值
        
        数据库优化：
        - 批量查询减少网络往返
        - 利用Neo4j参数化查询避免SQL注入
        - 使用图数据库原生ID进行精确查找，提高性能
        """
        # 构建查询参数 - 提取所有需要查询的文本块ID
        chunk_ids = [chunk['neo4j_id'] for chunk in chunks]
        
        # 使用高效的文本提取查询 - 使用UNWIND批量处理
        # UNWIND语句可以将列表展开为多行，一次处理多个ID
        query = f"""
        UNWIND $chunk_ids AS id
        MATCH (c) WHERE id(c) = id
        RETURN id, c.{text_property} AS chunk_text
        """
        
        # 执行批量查询，获取所有文本内容
        results = self.graph.query(query, params={"chunk_ids": chunk_ids})
        
        # 提取和清理文本内容
        chunk_texts = []
        for row in results:
            text = row.get("chunk_text", "")
            # 确保文本不为空，提供默认标识
            # 这是一种健壮性处理，防止空文本导致embedding计算失败
            if not text:
                text = f"chunk_{row['id']}"
                
            chunk_texts.append(text)
        
        return chunk_texts
    
    def _update_embeddings_batch(self, chunks: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                embedding_property: str) -> None:
        """
        批量更新文本块embeddings到数据库
        
        参数：
            chunks: 文本块列表，包含节点ID信息
            embeddings: 对应的embedding向量列表
            embedding_property: 存储embedding的属性名
            
        实现特点：
        - 批量更新优化，减少数据库交互次数
        - 内置错误检测和过滤
        - 批量操作失败时自动降级到单条更新
        
        数据一致性保障：
        - 严格的边界检查，确保索引不越界
        - None值过滤，防止存储无效的embedding
        - 降级策略确保数据写入不中断
        """
        # 构建更新数据，确保索引匹配
        update_data = []
        for i, chunk in enumerate(chunks):
            # 安全检查：确保索引不越界且embedding不为None
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": chunk['neo4j_id'],  # Neo4j原生ID，用于精确定位节点
                    "embedding": embeddings[i]  # 计算好的向量嵌入
                })
        
        # 批量更新 - 只在有数据时执行
        if update_data:
            try:
                # 使用UNWIND进行高效批量更新
                # 这种方式可以在一次事务中更新多个节点，大大提高效率
                query = f"""
                UNWIND $updates AS update
                MATCH (c) WHERE id(c) = update.id
                SET c.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"批量更新embeddings失败: {e}")
                # 降级策略：批量失败时回退到单个更新模式
                # 这是一种优雅降级策略，确保即使批量操作失败，也能部分完成任务
                for update in update_data:
                    try:
                        # 对每个节点单独执行更新操作
                        single_query = f"""
                        MATCH (c) WHERE id(c) = $id
                        SET c.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        # 记录单个节点更新失败，但不中断整体处理
                        print(f"单个embedding更新失败 (ID: {update['id']}): {e2}")