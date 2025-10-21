"""
Graph-RAG Agent 实体索引管理模块

此模块实现了知识图谱系统中的实体索引管理功能，负责创建和管理实体的向量索引，
支持基于向量相似度的实体查询。主要功能包括：

1. 实体向量索引创建
   - 批量处理实体节点的嵌入向量计算
   - 优化的索引创建和管理
   - 支持索引刷新和重建

2. 高效计算优化
   - 多级批处理策略
   - 并行计算框架
   - 智能批处理大小调整

3. 健壮性保障
   - 多级降级机制
   - 错误处理和恢复
   - 空文本和异常情况处理

4. 性能监控
   - 详细的时间统计
   - 处理进度跟踪
   - 资源利用优化

该模块是Graph-RAG系统中负责实体索引构建的核心组件，为实体查询提供向量支持。
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector

from model.get_models import get_embeddings_model, get_llm_model
from graph.core import BaseIndexer, connection_manager
from config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EntityIndexManager(BaseIndexer):
    """
    实体索引管理器
    
    功能：
    - 在Neo4j数据库中创建和管理实体的向量索引
    - 处理实体节点的embedding向量计算和存储
    - 支持基于向量相似度的实体查询
    - 提供高效的批处理和并行计算能力
    
    实现思路：
    - 继承自BaseIndexer基类，利用其通用批处理功能
    - 采用多级批处理和并行计算优化性能
    - 实现错误处理和降级策略确保稳定性
    - 支持灵活的索引配置和更新
    """
    
    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        初始化实体索引管理器
        
        参数：
            refresh_schema: 是否刷新Neo4j图数据库的schema
            batch_size: 批处理大小，控制每次处理的实体数量
            max_workers: 并行工作线程数，控制并发计算能力
        """
        # 设置批处理大小和工作线程数，优先使用配置文件中的值
        batch_size = batch_size or ENTITY_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 调用父类初始化方法
        super().__init__(batch_size, max_workers)
        
        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()
        
        # 初始化嵌入模型和语言模型
        self.embeddings = get_embeddings_model()
        self.llm = get_llm_model()
        
        # 创建必要的索引结构
        self._create_indexes()
    
    def _create_indexes(self) -> None:
        """
        创建必要的图数据库索引以优化查询性能
        
        实现细节：
        - 创建实体ID属性的B树索引，加速实体查找
        - 使用条件创建(IF NOT EXISTS)避免重复创建索引
        - 通过连接管理器批量创建索引
        
        索引作用：
        - 实体ID索引：加速通过ID查找特定实体的操作
        - 优化图数据库遍历性能，特别是在大型知识图谱中
        """
        # 定义实体索引创建查询列表
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"  # 实体ID索引 - 用于快速查找特定实体
        ]
        
        # 使用连接管理器批量执行索引创建
        connection_manager.create_multiple_indexes(index_queries)
        
    def clear_existing_index(self) -> None:
        """
        清除已存在的实体embedding相关索引
        
        实现细节：
        - 清除实体嵌入向量索引
        - 清除通用向量索引
        - 这一步是为了防止embedding模型切换导致的索引不兼容问题
        
        注意事项：
        - 在索引重建前执行此操作
        - 执行期间可能会影响查询性能
        - 索引重建完成前，向量搜索功能可能不可用
        """
        # 清除特定的实体嵌入向量索引
        connection_manager.drop_index("entity_embedding")
        # 清除通用向量索引，确保兼容性
        connection_manager.drop_index("vector")

    def create_entity_index(self, 
                          node_label: str = '__Entity__',
                          text_properties: List[str] = ['id', 'description'],
                          embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        创建实体的向量索引，支持批处理和并行优化
        
        参数：
            node_label: 实体节点的标签，默认使用__Entity__标签
            text_properties: 用于计算embedding的文本属性列表，默认使用id和description
            embedding_property: 存储embedding向量的属性名，默认使用embedding
            
        返回：
            Neo4jVector: 创建的向量存储对象，用于后续的相似度查询；如果创建失败则返回None
        
        实现流程：
        1. 清除已有的向量索引，避免索引冲突
        2. 查询所有需要处理的实体（没有嵌入向量的实体）
        3. 批量处理实体，计算并存储嵌入向量
        4. 创建Neo4j向量存储，构建向量索引
        5. 返回创建的向量存储对象
        
        设计理念：
        - 增量处理：只处理缺少embedding的实体，避免重复计算
        - 性能优化：采用多级批处理和并行计算提高效率
        - 健壮性：内置错误处理和降级策略确保稳定性
        """
        # 开始计时，用于性能统计和监控
        start_time = time.time()
        
        # 步骤1: 清除已有索引，防止模型切换或索引不兼容问题
        self.clear_existing_index()
        
        # 步骤2: 获取所有需要处理的实体（增量处理策略）
        # 只处理没有嵌入向量的实体，避免重复计算
        entities = self.graph.query(
            f"""
            MATCH (e:`{node_label}`)
            WHERE e.{embedding_property} IS NULL
            RETURN id(e) AS neo4j_id, e.id AS entity_id
            """
        )
        
        # 处理空结果情况
        if not entities:
            print("没有找到需要处理的实体节点")
            return None
            
        # 开始处理实体
        print(f"开始为 {len(entities)} 个实体生成embeddings")
        
        # 步骤3: 批量处理所有实体，计算并存储嵌入向量
        self._process_embeddings_in_batches(entities, node_label, text_properties, embedding_property)
        
        # 步骤4: 创建新的向量索引，用于相似度查询
        try:
            # 从现有图中创建Neo4j向量存储对象
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=text_properties,
                embedding_node_property=embedding_property
            )
            
            # 步骤5: 性能统计和返回结果
            end_time = time.time()
            print(f"索引创建成功，总耗时: {end_time - start_time:.2f}秒")
            print(f"其中: embedding计算: {self.embedding_time:.2f}秒, 数据库操作: {self.db_time:.2f}秒")
            
            return vector_store
        except Exception as e:
            print(f"创建向量索引时出错: {e}")
            return None
    
    def _process_embeddings_in_batches(self, entities: List[Dict[str, Any]], 
                                      node_label: str, text_properties: List[str], 
                                      embedding_property: str) -> None:
        """
        批量处理实体嵌入向量的生成和存储
        
        参数：
            entities: 需要处理的实体列表
            node_label: 实体节点标签
            text_properties: 用于计算嵌入的文本属性
            embedding_property: 存储嵌入向量的属性名
        
        实现流程：
        1. 根据实体数量动态调整最优批处理大小
        2. 定义批次处理函数，包含文本提取、嵌入计算和数据库更新
        3. 使用父类的批量处理方法，带进度显示
        4. 记录各阶段耗时，用于性能分析
        
        性能优化：
        - 动态批处理：根据实体数量调整批次大小
        - 并行计算：利用多线程加速embedding计算
        - 增量处理：只处理缺少embedding的实体
        """
        # 根据实体总数计算最优批处理大小，避免过大或过小的批次
        # 动态批处理大小可以根据数据量自动优化内存使用和计算效率
        entity_count = len(entities)
        optimal_batch_size = self.get_optimal_batch_size(entity_count)
        
        # 定义批次处理函数 - 每个批次独立执行三个步骤
        def process_batch(batch, batch_index):
            # 步骤1: 获取批次内所有实体的文本内容
            entity_texts = self._get_entity_texts_batch(batch, text_properties)
            
            # 步骤2: 计算嵌入向量并记录时间 - 最耗时的步骤
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(entity_texts)
            embedding_end = time.time()
            # 累加embedding计算时间，用于性能分析
            self.embedding_time += (embedding_end - embedding_start)
            
            # 步骤3: 更新数据库中的嵌入向量并记录时间
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            # 累加数据库操作时间，用于性能分析
            self.db_time += (db_end - db_start)
        
        # 使用父类的通用批处理方法，带进度显示
        self.batch_process_with_progress(
            entities, 
            process_batch, 
            optimal_batch_size, 
            "处理实体embedding"
        )
    
    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        计算一批文本的嵌入向量，带多级优化和降级策略
        
        参数：
            texts: 待计算嵌入的文本列表
            
        返回：
            List[List[float]]: 对应的嵌入向量列表
        
        实现特点：
        - 三级批处理策略：外层批次、内层子批次、并行单条
        - 多级降级机制：批量API -> 并行单条 -> 顺序单条
        - 健壮性处理，确保空文本和错误情况的正确处理
        - 使用零向量作为错误情况下的备用
        
        算法设计：
        - 第一级批处理：将大量实体分成合理大小的批次
        - 第二级批处理：每个批次内部再细分，避免内存压力
        - 三级降级策略：从高效到稳定的三种处理方式
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
                safe_text = text if text and text.strip() else "unknown entity"
                embedding_tasks.append(safe_text)
            
            # 分析批处理的最佳大小，避免单次处理过大批次导致内存溢出
            # 32是一个经验值，兼顾了计算效率和内存使用
            embed_batch_size = min(32, len(embedding_tasks))
            
            # 二级批处理：将整个批次进一步分割成较小的子批次
            # 这种二级批处理策略既保证了处理效率，又避免了内存压力
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # 第一级策略：尝试使用批量嵌入方法，性能最优
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
                    # 当批量和并行处理都失败时，使用这种最保守的策略确保处理继续
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings.embed_query(text))
                        except Exception as e2:
                            print(f"单个嵌入计算失败: {e2}")
                            # 添加零向量作为备用，保证结果列表与输入列表长度一致
                            if hasattr(self.embeddings, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings.embedding_size)
                            else:
                                # 使用OpenAI模型的默认嵌入维度1536
                                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def _get_entity_texts_batch(self, entities: List[Dict[str, Any]], text_properties: List[str]) -> List[str]:
        """
        批量获取实体的文本内容，用于嵌入计算
        
        参数：
            entities: 需要提取文本的实体列表
            text_properties: 需要提取的文本属性列表
            
        返回：
            List[str]: 实体的文本内容列表，顺序与输入实体列表对应
        
        实现特点：
        - 使用UNWIND批量处理多个实体，提高查询效率
        - 处理空属性的情况，确保每个实体都有有效的文本
        - 组合多个文本属性，丰富实体表示
        - 为无内容实体生成默认标识文本
        
        数据库优化：
        - 使用Neo4j原生ID进行精确查找
        - 参数化查询避免SQL注入风险
        - 条件属性选择处理空值情况
        """
        # 构建查询参数，提取实体Neo4j原生ID列表
        entity_ids = [entity['neo4j_id'] for entity in entities]
        
        # 构建属性选择部分的查询语句，处理可能为空的属性
        # 使用CASE WHEN语句确保即使属性为空也能正确处理
        property_selections = ", ".join([
            f"CASE WHEN e.{prop} IS NOT NULL THEN e.{prop} ELSE '' END AS {prop}_text"
            for prop in text_properties
        ])
        
        # 构建高效的批量查询 - 使用UNWIND一次性处理多个ID
        query = f"""
        UNWIND $entity_ids AS id
        MATCH (e) WHERE id(e) = id
        RETURN id, {property_selections}
        """
        
        # 执行批量查询，一次获取所有实体的文本属性
        results = self.graph.query(query, params={"entity_ids": entity_ids})
        
        # 组合多个文本属性为单个文本，用于embedding计算
        entity_texts = []
        for row in results:
            text_parts = []
            for prop in text_properties:
                prop_text = row.get(f"{prop}_text", "")
                if prop_text:  # 只添加非空文本，减少噪音
                    text_parts.append(prop_text)
            
            # 组合所有文本属性，确保至少有一些内容
            # 空格分隔不同属性，便于模型理解
            combined_text = " ".join(text_parts).strip()
            if not combined_text:  # 为无内容实体生成默认标识
                combined_text = f"entity_{row['id']}"
                
            entity_texts.append(combined_text)
        
        return entity_texts
    
    def _update_embeddings_batch(self, entities: List[Dict[str, Any]], 
                                embeddings: List[List[float]], 
                                embedding_property: str) -> None:
        """
        批量更新实体的嵌入向量到数据库
        
        参数：
            entities: 需要更新的实体列表
            embeddings: 计算好的嵌入向量列表
            embedding_property: 存储嵌入向量的属性名
        
        实现特点：
        - 使用UNWIND批量更新多个实体，提高写入效率
        - 验证嵌入向量的有效性，避免写入无效数据
        - 实现降级策略：批量更新失败时回退到单个更新
        - 详细的错误处理和日志记录
        
        数据一致性保障：
        - 严格的边界检查，防止索引越界
        - None值过滤，确保只写入有效的embedding
        - 降级机制确保即使部分失败也能完成大部分更新
        """
        # 构建更新数据，确保嵌入向量的有效性
        update_data = []
        for i, entity in enumerate(entities):
            # 安全检查：确保索引不越界且embedding不为None
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": entity['neo4j_id'],  # Neo4j原生ID，用于精确定位实体节点
                    "embedding": embeddings[i]  # 计算好的向量嵌入
                })
        
        # 批量更新嵌入向量 - 只在有有效数据时执行
        if update_data:
            try:
                # 使用UNWIND进行高效的批量更新
                # 这种方式可以在一次事务中更新多个节点，大大提高写入效率
                query = f"""
                UNWIND $updates AS update
                MATCH (e) WHERE id(e) = update.id
                SET e.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"批量更新embeddings失败: {e}")
                # 降级策略：批量更新失败时回退到单个更新
                # 这种优雅降级策略确保即使批量操作失败，也能尽可能完成任务
                for update in update_data:
                    try:
                        # 对每个实体单独执行更新操作
                        single_query = f"""
                        MATCH (e) WHERE id(e) = $id
                        SET e.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        # 记录单个实体更新失败，但不中断整体处理
                        print(f"单个embedding更新失败 (ID: {update['id']}): {e2}")