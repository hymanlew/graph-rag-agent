"""
Graph-RAG Agent 向量嵌入管理模块

此模块实现了知识图谱系统中的向量嵌入(Embedding)管理功能，负责实体和文本块的向量
计算、存储和增量更新。主要功能包括：

1. 增量向量更新机制
   - 仅处理需要更新的实体和Chunk的Embedding
   - 跟踪和管理向量更新状态
   - 支持选择性更新特定实体或文本块

2. 高效计算优化
   - 批量处理机制，减少API调用次数
   - 并行计算框架，充分利用多核性能
   - 多级降级策略，确保系统稳定性

3. 变更追踪与管理
   - 自动标记需要更新的元素
   - 支持按文件或实体进行批量标记
   - 维护更新时间戳和状态标志

4. 性能监控与统计
   - 详细的处理性能指标跟踪
   - 可视化进度和结果反馈
   - 错误处理和恢复机制

该模块是Graph-RAG系统中负责向量索引维护的核心组件，确保知识图谱中的元素
能够高效地进行语义相似度搜索。
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Optional

from rich.console import Console

from model.get_models import get_embeddings_model
from config.neo4jdb import get_db_manager
from config.settings import EMBEDDING_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EmbeddingManager:
    """
    向量嵌入管理器
    
    功能：
    - 管理知识图谱中实体和文本块的向量嵌入计算与更新
    - 实现增量更新策略，仅处理需要更新的元素
    - 提供高效的批量处理和并行计算框架
    - 维护向量更新状态和跟踪机制
    
    实现思路：
    - 采用标记-更新模式，先标记需要更新的元素，再批量处理
    - 使用批处理和并行计算优化性能
    - 实现多级错误处理和降级策略
    - 跟踪和报告处理性能指标
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        初始化嵌入管理器
        
        参数：
            batch_size: 批处理大小，控制每次处理的元素数量
            max_workers: 并行工作线程数，控制并发计算能力
        """
        # 初始化控制台输出工具
        self.console = Console()
        # 获取Neo4j图数据库连接
        self.graph = get_db_manager().graph
        # 初始化嵌入模型
        self.embeddings_model = get_embeddings_model()
        
        # 设置批处理大小和工作线程数，优先使用配置文件中的值
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 性能监控指标
        self.embedding_time = 0  # 嵌入计算耗时
        self.db_time = 0         # 数据库操作耗时
        self.total_time = 0      # 总处理耗时
        
        # 处理统计数据
        self.stats = {
            "entity_updates": 0,  # 实体更新数量
            "chunk_updates": 0,   # 文本块更新数量
            "total_updates": 0,   # 总更新数量
            "errors": 0           # 错误数量
        }
    
    def setup_embedding_tracking(self):
        """
        设置嵌入更新追踪机制
        
        实现细节：
        - 为实体和文本块节点添加创建时间属性
        - 这些时间戳用于后续的增量更新判断
        - 只处理尚未设置时间属性的节点，避免重复操作
        
        业务逻辑：
        - 时间戳是增量更新策略的关键，确保系统只处理变更的内容
        - 为节点添加元数据，支持版本控制和变更追踪
        - 初始化时运行，建立基线时间点
        """
        try:
            # 添加实体修改时间追踪
            # 为实体节点初始化创建时间，用于后续的增量更新判断
            self.graph.query("""
                MATCH (e:`__Entity__`)
                WHERE e.created_at IS NULL
                SET e.created_at = datetime()
            """)

            # 添加Chunk修改时间追踪
            # 为文本块节点初始化创建时间，建立增量更新基线
            self.graph.query("""
                MATCH (c:`__Chunk__`)
                WHERE c.created_at IS NULL
                SET c.created_at = datetime()
            """)

            self.console.print("[green]Embedding更新追踪设置完成[/green]")
            
        except Exception as e:
            # 异常处理，确保初始化失败不会中断程序
            self.console.print(f"[yellow]设置Embedding追踪时出错: {e}[/yellow]")
    
    def get_entities_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        获取需要更新Embedding的实体
        
        参数：
            limit: 返回的最大实体数量，避免一次性处理过多数据
            
        返回：
            List[Dict]: 需要更新的实体列表，包含实体ID和用于计算嵌入的文本
        
        选择条件：
        - 没有嵌入向量的实体
        - 被标记为需要重新计算嵌入的实体
        
        实现思路：
        - 使用LIMIT限制返回数量，防止内存溢出
        - 优先使用实体描述作为嵌入输入，若无则使用ID
        - Neo4j原生ID用于后续精确定位节点
        """
        # Cypher查询获取需要更新的实体
        # 查询条件设计确保只处理真正需要更新的实体，优化计算资源使用
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.embedding IS NULL 
        OR (e.needs_reembedding IS NOT NULL AND e.needs_reembedding = true)
        RETURN elementId(e) AS neo4j_id,  # Neo4j原生ID，用于后续精确定位
            e.id AS entity_id,            # 实体ID，用于标识
            CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text  # 优先使用描述作为嵌入输入
        LIMIT $limit  # 限制返回数量，防止内存溢出
        """
        
        result = self.graph.query(query, params={"limit": limit})
        # 安全处理：确保始终返回列表类型
        return result if result else []
    
    def get_chunks_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        获取需要更新Embedding的文本块
        
        参数：
            limit: 返回的最大Chunk数量
            
        返回：
            List[Dict]: 需要更新的Chunk列表，包含Chunk ID和文本内容
        
        选择条件：
        - 没有嵌入向量的文本块
        - 被标记为需要重新计算嵌入的文本块
        - 已更新但未重新计算嵌入的文本块（根据时间戳判断）
        
        实现思路：
        - 采用多条件过滤，确保增量更新的准确性
        - 利用时间戳比较实现智能更新判断
        - 限制返回数量，优化内存使用
        """
        # Cypher查询获取需要更新的文本块
        # 三层筛选条件确保只处理需要更新的文本块，实现高效增量更新
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.embedding IS NULL 
            OR c.needs_reembedding = true
            OR (c.last_updated IS NOT NULL AND 
                (c.last_embedded IS NULL OR c.last_updated > c.last_embedded))
        RETURN elementId(c) AS neo4j_id,  # Neo4j原生ID，用于后续精确定位
               c.id AS chunk_id,          # 文本块ID，用于标识
               c.text AS text              # 文本内容，用于计算嵌入
        LIMIT $limit  # 限制返回数量，防止内存溢出
        """
        
        result = self.graph.query(query, params={"limit": limit})
        # 安全处理：确保始终返回列表类型
        return result if result else []
    
    def update_entity_embeddings(self, entity_ids: Optional[List[str]] = None) -> int:
        """
        更新实体的嵌入向量
        
        参数：
            entity_ids: 要更新的实体ID列表，如果为None则自动检测需要更新的实体
            
        返回：
            int: 成功更新的实体数量
        
        实现流程：
        1. 根据参数决定是处理特定实体还是自动检测需要更新的实体
        2. 批量处理实体，每批包含指定数量的实体
        3. 对每批实体：
           - 提取用于计算嵌入的文本
           - 使用嵌入模型计算向量
           - 更新数据库中的嵌入向量
        4. 记录处理统计和性能指标
        
        业务逻辑：
        - 支持选择性更新和全量检测更新两种模式
        - 实时性能监控和统计，便于优化系统
        - 批量处理策略，提高系统效率和稳定性
        """
        # 记录开始时间，用于性能统计
        start_time = time.time()
        
        # 步骤1: 获取需要更新的实体
        if entity_ids:
            # 模式1: 处理指定实体 - 用于精确更新特定实体
            id_list = ", ".join([f"'{eid}'" for eid in entity_ids])
            query = f"""
            MATCH (e:`__Entity__`)
            WHERE e.id IN [{id_list}]
            RETURN elementId(e) AS neo4j_id,
                   e.id AS entity_id, 
                   CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
            """
            entities = self.graph.query(query)
        else:
            # 模式2: 自动检测需要更新的实体 - 用于常规增量更新
            # 使用batch_size * 5作为限制，确保获取足够数据但不会内存溢出
            entities = self.get_entities_needing_update(limit=self.batch_size * 5)
        
        # 空检查：如果没有需要更新的实体，提前返回
        if not entities:
            self.console.print("[yellow]没有需要更新Embedding的实体[/yellow]")
            return 0
        
        self.console.print(f"[cyan]开始更新 {len(entities)} 个实体的Embedding...[/cyan]")
        
        # 步骤2: 批量处理实体 - 核心处理逻辑
        updated_count = 0
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i+self.batch_size]
            
            # 步骤3a: 提取文本和ID信息
            texts = [entity["text"] for entity in batch]  # 用于计算嵌入的文本
            entity_ids = [entity["entity_id"] for entity in batch]  # 实体ID
            neo4j_ids = [entity["neo4j_id"] for entity in batch]  # Neo4j原生ID，用于精确定位
            
            # 步骤3b: 计算Embedding并测量时间
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)  # 调用内部方法批量计算嵌入
                self.embedding_time += time.time() - embedding_start  # 累积嵌入计算时间
                
                # 步骤3c: 准备更新数据，确保索引匹配
                # 安全检查：确保不会越界访问embeddings数组
                updates = []
                for j, entity_id in enumerate(entity_ids):
                    # 边界检查和空值检查，确保数据安全
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],  # 使用Neo4j原生ID精确定位
                            "embedding": embeddings[j]  # 计算得到的嵌入向量
                        })
                
                # 步骤3d: 更新数据库并测量时间
                db_start = time.time()
                if updates:  # 只有在有有效更新数据时才执行数据库操作
                    # 使用UNWIND批量更新，提高数据库操作效率
                    query = """
                    UNWIND $updates AS update
                    MATCH (e) WHERE elementId(e) = update.neo4j_id
                    SET e.embedding = update.embedding,        # 更新嵌入向量
                        e.last_embedded = datetime(),           # 更新嵌入时间戳
                        e.needs_reembedding = false             # 清除更新标记
                    RETURN count(e) AS updated                  # 返回更新成功数量
                    """
                    
                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated  # 累积更新成功数量
                    
                self.db_time += time.time() - db_start  # 累积数据库操作时间
                
                # 输出批处理结果，提供实时反馈
                self.console.print(f"[green]批次 {i//self.batch_size + 1} 更新完成，"
                                  f"处理了 {len(batch)} 个实体，"
                                  f"成功更新 {batch_updated} 个[/green]")
                
            except Exception as e:
                # 错误处理策略：记录错误但继续处理下一批，确保系统稳定性
                self.console.print(f"[red]更新实体Embedding时出错: {e}[/red]")
                self.stats["errors"] += 1
        
        # 步骤4: 更新统计信息
        self.stats["entity_updates"] += updated_count
        self.stats["total_updates"] += updated_count
        
        # 计算总时间并输出结果
        self.total_time += time.time() - start_time
        
        self.console.print(f"[blue]实体Embedding更新完成，共更新 {updated_count} 个实体，"
                          f"耗时: {time.time() - start_time:.2f}秒[/blue]")
        
        return updated_count
    
    def update_chunk_embeddings(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        更新文本块的嵌入向量
        
        参数：
            chunk_ids: 要更新的Chunk ID列表，如果为None则自动检测需要更新的文本块
            
        返回：
            int: 成功更新的Chunk数量
        
        实现流程：
        1. 根据参数决定是处理特定文本块还是自动检测需要更新的文本块
        2. 批量处理文本块，每批包含指定数量的文本块
        3. 对每批文本块：
           - 提取文本内容
           - 使用嵌入模型计算向量
           - 更新数据库中的嵌入向量
        4. 记录处理统计和性能指标
        
        业务逻辑：
        - 支持选择性更新和自动增量更新两种模式
        - 文本块是RAG系统中语义搜索的基本单位
        - 高效的批量处理确保系统性能
        """
        # 记录开始时间，用于性能统计
        start_time = time.time()
        
        # 步骤1: 获取需要更新的文本块
        if chunk_ids:
            # 模式1: 处理指定文本块 - 用于精确更新特定文本块
            id_list = ", ".join([f"'{cid}'" for cid in chunk_ids])
            query = f"""
            MATCH (c:`__Chunk__`)
            WHERE c.id IN [{id_list}]
            RETURN elementId(c) AS neo4j_id,
                   c.id AS chunk_id, 
                   c.text AS text
            """
            chunks = self.graph.query(query)
        else:
            # 模式2: 自动检测需要更新的文本块 - 用于常规增量更新
            # 使用batch_size * 5作为限制，确保获取足够数据但不会内存溢出
            chunks = self.get_chunks_needing_update(limit=self.batch_size * 5)
        
        # 空检查：如果没有需要更新的文本块，提前返回
        if not chunks:
            self.console.print("[yellow]没有需要更新Embedding的Chunk[/yellow]")
            return 0
        
        self.console.print(f"[cyan]开始更新 {len(chunks)} 个Chunk的Embedding...[/cyan]")
        
        # 步骤2: 批量处理文本块 - 核心处理逻辑
        updated_count = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            
            # 步骤3a: 提取文本和ID信息
            texts = [chunk["text"] for chunk in batch]  # 文本块内容，用于计算嵌入
            chunk_ids = [chunk["chunk_id"] for chunk in batch]  # 文本块ID
            neo4j_ids = [chunk["neo4j_id"] for chunk in batch]  # Neo4j原生ID，用于精确定位
            
            # 步骤3b: 计算Embedding并测量时间
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)  # 调用内部方法批量计算嵌入
                self.embedding_time += time.time() - embedding_start  # 累积嵌入计算时间
                
                # 步骤3c: 准备更新数据，确保索引匹配
                # 安全检查：确保不会越界访问embeddings数组
                updates = []
                for j, chunk_id in enumerate(chunk_ids):
                    # 边界检查和空值检查，确保数据安全
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],  # 使用Neo4j原生ID精确定位
                            "embedding": embeddings[j]  # 计算得到的嵌入向量
                        })
                
                # 步骤3d: 更新数据库并测量时间
                db_start = time.time()
                if updates:  # 只有在有有效更新数据时才执行数据库操作
                    # 使用UNWIND批量更新，提高数据库操作效率
                    query = """
                    UNWIND $updates AS update
                    MATCH (c) WHERE elementId(c) = update.neo4j_id
                    SET c.embedding = update.embedding,        # 更新嵌入向量
                        c.last_embedded = datetime(),           # 更新嵌入时间戳
                        c.needs_reembedding = false             # 清除更新标记
                    RETURN count(c) AS updated                  # 返回更新成功数量
                    """
                    
                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated  # 累积更新成功数量
                    
                self.db_time += time.time() - db_start  # 累积数据库操作时间
                
                # 输出批处理结果，提供实时反馈
                self.console.print(f"[green]批次 {i//self.batch_size + 1} 更新完成，"
                                  f"处理了 {len(batch)} 个Chunk，"
                                  f"成功更新 {batch_updated} 个[/green]")
                
            except Exception as e:
                # 错误处理策略：记录错误但继续处理下一批，确保系统稳定性
                self.console.print(f"[red]更新Chunk Embedding时出错: {e}[/red]")
                self.stats["errors"] += 1
        
        # 步骤4: 更新统计信息
        self.stats["chunk_updates"] += updated_count
        self.stats["total_updates"] += updated_count
        
        # 计算总时间并输出结果
        self.total_time += time.time() - start_time
        
        self.console.print(f"[blue]Chunk Embedding更新完成，共更新 {updated_count} 个Chunk，"
                          f"耗时: {time.time() - start_time:.2f}秒[/blue]")
        
        return updated_count
    
    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        计算一批文本的嵌入向量
        
        参数：
            texts: 待计算嵌入的文本列表
            
        返回：
            List[List[float]]: 对应的嵌入向量列表
        
        实现特点：
        - 两级批处理策略：先按批次分割，再进行子批次处理
        - 并行计算优化，充分利用多核处理能力
        - 三级降级机制：批量API -> 并行单条 -> 顺序单条
        - 健壮性处理，确保空文本和错误情况的正确处理
        
        算法设计：
        - 采用多级优化策略，平衡计算效率和系统稳定性
        - 三级降级机制确保在各种情况下都能继续处理
        - 零向量替换策略保证结果与输入长度匹配
        """
        # 初始化结果列表
        embeddings = []
        
        # 使用线程池实现并行计算，充分利用多核CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 预创建嵌入任务，进行文本预处理
            embedding_tasks = []
            for text in texts:
                # 添加强健性处理，确保文本不为空
                # 防止空文本导致embedding计算失败或返回None
                safe_text = text if text and text.strip() else "empty content"
                embedding_tasks.append(safe_text)
            
            # 分析批处理的最佳大小 - 避免单次处理过大批次导致内存溢出
            # 32是一个经验值，兼顾了计算效率和内存使用
            embed_batch_size = min(32, len(embedding_tasks))
            
            # 二级批处理：将整个批次进一步分割成较小的子批次
            # 这种二级批处理策略既保证了处理效率，又避免了内存压力
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # 第一级策略：尝试使用批量嵌入方法，性能最优
                    if hasattr(self.embeddings_model, 'embed_documents'):
                        # 大多数嵌入模型都提供批量处理方法，效率更高
                        sub_batch_embeddings = self.embeddings_model.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # 第二级降级：回退到单个嵌入，使用线程池并行
                        # 当模型不支持批量API时，使用并行单个调用提升效率
                        futures = [executor.submit(self.embeddings_model.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                self.console.print(f"[yellow]嵌入计算失败: {e}[/yellow]")
                                # 添加零向量作为备用，确保结果列表完整性
                                # 这是为了保证返回列表长度与输入一致，防止后续处理出错
                                if hasattr(self.embeddings_model, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings_model.embedding_size)
                                else:
                                    # 假设使用通用嵌入大小（如OpenAI的1536维）
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    self.console.print(f"[yellow]批量嵌入处理失败: {e}[/yellow]")
                    # 第三级降级：对每个文本单独处理 - 最稳定的方式
                    # 当批量和并行处理都失败时，使用这种最保守的策略确保处理继续
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings_model.embed_query(text))
                        except Exception as e2:
                            self.console.print(f"[yellow]单个嵌入计算失败: {e2}[/yellow]")
                            # 添加零向量作为备用，保证结果列表与输入列表长度一致
                            if hasattr(self.embeddings_model, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings_model.embedding_size)
                            else:
                                # 使用OpenAI模型的默认嵌入维度1536
                                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def mark_entities_for_update(self, entity_ids: List[str]) -> int:
        """
        标记实体需要更新嵌入向量
        
        参数：
            entity_ids: 需要标记的实体ID列表
            
        返回：
            int: 成功标记的实体数量
        
        实现细节：
        - 将实体的needs_reembedding标志设置为true
        - 更新last_updated时间戳
        - 这些标记将在后续更新过程中被检测并处理
        
        业务逻辑：
        - 实现标记-更新模式，分离标记和实际更新操作
        - 支持选择性更新特定实体，而不是全量更新
        - 时间戳用于后续增量更新判断
        """
        # 空检查：如果没有提供实体ID列表，直接返回
        if not entity_ids:
            return 0
            
        # 使用UNWIND批量标记实体为需要更新
        # 这种方式可以在一次事务中处理多个实体，提高效率
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})  # 通过实体ID精确定位节点
        SET e.needs_reembedding = true,          # 设置更新标记
            e.last_updated = datetime()          # 更新时间戳
        RETURN count(e) AS marked                # 返回标记成功数量
        """
        
        result = self.graph.query(query, params={"entity_ids": entity_ids})
        marked = result[0]["marked"] if result else 0  # 安全获取标记数量
        
        # 输出标记结果，提供操作反馈
        self.console.print(f"[blue]已标记 {marked} 个实体需要更新Embedding[/blue]")
        
        return marked
    
    def mark_chunks_for_update(self, chunk_ids: List[str]) -> int:
        """
        标记文本块需要更新嵌入向量
        
        参数：
            chunk_ids: 需要标记的文本块ID列表
            
        返回：
            int: 成功标记的文本块数量
        
        实现细节：
        - 将文本块的needs_reembedding标志设置为true
        - 更新last_updated时间戳
        - 这些标记将在后续更新过程中被检测并处理
        
        业务逻辑：
        - 支持选择性更新特定文本块，优化计算资源使用
        - 标记-更新模式确保操作的可追踪性
        - 时间戳用于增量更新策略
        """
        # 空检查：如果没有提供文本块ID列表，直接返回
        if not chunk_ids:
            return 0
            
        # 使用UNWIND批量标记文本块为需要更新
        # 这种方式可以在一次事务中处理多个文本块，提高效率
        query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})  # 通过文本块ID精确定位节点
        SET c.needs_reembedding = true,        # 设置更新标记
            c.last_updated = datetime()        # 更新时间戳
        RETURN count(c) AS marked              # 返回标记成功数量
        """
        
        result = self.graph.query(query, params={"chunk_ids": chunk_ids})
        marked = result[0]["marked"] if result else 0  # 安全获取标记数量
        
        # 输出标记结果，提供操作反馈
        self.console.print(f"[blue]已标记 {marked} 个Chunk需要更新Embedding[/blue]")
        
        return marked
    
    def mark_document_chunks_for_update(self, filename: str) -> int:
        """
        标记整个文档的所有文本块需要更新嵌入向量
        
        参数：
            filename: 文档文件名
            
        返回：
            int: 成功标记的文本块数量
        
        实现细节：
        - 通过文档与文本块的PART_OF关系，找到指定文档的所有文本块
        - 批量标记这些文本块需要重新计算嵌入
        - 适用于文档内容发生变更的情况
        
        业务逻辑：
        - 文档级别的批量更新，简化API使用
        - 利用图数据库的关系查询特性，高效定位相关文本块
        - 适用于文档编辑或替换场景
        """
        # 通过图数据库关系查询，找到特定文档包含的所有文本块
        # 利用PART_OF关系，从文档到文本块的级联更新
        query = """
        MATCH (d:`__Document__` {fileName: $filename})<-[:PART_OF]-(c:`__Chunk__`)  # 通过文档名和PART_OF关系找到相关文本块
        SET c.needs_reembedding = true,                                             # 设置更新标记
            c.last_updated = datetime()                                             # 更新时间戳
        RETURN count(c) AS marked                                                   # 返回标记成功数量
        """
        
        result = self.graph.query(query, params={"filename": filename})
        marked = result[0]["marked"] if result else 0  # 安全获取标记数量
        
        # 输出标记结果，提供操作反馈
        self.console.print(f"[blue]已标记文件 {filename} 的 {marked} 个Chunk需要更新Embedding[/blue]")
        
        return marked
    
    def mark_changed_files_chunks(self, changed_files: List[str]) -> int:
        """
        标记变更文件的所有文本块需要更新嵌入向量
        
        参数：
            changed_files: 变更的文件路径列表
            
        返回：
            int: 成功标记的文本块总数量
        
        实现细节：
        - 对每个变更的文件，提取文件名（不包含路径）
        - 调用mark_document_chunks_for_update标记该文件的所有文本块
        - 累计所有文件的标记数量
        - 适用于批量处理多个变更文件的场景
        
        业务逻辑：
        - 支持批量文件更新场景，简化API调用
        - 路径处理适配不同文件系统
        - 增量更新策略的实现关键
        """
        # 空检查：如果没有变更文件列表，直接返回
        if not changed_files:
            return 0
            
        total_marked = 0
        # 遍历所有变更文件，批量处理
        for filename in changed_files:
            # 获取文件名（不包含路径）
            # 处理不同操作系统的路径分隔符
            file_name = filename.split("/")[-1].split("\\")[-1]
            # 递归调用单文件标记方法
            marked = self.mark_document_chunks_for_update(file_name)
            total_marked += marked
        
        return total_marked
    
    def display_stats(self):
        """
        显示嵌入更新的统计信息
        
        输出内容：
        - 实体和文本块的更新数量
        - 总更新数量和错误数量
        - 详细的时间统计，包括总耗时、嵌入计算时间和数据库操作时间
        - 各阶段耗时占比分析
        
        业务逻辑：
        - 提供详细的性能分析和监控指标
        - 帮助识别系统瓶颈，优化性能
        - 直观展示更新结果，便于问题排查
        """
        # 使用rich库进行格式化输出，提高可读性
        self.console.print("\n[bold cyan]Embedding更新统计[/bold cyan]")
        self.console.print(f"[blue]实体更新: {self.stats['entity_updates']} 个[/blue]")
        self.console.print(f"[blue]Chunk更新: {self.stats['chunk_updates']} 个[/blue]")
        self.console.print(f"[blue]总更新: {self.stats['total_updates']} 个[/blue]")
        self.console.print(f"[blue]错误: {self.stats['errors']} 个[/blue]")
        
        # 计算并显示时间占比 - 性能分析的关键部分
        if self.total_time > 0:  # 避免除零错误
            embedding_percent = (self.embedding_time / self.total_time) * 100
            db_percent = (self.db_time / self.total_time) * 100
            self.console.print(f"[blue]总耗时: {self.total_time:.2f}秒，"
                              f"其中: 嵌入计算: {self.embedding_time:.2f}秒 ({embedding_percent:.1f}%)，"
                              f"数据库操作: {self.db_time:.2f}秒 ({db_percent:.1f}%)[/blue]")
        else:
            # 处理没有执行任何操作的情况
            self.console.print(f"[blue]总耗时: {self.total_time:.2f}秒，"
                              f"其中: 嵌入计算: {self.embedding_time:.2f}秒，"
                              f"数据库操作: {self.db_time:.2f}秒[/blue]")
    
    def process(self, entity_limit: int = 500, chunk_limit: int = 500) -> Dict[str, Any]:
        """
        执行完整的嵌入更新流程
        
        参数：
            entity_limit: 处理的最大实体数量
            chunk_limit: 处理的最大文本块数量
            
        返回：
            Dict: 处理结果统计，包含更新数量和时间信息
        
        实现流程：
        1. 设置嵌入更新追踪机制
        2. 更新实体嵌入向量
        3. 更新文本块嵌入向量
        4. 显示详细的处理统计信息
        5. 返回处理结果摘要
        
        业务逻辑：
        - 提供端到端的嵌入更新功能
        - 包含完整的错误处理和异常报告
        - 自动初始化追踪机制，确保系统一致性
        - 返回详细的处理结果，便于上层调用者了解执行情况
        """
        # 记录开始时间，用于计算总处理时间
        start_time = time.time()
        
        try:
            # 步骤1: 设置嵌入追踪
            # 初始化时间戳和元数据，建立基线
            self.setup_embedding_tracking()
            
            # 步骤2: 更新实体嵌入
            # 处理所有需要更新的实体
            entity_count = self.update_entity_embeddings()
            
            # 步骤3: 更新文本块嵌入
            # 处理所有需要更新的文本块
            chunk_count = self.update_chunk_embeddings()
            
            # 步骤4: 显示统计信息
            # 提供详细的性能和操作统计
            self.display_stats()
            
            # 计算总时间
            self.total_time = time.time() - start_time
            
            # 步骤5: 返回处理结果统计
            # 提供结构化的结果信息，便于上层调用者分析
            return {
                "entity_updates": entity_count,      # 实体更新数量
                "chunk_updates": chunk_count,        # 文本块更新数量
                "total_updates": entity_count + chunk_count,  # 总更新数量
                "total_time": self.total_time,       # 总耗时
                "embedding_time": self.embedding_time,  # 嵌入计算时间
                "db_time": self.db_time             # 数据库操作时间
            }
            
        except Exception as e:
            # 错误处理和报告
            # 记录错误并向上抛出异常，确保调用者能感知到异常
            self.console.print(f"[red]Embedding更新过程中出现错误: {e}[/red]")
            raise