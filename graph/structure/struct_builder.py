import time
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document

from graph.core import connection_manager, generate_hash
from config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE
from config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS


class GraphStructureBuilder:
    """
    图结构构建器
    
    功能：
    - 在Neo4j数据库中创建和管理文档(Document)和文本块(Chunk)节点
    - 文本块(Chunk)节点的创建和批量处理
    - 建立文档与文本块之间的PART_OF关系
    - 维护文本块之间的顺序(NEXT_CHUNK)关系
    - 记录文档的第一个文本块(FIRST_CHUNK)关系
    
    实现思路：
    - 采用批处理机制减少数据库往返，优化性能
    - 支持并行处理大规模文档数据
    - 使用哈希算法生成文本块唯一标识，使用 MERGE 语句确保数据一致性
    - 维护文档的完整结构和文本流顺序
    - 支持增量更新和数据一致性保障
    - 动态计算批处理大小以适应不同规模数据
    """
    
    def __init__(self, batch_size=100):
        """
        初始化图结构构建器
        
        参数：
            batch_size: 批处理大小，用于控制单次数据库操作的数据量
            较大的批处理大小可以提高性能，但会增加内存占用
            较小的批处理大小更适合内存受限环境
        
        实现细节：
        - 从连接管理器获取数据库连接
        - 刷新数据库模式以确保最新的节点标签和关系类型可用
        - 使用传入的批处理大小或默认配置值
        """
        self.graph = connection_manager.get_connection()
        self.graph.refresh_schema()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
            
    def clear_database(self):
        """
        清空数据库中的所有节点和关系
        
        注意事项：
        - 此操作不可撤销，会删除所有图数据
        - 仅用于初始化或重置数据库时使用
        
        实现思路：
        - 使用DETACH DELETE语句同时删除节点和相关关系
        - 无需WHERE条件，匹配所有节点
        """
        clear_query = """
            MATCH (n)
            DETACH DELETE n
            """
        self.graph.query(clear_query)
        
    def create_document(self, type: str, uri: str, file_name: str, domain: str) -> Dict:
        """
        创建或更新Document节点
        
        参数：
            type: 文档类型，如PDF、TXT等
            uri: 文档的统一资源标识符，用于定位文档
            file_name: 文件名，作为文档节点的唯一标识
            domain: 文档域，用于分类和组织文档
            
        返回：
            Dict: 创建或更新的文档节点信息
            
        实现思路：
        - 使用MERGE语句确保文件名唯一，避免重复创建
        - 使用SET语句更新文档的类型、URI和域信息
        - 这种实现支持文档的幂等操作，相同文件可重复调用
        """
        query = """
        MERGE(d:`__Document__` {fileName: $file_name}) 
        SET d.type=$type, d.uri=$uri, d.domain=$domain
        RETURN d;
        """
        doc = self.graph.query(
            query,
            {"file_name": file_name, "type": type, "uri": uri, "domain": domain}
        )
        return doc
        
    def create_relation_between_chunks(self, file_name: str, chunks: List) -> List[Dict]:
        """
        创建Chunk节点并建立文档-文档块关系
        
        参数：
            file_name: 文件名，用于关联文档节点
            chunks: 文本块列表，每个块包含文本内容
            
        返回：
            List[Dict]: 包含块ID和文档对象的列表
            
        实现思路：
        1. 遍历每个文本块，生成唯一哈希ID
        2. 计算每个块在文档中的位置、偏移量等元数据
        3. 构建块之间的顺序关系和文档-块关系
        4. 批量处理以提高数据库操作效率，写入数据库
        5. 记录处理时间用于性能监控
        """
        t0 = time.time()
        
        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0
        
        # 处理每个chunk
        for i, chunk in enumerate(chunks):
            # 将chunk列表内容合并为单个字符串
            page_content = ''.join(chunk)
            # 生成文本块的唯一标识哈希值
            current_chunk_id = generate_hash(page_content)
            # 计算文本块在文档中的位置（从1开始）
            position = i + 1
            # 处理第一个块特殊情况，设置previous_chunk_id为当前ID
            previous_chunk_id = current_chunk_id if i == 0 else lst_chunks_including_hash[-1]['chunk_id']
            
            # 计算内容偏移量，用于定位文本在原始文档中的位置
            if i > 0:
                last_page_content = ''.join(chunks[i-1])
                offset += len(last_page_content)
                
            # 标记第一个块
            firstChunk = (i == 0)
            
            # 创建metadata和Document对象，便于后续处理
            metadata = {
                "position": position,
                "length": len(page_content),  # 文本长度
                "content_offset": offset,    # 在原始文档中的偏移量
                "tokens": len(chunk)        # 令牌数量
            }
            chunk_document = Document(page_content=page_content, metadata=metadata)
            
            # 准备批处理数据
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,
                "previous_id": previous_chunk_id,
                "content_offset": offset,
                "tokens": len(chunk)
            }
            batch_data.append(chunk_data)
            
            # 收集结果信息
            lst_chunks_including_hash.append({
                'chunk_id': current_chunk_id,
                'chunk_doc': chunk_document
            })
            
            # 创建关系数据
            if firstChunk:
                # 第一个块与文档建立FIRST_CHUNK关系
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                # 非第一个块与前一个块建立NEXT_CHUNK关系
                relationships.append({
                    "type": "NEXT_CHUNK",
                    "previous_chunk_id": previous_chunk_id,
                    "current_chunk_id": current_chunk_id
                })
            
            # 当累积了一定量的数据时，进行批处理
            if len(batch_data) >= self.batch_size:
                self._process_batch(file_name, batch_data, relationships)
                batch_data = []
                relationships = []
        
        # 处理剩余的数据
        if batch_data:
            self._process_batch(file_name, batch_data, relationships)
        
        # 记录并打印处理时间
        t1 = time.time()
        print(f"创建关系耗时: {t1-t0:.2f}秒")
        
        return lst_chunks_including_hash
    
    def _process_batch(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        批量处理一组文本块和关系
        
        参数：
            file_name: 文件名
            batch_data: 批处理数据列表，包含所有待处理文本块信息
            relationships: 关系数据列表，包含所有待创建关系信息
        """
        if not batch_data:
            return
            
        # 分离FIRST_CHUNK和NEXT_CHUNK关系，便于针对性处理
        first_relationships = [r for r in relationships if r.get("type") == "FIRST_CHUNK"]
        next_relationships = [r for r in relationships if r.get("type") == "NEXT_CHUNK"]

        """
        使用优化的数据库操作方法处理批数据，实现思路：
        1. 分三个阶段执行数据库操作，每阶段专注于特定类型的操作
        2. 使用UNWIND语句高效处理批量数据
        3. 使用MERGE确保数据一致性，避免重复创建
        4. 仅在必要时（关系列表非空）执行相应查询
        
        数据库优化策略：
        - 利用Neo4j的批处理能力处理大量数据
        - 分离不同类型关系的处理，减少单次查询复杂度
        """
        # 第一阶段：创建Chunk节点并建立PART_OF关系
        query_chunks_and_part_of = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, 
            c.position = data.position, 
            c.length = data.length, 
            c.fileName = data.f_name,
            c.content_offset = data.content_offset, 
            c.tokens = data.tokens
        WITH c, data
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunks_and_part_of, params={"batch_data": batch_data})
        
        # 第二阶段：处理FIRST_CHUNK关系（文档到第一个块的关系）
        if first_relationships:
            query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """
            self.graph.query(query_first_chunk, params={
                "f_name": file_name,
                "relationships": first_relationships
            })
        
        # 第三阶段：处理NEXT_CHUNK关系（块之间的顺序关系）
        if next_relationships:
            query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            MERGE (pc)-[:NEXT_CHUNK]->(c)
            """
            self.graph.query(query_next_chunk, params={"relationships": next_relationships})
    
    def parallel_process_chunks(self, file_name: str, chunks: List, max_workers=None) -> List[Dict]:
        """
        并行处理文本块，提高大规模数据处理效率
        
        参数：
            file_name: 文件名
            chunks: 文本块列表
            max_workers: 并行工作线程数
            
        返回：
            List[Dict]: 包含块ID和文档对象的列表
            
        实现思路：
        1. 根据数据量自动选择处理策略：小数据集使用标准方法，大数据集使用并行方法
        2. 将文本块分成多个批次，每个批次分配给不同线程处理
        3. 并行计算每个批次的哈希值、位置信息和关系数据
        4. 合并所有批次结果后批量写入数据库（文本块顺序，文本块与文档之间关系）
        
        性能优化特点：
        - 动态批次大小计算，根据数据量和线程数自动调整
        - 并行文本处理与哈希计算，充分利用多核CPU
        - 内存中批处理后再写入数据库，减少数据库I/O
        - 完善的错误处理，确保单批次失败不影响整体处理
        - 进度显示，提高用户体验
        """
        max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        # 对于小数据集，直接使用单线程方法，避免线程开销
        if len(chunks) < 100:  # 小数据集阈值
            return self.create_relation_between_chunks(file_name, chunks)
        
        # 计算批次大小，确保每个线程有合理的数据量处理
        batch_size = max(10, len(chunks) // max_workers)
        chunk_batches = []
        
        # 将chunks分割成多个批次
        for i in range(0, len(chunks), batch_size):
            chunk_batches.append(chunks[i:i+batch_size])
        
        print(f"并行处理 {len(chunks)} 个块，每批次 {batch_size} 个，共 {len(chunk_batches)} 批次")
        
        # 定义批次处理函数
        def process_chunk_batch(batch, start_index):
            results = []
            current_chunk_id = ""
            batch_data = []
            relationships = []
            offset = 0
            
            # 处理非首个批次时，需要获取前一个批次的最后一个块信息
            if start_index > 0 and start_index < len(chunks):
                # 获取前一个chunk的ID作为起始点
                prev_chunk = chunks[start_index - 1]
                prev_content = ''.join(prev_chunk)
                current_chunk_id = generate_hash(prev_content)
                # 计算前面所有chunk的累计offset
                for j in range(start_index):
                    offset += len(''.join(chunks[j]))
            
            # 处理批次内的每个chunk
            for i, chunk in enumerate(batch):
                # 计算绝对索引（在整个文档中的位置）
                abs_index = start_index + i
                page_content = ''.join(chunk)
                previous_chunk_id = current_chunk_id
                current_chunk_id = generate_hash(page_content)
                position = abs_index + 1
                
                # 更新当前批次内的offset
                if i > 0:
                    last_page_content = ''.join(batch[i-1])
                    offset += len(last_page_content)
                    
                # 判断是否为文档的第一个块
                firstChunk = (abs_index == 0)
                
                # 创建metadata和Document对象
                metadata = {
                    "position": position,
                    "length": len(page_content),
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                chunk_document = Document(page_content=page_content, metadata=metadata)
                
                # 准备批处理数据
                chunk_data = {
                    "id": current_chunk_id,
                    "pg_content": chunk_document.page_content,
                    "position": position,
                    "length": chunk_document.metadata["length"],
                    "f_name": file_name,
                    "previous_id": previous_chunk_id,
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                batch_data.append(chunk_data)
                
                # 收集结果
                results.append({
                    'chunk_id': current_chunk_id,
                    'chunk_doc': chunk_document
                })
                
                # 创建关系数据
                if firstChunk:
                    relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
                else:
                    relationships.append({
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id
                    })
            
            # 返回处理结果
            return {
                "batch_data": batch_data,
                "relationships": relationships,
                "results": results
            }
        
        # 并行处理所有批次
        start_time = time.time()
        all_batch_data = []
        all_relationships = []
        all_results = []
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(process_chunk_batch, batch, i * batch_size): i
                for i, batch in enumerate(chunk_batches)
            }
            
            # 收集所有处理结果
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_batch_data.extend(result["batch_data"])
                    all_relationships.extend(result["relationships"])
                    all_results.extend(result["results"])
                except Exception as e:
                    # 单个批次失败不影响整体处理
                    print(f"处理批次时出错: {e}")
        
        # 写入数据库阶段
        print(f"并行处理完成，共 {len(all_batch_data)} 个块，开始写入数据库")
        
        # 数据库写入批次大小，可能与内存处理批次不同
        db_batch_size = 500
        total_batches = (len(all_batch_data) + db_batch_size - 1) // db_batch_size
        
        # 分批写入数据库，减少单次事务大小
        for i in range(0, len(all_batch_data), db_batch_size):
            batch = all_batch_data[i:i+db_batch_size]
            # 筛选出与当前批次相关的关系数据
            rel_batch = [r for r in all_relationships 
                         if r.get("type") == "FIRST_CHUNK" and any(b["id"] == r["chunk_id"] for b in batch)
                         or r.get("type") == "NEXT_CHUNK" and any(b["id"] == r["current_chunk_id"] for b in batch)]
            
            # 写入数据库
            self._create_chunks_and_relationships(file_name, batch, rel_batch)
            print(f"已写入批次 {i//db_batch_size + 1}/{total_batches}")
        
        # 记录总耗时
        end_time = time.time()
        print(f"写入数据库完成，耗时: {end_time - start_time:.2f}秒")
        
        return all_results
    
    def _create_chunks_and_relationships(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        执行创建文本块和关系的数据库查询
        
        参数：
            file_name: 文件名
            batch_data: 批处理数据列表
            relationships: 关系数据列表，包含FIRST_CHUNK和NEXT_CHUNK两种关系
            
        实现思路：
        1. 使用FOREACH和CASE语句在单个查询中处理不同类型的关系
        2. 通过批处理操作减少数据库往返次数
        3. 使用MERGE确保数据一致性，避免重复创建
        4. 分三个阶段执行不同类型的数据库操作
        
        技术特点：
        - 使用Cypher的条件语句根据关系类型执行不同操作
        - 利用Neo4j的批处理能力高效写入大量数据
        - 保持图结构的一致性和完整性
        """
        # 第一阶段：创建Chunk节点和PART_OF关系
        query_chunk_part_of = """
            UNWIND $batch_data AS data
            MERGE (c:`__Chunk__` {id: data.id})
            SET c.text = data.pg_content, 
                c.position = data.position, 
                c.length = data.length, 
                c.fileName = data.f_name,
                c.content_offset = data.content_offset, 
                c.tokens = data.tokens
            WITH data, c
            MATCH (d:`__Document__` {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunk_part_of, params={"batch_data": batch_data})
        
        # 第二阶段：创建FIRST_CHUNK关系（使用条件FOREACH避免重复处理）
        query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        self.graph.query(query_first_chunk, params={
            "f_name": file_name,
            "relationships": relationships
        })
        
        # 第三阶段：创建NEXT_CHUNK关系（使用条件FOREACH避免重复处理）
        query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        self.graph.query(query_next_chunk, params={"relationships": relationships})