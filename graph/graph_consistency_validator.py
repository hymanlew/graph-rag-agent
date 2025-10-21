import time
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table
from config.neo4jdb import get_db_manager

"""
图谱一致性验证模块

该模块负责验证和修复Neo4j图数据库中的一致性问题，确保图结构的完整性和正确性。

核心功能:
- 检查孤立节点（没有被引用的实体）
- 验证文档与文本块的关联关系
- 检测空文本块
- 验证文档链接完整性
- 检查文本块链的连续性
- 自动修复检测到的问题
- 生成详细的验证和修复报告

实现特点:
- 提供全面的一致性检查功能
- 支持自动修复大多数常见问题
- 详细的进度和结果可视化
- 性能监控和统计
- 安全的修复机制，避免误删重要数据
"""

class GraphConsistencyValidator:
    """
    图谱一致性验证器
    
    功能：
    - 全面检查图数据库中的结构和内容一致性问题
    - 识别并报告各种类型的数据异常
    - 提供自动化的修复机制解决一致性问题
    - 生成详细的验证和修复统计报告
    
    实现思路：
    - 采用多种验证策略检查不同类型的一致性问题
    - 实现隔离的修复方法，针对每种问题类型单独处理
    - 使用Rich库提供美观的结果展示
    - 记录详细的性能指标和统计数据
    - 实现非侵入式验证，避免影响正常数据
    """
    
    def __init__(self):
        """
        初始化图谱一致性验证器
        
        实现细节：
        - 创建Rich控制台对象用于美化输出
        - 获取数据库管理器连接
        - 初始化性能计时器变量
        - 初始化统计信息字典，用于记录各类问题的数量
        
        统计指标说明：
        - orphan_entities: 孤立实体数量
        - dangling_chunks: 悬空文本块数量
        - empty_chunks: 空文本块数量
        - broken_doc_links: 断开的文档链接数量
        - broken_chunk_chains: 断开的文本块链数量
        - total_issues: 问题总数
        - repaired_issues: 已修复问题数
        """
        # 创建Rich控制台对象，用于美化输出
        self.console = Console()
        # 获取数据库连接
        self.graph = get_db_manager().graph
        
        # 性能计时器，记录验证和修复操作的耗时
        self.validation_time = 0
        self.repair_time = 0
        
        # 初始化验证统计信息字典
        self.validation_stats = {
            "orphan_entities": 0,
            "dangling_chunks": 0,
            "empty_chunks": 0,
            "broken_doc_links": 0,
            "broken_chunk_chains": 0,
            "total_issues": 0,
            "repaired_issues": 0
        }
    
    def check_orphan_entities(self) -> Tuple[List[str], int]:
        """
        检查孤立的实体节点（没有被任何Chunk引用）
        
        返回：
            Tuple: (孤立实体ID列表, 孤立实体总数)
            
        实现思路：
        1. 查找没有MENTIONS关系指向的实体节点
        2. 排除手动编辑过或受保护的实体
        3. 首先获取孤立实体的总数
        4. 然后获取具体的孤立实体ID列表（限制最多1000个）
        5. 更新统计信息并返回结果
        
        数据安全考虑：
        - 不检查标记为manual_edit或protected的实体，避免误删用户手动创建的数据
        - 限制返回ID数量，避免内存占用过大
        """
        # 统计孤立实体数量的查询
        query = """
        MATCH (e:`__Entity__`)
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        RETURN e.id AS entity_id, count(e) AS count
        """
        
        result = self.graph.query(query)
        
        orphan_ids = []
        orphan_count = 0
        
        if result:
            # 获取孤立实体总数
            orphan_count = result[0]["count"]
            # 获取最多1000个孤立实体ID，用于后续可能的修复操作
            id_query = """
            MATCH (e:`__Entity__`)
            WHERE NOT (e)<-[:MENTIONS]-()
              AND NOT e.manual_edit = true
              AND NOT e.protected = true
            RETURN e.id AS entity_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            orphan_ids = [r["entity_id"] for r in id_result]
        
        # 更新统计信息
        self.validation_stats["orphan_entities"] = orphan_count
        
        return orphan_ids, orphan_count
    
    def check_dangling_chunks(self) -> Tuple[List[str], int]:
        """
        检查悬空的Chunk节点（没有关联到Document）
        
        返回：
            Tuple: (悬空Chunk ID列表, 悬空Chunk总数)
            
        实现思路：
        1. 查找没有PART_OF关系指向文档的Chunk节点
        2. 计算悬空Chunk的总数
        3. 获取具体的悬空Chunk ID列表（限制最多1000个）
        4. 更新统计信息并返回结果
        
        业务意义：
        悬空Chunk节点无法被关联到特定文档，影响信息检索的完整性
        这种情况通常发生在文档导入过程中断或出错时
        """
        # 统计悬空Chunk数量的查询
        query = """
        MATCH (c:`__Chunk__`)
        WHERE NOT (c)-[:PART_OF]->()
        RETURN c.id AS chunk_id, count(c) AS count
        """
        
        result = self.graph.query(query)
        
        dangling_ids = []
        dangling_count = 0
        
        if result:
            # 获取悬空Chunk总数
            dangling_count = result[0]["count"]
            # 获取最多1000个悬空Chunk ID
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE NOT (c)-[:PART_OF]->()
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            dangling_ids = [r["chunk_id"] for r in id_result]
        
        # 更新统计信息
        self.validation_stats["dangling_chunks"] = dangling_count
        
        return dangling_ids, dangling_count
    
    def check_empty_chunks(self) -> Tuple[List[str], int]:
        """
        检查空的Chunk节点（没有文本内容）
        
        返回：
            Tuple: (空Chunk ID列表, 空Chunk总数)
            
        实现思路：
        1. 查找text属性为空或NULL的Chunk节点
        2. 计算空Chunk的总数
        3. 获取具体的空Chunk ID列表（限制最多1000个）
        4. 更新统计信息并返回结果
        
        业务意义：
        空Chunk节点可能导致检索结果不准确，也浪费存储空间
        这些节点通常由文档解析错误或处理中断导致
        """
        # 统计空Chunk数量的查询
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.text IS NULL OR c.text = ''
        RETURN c.id AS chunk_id, count(c) AS count
        """
        
        result = self.graph.query(query)
        
        empty_ids = []
        empty_count = 0
        
        if result:
            # 获取空Chunk总数
            empty_count = result[0]["count"]
            # 获取最多1000个空Chunk ID
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE c.text IS NULL OR c.text = ''
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            empty_ids = [r["chunk_id"] for r in id_result]
        
        # 更新统计信息
        self.validation_stats["empty_chunks"] = empty_count
        
        return empty_ids, empty_count
    
    def check_broken_doc_links(self) -> int:
        """
        检查文档链接是否完整（Document应该有FIRST_CHUNK关系）
        
        返回：
            int: 有问题的文档数量
            
        实现思路：
        1. 查找没有FIRST_CHUNK关系的Document节点
        2. 计算这类文档的总数
        3. 更新统计信息并返回结果
        
        业务意义：
        没有FIRST_CHUNK关系的文档无法确定起始文本块
        这会导致文档浏览和检索功能出现异常
        """
        # 统计没有FIRST_CHUNK关系的文档数量
        query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()
        RETURN count(d) AS count
        """
        
        result = self.graph.query(query)
        
        # 获取统计结果，如果没有结果则默认为0
        count = result[0]["count"] if result else 0
        # 更新统计信息
        self.validation_stats["broken_doc_links"] = count
        
        return count
    
    def check_broken_chunk_chains(self) -> int:
        """
        检查文本块链是否完整（前后关系）
        
        返回：
            int: 有问题的链数量
            
        实现思路：
        1. 查找位置大于1但没有前向NEXT_CHUNK关系的Chunk节点
        2. 计算这类不连续Chunk的总数
        3. 更新统计信息并返回结果
        
        业务意义：
        断开的文本块链会导致文档内容读取不完整
        用户无法按照正确顺序浏览整个文档的内容
        """
        # 统计没有前向NEXT_CHUNK关系的Chunk数量
        query = """
        MATCH (c:`__Chunk__`)-[:PART_OF]->(d:`__Document__`)
        WHERE c.position > 1 AND NOT (c)<-[:NEXT_CHUNK]-()
        RETURN count(c) AS count
        """
        
        result = self.graph.query(query)
        
        # 获取统计结果，如果没有结果则默认为0
        count = result[0]["count"] if result else 0
        # 更新统计信息
        self.validation_stats["broken_chunk_chains"] = count
        
        return count
    
    def validate_graph(self) -> Dict[str, Any]:
        """
        执行全面的图谱验证，检查所有类型的一致性问题
        
        返回：
            Dict: 包含验证结果的详细字典
                - validation_time: 验证耗时
                - validation_stats: 验证统计信息
                - orphan_ids: 孤立实体ID列表
                - dangling_ids: 悬空Chunk ID列表
                - empty_ids: 空Chunk ID列表
                
        实现思路：
        1. 按顺序执行各类一致性检查
        2. 对发现的问题进行实时可视化报告
        3. 计算并记录总问题数量
        4. 记录验证操作耗时
        5. 返回完整的验证结果供后续处理使用
        
        验证类型说明：
        - 孤立实体：没有被任何Chunk引用的实体
        - 悬空Chunk：没有关联到Document的Chunk
        - 空Chunk：没有文本内容的Chunk
        - 断开的文档链接：没有FIRST_CHUNK关系的文档
        - 断开的文本块链：缺少前向NEXT_CHUNK关系的Chunk
        """
        # 记录开始时间
        start_time = time.time()
        
        # 1. 检查孤立实体
        orphan_ids, orphan_count = self.check_orphan_entities()
        if orphan_count > 0:
            self.console.print(f"[yellow]发现 {orphan_count} 个孤立实体节点[/yellow]")
        
        # 2. 检查悬空Chunk
        dangling_ids, dangling_count = self.check_dangling_chunks()
        if dangling_count > 0:
            self.console.print(f"[yellow]发现 {dangling_count} 个悬空Chunk节点[/yellow]")
        
        # 3. 检查空Chunk
        empty_ids, empty_count = self.check_empty_chunks()
        if empty_count > 0:
            self.console.print(f"[yellow]发现 {empty_count} 个空Chunk节点[/yellow]")
        
        # 4. 检查文档链接完整性
        broken_doc_count = self.check_broken_doc_links()
        if broken_doc_count > 0:
            self.console.print(f"[yellow]发现 {broken_doc_count} 个没有FIRST_CHUNK关系的文档[/yellow]")
        
        # 5. 检查文本块链完整性
        broken_chain_count = self.check_broken_chunk_chains()
        if broken_chain_count > 0:
            self.console.print(f"[yellow]发现 {broken_chain_count} 个断开的Chunk链[/yellow]")
        
        # 计算总问题数
        total_issues = (orphan_count + dangling_count + empty_count + 
                       broken_doc_count + broken_chain_count)
        self.validation_stats["total_issues"] = total_issues
        
        # 计算验证耗时
        self.validation_time = time.time() - start_time
        
        # 输出验证总结信息
        self.console.print(f"[blue]图谱验证完成，耗时: {self.validation_time:.2f}秒[/blue]")
        self.console.print(f"[blue]共发现 {total_issues} 个一致性问题[/blue]")
        
        # 返回详细的验证结果
        return {
            "validation_time": self.validation_time,
            "validation_stats": self.validation_stats,
            "orphan_ids": orphan_ids,
            "dangling_ids": dangling_ids,
            "empty_ids": empty_ids
        }
    
    def repair_orphan_entities(self, orphan_ids: List[str] = None) -> int:
        """
        修复孤立实体节点（删除未使用的实体）
        
        参数：
            orphan_ids: 要修复的孤立实体ID列表，如果为None则自动检测
            
        返回：
            int: 成功删除的节点数量
            
        实现思路：
        1. 如果没有提供ID列表，先执行检测获取孤立实体ID
        2. 检查是否有需要修复的实体
        3. 执行批量删除操作，确保再次验证实体确实是孤立的
        4. 返回删除的实体数量并更新统计
        
        安全机制：
        - 删除前再次验证实体确实是孤立的，并且不是手动编辑或受保护的
        - 只删除确实没有被引用的实体，避免误删重要数据
        """
        # 如果没有提供ID列表，自动检测孤立实体
        if orphan_ids is None:
            orphan_ids, _ = self.check_orphan_entities()
        
        # 如果没有需要修复的实体，直接返回
        if not orphan_ids:
            return 0
        
        # 执行批量删除操作，使用UNWIND高效处理
        delete_query = """
        UNWIND $orphan_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        DELETE e
        RETURN count(*) AS deleted
        """
        
        # 执行删除操作
        result = self.graph.query(delete_query, params={"orphan_ids": orphan_ids})
        
        # 获取删除的实体数量
        deleted = result[0]["deleted"] if result else 0
        
        # 输出删除结果
        self.console.print(f"[green]已删除 {deleted} 个孤立实体节点[/green]")
        
        return deleted
    
    def repair_dangling_chunks(self, dangling_ids: List[str] = None) -> int:
        """
        修复悬空Chunk节点（删除无关联的文本块）
        
        参数：
            dangling_ids: 要修复的悬空Chunk ID列表，如果为None则自动检测
            
        返回：
            int: 成功删除的节点数量
            
        实现思路：
        1. 如果没有提供ID列表，先执行检测获取悬空Chunk ID
        2. 检查是否有需要修复的Chunk
        3. 执行批量删除操作，使用DETACH确保同时删除相关关系
        4. 返回删除的Chunk数量并输出结果
        
        修复策略：
        - 直接删除没有关联到文档的Chunk节点
        - 使用DETACH DELETE确保彻底移除节点及所有关系
        - 删除前再次验证节点确实是悬空的
        """
        # 如果没有提供ID列表，自动检测悬空Chunk
        if dangling_ids is None:
            dangling_ids, _ = self.check_dangling_chunks()
        
        # 如果没有需要修复的Chunk，直接返回
        if not dangling_ids:
            return 0
        
        # 执行批量删除操作，使用DETACH确保同时删除所有关系
        delete_query = """
        UNWIND $dangling_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE NOT (c)-[:PART_OF]->()
        DETACH DELETE c
        RETURN count(*) AS deleted
        """
        
        # 执行删除操作
        result = self.graph.query(delete_query, params={"dangling_ids": dangling_ids})
        
        # 获取删除的Chunk数量
        deleted = result[0]["deleted"] if result else 0
        
        # 输出删除结果
        self.console.print(f"[green]已删除 {deleted} 个悬空Chunk节点[/green]")
        
        return deleted
    
    def repair_empty_chunks(self, empty_ids: List[str] = None) -> int:
        """
        修复空Chunk节点（添加占位符文本）
        
        参数：
            empty_ids: 要修复的空Chunk ID列表，如果为None则自动检测
            
        返回：
            int: 成功修复的节点数量
            
        实现思路：
        1. 如果没有提供ID列表，先执行检测获取空Chunk ID
        2. 检查是否有需要修复的Chunk
        3. 为空Chunk添加占位符文本和修复标记
        4. 返回修复的Chunk数量并输出结果
        
        修复策略：
        - 选择修复而非删除，保留节点结构完整性
        - 添加占位符文本，使节点文本不再为空
        - 设置repaired标记，便于后续识别和处理
        - 不删除节点，避免破坏文档结构
        """
        # 如果没有提供ID列表，自动检测空Chunk
        if empty_ids is None:
            empty_ids, _ = self.check_empty_chunks()
        
        # 如果没有需要修复的Chunk，直接返回
        if not empty_ids:
            return 0
        
        # 为空Chunk添加占位符文本
        repair_query = """
        UNWIND $empty_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE c.text IS NULL OR c.text = ''
        SET c.text = '[Empty Chunk]', c.repaired = true
        RETURN count(*) AS repaired
        """
        
        # 执行修复操作
        result = self.graph.query(repair_query, params={"empty_ids": empty_ids})
        
        # 获取修复的Chunk数量
        repaired = result[0]["repaired"] if result else 0
        
        # 输出修复结果
        self.console.print(f"[green]已修复 {repaired} 个空Chunk节点[/green]")
        
        return repaired
    
    def repair_broken_doc_links(self) -> int:
        """
        修复断开的文档链接（创建缺失的FIRST_CHUNK关系）
        
        返回：
            int: 成功修复的关系数量
            
        实现思路：
        1. 查找没有FIRST_CHUNK关系的文档
        2. 找到属于该文档且位置为1或null的文本块
        3. 为每个文档选择位置最靠前的文本块作为第一个块
        4. 创建文档到第一个块的FIRST_CHUNK关系
        5. 返回修复的关系数量并输出结果
        
        修复策略：
        - 优先选择position=1的文本块作为第一个块
        - 当没有position=1的块时，选择position=null的块
        - 使用ORDER BY和LIMIT确保每个文档只选择一个块
        - 使用MERGE避免重复创建关系
        """
        # 修复断开文档链接的查询
        repair_query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()
        
        MATCH (c:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c.position = 1 OR c.position IS NULL
        
        WITH d, c ORDER BY c.position ASC LIMIT 1
        MERGE (d)-[r:FIRST_CHUNK]->(c)
        
        RETURN count(r) AS repaired
        """
        
        # 执行修复操作
        result = self.graph.query(repair_query)
        
        # 获取修复的关系数量
        repaired = result[0]["repaired"] if result else 0
        
        # 输出修复结果
        self.console.print(f"[green]已修复 {repaired} 个断开的文档链接[/green]")
        
        return repaired
    
    def repair_broken_chunk_chains(self) -> int:
        """
        修复断开的Chunk链（重建NEXT_CHUNK关系）
        
        返回：
            int: 成功修复的关系数量
            
        实现思路：
        1. 按文档分组获取所有文本块
        2. 过滤出有明确position属性的文本块
        3. 按position属性排序文本块
        4. 为排序后的相邻文本块之间创建NEXT_CHUNK关系
        5. 只创建之前不存在的关系
        6. 返回修复的关系数量并输出结果
        
        修复策略：
        - 基于position属性重建文本块之间的顺序关系
        - 确保每个文档内的文本块形成完整的链
        - 使用MERGE避免重复创建已存在的关系
        - 一次性处理每个文档的所有文本块
        """
        # 修复断开Chunk链的查询
        repair_query = """
        MATCH (d:`__Document__`)
        WITH d
        MATCH (c1:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c1.position IS NOT NULL
        WITH d, c1 ORDER BY c1.position ASC
        WITH d, collect(c1) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH d, chunks[i] AS current, chunks[i+1] AS next
        WHERE NOT (current)-[:NEXT_CHUNK]->(next)
        MERGE (current)-[r:NEXT_CHUNK]->(next)
        RETURN count(r) AS repaired
        """
        
        # 执行修复操作
        result = self.graph.query(repair_query)
        
        # 获取修复的关系数量
        repaired = result[0]["repaired"] if result else 0
        
        # 输出修复结果
        self.console.print(f"[green]已修复 {repaired} 个断开的Chunk链[/green]")
        
        return repaired
    
    def repair_graph(self) -> Dict[str, Any]:
        """
        执行全面的图谱修复操作
        
        返回：
            Dict: 包含修复结果的详细字典
                - validation_time: 验证耗时
                - repair_time: 修复耗时
                - validation_stats: 验证统计信息
                - repairs: 各类问题修复数量
                
        实现思路：
        1. 先执行全面验证，获取所有需要修复的问题
        2. 按顺序修复各类问题，优先使用已获取的ID列表
        3. 计算总修复数量和修复耗时
        4. 更新统计信息并输出修复结果
        5. 返回完整的修复报告
        
        修复顺序策略：
        - 先处理节点问题（孤立实体、悬空Chunk、空Chunk）
        - 再处理关系问题（文档链接、Chunk链）
        - 这种顺序确保在修复关系前节点结构已经稳定
        """
        # 记录开始时间
        start_time = time.time()
        
        # 先进行全面验证，获取需要修复的问题
        validation_result = self.validate_graph()
        
        # 根据验证结果进行修复，按照特定顺序处理各类问题
        repairs = {
            "orphan_entities": self.repair_orphan_entities(validation_result.get("orphan_ids", [])),
            "dangling_chunks": self.repair_dangling_chunks(validation_result.get("dangling_ids", [])),
            "empty_chunks": self.repair_empty_chunks(validation_result.get("empty_ids", [])),
            "broken_doc_links": self.repair_broken_doc_links(),
            "broken_chunk_chains": self.repair_broken_chunk_chains()
        }
        
        # 计算总修复数量
        total_repaired = sum(repairs.values())
        self.validation_stats["repaired_issues"] = total_repaired
        
        # 计算修复耗时
        self.repair_time = time.time() - start_time
        
        # 输出修复总结信息
        self.console.print(f"[blue]图谱修复完成，耗时: {self.repair_time:.2f}秒[/blue]")
        self.console.print(f"[blue]共修复 {total_repaired} 个一致性问题[/blue]")
        
        # 返回详细的修复结果
        return {
            "validation_time": self.validation_time,
            "repair_time": self.repair_time,
            "validation_stats": self.validation_stats,
            "repairs": repairs
        }
    
    def display_graph_stats(self):
        """
        显示图谱统计信息，包括节点和关系的分布情况
        
        实现思路：
        1. 获取所有类型节点的统计信息
        2. 获取所有关系类型的统计信息
        3. 使用Rich库的Table组件美化输出
        4. 按照数量排序显示关系类型
        5. 计算并显示节点和关系的总数
        
        展示内容：
        - 各类节点数量（文档、文本块、实体）
        - 各类关系数量，按数量降序排列
        - 节点和关系的总计数量
        """
        # 获取图谱节点统计信息
        stats_query = """
        MATCH (n)
        RETURN 
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """
        
        stats_result = self.graph.query(stats_query)
        
        # 检查是否成功获取统计数据
        if not stats_result:
            self.console.print("[yellow]无法获取图谱统计信息[/yellow]")
            return
        
        node_stats = stats_result[0]
        
        # 获取关系类型统计信息
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """
        
        rel_result = self.graph.query(rel_query)
        
        # 创建并显示节点统计表
        node_table = Table(title="图谱节点统计")
        node_table.add_column("节点类型", style="cyan")
        node_table.add_column("数量", justify="right")
        
        node_table.add_row("__Document__", str(node_stats["doc_count"]))
        node_table.add_row("__Chunk__", str(node_stats["chunk_count"]))
        node_table.add_row("__Entity__", str(node_stats["entity_count"]))
        node_table.add_row("总计", str(node_stats["total_nodes"]), style="bold")
        
        self.console.print(node_table)
        
        # 创建并显示关系统计表（如果有数据）
        if rel_result:
            rel_table = Table(title="图谱关系统计")
            rel_table.add_column("关系类型", style="cyan")
            rel_table.add_column("数量", justify="right")
            
            total_rels = 0
            # 添加各类关系数据
            for rel in rel_result:
                rel_table.add_row(rel["rel_type"], str(rel["count"]))
                total_rels += rel["count"]
                
            # 添加总计行
            rel_table.add_row("总计", str(total_rels), style="bold")
            
            self.console.print(rel_table)
    
    def process(self, repair: bool = True) -> Dict[str, Any]:
        """
        执行完整的图谱一致性验证和修复流程
        
        参数：
            repair: 是否执行修复操作
            
        返回：
            Dict: 包含处理结果的详细字典
                - validation_result: 验证结果
                - repair_result: 修复结果（如果执行了修复）
                - total_time: 总耗时
                
        实现思路：
        1. 首先显示图谱的基本统计信息
        2. 执行全面的一致性验证
        3. 根据repair参数和验证结果决定是否执行修复
        4. 返回完整的处理结果
        5. 捕获并报告处理过程中的异常
        
        流程控制：
        - 只有当repair=True且存在问题时才执行修复
        - 无论修复是否执行，始终返回验证结果
        - 错误处理确保问题能够被正确报告而不会静默失败
        """
        try:
            # 显示图谱基本统计信息，提供上下文
            self.display_graph_stats()
            
            # 执行全面的一致性验证
            validation_result = self.validate_graph()
            
            # 如果需要修复并且存在问题，执行修复操作
            if repair and validation_result["validation_stats"]["total_issues"] > 0:
                repair_result = self.repair_graph()
                return {
                    "validation_result": validation_result,
                    "repair_result": repair_result,
                    "total_time": self.validation_time + self.repair_time
                }
            
            # 如果不需要修复或没有问题，只返回验证结果
            return {
                "validation_result": validation_result,
                "total_time": self.validation_time
            }
            
        except Exception as e:
            # 捕获并显示处理过程中的异常
            self.console.print(f"[red]图谱一致性验证过程中出现错误: {e}[/red]")
            # 重新抛出异常，让调用者知道处理失败
            raise