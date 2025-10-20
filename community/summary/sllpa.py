from typing import List, Dict
from .base import BaseSummarizer
import time

from config.settings import BATCH_SIZE

"""
SLLPA算法社区摘要模块

本模块实现了基于SLLPA (Speaker-Listener Label Propagation Algorithm)算法的社区摘要生成器。
SLLPA是一种高效的社区检测算法，特别适合大规模图数据。该模块负责收集SLLPA算法识别的社区信息，
为后续的摘要生成提供数据支持。

主要功能：
- 社区信息收集：从图数据库中获取SLLPA算法识别的社区及其实体和关系信息
- 批处理支持：当社区数量较多时，采用分批处理策略避免查询超时
- 优先级排序：基于社区权重优先处理重要性更高的社区
- 异常处理：提供备用的信息收集方法，确保系统稳定性

设计特点：
- 自适应查询：根据社区规模动态调整查询策略
- 资源优化：限制实体数量和处理批次，避免系统资源过度消耗
- 容错机制：在主查询失败时自动切换到备用查询
- 与Leiden实现类似但针对SLLPA算法特性做了优化
"""

class SLLPASummarizer(BaseSummarizer):
    """SLLPA算法的社区摘要生成器
    
    继承自BaseSummarizer抽象类，实现了特定于SLLPA算法的社区信息收集逻辑。
    SLLPA (Speaker-Listener Label Propagation Algorithm)是一种高效的社区检测算法，
    该实现主要负责收集和处理SLLPA算法生成的社区信息。
    
    特性：
    - 处理SLLPA算法生成的社区结构（所有社区都在level 0）
    - 支持大量社区的分批处理
    - 基于社区权重的优先级排序
    - 提供主备两种信息收集策略
    - 针对SLLPA算法的特性进行了优化
    """
    
    def collect_community_info(self) -> List[Dict]:
        """收集SLLPA社区信息
        
        实现BaseSummarizer的抽象方法，从图数据库中收集SLLPA算法生成的社区信息。
        该方法采用自适应策略，根据社区数量选择一次性查询或分批查询。
        
        返回：
            社区信息列表，每个社区包含communityId、nodes和rels字段
            
        处理流程：
        1. 查询社区总数
        2. 根据社区数量选择适当的查询策略
        3. 获取社区内的实体和实体间关系
        4. 格式化结果数据
        5. 异常处理和回退策略
        """
        start_time = time.time()
        print("收集SLLPA社区信息...")
        
        try:
            # 查询社区总数，判断查询策略
            count_result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0  // SLLPA的所有社区都在level 0
            RETURN count(c) AS community_count
            """)
            
            community_count = count_result[0]['community_count'] if count_result else 0
            if not community_count:
                print("没有找到SLLPA社区")
                return []
                
            print(f"找到 {community_count} 个SLLPA社区，开始收集详细信息")
            
            # 当社区数量超过阈值时使用批处理
            if community_count > 1000:
                return self._collect_info_in_batches(community_count)
            
            # 当社区数量较少时，一次性查询所有社区信息
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c ORDER BY c.community_rank DESC NULLS LAST  // 按社区权重降序排序
            LIMIT 200  // 限制处理的社区数量
            
            // 获取社区中的实体
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) AS nodes
            WHERE size(nodes) > 1  // 只处理包含多个实体的社区
            
            // 获取实体间的关系
            CALL {  
                WITH nodes
                MATCH (n1:__Entity__)
                WHERE n1 IN nodes
                MATCH (n2:__Entity__)
                WHERE n2 IN nodes AND id(n1) < id(n2)
                MATCH (n1)-[r]->(n2)
                WITH collect(distinct r) as relationships
                LIMIT 100  // 限制关系数量，避免查询过大
                RETURN relationships
            }
            
            // 返回格式化的结果
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: n.description, 
                       type: [el in labels(n) WHERE el <> '__Entity__'][0]  // 提取实体类型
                   }] AS nodes,
                   [r in relationships | {
                       start: startNode(r).id, 
                       type: type(r), 
                       end: endNode(r).id, 
                       description: r.description
                   }] AS rels
            """)
            
            elapsed_time = time.time() - start_time
            print(f"收集到 {len(result)} 个SLLPA社区信息，耗时: {elapsed_time:.2f}秒")
            return result
            
        except Exception as e:
            print(f"收集SLLPA社区信息出错: {e}")
            # 执行备用信息收集方法
            return self._collect_info_fallback()
    
    def _collect_info_in_batches(self, total_count: int) -> List[Dict]:
        """分批收集社区信息
        
        当社区数量较多时，采用分批处理策略收集社区信息，避免单次查询过大导致超时。
        对于SLLPA算法特有的优化：在处理大量社区时，限制每个社区中的实体数量，
        以进一步优化查询性能和资源使用。
        
        参数：
            total_count: 社区总数
            
        返回：
            所有批次收集的社区信息列表
            
        实现步骤：
        1. 计算合适的批次大小（默认50，可根据配置调整）
        2. 计算总批次数
        3. 循环处理每一批社区
        4. 对每批执行社区信息查询，限制每个社区的实体数量
        5. 合并所有批次结果
        6. 处理异常并跳过出错的批次
        """
        # 确定批次大小，确保在合理范围内
        batch_size = 50  # 默认批处理大小
        if BATCH_SIZE:
            batch_size = min(50, max(10, BATCH_SIZE // 2))  # 调整为适合社区收集的批次大小
            
        total_batches = (total_count + batch_size - 1) // batch_size
        all_results = []
        
        print(f"使用批处理收集SLLPA社区信息，共 {total_batches} 批")
        
        for batch in range(total_batches):
            # 限制最大处理批次，避免资源过度消耗
            if batch > 20:  # 限制批次
                print(f"已达到最大批次限制(20)，停止收集")
                break
                
            # 计算当前批次的偏移量
            skip = batch * batch_size
            
            try:
                # 执行当前批次的查询，对SLLPA特有的优化：限制实体数量
                batch_result = self.graph.query("""
                MATCH (c:`__Community__`)
                WHERE c.level = 0
                WITH c ORDER BY c.community_rank DESC NULLS LAST  // 按社区权重排序
                SKIP $skip LIMIT $batch_size  // 分页处理
                
                // 获取社区中的实体
                MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
                WITH c, collect(e) as nodes
                WHERE size(nodes) > 1  // 只处理包含多个实体的社区
                
                // 获取实体间的关系
                CALL {  
                    WITH nodes
                    WITH nodes[0..20] AS limited_nodes  // 限制实体数量，避免查询过大
                    MATCH (n1)-->(n2)
                    WHERE n1 IN limited_nodes AND n2 IN limited_nodes
                    RETURN collect(distinct relationship(n1, n2)) as relationships
                }
                
                // 返回格式化的结果
                RETURN c.id AS communityId,
                       [n in nodes | {
                           id: n.id, 
                           description: n.description, 
                           type: [el in labels(n) WHERE el <> '__Entity__'][0]
                       }] AS nodes,
                       [r in relationships | {
                           start: startNode(r).id, 
                           type: type(r), 
                           end: endNode(r).id, 
                           description: r.description
                       }] AS rels
                """, params={"skip": skip, "batch_size": batch_size})
                
                # 合并当前批次结果
                all_results.extend(batch_result)
                print(f"批次 {batch+1}/{total_batches} 完成，收集到 {len(batch_result)} 个社区")
                
            except Exception as e:
                print(f"批次 {batch+1} 处理出错: {e}")
                # 出错时继续处理下一批次
                continue
        
        return all_results
    
    def _collect_info_fallback(self) -> List[Dict]:
        """备用的信息收集方法
        
        当主信息收集方法失败时，使用简化的查询策略获取基本的社区信息。
        该方法优先级较低，但更加稳定，只获取核心信息，不包含关系数据。
        
        返回：
            简化的社区信息列表，仅包含communityId和nodes字段
            
        实现特点：
        - 使用更简单的查询结构
        - 限制处理的社区数量
        - 不查询关系数据
        - 使用coalesce处理缺失的描述信息
        """
        try:
            print("尝试使用简化查询收集社区信息...")
            # 使用简化的查询语句，仅获取基本信息
            result = self.graph.query("""
            MATCH (c:`__Community__`)
            WHERE c.level = 0
            WITH c LIMIT 50  // 限制获取的社区数量
            MATCH (c)<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect(e) as nodes
            WHERE size(nodes) > 1  // 只处理包含多个实体的社区
            RETURN c.id AS communityId,
                   [n in nodes | {
                       id: n.id, 
                       description: coalesce(n.description, 'No description'),  // 处理空描述
                       type: labels(n)[0]  // 获取实体类型
                   }] AS nodes,
                   [] AS rels  // 简化版本不包含关系信息
            """)
            
            print(f"使用简化查询收集到 {len(result)} 个社区信息")
            return result
        except Exception as e:
            print(f"简化查询也失败: {e}")
            return []