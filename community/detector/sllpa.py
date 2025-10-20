from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from config.settings import GDS_CONCURRENCY

"""
SLLPA社区检测算法实现

本模块实现了基于结构化标签传播算法(Structural Label Propagation Algorithm)的社区检测器。
SLLPA是一种高效的社区检测算法，特别适合处理大规模图数据，通过迭代标签传播过程来识别社区。

SLLPA算法的核心优势：
- 高效处理大规模图数据
- 算法复杂度低，线性时间复杂度
- 可以检测重叠社区
- 参数易于理解和调优
- 执行速度快，内存占用相对较低

实现特性：
- 根据系统资源自动优化算法参数
- 包含备用参数配置，提高鲁棒性
- 提供简化的回退保存策略
- 完整的错误处理和异常捕获
- 支持社区关系的持久化存储
"""

class SLLPADetector(GraphProjectionMixin, BaseCommunityDetector):
    """SLLPA算法社区检测实现
    
    实现了基于结构化标签传播算法的社区检测器，通过多继承方式集成了图投影功能和基础检测框架。
    SLLPA是一种高效的社区发现算法，特别适合大规模网络，通过迭代传播标签来识别紧密连接的节点组。
    
    关键特性：
    - 根据系统内存自动调整算法参数
    - 支持重叠社区检测（一个节点可以属于多个社区）
    - 提供备用参数配置，处理异常情况
    - 实现了详细的错误处理和回退机制
    - 包含简化的社区保存策略
    """
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行SLLPA算法检测社区
        
        实现步骤：
        1. 检查图投影是否已创建
        2. 应用SLLPA算法检测社区，保存结果到图投影中
        3. 收集算法执行结果和迭代次数
        4. 异常处理和回退机制
        
        返回：
            包含社区检测统计信息的字典
            
        异常：
            ValueError: 如果图投影未创建
            其他异常将触发回退策略
        """
        # 检查图投影是否已创建
        if not self.G:
            raise ValueError("请先创建图投影")
            
        print("开始执行SLLPA社区检测...")
        
        try:
            # 执行SLLPA算法进行社区检测
            # writeProperty指定将检测结果写入节点的communityIds属性
            # 其他参数通过_get_optimized_sllpa_params()获取，根据系统资源优化
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                **self._get_optimized_sllpa_params()
            )
            
            # 提取算法结果
            community_count = result.get('communityCount', 0)
            iterations = result.get('iterations', 0)
            
            print(f"SLLPA算法完成: {community_count} 个社区, "
                  f"{iterations} 次迭代")
            
            # 返回检测结果统计
            return {
                'communityCount': community_count,  # 检测到的社区数量
                'iterations': iterations            # 算法执行的迭代次数
            }
            
        except Exception as e:
            print(f"SLLPA算法执行失败: {e}")
            # 执行回退策略
            return self._execute_fallback_sllpa()
    
    def _execute_fallback_sllpa(self) -> Dict[str, Any]:
        """执行备用SLLPA算法
        
        当主要SLLPA算法参数配置失败时，使用更保守的参数配置进行回退执行。
        备用策略特点：
        - 减少最大迭代次数，加快收敛
        - 提高关联强度阈值，产生更少但更紧密的社区
        - 降低并发度，提高在资源受限环境下的稳定性
        
        返回：
            包含回退执行结果的字典
            
        异常：
            ValueError: 如果回退执行也失败
        """
        print("尝试使用备用参数...")
        
        try:
            # 使用更保守的参数配置
            result = self.gds.sllpa.write(
                self.G,
                writeProperty="communityIds",
                maxIterations=50,        # 减少迭代次数，加快收敛
                minAssociationStrength=0.2,  # 提高阈值，产生更少但更紧密的社区
                concurrency=1           # 单线程执行，提高稳定性
            )
            
            return {
                'communityCount': result.get('communityCount', 0),
                'iterations': result.get('iterations', 0),
                'note': '使用了备用参数'
            }
        except Exception as e:
            raise ValueError(f"SLLPA算法执行失败: {e}")
    
    def _get_optimized_sllpa_params(self) -> Dict[str, Any]:
        """获取优化的SLLPA参数
        
        根据系统内存大小自动调整SLLPA算法参数，实现资源感知的优化配置。
        内存越大，参数配置越激进，可能产生更精确的社区划分，但消耗更多资源。
        
        参数说明：
        - maxIterations: 最大迭代次数，影响算法精度和执行时间
        - minAssociationStrength: 最小关联强度阈值，决定节点可以属于的社区数量
          （值越低，节点可属于的社区越多）
        - concurrency: 并发度，控制算法执行的并行度
        
        返回：
            根据系统资源优化的参数字典
        """
        # 根据系统内存大小选择参数配置
        if self.memory_mb > 32 * 1024:  # >32GB 大内存系统
            return {
                'maxIterations': 100,         # 最多100次迭代，追求更高精度
                'minAssociationStrength': 0.05,  # 较低的阈值，允许多社区归属
                'concurrency': GDS_CONCURRENCY  # 最大并发度
            }
        elif self.memory_mb > 16 * 1024:  # >16GB 中等内存系统
            return {
                'maxIterations': 80,          # 最多80次迭代，平衡精度和性能
                'minAssociationStrength': 0.08,  # 稍高的阈值
                'concurrency': max(1, GDS_CONCURRENCY - 1)  # 略降低并发度
            }
        else:  # 小内存系统
            return {
                'maxIterations': 50,          # 最多50次迭代，加快收敛
                'minAssociationStrength': 0.1,   # 更高的阈值，减少社区重叠
                'concurrency': max(1, GDS_CONCURRENCY // 2)  # 大幅降低并发度
            }
    
    def save_communities(self) -> Dict[str, int]:
        """保存SLLPA算法结果
        
        将检测到的社区关系保存到图数据库中，包括：
        1. 创建社区节点约束
        2. 处理实体节点与多个社区的关联关系（重叠社区）
        3. 构建社区节点并建立实体到社区的关系
        
        社区数据结构：
        - 社区节点使用`__Community__`标签，id格式为"level-communityId"
        - 实体节点通过`:IN_COMMUNITY`关系连接到所属的多个社区
        - 社区节点包含level和algorithm属性
        
        返回：
            包含保存结果统计信息的字典
            
        异常：
            任何异常都会触发回退保存策略
        """
        print("开始保存SLLPA社区检测结果...")
        
        try:
            # 创建社区节点ID的唯一约束，提高查询效率和数据一致性
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )
            
            # 保存社区关系（处理重叠社区）
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL
            WITH count(e) AS entities_with_communities
            
            CALL {
                WITH entities_with_communities
                MATCH (e:`__Entity__`)
                WHERE e.communityIds IS NOT NULL
                WITH collect(e) AS entities
                CALL {
                    WITH entities
                    UNWIND entities AS e
                    UNWIND range(0, size(e.communityIds) - 1, 1) AS index
                    MERGE (c:`__Community__` {id: '0-'+toString(e.communityIds[index])})
                    ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
                    MERGE (e)-[:IN_COMMUNITY]->(c)
                }
                RETURN count(*) AS processed_count
            }
            
            RETURN CASE 
                WHEN entities_with_communities > 0 THEN entities_with_communities 
                ELSE 0 
            END AS total_count
            """)
            
            total_count = result[0]['total_count'] if result else 0
            print(f"已保存 {total_count} 个SLLPA社区关系")
            
            return {'saved_communities': total_count}
            
        except Exception as e:
            print(f"保存SLLPA社区结果失败: {e}")
            # 执行回退保存策略
            return self._save_communities_fallback()
    
    def _save_communities_fallback(self) -> Dict[str, int]:
        """备用社区保存方法
        
        当主保存方法失败时使用的简化保存策略。与主方法相比：
        - 仅保存节点的主要社区（第一个社区ID）
        - 忽略重叠社区的复杂处理
        - 使用更简单的查询逻辑，减少内存消耗
        
        返回：
            包含简化保存结果的字典
            
        异常：
            ValueError: 如果简化保存方法也失败
        """
        print("尝试使用简化方法保存社区...")
        
        try:
            # 仅保存节点的第一个社区ID，忽略重叠社区
            result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communityIds IS NOT NULL AND size(e.communityIds) > 0
            WITH e, e.communityIds[0] AS primary_community
            MERGE (c:`__Community__` {id: '0-' + toString(primary_community)})
            ON CREATE SET c.level = 0, c.algorithm = 'SLLPA'
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) as count
            """)
            
            count = result[0]['count'] if result else 0
            print(f"使用简化方法保存了 {count} 个社区关系")
            
            return {
                'saved_communities': count,
                'note': '使用了简化保存方法'
            }
        except Exception as e:
            raise ValueError(f"无法保存社区结果: {e}")