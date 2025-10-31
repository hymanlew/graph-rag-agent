from typing import Dict, Any
from .base import BaseCommunityDetector
from .projections import GraphProjectionMixin

from config.settings import GDS_CONCURRENCY

"""
Leiden 社区检测算法实现，用于发现图中的社区结构（即密集连接的子图）。Leiden算法是Louvain算法的改进版本，
通过更严格的社区质量优化和更好的社区分隔处理，提供更精确的社区检测结果。

它区别于 WCC 算法的社区检测，WCC 是指在忽略关系方向的情况下，节点之间可以互相到达的最大子图。
- WCC 基于连通性，结果是将图分割成连通的子图，每个子较内部是相似关系连通的，但内部只是连通而非密集（即可能只是链式连接，而没有形成簇）。
- Leiden 基于模块度优化，结果是密集连接的社区，社区内部节点连接密集（具有较高的相似性，且权重高），而社区之间连接稀疏（相似性较低）。
- 在WCC中只关心节点之间是否存在"SIMILAR"相似关系，而不关心关系的强度（除非您通过权重考虑，但WCC一般不权重）。
- 而Leiden算法则使用了权重（weight）来考虑关系的强度。

在实际应用中，可先使用WCC来获取连通分量（特点：算法简单快速，初步分割分连通子较，适合大规模图），然后对每个连通分量
使用Leiden进行更细致的社区划分（特点：计算更复杂，但能发现更准确的社区结构），或者直接在整个图上使用Leiden。这取决于业务需求。

应用场景：
Leiden常用于复杂网络中的社区发现，例如社交网络中的朋友圈、生物网络中的功能模块等。
WCC通常用于网络分析中的基础连通性分析，例如在社会网络中找出互相关联的群体，在推荐系统中找出连通的物品集等。

实现特性：
- 根据系统资源自动优化算法参数
- 包含备用参数配置，提高鲁棒性
- 支持多层次社区结构存储
- 实现了详细的错误处理和回退机制
- 集成了图投影功能，提高执行效率
"""

class LeidenDetector(GraphProjectionMixin, BaseCommunityDetector):
    """Leiden算法社区检测实现
    
    实现了基于Leiden算法的社区检测，通过多继承方式集成了图投影功能和基础检测框架。
    这个实现针对不同的系统资源配置自动优化算法参数，并包含回退机制以提高鲁棒性。
    
    关键特性：
    - 自动检测图的连通分量
    - 根据系统内存自动调整算法参数
    - 支持多层次社区结构检测和存储
    - 提供备用参数配置，处理异常情况
    - 将社区关系保存到图数据库中
    """
    
    def detect_communities(self) -> Dict[str, Any]:
        """执行Leiden算法社区检测
        
        实现步骤：
        1. 检查图投影是否已创建
        2. 使用弱连通分量算法分析图的连通性
        3. 应用Leiden算法检测社区，保存结果到图投影中
        4. 收集算法执行结果和性能指标
        5. 异常处理和回退机制
        
        返回：
            包含社区检测统计信息的字典
            
        异常：
            ValueError: 如果图投影未创建
            其他异常将触发回退策略
        """
        # 检查图投影是否已创建
        if not self.G:
            raise ValueError("请先创建图投影")
            
        print("开始执行Leiden社区检测...")
        
        try:
            # 检查连通分量，了解图的结构
            wcc = self.gds.wcc.stats(self.G)
            print(f"图包含 {wcc.get('componentCount', 0)} 个连通分量")
            
            # 执行Leiden算法进行社区检测
            # 参数说明：
            # - writeProperty: 将结果写入图节点的哪个属性
            # - includeIntermediateCommunities: 是否包含中间层级的社区结构
            # - relationshipWeightProperty: 边权重属性
            # - 其他参数通过_get_optimized_leiden_params()获取
            result = self.gds.leiden.write(
                self.G,
                writeProperty="communities",
                includeIntermediateCommunities=True,
                relationshipWeightProperty="weight",
                **self._get_optimized_leiden_params()
            )
            
            # 返回检测结果统计信息
            return {
                'componentCount': wcc.get('componentCount', 0),      # 连通分量数量
                'componentDistribution': wcc.get('componentDistribution', {}),  # 连通分量分布
                'communityCount': result.get('communityCount', 0),   # 检测到的社区数量
                'modularity': result.get('modularity', 0),           # 模块化度（社区质量指标）
                'ranLevels': result.get('ranLevels', 0)              # 执行的层次数
            }
            
        except Exception as e:
            print(f"Leiden算法执行失败: {e}")
            # 执行回退策略
            return self._execute_fallback_leiden()
    
    def _execute_fallback_leiden(self) -> Dict[str, Any]:
        """执行备用Leiden算法
        
        当主要Leiden算法参数配置失败时，使用更保守的参数配置进行回退执行。
        备用策略特点：
        - 不包含中间层级社区，减少内存消耗
        - 使用较低的gamma值，倾向于检测较大的社区
        - 降低精度要求，提高稳定性
        - 限制最大层次数
        - 降低并发度，提高在资源受限环境下的稳定性
        
        返回：
            包含回退执行结果的字典
            
        异常：
            ValueError: 如果回退执行也失败
        """
        print("尝试使用备用参数...")
        
        try:
            # 使用更保守的参数配置
            result = self.gds.leiden.write(
                self.G,
                writeProperty="communities",
                includeIntermediateCommunities=False,  # 不保存中间层级，减少内存使用
                gamma=0.5,         # 降低模块度解析参数，得到更少更大的社区
                tolerance=0.001,   # 较低的精度要求，提高执行速度
                maxLevels=2,       # 限制最大层次数，减少计算复杂度
                concurrency=1      # 单线程执行，提高稳定性
            )
            
            return {
                'communityCount': result.get('communityCount', 0),
                'modularity': result.get('modularity', 0),
                'ranLevels': result.get('ranLevels', 0),
                'note': '使用了备用参数'
            }
        except Exception as e:
            raise ValueError(f"Leiden算法执行失败: {e}")
    
    def _get_optimized_leiden_params(self) -> Dict[str, Any]:
        """获取优化的Leiden算法参数
        
        根据系统内存大小自动调整Leiden算法参数，实现资源感知的优化配置。
        内存越大，参数配置越激进，可能产生更高质量的社区，但消耗更多资源。
        
        参数说明：
        - gamma: 模块化度解析参数，值越高，检测的社区数量越多
        - tolerance: 精度要求，值越低，结果越精确但计算越慢
        - maxLevels: 最大层次数，决定层次化社区检测的深度
        - concurrency: 并发度，控制算法执行的并行度
        
        返回：
            根据系统资源优化的参数字典
        """
        # 根据系统内存大小选择参数配置
        if self.memory_mb > 32 * 1024:  # >32GB 大内存系统
            return {
                'gamma': 1.0,        # 标准模块化度解析
                'tolerance': 0.0001, # 高精度要求
                'maxLevels': 10,     # 最多10层层次结构
                'concurrency': GDS_CONCURRENCY  # 最大并发度
            }
        elif self.memory_mb > 16 * 1024:  # >16GB 中等内存系统
            return {
                'gamma': 1.0,        # 标准模块化度解析
                'tolerance': 0.0005, # 适中精度
                'maxLevels': 5,      # 最多5层层次结构
                'concurrency': max(1, GDS_CONCURRENCY - 1)  # 略降低并发度
            }
        else:  # 小内存系统
            return {
                'gamma': 0.8,        # 较低的模块化度解析，产生更大的社区
                'tolerance': 0.001,  # 更低的精度要求，提高执行速度
                'maxLevels': 3,      # 最多3层层次结构
                'concurrency': max(1, GDS_CONCURRENCY // 2)  # 大幅降低并发度
            }
    
    def save_communities(self) -> Dict[str, int]:
        """保存Leiden算法的社区检测结果
        
        将检测到的社区关系保存到图数据库中，包括：
        1. 创建社区节点约束
        2. 保存基础社区关系（第一层社区）
        3. 保存更高层级的社区关系，构建层次化结构
        
        社区数据结构：
        - 社区节点使用`__Community__`标签，id格式为"level-communityId"
        - 实体节点通过`:IN_COMMUNITY`关系连接到所属社区
        - 社区节点之间也通过`:IN_COMMUNITY`关系构建层次结构
        
        返回：
            包含保存结果统计信息的字典
            
        异常：
            任何异常都会触发回退保存策略
        """
        print("开始保存Leiden社区检测结果...")
        
        try:
            # 创建社区节点ID的唯一约束，提高查询效率和数据一致性
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
            )
            
            # 保存基础社区关系（第一层社区）
            base_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 0
            WITH collect({entityId: id(e), community: e.communities[0]}) AS data
            UNWIND data AS item
            MERGE (c:`__Community__` {id: '0-' + toString(item.community)})
            ON CREATE SET c.level = 0
            WITH item, c
            MATCH (e) WHERE id(e) = item.entityId
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) AS base_count
            """)
            
            base_count = base_result[0]['base_count'] if base_result else 0
            
            # 保存更高层级社区关系，构建层次化社区结构
            higher_result = self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 1
            WITH e, e.communities AS communities
            UNWIND range(1, size(communities) - 1) AS index
            WITH e, index, communities[index] AS current_community, 
                 communities[index-1] AS previous_community
            
            MERGE (current:`__Community__` {id: toString(index) + '-' + 
                                              toString(current_community)})
            ON CREATE SET current.level = index
            
            WITH e, current, previous_community, index
            MATCH (previous:`__Community__` {id: toString(index - 1) + '-' + 
                                              toString(previous_community)})
            MERGE (previous)-[:IN_COMMUNITY]->(current)
            
            RETURN count(*) AS higher_count
            """)
            
            higher_count = higher_result[0]['higher_count'] if higher_result else 0
            
            # 返回保存的社区关系总数
            return {'saved_communities': base_count + higher_count}
            
        except Exception as e:
            print(f"社区保存失败: {e}")
            # 执行回退保存策略
            return self._save_communities_fallback()