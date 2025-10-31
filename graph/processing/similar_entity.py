import os
import time
# 导入Neo4j图数据科学库
from graphdatascience import GraphDataScience
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass
from config.settings import similarity_threshold, BATCH_SIZE, GDS_MEMORY_LIMIT
from graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

"""
GraphDataScience 库允许使用纯 Python 代码来操作图数据、运行图算法以及构建机器学习管道，而无需编写复杂的 Cypher 查询。

功能模块	    核心用途	                                关键技术点
图投影与管理 	将数据库中的图数据加载到内存中进行高性能计算。	使用优化的内存格式，提升计算效率。
图算法执行 	在图数据上运行各类分析算法，以发现洞见和模式。	提供超过60种算法，涵盖社区检测、中心性分析、路径发现、节点嵌入和链接预测等。
机器学习管道 	构建端到端的图机器学习工作流。	            支持节点分类、链接预测等任务的管道，可进行特征工程、模型训练和预测。
结果处理 	灵活保存和分析算法计算结果。	                可将结果写回数据库、以CSV格式导出或流式传输到其他应用。

金融反欺诈：通过分析资金流转图，利用链接预测等算法识别潜在的欺诈团伙和异常交易模式。
智能推荐系统：在用户-商品二部图上运行Personalized PageRank等算法，为用户发现可能感兴趣的商品或内容。
社交网络分析：识别社交网络中的关键影响者（中心性分析）或发现潜在的社区结构（社区检测）。
知识图谱丰富：利用图算法从已有的知识图谱中发掘隐藏的关系和潜在的新知识。
"""

# dataclass 会自动生成特殊方法（如 __init__、__repr__、__eq__ 等），它只是简化了类的初始化
# property 才是自动实现 getter 和 setter，@dataclass 不会生成这些，且 Python 中通常不鼓励使用 getter/setter
# 直接操作对应属性即可，不需要使用 @property
@dataclass
class GDSConfig:
    """
    Neo4j GDS(图数据科学库)配置参数

    配置参数说明：
    - uri/username/password: Neo4j数据库连接信息
    - similarity_threshold: 相似度阈值，影响KNN算法的结果精确度
    - word_edit_distance: 单词编辑距离，用于文本相似度比较
    - batch_size: 批处理大小，优化内存使用
    - memory_limit: 内存限制，防止GDS操作占用过多资源
    """
    uri: str = os.environ["NEO4J_URI"]
    username: str = os.environ["NEO4J_USERNAME"]
    password: str = os.environ["NEO4J_PASSWORD"]
    similarity_threshold: float = similarity_threshold
    word_edit_distance: int = 3
    batch_size: int = 500
    memory_limit: int = 6  # 单位：GB

    # __post_init__ 是 @dataclass 提供的一个钩子方法。当自动生成 __init__ 方法后，会被自动调用。
    # 这个方法的主要用途是进行一些初始化后的处理，比如验证数据、计算派生属性等。
    def __post_init__(self):

        # globals() 是 Python内 置函数，不需要导入任何包。返回当前模块中定义的所有全局变量（包括函数、类、变量等）字典
        # 只返回当前模块的全局变量，不推荐使用
        if 'BATCH_SIZE' in globals() and BATCH_SIZE:
            self.batch_size = BATCH_SIZE
            
        if 'GDS_MEMORY_LIMIT' in globals() and GDS_MEMORY_LIMIT:
            self.memory_limit = GDS_MEMORY_LIMIT

"""
在 Neo4j 中，弱连通分量（WCC）是图论中的概念，指在忽略关系方向的情况下，节点之间可以互相到达的最大子图。
即忽略关系方向的情况下，整个图是连通的，那么该图就是弱连通的。
在无向图中，连通分量就是弱连通分量。

WCC 算法用于找到图中的连通组件，每个组件内的节点都是连通的（通过任意边），而不同组件之间没有连接。
此类代码 WCC 是基于"SIMILAR"关系（边）来构建连通分量的。这里"SIMILAR"关系可以理解为无向的，或者即使是有向的，在WCC中也会忽略方向。

应用场景：WCC通常用于网络分析中的基础连通性分析，例如在社会网络中找出互相关联的群体，在推荐系统中找出连通的物品集等。
"""
class SimilarEntityDetector:
    """
    相似实体检测器，使用Neo4j GDS库实现实体相似性分析和社区识别。
    
    核心功能模块：
    1. 实体投影管理：创建内存中的实体关系子图
    2. 相似度计算：使用 KNN 算法识别相似实体并创建关系
    3. 社区检测：通过 WCC 弱连通分量算法将相似实体分组到社区
    4. 重复实体识别：基于社区和文本相似度找出潜在重复
    
    技术特点：
    - 基于图算法的实体相似度分析，相比传统方法更精确
    - 多级错误处理和降级策略确保稳定性
    - 完整的性能监控和统计报告
    - 高效的内存管理和资源控制
    """
    
    def __init__(self, config: GDSConfig = None):
        """
        初始化相似实体检测器
        
        Args:
            config: GDS配置参数，包含连接信息和算法阈值
        """
        # 使用提供的配置或默认配置
        self.config = config or GDSConfig()
        
        # 初始化GDS客户端
        self.gds = GraphDataScience(
            self.config.uri,
            auth=(self.config.username, self.config.password)
        )
        
        # 获取Neo4j数据库连接
        self.graph = connection_manager.get_connection()
        
        # 投影图名称和引用
        self.projection_name = "entities"
        self.G = None
        
        # 性能监控指标
        self.projection_time = 0   # 投影创建时间
        self.knn_time = 0          # KNN算法执行时间
        self.wcc_time = 0          # WCC算法执行时间
        self.query_time = 0        # 重复实体查询时间
        
        # 创建索引来优化重复实体检测
        self._create_indexes()
    
    def _create_indexes(self):
        """
        创建必要的索引以优化查询性能

        langchain-neo4j graph.add_graph_documents 方法，设置 baseEntityLabel=true，它会将所有的实体节点都添加一个 __Entity__ 标签，
        并会为这些节点计算一个弱连通分量（WCC）社区属性 e.wcc。
        - 创建 WCC 索引：自动执行 CREATE INDEX IF NOT EXISTS FOR (e:\Entity`) ON (e.wcc)`
        - 计算 WCC 社区：使用 Neo4j 的 GDS 图算法计算弱连通分量（WCC）
        - 设置实体属性：为每个实体节点设置 wcc 属性及社区 ID
        通过为每个节点分配一个 WCC 标识，可以将节点分组到不同的社区中。

        索引策略说明：
        1. 在实体ID上创建索引，加速实体查找和匹配
        2. 在社区属性(wcc)上创建索引，提高社区相关查询效率
        这两个索引对于大规模图数据库中的重复实体检测至关重要
        """
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.wcc)"
        ]
        
        connection_manager.create_multiple_indexes(index_queries)
    
    @timer
    def create_entity_projection(self) -> Tuple[Any, Dict[str, Any]]:
        """
        创建实体的内存投影子图
        
        功能说明：
        在Neo4j GDS中创建内存中的子图投影，这是后续图算法执行的基础。
        投影过程会将实体节点和它们的嵌入向量加载到内存中，提高算法执行效率。
        
        实现细节：
        1. 先清理可能存在的旧投影
        2. 验证实体数量，确保有足够数据
        3. 尝试创建投影，包含两级降级策略
        4. 记录投影创建时间
        
        Returns:
            Tuple[Any, Dict[str, Any]]: 投影图对象和结果信息
        """
        start_time = time.time()
        
        # 如果已存在，先清除旧的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"清除旧投影时出错 (可忽略): {e}")
        
        # 获取实体总数
        entity_count = self._get_entity_count()
        if entity_count == 0:
            print("没有找到有效的实体节点，请确保数据已经正确导入")
            return None, {"status": "error", "message": "No entities found"}
        
        # 创建新的投影图 - 主要方法
        try:
            self.G, result = self.gds.graph.project(
                self.projection_name,          # 图名称
                "__Entity__",                  # 节点投影
                "*",                           # 关系投影（所有类型）
                nodeProperties=["embedding"]    # 配置参数
            )
        except Exception as e:
            print(f"创建投影时出错: {e}")
            # 降级策略1：使用保守配置重试
            try:
                print("尝试使用保守配置重新创建投影...")
                config = {
                    "nodeProjection": {"__Entity__": {"properties": ["embedding"]}},
                    "relationshipProjection": {"*": {"orientation": "UNDIRECTED"}},
                    "nodeProperties": ["embedding"]
                }
                self.G, result = self.gds.graph.project(
                    self.projection_name,
                    config
                )
            except Exception as e2:
                print(f"二次尝试仍然失败: {e2}")
                return None, {"status": "error", "message": str(e2)}
        
        # 记录并报告投影时间
        self.projection_time = time.time() - start_time
        
        if self.G:
            print(f"投影创建成功，耗时: {self.projection_time:.2f}秒")
            return self.G, result
        else:
            print("投影创建失败")
            return None, {"status": "error", "message": "Failed to create projection"}
    
    def _get_entity_count(self) -> int:
        """
        获取实体总数
        
        功能说明：
        统计数据库中具有嵌入向量的实体节点数量，用于验证数据完整性。
        只计算具有embedding属性的实体，确保后续算法有必要的特征向量。
        
        Returns:
            int: 实体数量
        """
        result = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL
            RETURN count(e) AS count
            """
        )
        return result[0]["count"] if result else 0
    
    @timer
    def detect_similar_entities(self) -> Dict[str, Any]:
        """
        使用KNN算法检测相似实体并创建SIMILAR关系
        
        算法原理：
        KNN(K-Nearest Neighbors)算法查找每个实体的最相似邻居，基于嵌入向量的相似度算法
        为每个实体创建到其相似实体的SIMILAR关系，并存储相似度得分。
        
        实现特点：
        1. 双重操作：先使用mutate在内存中计算，再使用write写入数据库
        2. 相似度阈值控制：通过配置的阈值过滤低相似度对
        3. 多级降级策略：主算法失败时自动尝试更保守的参数
        4. 详细的性能统计和结果报告
        
        Returns:
            Dict[str, Any]: 算法结果统计，包含创建的关系数和执行时间
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
        
        start_time = time.time()
        print("开始检测相似实体...")
        
        try:
            # 使用KNN算法找出相似实体 - 先在内存中计算
            mutate_result = self.gds.knn.mutate(
                self.G,
                nodeProperties=['embedding'],
                mutateRelationshipType='SIMILAR',
                mutateProperty='score',
                similarityCutoff=self.config.similarity_threshold,
                topK=10
            )
            
            # 将KNN结果写入数据库
            write_result = self.gds.knn.write(
                self.G,
                nodeProperties=['embedding'],
                writeRelationshipType='SIMILAR',
                writeProperty='score',
                similarityCutoff=self.config.similarity_threshold,
                topK=10
            )
            
            # 记录并报告执行时间
            self.knn_time = time.time() - start_time
            print(f"KNN完成，写入 {write_result['relationshipsWritten']} 个关系, 用时: {self.knn_time:.2f}秒")
            
            return {
                "status": "success",
                "relationshipsWritten": write_result['relationshipsWritten'],
                "knnTime": self.knn_time
            }
            
        except Exception as e:
            print(f"KNN算法执行失败: {e}")
            # 降级策略：使用备用参数重试
            try:
                print("尝试使用备用参数重新执行KNN...")
                fallback_params = {
                    "nodeProperties": ["embedding"],
                    "writeRelationshipType": "SIMILAR",
                    "writeProperty": "score",
                    "similarityCutoff": self.config.similarity_threshold,
                    "topK": 5,  # 降低topK值减少计算量
                    "sampleRate": 0.5  # 降低采样率减少内存消耗
                }
                
                fallback_result = self.gds.knn.write(self.G, **fallback_params)
                self.knn_time = time.time() - start_time
                
                print(f"备用KNN执行完成，写入 {fallback_result['relationshipsWritten']} 个关系, 用时: {self.knn_time:.2f}秒")
                
                return {
                    "status": "success",
                    "relationshipsWritten": fallback_result['relationshipsWritten'],
                    "knnTime": self.knn_time,
                    "note": "使用了备用参数"
                }
                
            except Exception as e2:
                print(f"备用KNN也失败了: {e2}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
    @timer
    def detect_communities(self) -> Dict[str, Any]:
        """
        使用WCC算法检测社区并将结果写入节点的wcc属性
        
        算法原理：
        WCC(Weakly Connected Components)弱连通分量算法将相似实体分组到不同社区。
        具有相似关系的实体会被分配相同的社区ID，便于后续的重复检测。
        
        实现特点：
        1. 基于SIMILAR（相似）关系网络构建社区
        2. 使用consecutiveIds优化存储和查询
        3. 实现降级机制处理可能的算法失败
        4. 返回详细的社区统计信息
        
        Returns:
            Dict[str, Any]: 社区检测结果统计，包含社区数量和执行时间
        """
        if not self.G:
            raise ValueError("请先创建实体投影")
        
        start_time = time.time()
        print("开始检测社区...")
        
        try:
            # 使用WCC算法检测社区
            result = self.gds.wcc.write(
                self.G,
                writeProperty="wcc",
                relationshipTypes=["SIMILAR"],
                consecutiveIds=True  # 使用连续ID优化存储
            )
            
            # 记录执行时间
            self.wcc_time = time.time() - start_time
            
            # 提取社区统计信息
            community_count = result.get("communityCount", 0)
            print(f"社区检测完成，找到 {community_count} 个社区, 用时: {self.wcc_time:.2f}秒")
            
            return {
                "status": "success",
                "communityCount": community_count,
                "wccTime": self.wcc_time
            }
        
        except Exception as e:
            print(f"WCC算法执行失败: {e}")
            # 降级策略：使用简化参数重试
            try:
                print("尝试使用备用参数重新执行WCC...")
                fallback_result = self.gds.wcc.write(
                    self.G,
                    writeProperty="wcc",
                    relationshipTypes=["SIMILAR"]
                )
                
                self.wcc_time = time.time() - start_time
                community_count = fallback_result.get("communityCount", 0)
                
                print(f"备用WCC执行完成，找到 {community_count} 个社区, 用时: {self.wcc_time:.2f}秒")
                
                return {
                    "status": "success",
                    "communityCount": community_count,
                    "wccTime": self.wcc_time,
                    "note": "使用了备用参数"
                }
                
            except Exception as e2:
                print(f"备用WCC也失败了: {e2}")
                return {
                    "status": "error",
                    "message": str(e)
                }
        
    @timer
    def find_potential_duplicates(self) -> List[Any]:
        """
        查找潜在的重复实体
        
        算法设计：
        1. 首先识别包含多个实体的社区
        2. 在这些社区内，使用文本距离算法进一步筛选相似实体
        3. 合并有重叠元素的实体组
        4. 移除完全包含在其他组中的子组
        
        核心优化：
        - 使用APOC库的text.distance函数高效计算编辑距离
        - 使用apoc.coll.union/intersection优化集合操作
        - 通过distinct和过滤条件减少重复计算
        - 使用排序确保结果一致性
        
        Returns:
            List[Any]: 潜在重复实体的候选列表，每个元素是一组可能重复的实体ID
        """
        query_start = time.time()
        
        # 查找包含多个实体的社区 - 预筛选
        community_counts = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL AND size(e.id) > 1
            WITH e.wcc AS community, count(*) AS count
            WHERE count > 1
            RETURN community, count
            ORDER BY count DESC
            """
        )
        
        if not community_counts:
            print("没有找到可能包含重复实体的社区")
            return []
        
        # 为有效社区查找潜在重复 - 核心查询
        results = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 1  // 长度大于1个字符
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // 添加文本距离计算 - 编辑距离小于阈值的实体被认为相似
            WITH distinct
                [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] 
                AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // 如果组之间有共同元素，则合并组 - 使用累积聚合函数
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, 
                index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // 额外过滤 - 移除被其他组完全包含的子组
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, 
                combinedResultIndex, 
                allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={'distance': self.config.word_edit_distance}
        )
        
        # 记录查询时间
        self.query_time = time.time() - query_start
        
        # 处理和转换查询结果
        processed_results = []
        for record in results:
            if "combinedResult" in record and isinstance(record["combinedResult"], list):
                processed_results.append(record["combinedResult"])
        
        print(f"潜在重复实体查找完成，找到 {len(processed_results)} 组候选实体, 用时: {self.query_time:.2f}秒")
        
        return processed_results
    
    def cleanup(self) -> None:
        """
        清理内存中的投影图
        
        资源管理：
        释放GDS创建的内存投影，避免长时间占用内存资源。
        实现了异常处理，确保即使发生错误也能将G置为None。
        """
        if self.G:
            try:
                self.G.drop()
                print("投影图清理完成")
            except Exception as e:
                print(f"清理投影图时出错: {str(e)}")
            finally:
                self.G = None

    @timer
    def process_entities(self) -> Tuple[List[Any], Dict[str, Any]]:
        """
        执行完整的实体处理流程：社区检测、中心性分析、路径发现
        
        工作流程：
        1. 创建实体投影：将实体和关系加载到内存
        2. 检测相似实体：使用KNN算法识别相似关系
        3. 检测社区：使用WCC算法将相似实体分组
        4. 查找潜在重复：识别可能重复的实体组
        5. 资源清理：释放内存中的投影图
        6. 性能统计：生成详细的处理时间和结果统计

        Returns:
            Tuple[List[Any], Dict[str, Any]]: 
                - 潜在重复实体的列表
                - 包含详细性能统计和处理结果的字典
        """
        start_time = time.time()
        duplicates = []
        
        try:
            # 步骤1：创建实体投影
            self.G, projection_result = self.create_entity_projection()
            
            if not self.G:
                print("实体投影创建失败，无法继续处理")
                return [], {"status": "error", "message": "投影创建失败"}
                
            # 步骤2：检测相似实体
            knn_result = self.detect_similar_entities()
            
            if knn_result.get('status') == 'error':
                print(f"相似实体检测失败: {knn_result.get('message')}")
                return [], {"status": "error", "message": "相似实体检测失败"}
                
            # 步骤3：检测社区
            wcc_result = self.detect_communities()
            
            if wcc_result.get('status') == 'error':
                print(f"社区检测失败: {wcc_result.get('message')}")
                return [], {"status": "error", "message": "社区检测失败"}
                
            # 步骤4：查找潜在重复
            duplicates = self.find_potential_duplicates()
            
            # 计算总处理时间
            total_time = time.time() - start_time
            
            # 准备性能统计
            time_records = {
                "投影时间": self.projection_time,
                "KNN时间": self.knn_time,
                "WCC时间": self.wcc_time,
                "查询时间": self.query_time
            }
            
            # 获取并扩展性能统计
            stats = get_performance_stats(total_time, time_records)
            stats.update({
                "status": "success",
                "候选实体组数": len(duplicates),
                "关系数量": knn_result.get('relationshipsWritten', 0),
                "社区数量": wcc_result.get('communityCount', 0)
            })
            
            # 打印性能统计报告
            print_performance_stats(stats)
            
            return duplicates, stats
            
        except Exception as e:
            print(f"实体处理过程中发生错误: {e}")
            return [], {"status": "error", "message": str(e)}
            
        finally:
            # 确保清理投影图 - 资源释放保障
            self.cleanup()