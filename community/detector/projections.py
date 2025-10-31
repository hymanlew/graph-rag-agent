from abc import ABC
from typing import Dict, Any, Tuple


class GraphProjectionMixin(ABC):
    """图投影功能的混入类
    
    提供图投影相关功能的混入类，被各种社区检测器继承使用。用于在社区检测算法执行前创建内存中的图投影。
    它将图数据库中的数据高效地加载到内存中，实现了多种投影策略，根据图的大小和系统资源自动选择合适的方式创建投影。
    包含多级回退机制，确保在各种情况下都能创建有效的图投影。
    
    混入类设计的优势：
    - 代码复用：避免在多个检测器中重复实现投影逻辑
    - 集中管理：统一管理和更新投影策略
    - 灵活组合：与不同的检测器类无缝集成
    - 易于扩展：可以方便地添加新的投影策略

    核心功能：
    - 创建标准图投影
    - 提供节点数量检查和限制机制
    - 实现多级投影策略，包括过滤投影和保守投影
    - 支持内存优化和异常处理
    - 确保投影操作的健壮性和资源安全性

    设计原则：
    - 资源感知：根据系统资源限制自动调整投影策略
    - 多级回退：提供多层次的投影策略，从标准到最小化
    - 数据优化：优先处理重要节点，确保社区检测质量
    - 错误处理：全面的异常捕获和回退机制
    """

    # 使用的是混入机制，是在继承类（LeidenDetector）中使用的，并且会根据 MRO 顺序自动获取真正的 gds 对象
    # 一旦在某个类中找到属性，查找就会停止，不会继续向后查找！
    # 只有类型注解，不赋值，不会阻止MRO查找
    # 如果赋值时，会阻止MRO查找，导致找不到属性
    gds: GraphDataScience

    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影
        
        创建内存中的图投影，用于执行社区检测算法。实现了多级投影策略，
        根据节点数量和系统资源自动选择合适的方式。
        
        投影策略优先级：
        1. 标准投影：适用于中小型图
        2. 过滤投影：当节点数量超过限制时，选择关系最丰富的节点
        3. 保守投影：使用最小配置的投影
        4. 最小化投影：仅包含最重要的节点
        
        返回：
            投影图对象和结果信息的元组
            
        可参照比对 @SimilarEntityDetector - create_entity_projection
        """
        print("开始创建社区检测的图投影...")
        
        # 检查节点数量
        node_count = self._get_node_count()
        # 如果节点数量超过限制，使用过滤投影
        if node_count > self.node_count_limit:
            print(f"警告: 节点数量({node_count})超过限制({self.node_count_limit})")
            return self._create_filtered_projection(node_count)
        
        # 删除已存在的投影
        try:
            self.gds.graph.drop(self.projection_name, failIfMissing=False)
        except Exception as e:
            print(f"删除旧投影时出错 (可忽略): {e}")
        
        # 创建标准投影
        try:
            # 投影配置说明：
            # - 节点标签：__Entity__
            # - 关系：所有类型的关系
            # - 关系方向：无向（UNDIRECTED）
            # - 关系权重：通过计数聚合计算
            self.G, result = self.gds.graph.project(
                self.projection_name,
                "__Entity__",
                {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                    }
                },
            )
            print(f"图投影创建成功: {result.get('nodeCount', 0)} 节点, "
                  f"{result.get('relationshipCount', 0)} 关系")
            return self.G, result
        except Exception as e:
            print(f"标准投影创建失败: {e}")
            # 标准投影失败，尝试保守投影
            return self._create_conservative_projection()
    
    def _get_node_count(self) -> int:
        """获取节点数量
        
        查询图数据库中__Entity__标签的节点总数，用于决定使用哪种投影策略。
        
        返回：
            节点数量
        """
        result = self.graph.query(
            "MATCH (e:__Entity__) RETURN count(e) AS count"
        )
        return result[0]["count"] if result else 0
    
    def _create_filtered_projection(self, total_node_count: int) -> Tuple[Any, Dict]:
        """创建过滤后的投影
        
        当图的节点数量超过限制时，创建一个过滤投影，仅包含关系最丰富的节点。
        这种策略在处理大规模图时可以在有限资源下识别最重要的社区。
        
        参数：
            total_node_count: 原始图的节点总数
            
        返回：
            过滤后的投影图对象和结果信息的元组
            
        实现步骤：
        1. 根据关系数量识别重要节点
        2. 创建仅包含这些重要节点的投影
        3. 如果失败，回退到保守投影
        """
        print("创建过滤后的投影...")
        
        try:
            # 获取重要节点（基于关系数量）
            # 查询关系数量最多的节点，按关系数降序排列，限制数量为节点限制
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT toInteger($limit)
                RETURN collect(id(e)) AS important_nodes
                """,
                params={"limit": self.node_count_limit}
            )
            
            # 如果没有找到重要节点，使用保守投影
            if not result or not result[0]["important_nodes"]:
                return self._create_conservative_projection()
            
            important_nodes = result[0]["important_nodes"]
            
            # 创建过滤投影
            config = {
                "nodeProjection": {
                    "__Entity__": {
                        "properties": ["*"],
                        "filter": f"id(node) IN {important_nodes}"  # 仅包含重要节点
                    }
                },
                "relationshipProjection": {
                    "_ALL_": {
                        "type": "*",
                        "orientation": "UNDIRECTED",
                        "properties": {"weight": {"property": "*", "aggregation": "COUNT"}}
                    }
                }
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"过滤投影创建成功: {result.get('nodeCount', 0)} 节点, "
                  f"{result.get('relationshipCount', 0)} 关系")
            return self.G, result
            
        except Exception as e:
            print(f"过滤投影创建失败: {e}")
            # 过滤投影失败，尝试保守投影
            return self._create_conservative_projection()
    
    def _create_conservative_projection(self) -> Tuple[Any, Dict]:
        """创建保守配置的投影
        
        当标准投影和过滤投影都失败时，使用更简单的配置创建保守投影。
        这种投影配置最少，减少了内存消耗，增加了成功创建的可能性。
        
        返回：
            保守投影的图对象和结果信息的元组
            
        实现特点：
        - 仅指定必要的节点标签和关系类型
        - 不包含复杂的属性配置
        - 如果失败，回退到最小化投影
        """
        print("尝试使用保守配置创建投影...")
        
        try:
            # 使用最小配置
            config = {
                "nodeProjection": "__Entity__",  # 仅指定节点标签
                "relationshipProjection": "*"  # 所有关系
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                config
            )
            print(f"保守投影创建成功: {result.get('nodeCount', 0)} 节点")
            return self.G, result
            
        except Exception as e:
            print(f"保守投影创建失败: {e}")
            # 保守投影失败，尝试最小化投影
            return self._create_minimal_projection()
    
    def _create_minimal_projection(self) -> Tuple[Any, Dict]:
        """创建最小化投影
        
        当其他投影策略都失败时的最后尝试，创建仅包含最关键节点的最小投影。
        这种投影配置仅保留图中关系最丰富的前1000个节点，牺牲完整性以确保可用性。
        
        返回：
            最小化投影的图对象和结果信息的元组
            
        异常：
            ValueError: 如果无法创建任何投影
            
        实现步骤：
        1. 获取关系数量最多的前1000个节点
        2. 创建仅包含这些节点的最小投影
        3. 如果仍然失败，抛出异常
        """
        print("尝试创建最小化投影...")
        
        try:
            # 获取最重要的节点（关系数量最多的1000个节点）
            result = self.graph.query(
                """
                MATCH (e:__Entity__)-[r]-()
                WITH e, count(r) AS rel_count
                ORDER BY rel_count DESC
                LIMIT 1000
                RETURN collect(id(e)) AS critical_nodes
                """
            )
            
            # 如果无法获取关键节点，抛出异常
            if not result or not result[0]["critical_nodes"]:
                raise ValueError("无法获取关键节点")
            
            critical_nodes = result[0]["critical_nodes"]
            
            # 创建最小化投影
            minimal_config = {
                "nodeProjection": {
                    "__Entity__": {
                        "filter": f"id(node) IN {critical_nodes}"  # 仅包含关键节点
                    }
                },
                "relationshipProjection": "*"  # 所有关系
            }
            
            self.G, result = self.gds.graph.project(
                self.projection_name,
                minimal_config
            )
            print(f"最小化投影创建成功: {result.get('nodeCount', 0)} 节点")
            return self.G, result
            
        except Exception as e:
            print(f"所有投影方法均失败: {e}")
            # 所有投影策略都失败，抛出异常
            raise ValueError("无法创建必要的图投影")