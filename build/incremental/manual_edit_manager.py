"""
手动编辑管理模块

此模块负责在Graph-RAG系统中管理和保留用户对知识图谱的手动编辑。
在增量更新过程中，确保系统自动更新不会覆盖用户的重要手动修改，
同时提供冲突检测和解决机制，保障知识图谱的完整性和用户编辑的持久性。

主要功能点：
- 手动编辑的识别和追踪
- 编辑状态的属性管理和初始化
- 变更文件相关编辑的保护
- 自动更新与手动编辑冲突的智能解决
- 编辑操作的性能监控和统计
"""

import time
from datetime import datetime
from typing import Dict, List, Any

# Rich库用于终端美化输出
from rich.console import Console
from rich.table import Table
# 数据库连接管理
from config.neo4jdb import get_db_manager
# 配置参数，包括冲突解决策略、并行处理参数等
from config.settings import conflict_strategy, MAX_WORKERS, BATCH_SIZE

class ManualEditManager:
    """
    手动编辑同步管理器，负责识别、保留和处理Neo4j数据库中的手动编辑。
    
    该类是Graph-RAG系统中的关键组件，解决了自动化知识抽取与用户自定义编辑之间的协作问题。
    在知识图谱构建过程中，系统会自动从文档中抽取实体和关系，但用户可能会对这些自动生成的
    内容进行手动修改、补充或调整。该管理器确保在系统进行增量更新时，这些宝贵的人工编辑
    不会被覆盖，同时提供灵活的冲突解决策略。
    
    主要功能：
    1. 识别手动编辑的节点和关系 - 通过多种属性标志和时间戳机制
    2. 保留手动编辑，确保增量更新不会覆盖它 - 实现保护机制
    3. 解决自动更新和手动编辑之间的冲突 - 提供多种冲突解决策略
    4. 编辑操作的性能监控和统计 - 记录各阶段处理时间和资源消耗
    """
    
    def __init__(self):
        """初始化手动编辑同步管理器
        
        初始化过程中，建立数据库连接，设置性能参数，并准备实体属性和统计记录结构。
        为后续的手动编辑识别、保护和冲突解决做准备。
        """
        # 初始化控制台输出对象，用于美化日志输出
        self.console = Console()
        # 获取Neo4j数据库连接管理器
        self.graph = get_db_manager().graph
        
        # 设置并行工作线程数和批处理大小，用于性能优化
        self.max_workers = MAX_WORKERS
        self.batch_size = BATCH_SIZE

        # 初始化实体和关系的必要属性，确保数据库中的节点和关系具有一致的属性结构
        self.initialize_entity_properties()

        # 性能计时器，用于监控各阶段处理时间
        self.detection_time = 0
        self.sync_time = 0
        
        # 编辑统计字典，记录手动编辑处理的关键指标
        self.edit_stats = {
            "manual_entities": 0,      # 手动编辑的实体数量
            "manual_relations": 0,     # 手动编辑的关系数量
            "preserved_edits": 0,      # 成功保留的编辑数量
            "conflicts_resolved": 0    # 解决的冲突数量
        }

    def initialize_entity_properties(self):
        """
        初始化实体和关系的常用属性，确保数据结构一致性
        
        该方法为知识图谱中的所有实体和关系节点添加必要的元数据属性，
        用于追踪手动编辑状态、创建和修改信息。这些属性是区分手动编辑和
        系统自动生成内容的基础。仅对缺少属性的节点进行初始化，不覆盖已有的值。
        """
        try:
            # 初始化实体的manual_edit属性，标记是否为手动编辑
            self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.manual_edit IS NULL
            SET e.manual_edit = false
            """)
            
            # 初始化实体的created_by属性，记录创建者信息
            self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.created_by IS NULL
            SET e.created_by = null
            """)
            
            # 初始化实体的edited_by属性，记录编辑者信息
            self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.edited_by IS NULL
            SET e.edited_by = null
            """)
            
            # 初始化实体的created_at属性，记录创建时间
            self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.created_at IS NULL
            SET e.created_at = datetime()
            """)
            
            # 初始化实体的system_generated属性，标记是否为系统生成
            self.graph.query("""
            MATCH (e:`__Entity__`)
            WHERE e.system_generated IS NULL
            SET e.system_generated = true
            """)
            
            # 同样为关系初始化manual_edit属性
            self.graph.query("""
            MATCH ()-[r]->()
            WHERE r.manual_edit IS NULL
            SET r.manual_edit = false
            """)
            
            # 初始化关系的created_by属性
            self.graph.query("""
            MATCH ()-[r]->()
            WHERE r.created_by IS NULL
            SET r.created_by = null
            """)
            
            # 初始化关系的edited_by属性
            self.graph.query("""
            MATCH ()-[r]->()
            WHERE r.edited_by IS NULL
            SET r.edited_by = null
            """)
            
            self.console.print("[green]实体和关系属性初始化完成[/green]")
        except Exception as e:
            self.console.print(f"[yellow]初始化属性时出错: {e}[/yellow]")
    
    def _setup_manual_edit_tracking(self):
        """
        设置手动编辑追踪机制，通过Neo4j触发器自动记录节点和关系的变更
        
        该方法配置数据库层面的自动追踪机制，利用Neo4j的APOC插件创建触发器，
        当节点和关系发生变化时，自动更新时间戳和其他元数据。这种方法比应用层
        追踪更可靠，可以捕获所有类型的数据库操作，包括直接在数据库中执行的操作。
        
        特别处理集群模式，确保只在主节点上设置触发器，并提供优雅的错误处理，
        即使APOC插件不可用也能通过其他方式继续工作。
        """
        try:
            # 检查是否是集群模式，如果是，确保只在主节点上设置触发器
            try:
                cluster_role = self.graph.query("CALL dbms.cluster.role()")
                if cluster_role and cluster_role[0].get("role") == "FOLLOWER":
                    self.console.print("[yellow]当前节点为FOLLOWER，无法设置触发器，跳过...[/yellow]")
                    return
            except Exception as e:
                self.console.print(f"[yellow]检查集群角色时出错: {e}[/yellow]")
                
            # 查询当前数据库名称，确保触发器安装到正确的数据库
            try:
                db_info = self.graph.query("CALL db.info()")
                db_name = db_info[0]["name"] if db_info and "name" in db_info[0] else "neo4j"
            except:
                db_name = "neo4j"  # 默认数据库名称
                
            # 添加时间戳触发器，自动追踪实体节点的创建和修改时间
            try:
                self.graph.query(f"""
                CALL apoc.trigger.install(
                '{db_name}',
                'updateNodeTimestamps',
                '
                UNWIND $assignedLabels AS label
                UNWIND $createdNodes AS n
                WITH n WHERE label = "__Entity__" AND label IN labels(n)
                SET n.updated_at = datetime(),
                    n.created_at = coalesce(n.created_at, datetime())
                ',
                {{phase: 'after'}}
                )
                """)
                
                self.console.print("[green]成功设置节点时间戳追踪[/green]")
                
                # 添加关系时间戳触发器，自动追踪关系的创建和修改时间
                self.graph.query(f"""
                CALL apoc.trigger.install(
                '{db_name}',
                'updateRelationshipTimestamps',
                '
                UNWIND $createdRelationships AS r
                SET r.updated_at = datetime(),
                    r.created_at = coalesce(r.created_at, datetime())
                ',
                {{phase: 'after'}}
                )
                """)
                
                self.console.print("[green]成功设置关系时间戳追踪[/green]")
            except Exception as trigger_error:
                self.console.print(f"[yellow]设置触发器时出错: {trigger_error}[/yellow]")
                
        except Exception as e:
            # 如果触发器设置失败，降级使用基础检测方法
            self.console.print(f"[yellow]设置手动编辑追踪时出错 (可能APOC未安装): {e}[/yellow]")
            self.console.print("[yellow]将使用基础的手动编辑检测方法[/yellow]")

    def detect_manual_edits(self) -> Dict[str, int]:
        """
        检测数据库中的手动编辑实体和关系
        
        该方法采用多维度检测策略，通过检查实体和关系的元数据属性来识别手动编辑内容。
        实现了动态查询构建，确保在不同版本的数据库结构下都能正常工作，即使某些属性不存在。
        
        Returns:
            Dict: 包含手动编辑实体、关系和时间戳实体数量的统计信息
        """
        # 记录检测开始时间，用于性能监控
        start_time = time.time()
        
        # 检查属性是否存在，确保查询与数据库结构兼容
        try:
            # 获取数据库中所有存在的属性键
            props_result = self.graph.query("""
            CALL db.propertyKeys() YIELD propertyKey
            RETURN collect(propertyKey) AS all_props
            """)
            
            all_props = props_result[0]["all_props"] if props_result and props_result[0]["all_props"] else []
            
            # 1. 检测手动创建的实体节点 - 构建动态查询条件
            entity_clauses = []
            # 通过manual_edit标志检测
            if "manual_edit" in all_props:
                entity_clauses.append("e.manual_edit = true")
            # 通过创建者信息检测
            if "created_by" in all_props:
                entity_clauses.append("e.created_by IS NOT NULL")
            # 通过编辑者信息检测
            if "edited_by" in all_props:
                entity_clauses.append("e.edited_by IS NOT NULL")
                
            # 处理边界情况：如果没有可用条件，确保查询安全执行
            if not entity_clauses:
                entity_clauses.append("false")
                
            # 构建并执行实体检测查询
            manual_entity_query = f"""
            MATCH (e:`__Entity__`)
            WHERE {" OR ".join(entity_clauses)}
            RETURN count(e) AS manual_entities
            """
            
            entity_result = self.graph.query(manual_entity_query)
            manual_entities = entity_result[0]["manual_entities"] if entity_result else 0
            
            # 2. 检测手动创建的关系 - 使用类似的动态查询构建
            rel_clauses = []
            if "manual_edit" in all_props:
                rel_clauses.append("r.manual_edit = true")
            if "created_by" in all_props:
                rel_clauses.append("r.created_by IS NOT NULL")
            if "edited_by" in all_props:
                rel_clauses.append("r.edited_by IS NOT NULL")
                
            # 处理边界情况
            if not rel_clauses:
                rel_clauses.append("false")
                
            manual_relation_query = f"""
            MATCH ()-[r]->()
            WHERE {" OR ".join(rel_clauses)}
            RETURN count(r) AS manual_relations
            """
            
            relation_result = self.graph.query(manual_relation_query)
            manual_relations = relation_result[0]["manual_relations"] if relation_result else 0
            
            # 3. 检测通过时间戳识别的可能手动编辑 - 额外的检测维度
            timestamp_entities = 0
            if "created_at" in all_props and "system_generated" in all_props:
                timestamp_query = """
                MATCH (e:`__Entity__`) 
                WHERE e.created_at IS NOT NULL 
                AND e.system_generated = false
                RETURN count(e) AS timestamp_entities
                """
                
                try:
                    timestamp_result = self.graph.query(timestamp_query)
                    timestamp_entities = timestamp_result[0]["timestamp_entities"] if timestamp_result else 0
                except:
                    timestamp_entities = 0
            
        except Exception as e:
            # 错误处理：确保即使查询失败也能返回默认值
            self.console.print(f"[yellow]检测手动编辑时出错: {e}[/yellow]")
            manual_entities = 0
            manual_relations = 0
            timestamp_entities = 0
        
        # 更新统计数据
        self.edit_stats["manual_entities"] = manual_entities
        self.edit_stats["manual_relations"] = manual_relations
        
        # 计算检测时间
        self.detection_time = time.time() - start_time
        
        # 输出检测结果
        self.console.print(f"[blue]手动编辑检测完成，耗时: {self.detection_time:.2f}秒[/blue]")
        self.console.print(f"[blue]检测到 {manual_entities} 个手动编辑的实体节点，"
                        f"{manual_relations} 个手动编辑的关系[/blue]")
        
        # 返回统计结果
        return {
            "manual_entities": manual_entities,
            "manual_relations": manual_relations,
            "timestamp_entities": timestamp_entities
        }
    
    def mark_manual_edit(self, entity_id: str, edit_info: Dict[str, Any]) -> bool:
        """
        标记指定实体为手动编辑状态
        
        此方法用于将系统识别为手动修改过的实体进行明确标记，确保这些实体在后续的
        增量更新过程中得到特殊处理。通过添加元数据如编辑者、编辑时间和编辑注释，
        可以追踪和管理人工编辑的历史记录，为冲突解决提供依据。
        
        该方法是保护机制的核心组成部分，通常在以下场景中被调用：
        1. 用户通过界面进行手动编辑后
        2. 系统自动检测到可能的手动修改时
        3. 需要明确标记某个实体为受保护状态时
        
        Args:
            entity_id: 实体ID，用于在数据库中唯一标识实体
            edit_info: 编辑信息字典，包含编辑者、编辑时间等元数据
            
        Returns:
            bool: 标记操作是否成功
        """
        # 准备编辑信息，设置默认值确保字段完整性
        params = {
            "entity_id": entity_id,
            "edited_by": edit_info.get("edited_by", "manual"),  # 默认编辑者为"manual"
            "edit_time": edit_info.get("edit_time", datetime.now().isoformat()),  # 默认时间为当前时间
            "edit_comment": edit_info.get("edit_comment", ""),  # 可选编辑注释
            "manual_edit": True  # 设置手动编辑标志
        }
        
        # 执行标记查询，将实体标记为手动编辑并添加元数据
        query = """
        MATCH (e:`__Entity__` {id: $entity_id})
        SET e.manual_edit = $manual_edit,
            e.edited_by = $edited_by,
            e.edit_time = $edit_time,
            e.edit_comment = $edit_comment
        RETURN e.id AS entity_id
        """
        
        try:
            # 执行查询并验证结果
            result = self.graph.query(query, params=params)
            return bool(result and result[0]["entity_id"])
        except Exception as e:
            # 捕获并记录错误，但不中断程序执行
            self.console.print(f"[red]标记实体为手动编辑时出错: {e}[/red]")
            return False
    
    def preserve_manual_edits(self, changed_files: List[str]) -> int:
        """
        保护与变更文件相关的手动编辑，确保增量更新过程不覆盖人工修改
        
        该方法是增量更新流程中的关键环节，它通过分析知识图谱的文档-块-实体关系，
        识别与变更文件相关联且经过手动编辑的实体，然后为这些实体添加保护标志，
        确保它们在后续的自动更新过程中不会被意外删除或修改。
        
        保护机制采用两级标记策略：先标记需要保留的编辑，再应用保护规则，
        提供了更精细的控制和更大的灵活性。这种设计既确保了手动编辑的安全性，
        又保持了系统更新的效率。
        
        工作原理：
        1. 首先通过图查询识别与变更文件相关联的所有实体
        2. 筛选出其中被手动编辑过的实体（通过manual_edit或其他编辑属性判断）
        3. 为这些实体添加preserve_edit标志，标记为需要保留
        4. 进一步为这些实体添加protected标志，提供更高级别的保护
        
        该方法是ManualEditManager的核心功能之一，直接影响到用户体验，
        确保用户的手动编辑在系统更新过程中得到尊重和保护。
        
        Args:
            changed_files: 变更的文件列表，通常是本次增量更新中被修改或新增的文件
            
        Returns:
            int: 成功保留的手动编辑实体数量
        """
        # 记录开始时间，用于性能监控
        start_time = time.time()
        
        # 1. 标记与变更文件相关的手动编辑节点 - 通过图谱关系查找
        # 查询逻辑：找到属于变更文件的文档节点，关联到文本块，再关联到被提及的实体
        # 筛选条件：实体必须是手动编辑过的（通过manual_edit、created_by或edited_by属性判断）
        preserve_query = """
        MATCH (d:`__Document__`)<-[:PART_OF]-(c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
        WHERE d.fileName IN $changed_files
          AND (e.manual_edit = true OR e.created_by IS NOT NULL OR e.edited_by IS NOT NULL)
        SET e.preserve_edit = true
        RETURN count(e) AS preserved_count
        """
        
        try:
            # 执行查询并获取结果
            result = self.graph.query(preserve_query, params={"changed_files": changed_files})
            preserved_count = result[0]["preserved_count"] if result else 0
        except Exception as e:
            # 错误处理：确保即使查询失败也能继续执行
            self.console.print(f"[yellow]标记保留手动编辑时出错: {e}[/yellow]")
            preserved_count = 0
        
        # 2. 创建保护规则，为已标记的实体添加更高级别的保护
        protection_query = """
        MATCH (e:`__Entity__`) 
        WHERE e.preserve_edit = true
        SET e.protected = true
        RETURN count(e) AS protected_count
        """
        
        try:
            protection_result = self.graph.query(protection_query)
            protected_count = protection_result[0]["protected_count"] if protection_result else 0
        except Exception as e:
            self.console.print(f"[yellow]创建保护规则时出错: {e}[/yellow]")
            protected_count = 0
        
        # 更新统计信息
        self.edit_stats["preserved_edits"] = preserved_count
        
        # 计算同步时间
        self.sync_time = time.time() - start_time
        
        # 输出操作结果
        self.console.print(f"[blue]手动编辑保护完成，耗时: {self.sync_time:.2f}秒[/blue]")
        self.console.print(f"[blue]已保护 {preserved_count} 个手动编辑，"
                          f"{protected_count} 个节点被标记为受保护[/blue]")
        
        return preserved_count
    
    def resolve_conflicts(self, conflict_strategy: str = conflict_strategy) -> int:
        """
        解决自动更新和手动编辑之间的冲突
        
        冲突检测与解决是增量更新流程中的重要环节。当系统尝试自动更新已被用户手动修改的
        实体时，会产生潜在冲突。该方法提供三种不同的冲突解决策略，适应不同的业务需求。
        冲突解决是确保系统自动化与人工编辑和谐共存的关键机制，直接关系到知识图谱的质量和用户体验。
        
        支持的冲突解决策略：
        1. manual_first（默认）：优先保留手动编辑，这是最常用的策略，确保用户编辑不会丢失
           - 实现方式：移除系统生成标记，保留手动编辑内容
           - 适用场景：用户编辑被认为更权威或更准确的情况
        
        2. auto_first：优先使用自动更新，适用于需要保持数据最新性的场景
           - 实现方式：移除手动编辑标记，应用系统更新
           - 适用场景：系统数据来源更权威或需要频繁更新的情况
        
        3. merge：尝试合并两种编辑，适用于需要综合双方信息的场景
           - 实现方式：保留两种编辑标记，添加合并时间戳
           - 适用场景：希望保留双方修改痕迹，用于后续人工审查的情况
        
        该方法采用动态查询构建，确保在不同版本的数据库结构下都能正常工作。同时通过
        详细的错误处理和日志记录，确保即使在处理单个实体时出现问题，也不会影响整体流程。
        
        Args:
            conflict_strategy: 冲突解决策略，可选值为：
                            "manual_first"（优先保留手动编辑），
                            "auto_first"（优先自动更新），
                            "merge"（尝试合并）
            
        Returns:
            int: 成功解决的冲突数量
        """
        start_time = time.time()
        
        # 检查数据库中存在的属性，构建兼容查询
        try:
            props_result = self.graph.query("""
            CALL db.propertyKeys() YIELD propertyKey
            RETURN collect(propertyKey) AS all_props
            """)
            
            all_props = props_result[0]["all_props"] if props_result and props_result[0]["all_props"] else []
            
            # 构建动态查询条件，用于识别手动编辑的实体
            where_clauses = []
            if "manual_edit" in all_props:
                where_clauses.append("e.manual_edit = true")
            if "edited_by" in all_props:
                where_clauses.append("e.edited_by IS NOT NULL")
                
            # 系统生成条件，用于识别可能存在冲突的节点
            system_cond = "e.system_generated = true" if "system_generated" in all_props else "true"
            
            # 处理边界情况，确保查询安全
            if not where_clauses:
                where_clauses.append("false")
            
            # 查找可能的冲突节点 - 同时具有手动编辑标记和系统生成标记的实体
            conflict_query = f"""
            MATCH (e:`__Entity__`)
            WHERE ({" OR ".join(where_clauses)})
            AND {system_cond}
            RETURN e.id AS entity_id, e.type AS entity_type
            """
            
            conflicts = self.graph.query(conflict_query)
        except Exception as e:
            # 错误处理：即使查询失败也能继续执行
            self.console.print(f"[yellow]查找冲突节点时出错: {e}[/yellow]")
            conflicts = []
        
        resolved_count = 0
        
        # 基于策略逐一对冲突实体进行处理
        if conflicts:
            for conflict in conflicts:
                entity_id = conflict["entity_id"]
                
                if conflict_strategy == "manual_first":
                    # 优先保留手动编辑，移除系统生成标记
                    # 这种策略确保用户的修改不会被系统更新覆盖
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'manual_preferred'
                    """
                    
                    # 动态添加属性设置，确保查询与数据库结构兼容
                    if "system_generated" in all_props:
                        resolution_query += ",\n    e.system_generated = false"
                    
                    resolution_query += "\nRETURN e.id"
                
                elif conflict_strategy == "auto_first":
                    # 优先自动更新，移除手动编辑标记
                    # 这种策略适用于需要保持数据最新性的场景
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'auto_preferred'
                    """
                    
                    # 动态添加属性设置
                    if "manual_edit" in all_props:
                        resolution_query += ",\n    e.manual_edit = false"
                    
                    if "edited_by" in all_props:
                        resolution_query += ",\n    e.edited_by = null"
                    
                    resolution_query += "\nRETURN e.id"
                
                else:  # "merge" 策略
                    # 尝试合并两种编辑，添加合并时间戳
                    resolution_query = """
                    MATCH (e:`__Entity__` {id: $entity_id})
                    SET e.conflict_resolved = true,
                        e.conflict_resolution = 'merged',
                        e.merged_at = datetime()
                    RETURN e.id
                    """
                
                try:
                    # 执行冲突解决查询
                    result = self.graph.query(resolution_query, params={"entity_id": entity_id})
                    if result and result[0]:
                        resolved_count += 1
                except Exception as e:
                    # 记录特定实体的冲突解决错误
                    self.console.print(f"[red]解决实体 {entity_id} 的冲突时出错: {e}[/red]")
        
        # 更新统计信息
        self.edit_stats["conflicts_resolved"] = resolved_count
        
        resolution_time = time.time() - start_time
        
        # 输出操作结果
        self.console.print(f"[blue]冲突解决完成，耗时: {resolution_time:.2f}秒[/blue]")
        self.console.print(f"[blue]已解决 {resolved_count} 个冲突，使用策略: {conflict_strategy}[/blue]")
        
        return resolved_count
    
    def display_edit_stats(self):
        """
        显示手动编辑的统计信息和性能指标
        
        该方法使用Rich库创建格式化的表格输出，展示手动编辑处理过程中的关键指标，
        包括手动编辑的实体和关系数量、保留的编辑数量、解决的冲突数量等。
        同时还显示了各阶段的处理时间，便于性能监控和优化。
        
        这种可视化的统计报告不仅有助于监控当前操作的执行情况，还为系统性能优化
        提供了数据支持。通过分析统计信息，可以识别潜在的性能瓶颈，调整系统配置。
        
        该方法主要在process方法的最后阶段被调用，为用户提供完整的手动编辑处理
        结果概览，增强了系统的可观测性和用户体验。
        """
        # 创建统计表格
        stats_table = Table(title="手动编辑统计")
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("值", justify="right")
        
        # 添加统计数据到表格
        for key, value in self.edit_stats.items():
            stats_table.add_row(key, str(value))
        
        # 输出表格
        self.console.print(stats_table)
        
        # 显示时间统计信息
        self.console.print(f"[blue]检测耗时: {self.detection_time:.2f}秒, "
                          f"同步耗时: {self.sync_time:.2f}秒[/blue]")
        
    def process(self, changed_files: List[str], conflict_strategy: str = "manual_first") -> Dict[str, Any]:
        """
        执行完整的手动编辑同步流程
        
        该方法是ManualEditManager的核心方法，它集成了整个手动编辑管理流程，
        按照正确的顺序调用各个处理步骤，确保增量更新过程中手动编辑得到妥善处理。
        整个流程包括：设置追踪机制、检测手动编辑、保留重要编辑、解决潜在冲突，
        最后显示统计信息。
        
        这是增量更新系统中的关键环节，确保了自动化知识抽取与人工编辑之间的
        无缝协作，提高了知识图谱的质量和可靠性。手动编辑保护是Graph-RAG系统
        区别于传统自动知识抽取系统的重要特性，使用户可以通过人工干预来修正、
        补充和优化自动生成的知识。
        
        处理流程详解：
        1. 设置手动编辑追踪 - 配置数据库层面的触发器，用于自动记录变更
        2. 检测手动编辑 - 扫描数据库中所有的手动编辑内容，建立初始统计
        3. 保留手动编辑 - 为与变更文件相关的手动编辑添加保护标记
        4. 解决冲突 - 采用指定的策略处理自动更新与手动编辑之间的矛盾
        5. 显示统计 - 生成并展示完整的处理结果报告
        
        该方法在增量更新流程中被调用，通常是在处理变更文件之前，确保在进行
        任何可能影响已有内容的操作前，先保护好用户的手动编辑。
        
        Args:
            changed_files: 变更的文件列表，通常是本次增量更新中被修改或新增的文件
            conflict_strategy: 冲突解决策略，默认为"manual_first"（优先保留手动编辑）
            
        Returns:
            Dict: 包含处理结果统计、时间消耗和状态信息的详细报告
            
        Raises:
            Exception: 如果处理过程中发生错误，会向上传播异常
        """
        try:
            # 1. 设置手动编辑追踪 - 配置数据库层面的自动记录机制
            self._setup_manual_edit_tracking()
            
            # 2. 检测手动编辑 - 识别数据库中所有的手动编辑内容
            edit_stats = self.detect_manual_edits()
            
            # 3. 保留手动编辑 - 为与变更文件相关的手动编辑添加保护标记
            preserved_count = self.preserve_manual_edits(changed_files)
            
            # 4. 解决冲突 - 根据指定策略处理自动更新与手动编辑之间的冲突
            resolved_count = self.resolve_conflicts(conflict_strategy)
            
            # 5. 显示统计 - 输出处理结果和性能指标
            self.display_edit_stats()
            
            # 返回详细的处理结果报告
            return {
                "detection_time": self.detection_time,  # 检测阶段耗时
                "sync_time": self.sync_time,            # 同步阶段耗时
                "edit_stats": self.edit_stats,          # 编辑统计信息
                "preserved_count": preserved_count,     # 保留的编辑数量
                "resolved_count": resolved_count        # 解决的冲突数量
            }
            
        except Exception as e:
            # 记录错误并向上传播，确保调用者能够感知异常
            self.console.print(f"[red]手动编辑同步过程中出现错误: {e}[/red]")
            raise