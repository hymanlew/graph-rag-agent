"""
时间知识图谱工具函数模块

该模块提供了构建和操作时间知识图谱的核心功能，主要用于从预处理的数据库中加载数据，
并构建带有时间维度的知识图谱结构。这些函数是知识图谱问答系统的重要组成部分，
特别是在处理具有时效性信息的问答场景中。

主要功能：
- 从HuggingFace加载预处理的时间知识图谱数据
- 构建带有实体规范化和时间属性的NetworkX图结构
- 加载实体映射和事件时间信息
- 添加带时间属性的图边和节点

设计特点：
- 支持实体规范化，确保实体引用一致性
- 保留事件的时间维度信息
- 灵活的图构建选项，支持使用名称或ID作为节点标识
- 与项目中的数据库接口紧密集成
"""

import sqlite3
import networkx as nx
from typing import Any
from datasets import load_dataset

from db_interface import get_all_triplets


def load_db_from_hf(db_path: str = "temporal_graph.db", hf_dataset_name: str = "TomoroAI/temporal_cookbook_db") -> sqlite3.Connection:
    """
    从HuggingFace加载预处理的时间知识图谱数据库
    
    该函数连接到HuggingFace数据集仓库，下载预处理的时间知识图谱数据，
    并将其保存到SQLite数据库中。这是构建时间知识图谱的第一步，
    为后续的图构建和查询提供数据基础。
    
    Args:
        db_path: SQLite数据库文件的保存路径，默认为"temporal_graph.db"
        hf_dataset_name: HuggingFace数据集名称，默认为"TomoroAI/temporal_cookbook_db"
        
    Returns:
        sqlite3.Connection: 连接到加载完成的SQLite数据库的连接对象
        
    业务流程：
    1. 创建或连接到指定路径的SQLite数据库
    2. 定义需要加载的表名列表（transcripts、chunks、events、triplets、entities）
    3. 遍历每个表名，执行以下操作：
       a. 从HuggingFace加载对应表的数据
       b. 将数据转换为DataFrame格式
       c. 将DataFrame写入SQLite数据库
       d. 提交事务
    4. 完成所有表的加载后返回数据库连接
    
    技术特点：
    - 使用datasets库从HuggingFace高效加载数据集
    - 利用pandas的DataFrame简化数据处理和数据库写入
    - 事务提交确保数据完整性
    - 提供详细的加载进度输出
    """
    conn = sqlite3.connect(db_path)
    table_names = [
        "transcripts",
        "chunks",
        "events",
        "triplets",
        "entities",
    ]

    for table in table_names:
        print(f"Loading {table}...")
        ds = load_dataset(hf_dataset_name, name=table, split="train")
        df = ds.to_pandas()
        df.to_sql(table, conn, if_exists="replace", index=False)

        conn.commit()
    print("✅ All tables written to SQLite.")

    return conn

def build_graph(
        conn: sqlite3.Connection,
        *,
        nodes_as_names: bool = False
        ) -> nx.MultiDiGraph:
    """
    构建时间知识图谱
    
    该函数从SQLite数据库中加载数据，并构建一个完整的时间知识图谱。
    图谱使用NetworkX的MultiDiGraph实现，支持多标签边和方向性，
    并包含实体规范化和时间属性信息。这是将结构化数据转换为图结构的关键步骤。
    
    Args:
        conn: SQLite数据库连接对象
        nodes_as_names: 是否使用实体名称而不是ID作为图节点标识符，默认为False
        
    Returns:
        nx.MultiDiGraph: 构建完成的时间知识图谱
        
    业务流程：
    1. 创建空的有向多重图（MultiDiGraph）
    2. 加载实体规范化映射和事件时间信息
    3. 获取所有三元组（subject-predicate-object关系）
    4. 遍历每个有效三元组，将其添加到图中
    5. 返回构建完成的图
    
    技术特点：
    - 使用实体规范化确保实体引用一致性
    - 保留所有事件的时间属性
    - 支持灵活的节点标识方式（ID或名称）
    - 高效处理大量三元组数据
    
    业务意义：
    - 构建带时间维度的知识表示，支持时间相关查询
    - 提供实体间关系的图状视图，便于复杂关系发现
    - 为基于图的问答和推理提供数据基础
    """
    graph = nx.MultiDiGraph()

    # Always load canonical mappings
    entity_to_canonical, canonical_names = _load_entity_maps(conn)
    event_temporal_map = _load_event_temporal(conn)

    for t in get_all_triplets(conn):
        if not t["subject_id"]:
            continue

        event_attrs = event_temporal_map.get(t["event_id"])
        _add_triplet_edge(
            graph,
            t,
            entity_to_canonical,
            canonical_names,
            event_attrs,
            nodes_as_names,
        )

    return graph

def _load_entity_maps(conn: sqlite3.Connection) -> tuple[dict[bytes, bytes], dict[bytes, str]]:
    """
    加载实体规范化映射
    
    该函数从数据库中加载实体数据，并构建两个关键映射：
    1. 实体ID到规范实体ID的映射
    2. 规范实体ID到规范实体名称的映射
    这对于确保知识图谱中实体引用的一致性至关重要，特别是在处理可能有多个引用形式的同一实体时。
    
    Args:
        conn: SQLite数据库连接对象
        
    Returns:
        tuple[dict[bytes, bytes], dict[bytes, str]]: 包含两个映射的元组
            - 第一个字典：实体ID到规范实体ID的映射
            - 第二个字典：规范实体ID到规范实体名称的映射
            
    业务流程：
    1. 创建数据库游标
    2. 执行SQL查询，获取所有实体的ID、名称和解析ID
    3. 遍历查询结果，构建映射：
       a. 对于有解析ID的实体，将其映射到解析ID
       b. 对于没有解析ID的实体，将其映射到自身
       c. 记录规范实体的名称
    4. 返回构建好的映射字典
    
    技术特点：
    - 使用SQL查询高效获取实体数据
    - 处理实体规范化和实体解析逻辑
    - 支持字节类型的ID处理
    
    业务意义：
    - 解决实体消歧问题，确保同一实体的不同引用指向同一规范形式
    - 为图构建提供一致的实体标识
    - 支持基于规范名称的实体展示和查询
    """
    cur = conn.cursor()

    # Get all entities with their resolved IDs
    cur.execute("""
        SELECT id, name, resolved_id
        FROM entities
    """)

    entity_to_canonical: dict[bytes, bytes] = {}
    canonical_names: dict[bytes, str] = {}

    for row in cur.fetchall():
        entity_id = row[0]
        name = row[1]
        resolved_id = row[2]

        if resolved_id:
            # If entity has a resolved_id, map to that
            entity_to_canonical[entity_id] = resolved_id
            # Store name of the canonical entity
            canonical_names[resolved_id] = name
        else:
            # If no resolved_id, entity is its own canonical version
            entity_to_canonical[entity_id] = entity_id
            canonical_names[entity_id] = name

    return entity_to_canonical, canonical_names

def _load_event_temporal(conn: sqlite3.Connection) -> dict[bytes, dict[str, Any]]:
    """
    加载事件的时间和描述属性
    
    该函数从数据库中加载所有事件的时间和描述信息，并构建事件ID到属性字典的映射。
    这些时间属性对于构建具有时间维度的知识图谱至关重要，支持基于时间的查询和推理。
    
    Args:
        conn: SQLite数据库连接对象
        
    Returns:
        dict[bytes, dict[str, Any]]: 事件ID到属性字典的映射
            属性包括：语句内容、语句类型、时间类型、创建时间、有效时间、
            过期时间、失效时间和失效原因
            
    业务流程：
    1. 创建数据库游标
    2. 执行SQL查询，获取事件表中的关键属性
    3. 遍历查询结果，构建事件ID到属性字典的映射
    4. 返回构建好的映射字典
    
    技术特点：
    - 选择性加载事件的关键属性，优化内存使用
    - 支持多种时间类型属性的处理
    - 使用字节类型的ID作为映射键
    
    业务意义：
    - 为知识图谱提供时间维度信息
    - 支持基于时间的查询和推理（如：查询某个时间点有效的关系）
    - 保留事件的完整上下文和时间演变信息
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT  id,
                statement,
                statement_type,
                temporal_type,
                created_at,
                valid_at,
                expired_at,
                invalid_at,
                invalidated_by
        FROM events
    """)
    event_map: dict[bytes, dict[str, Any]] = {}
    for (
        eid,
        statement,
        statement_type,
        temporal_type,
        created_at,
        valid_at,
        expired_at,
        invalid_at,
        invalidated_by,
    ) in cur.fetchall():
        event_map[eid] = {
            "statement": statement,
            "statement_type": statement_type,
            "temporal_type": temporal_type,
            "created_at": created_at,
            "valid_at": valid_at,
            "expired_at": expired_at,
            "invalid_at": invalid_at,
            "invalidated_by": invalidated_by,
        }
    return event_map


def _add_triplet_edge(
        graph: nx.MultiDiGraph, t: dict,
        entity_to_canonical: dict[bytes, bytes],
        canonical_names: dict[bytes, str],
        event_attrs: dict[str, Any] | None = None,
        use_names: bool = False,
        ) -> None:
    """
    向知识图谱添加三元组边
    
    该函数处理单个三元组数据，将其作为边添加到知识图谱中。
    它负责实体规范化、节点创建和边属性设置，包括时间信息。
    该函数是图构建过程中的核心组件，将结构化的三元组数据转换为图结构。
    
    Args:
        graph: 要添加边的NetworkX有向多重图
        t: 三元组字典，包含subject_id、object_id、predicate等信息
        entity_to_canonical: 实体ID到规范实体ID的映射
        canonical_names: 规范实体ID到规范实体名称的映射
        event_attrs: 可选的事件属性字典，包含时间信息等
        use_names: 是否使用名称而不是ID作为节点标识符
        
    业务流程：
    1. 验证三元组数据的有效性（至少需要主语ID）
    2. 获取主语和宾语的规范实体ID
    3. 获取主语和宾语的规范实体名称
    4. 根据配置决定使用ID还是名称作为节点标识
    5. 添加主语节点到图中，设置节点属性
    6. 构建边属性字典，包括谓词信息和三元组元数据
    7. 如果有事件属性，将其合并到边属性中
    8. 处理宾语为空的特殊情况（自环边）
    9. 处理常规情况：添加宾语节点和从主语到宾语的边
    
    技术特点：
    - 处理实体规范化，确保实体引用一致性
    - 支持空宾语的特殊情况处理（自环边）
    - 灵活的节点标识方式（ID或名称）
    - 完整保留事件的时间和描述属性
    
    业务意义：
    - 将结构化的三元组数据转换为图结构
    - 保留实体间关系的方向和丰富的属性信息
    - 为基于图的查询和推理提供数据基础
    - 支持时间相关的知识表示和查询
    """
    subj_id = t["subject_id"]
    obj_id = t["object_id"]

    if subj_id is None:
        return

    # Get canonical IDs
    canonical_subj = entity_to_canonical.get(subj_id, subj_id)
    canonical_obj = entity_to_canonical.get(obj_id, obj_id) if obj_id else None

    # Get canonical names
    subj_name = canonical_names.get(canonical_subj, t["subject_name"]) if canonical_subj is not None else t["subject_name"]
    obj_name = canonical_names.get(canonical_obj, t["object_name"]) if canonical_obj is not None else t["object_name"]

    subj_node = subj_name if use_names else canonical_subj
    obj_node  = obj_name  if use_names else canonical_obj

    # Add nodes with canonical names
    graph.add_node(
        subj_node,
        canonical_id=canonical_subj,
        name=subj_name,
    )

    # Core edge attributes (triplet-specific)
    edge_attrs: dict[str, Any] = {
        "predicate": t["predicate"],
        "triplet_id": t["id"],
        "event_id": t["event_id"],
        "value": t["value"],
        "canonical_subject_name": subj_name,
        "canonical_object_name": obj_name,
    }

    # Merge in temporal data, if we have it
    if event_attrs:
        edge_attrs.update(event_attrs)

    if canonical_obj is None:
        # Handle self-loops for null objects
        graph.add_edge(
            subj_node, subj_node,
            key=t["predicate"],
            **edge_attrs,
            literal_object=t["object_name"],
        )
    else:
        graph.add_node(
            obj_node,
            canonical_id=canonical_obj,
            name=obj_name,
        )
        graph.add_edge(
            subj_node, obj_node,
            key=t["predicate"],
            **edge_attrs,
        )
