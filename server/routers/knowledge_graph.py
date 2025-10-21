"""
知识图谱API路由模块

该模块提供了一套完整的知识图谱操作API，支持知识图谱的查询、构建、修改和分析功能。
它是系统的核心组件之一，为上层应用提供了与知识图谱交互的接口。

主要功能：
- 知识图谱数据获取和可视化
- 从文本中提取知识图谱
- 实体和关系的完整CRUD操作
- 多种图查询和推理功能
  - 最短路径查询
  - 一到两跳关系查询
  - 共同邻居分析
  - 所有可能路径查询
  - 实体循环检测
  - 实体影响力分析
  - 社区检测
- 实体和关系的搜索功能

设计特点：
- 采用FastAPI构建的RESTful API
- 支持复杂的图查询和推理操作
- 提供全面的错误处理和异常捕获
- 与Neo4j图数据库紧密集成
- 结构化的请求和响应数据模型
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import traceback
from services.kg_service import (
    get_knowledge_graph, 
    extract_kg_from_message, 
    get_chunks,
    get_shortest_path,
    get_one_two_hop_paths,
    get_common_neighbors,
    get_all_paths,
    get_entity_cycles,
    get_entity_influence,
    get_simplified_community,
)
from server_config.database import get_db_manager
from models.schemas import (ReasoningRequest, EntityData, EntityDeleteData, EntitySearchFilter, EntityUpdateData,
                            RelationData, RelationDeleteData, RelationSearchFilter, RelationUpdateData)

# 创建路由器
# 初始化知识图谱相关的API路由组，统一管理所有知识图谱操作端点
router = APIRouter()


@router.get("/knowledge_graph")
async def knowledge_graph(limit: int = 100, query: Optional[str] = None):
    """
    获取知识图谱数据
    
    该端点提供了知识图谱数据的基础查询功能，返回知识图谱的节点和连接信息，
    支持设置返回数据的数量限制和可选的查询条件，适用于知识图谱的可视化和基础分析。
    
    Args:
        limit: 节点数量限制，默认返回100个节点及其相关连接，用于控制数据量
        query: 可选的查询条件，用于筛选特定的节点或关系
        
    Returns:
        Dict: 知识图谱数据，包含两个主要字段：
            - nodes: 节点列表，每个节点包含id、标签和属性等信息
            - links: 连接列表，表示节点之间的关系
    
    业务流程：
    1. 接收API请求，获取limit和query参数
    2. 调用kg_service中的get_knowledge_graph函数获取数据
    3. 返回格式化的知识图谱数据
    
    业务意义：
        - 为知识图谱可视化组件提供数据支持
        - 提供知识图谱的整体视图
        - 支持基于特定条件的知识图谱数据筛选
    """
    # 调用知识图谱服务获取数据，传入限制数量和查询条件
    # get_knowledge_graph函数负责从数据库中查询并构建知识图谱数据结构
    return get_knowledge_graph(limit, query)


@router.get("/knowledge_graph_from_message")
async def knowledge_graph_from_message(message: Optional[str] = None, query: Optional[str] = None):
    """
    从消息文本中提取知识图谱数据
    
    该端点提供了文本理解和知识抽取功能，能够从用户提供的消息文本中自动识别实体和它们之间的关系，
    构建成结构化的知识图谱数据。这是系统智能化处理非结构化文本的关键能力。
    
    Args:
        message: 消息文本，将从中提取实体和关系信息
        query: 可选的查询内容，用于过滤或聚焦于特定主题的知识抽取
        
    Returns:
        Dict: 知识图谱数据，包含两个主要字段：
            - nodes: 从文本中识别出的实体节点列表
            - links: 实体之间的关系连接列表
            如果消息为空，返回空的知识图谱结构
    
    业务流程：
    1. 检查消息是否为空，如为空直接返回空知识图谱
    2. 调用kg_service中的extract_kg_from_message函数进行知识抽取
    3. 返回从文本中提取的结构化知识图谱数据
    
    业务意义：
        - 实现从非结构化文本到结构化知识的转换
        - 支持自动构建和扩展知识图谱
        - 提供基于文本的实体关系可视化
        - 增强系统对文本内容的理解能力
    """
    # 检查消息是否为空，避免处理空文本
    if not message:
        return {"nodes": [], "links": []}
    
    # 调用知识抽取服务，从文本中提取实体和关系
    # extract_kg_from_message函数负责自然语言处理和实体关系抽取
    return extract_kg_from_message(message, query)

@router.get("/chunks")
async def chunks(limit: int = 10, offset: int = 0):
    """
    获取数据库中的文本块
    
    该端点提供了知识图谱系统中文本块(文档片段)的分页查询功能。这些文本块通常是知识图谱构建的基础
    数据源，包含了原始的非结构化或半结构化信息。通过这个端点可以查看和管理知识图谱的原始内容。
    
    Args:
        limit: 返回数量限制，默认每页返回10个文本块，用于分页控制
        offset: 偏移量，指定从第几个文本块开始返回，用于分页查询
        
    Returns:
        Dict: 文本块数据和总数，包含两个主要字段：
            - chunks: 返回的文本块列表
            - total: 数据库中文本块的总数
    
    业务流程：
    1. 接收API请求，获取limit和offset参数
    2. 调用kg_service中的get_chunks函数进行分页查询
    3. 返回查询到的文本块数据和总数信息
    
    业务意义：
        - 提供对知识图谱数据源的管理和查看功能
        - 支持文本块的分页浏览和检索
        - 便于监控和维护知识图谱的内容质量
        - 为管理员提供知识图谱内容审核的入口
    """
    # 调用知识图谱服务获取文本块数据，支持分页
    # get_chunks函数负责从数据库中查询文本块并处理分页逻辑
    return get_chunks(limit, offset)

@router.post("/kg_reasoning")
async def knowledge_graph_reasoning(request: ReasoningRequest):
    """
    执行知识图谱推理
    
    该端点提供了高级知识图谱推理和分析功能，支持多种图算法和查询模式，能够从知识图谱中发现隐含的
    关系和模式。这是系统智能化分析和知识挖掘的核心能力，为用户提供深度知识发现和关系分析。
    
    支持的推理类型：
    - shortest_path: 查找两个实体之间的最短路径
    - one_two_hop: 查询两个实体之间的一到两跳关系
    - common_neighbors: 查找两个实体的共同邻居
    - all_paths: 查找两个实体之间的所有可能路径（有深度限制）
    - entity_cycles: 检测实体相关的循环关系
    - entity_influence: 分析实体在图中的影响力
    - entity_community: 社区检测，识别实体所在的社区结构
    
    Args:
        request: 推理请求对象，包含以下关键字段：
            - reasoning_type: 推理类型，指定要执行的分析算法
            - entity_a: 起始实体
            - entity_b: 目标实体（部分推理类型需要）
            - max_depth: 最大查询深度，范围限制在1-5之间
            - algorithm: 算法选择（部分推理类型需要）
    
    Returns:
        Dict: 推理结果，通常包含：
            - nodes: 推理结果涉及的节点列表
            - links: 推理结果涉及的关系连接列表
            - 可能包含特定推理类型的额外信息字段
            - 如发生错误，将包含error字段说明错误原因
    
    业务流程：
    1. 接收并验证推理请求参数
    2. 获取数据库连接
    3. 根据推理类型执行相应的查询和分析
    4. 异常情况下捕获错误并返回友好的错误信息
    5. 返回推理结果数据
    
    业务意义：
        - 提供深度知识发现和关系分析能力
        - 支持复杂问题的智能推理和解答
        - 发现知识图谱中的隐含模式和关系
        - 支持基于图的高级分析和决策支持
        - 为用户提供可视化的知识关系探索工具
    """
    try:
        # 获取数据库连接管理器和驱动
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 对参数进行处理，确保安全传递给Neo4j
        reasoning_type = request.reasoning_type
        entity_a = request.entity_a.strip()
        entity_b = request.entity_b.strip() if request.entity_b else None
        max_depth = min(max(1, request.max_depth), 5)  # 确保在1-5的范围内
        algorithm = request.algorithm
        
        # 日志记录推理请求详情
        print(f"推理请求: 类型={reasoning_type}, 实体A={entity_a}, 实体B={entity_b}, 深度={max_depth}, 算法={algorithm}")
        
        # 社区检测系统 - 特殊处理的推理类型
        if reasoning_type == "entity_community":
            return await process_community_detection(entity_a, max_depth, algorithm)
            
        # 其他推理类型的处理
        if reasoning_type == "shortest_path":
            if not entity_b:
                return {"error": "最短路径查询需要指定两个实体", "nodes": [], "links": []}
            result = get_shortest_path(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "one_two_hop":
            if not entity_b:
                return {"error": "一到两跳关系查询需要指定两个实体", "nodes": [], "links": []}
            result = get_one_two_hop_paths(driver, entity_a, entity_b)
        elif reasoning_type == "common_neighbors":
            if not entity_b:
                return {"error": "共同邻居查询需要指定两个实体", "nodes": [], "links": []}
            # 获取两个实体的共同邻居，发现它们之间的间接关联
            result = get_common_neighbors(driver, entity_a, entity_b)
        elif reasoning_type == "all_paths":
            if not entity_b:
                return {"error": "关系路径查询需要指定两个实体", "nodes": [], "links": []}
            # 查找两个实体之间的所有可能路径，限制最大深度
            result = get_all_paths(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "entity_cycles":
            # 检测与目标实体相关的循环关系
            result = get_entity_cycles(driver, entity_a, max_depth)
        elif reasoning_type == "entity_influence":
            # 分析实体在知识图谱中的影响力和重要性
            result = get_entity_influence(driver, entity_a, max_depth)
        else:
            # 处理未知的推理类型
            return {"error": "未知的推理类型", "nodes": [], "links": []}
        
        # 返回推理结果
        return result
    except Exception as e:
        # 捕获并记录异常，返回友好的错误信息
        print(f"推理查询异常: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈信息用于调试
        return {"error": str(e), "nodes": [], "links": []}

async def process_community_detection(entity_id: str, max_depth: int, algorithm: str):
    """
    执行专业社区检测流程
    
    该函数负责执行实体社区检测的完整流程，首先尝试从数据库中查找已有的社区信息，
    如果不存在，则使用简化版本的社区检测算法生成结果。社区检测是一种识别知识图谱中
    紧密连接的实体集合的重要方法。
    
    Args:
        entity_id: 目标实体的ID
        max_depth: 社区检测的最大深度，控制社区范围大小
        algorithm: 使用的社区检测算法名称
    
    Returns:
        Dict: 社区检测结果，包含：
            - nodes: 社区中的实体节点列表
            - links: 节点之间的关系连接列表
            - community_info: 社区的统计信息（如存在）
            - 错误情况下包含error字段
    
    业务流程：
    1. 尝试从数据库获取实体的已有社区信息
    2. 如果找到有效社区信息，直接返回
    3. 如果未找到，获取数据库连接并调用简化版社区检测函数
    4. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 识别知识图谱中实体所属的社区或群体
        - 发现实体周围的重要关联和结构模式
        - 支持基于社区的知识组织和可视化
        - 为用户提供实体上下文和关系网络的整体视图
    """
    try:
        # 首先检查实体是否已存在于社区中 - 尝试从缓存或预计算结果中获取
        community_info = await get_entity_community_from_db(entity_id)
        if community_info and community_info.get("nodes") and community_info.get("links"):
            print(f"实体 {entity_id} 已有社区信息，直接返回")
            return community_info
            
        # 实体没有社区信息，使用简化版本返回查询结果 - 动态计算社区结构
        print(f"实体 {entity_id} 没有社区信息，使用简化版本")
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        # 调用简化版社区检测函数，基于图算法生成实体的社区结构
        return get_simplified_community(driver, entity_id, max_depth)
    except Exception as e:
        # 捕获并记录异常，返回友好的错误信息
        print(f"处理社区检测失败: {str(e)}")
        traceback.print_exc()  # 打印详细的堆栈信息用于调试
        return {"error": str(e), "nodes": [], "links": []}

async def get_entity_community_from_db(entity_id: str):
    """
    从数据库中获取实体的社区信息
    
    该函数负责从数据库中查询指定实体所属的社区信息，并将其转换为适合前端可视化的格式。
    社区信息包括社区内的所有实体、实体之间的关系，以及社区的统计摘要信息。
    
    Args:
        entity_id: 目标实体的ID
    
    Returns:
        Dict: 社区信息字典，包含：
            - nodes: 社区中的实体节点列表
            - links: 节点之间的关系连接列表
            - community_info: 社区统计信息
              - id: 社区ID
              - entity_count: 实体数量
              - relation_count: 关系数量
              - summary: 社区摘要描述
            如果未找到社区信息，返回None
    
    业务流程：
    1. 获取数据库连接和图对象
    2. 查询实体所属的社区ID
    3. 如果未找到社区，返回None
    4. 根据社区ID查询社区内的所有实体和关系
    5. 构建适合可视化的节点和连接列表
    6. 计算社区统计信息并返回完整的社区数据
    
    业务意义：
        - 提供实体所属社区的完整视图
        - 支持基于社区的知识组织和浏览
        - 为知识图谱可视化提供结构化数据
        - 帮助用户理解实体在更大知识网络中的位置和上下文
    """
    try:
        # 获取数据库连接管理器和图对象
        db_manager = get_db_manager()
        graph = db_manager.get_graph()
        
        # 查询实体所属的社区 - 找出实体关联的社区节点
        community_result = graph.query("""
        MATCH (e:__Entity__ {id: $entity_id})-[:IN_COMMUNITY]->(c:__Community__)
        RETURN c.id AS community_id
        LIMIT 1
        """, params={"entity_id": entity_id})
        
        # 检查是否找到社区
        if not community_result:
            return None
            
        # 提取社区ID
        community_id = community_result[0].get("community_id")
        if not community_id:
            return None
            
        # 获取该社区的所有节点和关系 - 执行复杂的图查询
        community_data = graph.query("""
        // 获取社区中的所有实体
        MATCH (c:__Community__ {id: $community_id})<-[:IN_COMMUNITY]-(e:__Entity__)
        WITH c, collect({
            id: e.id,
            description: e.description,
            labels: labels(e)
        }) AS entities
        
        // 获取社区摘要
        OPTIONAL MATCH (c)
        WHERE c.summary IS NOT NULL
        
        // 获取实体间的关系
        CALL {
            WITH c
            MATCH (c)<-[:IN_COMMUNITY]-(e1:__Entity__)-[r]->(e2:__Entity__)-[:IN_COMMUNITY]->(c)
            RETURN collect({
                source: e1.id,
                target: e2.id,
                type: type(r)
            }) AS relationships
        }
        
        // 返回社区信息
        RETURN 
            c.id AS community_id,
            c.summary AS summary,
            entities,
            relationships
        """, params={"community_id": community_id})
        
        # 检查查询结果
        if not community_data:
            return None
            
        # 初始化可视化数据结构
        nodes = []
        links = []
        # 获取社区摘要，默认为"无社区摘要"
        community_summary = community_data[0].get("summary", "无社区摘要")
        
        # 处理节点
        for entity in community_data[0].get("entities", []):
            entity_labels = entity.get("labels", [])
            group = [label for label in entity_labels if label != "__Entity__"]
            group = group[0] if group else "Unknown"
            
            # 标记中心实体
            if entity.get("id") == entity_id:
                group = "Center"
                
            nodes.append({
                "id": entity.get("id"),
                "label": entity.get("id"),
                "description": entity.get("description", ""),
                "group": group
            })
        
        # 处理关系
        for rel in community_data[0].get("relationships", []):
            links.append({
                "source": rel.get("source"),
                "target": rel.get("target"),
                "label": rel.get("type"),
                "weight": 1
            })
        
        # 获取社区统计信息
        stats = {
            "id": community_id,
            "entity_count": len(nodes),
            "relation_count": len(links),
            "summary": community_summary
        }
        
        return {
            "nodes": nodes,
            "links": links,
            "community_info": stats
        }
            
    except Exception as e:
        print(f"获取社区信息失败: {str(e)}")
        return None

@router.get("/entity_types")
def get_entity_types():
    """
    获取知识图谱中的所有实体类型
    
    该端点提供了知识图谱中存在的所有实体类型的查询功能，返回一个分类列表，
    用于前端界面的类型过滤、数据统计和知识组织。实体类型是知识图谱分类体系的重要组成部分。
    
    Returns:
        Dict: 包含实体类型列表的字典
            - entity_types: 知识图谱中存在的所有实体类型名称列表
    
    Raises:
        HTTPException: 当查询过程中发生错误时抛出500异常
    
    业务流程：
    1. 获取数据库连接管理器
    2. 执行Cypher查询，获取所有实体的非__Entity__标签作为类型
    3. 将查询结果转换为列表格式
    4. 返回实体类型列表
    5. 异常情况下捕获错误并抛出友好的异常信息
    
    业务意义：
        - 提供知识图谱的类型体系概览
        - 支持基于类型的实体过滤和查询
        - 为用户界面提供类型选择器的数据
        - 帮助理解知识图谱的组织结构
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 查询实体类型 - 从所有实体中提取非__Entity__的标签作为类型
        result = db_manager.execute_query("""
        MATCH (e:__Entity__)
        RETURN DISTINCT
        CASE WHEN size(labels(e)) > 1 
             THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] 
             ELSE 'Unknown' 
        END AS entity_type
        ORDER BY entity_type
        """)
        
        # 处理查询结果，提取实体类型列表
        # DataFrame处理方式 - 假设返回结果是pandas DataFrame格式
        entity_types = result['entity_type'].tolist() if 'entity_type' in result.columns else []
        
        # 返回格式化的实体类型列表
        return {"entity_types": entity_types}
    except Exception as e:
        # 记录错误信息并抛出异常
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        raise HTTPException(status_code=500, detail=f"获取实体类型失败: {str(e)}")


@router.get("/relation_types")
def get_relation_types():
    """
    获取知识图谱中的所有关系类型
    
    该端点提供了知识图谱中存在的所有关系类型的查询功能，返回一个关系类型列表，
    用于前端界面的关系过滤、数据统计和知识结构分析。关系类型是知识图谱中实体连接方式的描述。
    
    Returns:
        Dict: 包含关系类型列表的字典
            - relation_types: 知识图谱中存在的所有关系类型名称列表
    
    Raises:
        HTTPException: 当查询过程中发生错误时抛出500异常
    
    业务流程：
    1. 获取数据库连接管理器
    2. 执行Cypher查询，获取所有关系的类型
    3. 将查询结果转换为列表格式
    4. 返回关系类型列表
    5. 异常情况下捕获错误并抛出友好的异常信息
    
    业务意义：
        - 提供知识图谱的关系类型体系概览
        - 支持基于关系类型的查询和过滤
        - 为用户界面提供关系类型选择器的数据
        - 帮助理解知识图谱中实体之间的连接模式
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 查询关系类型 - 获取所有关系的类型名称
        result = db_manager.execute_query("""
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS relation_type
        ORDER BY relation_type
        """)
        
        # 处理查询结果，提取关系类型列表
        # DataFrame处理方式 - 假设返回结果是pandas DataFrame格式
        relation_types = result['relation_type'].tolist() if 'relation_type' in result.columns else []
        
        # 返回格式化的关系类型列表
        return {"relation_types": relation_types}
    except Exception as e:
        # 记录错误信息并抛出异常
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        raise HTTPException(status_code=500, detail=f"获取关系类型失败: {str(e)}")


@router.post("/entities/search")
def search_entities(filters: EntitySearchFilter):
    """
    搜索知识图谱中的实体
    
    该端点提供了基于多种条件的实体搜索功能，支持按类型过滤和关键词搜索，
    允许用户在知识图谱中精确定位所需的实体。这是知识图谱查询和探索的核心功能之一。
    
    Args:
        filters: 实体搜索过滤条件对象，包含以下字段：
            - type: 可选，实体类型过滤
            - term: 可选，关键词搜索条件
            - limit: 返回结果数量限制
    
    Returns:
        Dict: 包含搜索结果的字典
            - entities: 符合搜索条件的实体列表，每个实体包含id、name、type、description等信息
    
    Raises:
        HTTPException: 当搜索过程中发生错误时抛出500异常
    
    业务流程：
    1. 获取数据库连接管理器
    2. 构建搜索条件和参数
    3. 执行Cypher查询，搜索符合条件的实体
    4. 处理查询结果，构建规范化的实体数据结构
    5. 返回格式化的搜索结果
    6. 异常情况下捕获错误并抛出友好的异常信息
    
    业务意义：
        - 提供灵活的实体查询能力
        - 支持基于类型和关键词的精确搜索
        - 为知识图谱的内容发现提供入口
        - 支持用户快速定位和访问特定知识
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 构建查询条件
        conditions = ["e:__Entity__"]  # 基础条件：必须是__Entity__类型
        params = {}  # 查询参数字典
        
        # 如果指定了实体类型过滤条件
        if filters.type:
            conditions.append(f"e:{filters.type}")
        
        # 如果指定了关键词搜索条件
        if filters.term:
            conditions.append("e.id CONTAINS $term")
            params["term"] = filters.term
        
        # 构建完整查询语句
        query = f"""
        MATCH (e)
        WHERE {' AND '.join(conditions)}
        RETURN e.id AS id,
               COALESCE(e.id, '') AS name,
               CASE WHEN size(labels(e)) > 1 
                    THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] 
                    ELSE 'Unknown' 
               END AS type,
               COALESCE(e.description, '') AS description
        LIMIT {filters.limit}
        """
        
        # 执行查询
        result = db_manager.execute_query(query, params)
        
        # 检查结果是否为None
        if result is None:
            return {"entities": []}
            
        # 处理查询结果，构建实体列表
        # DataFrame处理方式 - 假设返回结果是pandas DataFrame格式
        entities = []
        if not result.empty:
            for _, row in result.iterrows():
                # 构建规范化的实体数据结构
                entity = {
                    "id": row['id'] if 'id' in row and row['id'] is not None else '',
                    "name": row['name'] if 'name' in row and row['name'] is not None else '',
                    "type": row['type'] if 'type' in row and row['type'] is not None else 'Unknown',
                    "description": row['description'] if 'description' in row and row['description'] is not None else '',
                    "properties": {}
                }
                entities.append(entity)
        
        # 返回格式化的搜索结果
        return {"entities": entities}
    except Exception as e:
        # 记录错误信息并抛出异常
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        raise HTTPException(status_code=500, detail=f"搜索实体失败: {str(e)}")


@router.post("/relations/search")
def search_relations(filters: RelationSearchFilter):
    """
    搜索知识图谱中的关系
    
    该端点提供了基于多种条件的关系搜索功能，支持按源实体、目标实体和关系类型进行过滤，
    允许用户在知识图谱中查询实体之间的特定连接关系。这对于理解实体间的相互作用和依赖关系至关重要。
    
    Args:
        filters: 关系搜索过滤条件对象，包含以下字段：
            - source: 可选，源实体ID
            - target: 可选，目标实体ID
            - type: 可选，关系类型
            - limit: 返回结果数量限制
    
    Returns:
        Dict: 包含搜索结果的字典
            - relations: 符合搜索条件的关系列表，每个关系包含source、type、target、description、weight等信息
    
    Raises:
        HTTPException: 当搜索过程中发生错误时抛出500异常
    
    业务流程：
    1. 获取数据库连接管理器
    2. 构建搜索条件和参数
    3. 动态生成WHERE子句
    4. 执行Cypher查询，搜索符合条件的关系
    5. 处理查询结果，构建规范化的关系数据结构
    6. 返回格式化的搜索结果
    7. 异常情况下捕获错误并抛出友好的异常信息
    
    业务意义：
        - 提供灵活的关系查询能力
        - 支持基于源实体、目标实体和关系类型的精确搜索
        - 帮助用户理解实体之间的相互连接和影响
        - 支持知识图谱中的路径发现和关系分析
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 构建查询条件
        conditions = []  # 条件列表
        params = {}  # 查询参数字典
        
        # 如果指定了源实体过滤条件
        if filters.source:
            conditions.append("e1.id = $source")
            params["source"] = filters.source
        
        # 如果指定了目标实体过滤条件
        if filters.target:
            conditions.append("e2.id = $target")
            params["target"] = filters.target
        
        # 如果指定了关系类型过滤条件
        if filters.type:
            conditions.append("type(r) = $relType")
            params["relType"] = filters.type
        
        # 构建WHERE子句 - 如果有条件则生成WHERE语句
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # 构建完整查询语句
        query = f"""
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        {where_clause}
        RETURN e1.id AS source,
               type(r) AS type,
               e2.id AS target,
               COALESCE(r.description, '') AS description,
               COALESCE(r.weight, 0.5) AS weight
        LIMIT {filters.limit}
        """
        
        # 执行查询
        result = db_manager.execute_query(query, params)
        
        # 处理查询结果，构建关系列表
        # DataFrame处理方式 - 假设返回结果是pandas DataFrame格式
        relations = []
        if not result.empty:
            for _, row in result.iterrows():
                # 构建规范化的关系数据结构
                relation = {
                    "source": row['source'] if 'source' in row else None,
                    "type": row['type'] if 'type' in row else None,
                    "target": row['target'] if 'target' in row else None,
                    "description": row['description'] if 'description' in row else '',
                    "weight": row['weight'] if 'weight' in row else 0.5,
                    "properties": {}
                }
                relations.append(relation)
        
        # 返回格式化的搜索结果
        return {"relations": relations}
    except Exception as e:
        # 记录错误信息并抛出异常
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        raise HTTPException(status_code=500, detail=f"搜索关系失败: {str(e)}")


@router.post("/entity/create")
def create_entity(entity_data: EntityData):
    """
    创建新的知识图谱实体
    
    该端点提供了向知识图谱中添加新实体的功能，支持设置实体的ID、名称、类型和描述。
    在创建前会先检查实体是否已存在，避免重复创建，确保知识图谱数据的唯一性和一致性。
    
    Args:
        entity_data: 实体数据对象，包含以下字段：
            - id: 实体唯一标识符
            - name: 实体名称
            - type: 实体类型
            - description: 实体描述信息
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True, "id": 实体ID}
            - 失败时: {"success": False, "message": 错误消息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查实体是否已存在
    3. 如果实体已存在，返回失败结果
    4. 如果实体不存在，构建并执行创建实体的Cypher查询
    5. 返回创建结果，包含成功状态和实体ID
    6. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 支持知识图谱的动态扩展和内容添加
        - 确保实体的唯一性和数据一致性
        - 提供标准化的实体创建接口
        - 为知识图谱内容管理提供基础功能
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查实体是否已存在 - 避免重复创建
        check_query = """
        MATCH (e:__Entity__ {id: $id})
        RETURN count(e) AS count
        """
        
        check_result = db_manager.execute_query(check_query, {"id": entity_data.id})
        
        # 检查结果，判断实体是否已存在
        if not check_result.empty and check_result.iloc[0]['count'] > 0:
            return {"success": False, "message": f"实体ID '{entity_data.id}' 已存在"}
        
        # 创建实体，设置基本属性 - 同时添加__Entity__基类和指定的类型标签
        create_query = f"""
        CREATE (e:__Entity__:{entity_data.type} {{
            id: $id,
            name: $name,
            description: $description
        }})
        RETURN e.id AS id
        """
        
        # 准备查询参数
        params = {
            "id": entity_data.id,
            "name": entity_data.name,
            "description": entity_data.description
        }
        
        # 执行创建查询
        result = db_manager.execute_query(create_query, params)
        
        # 处理创建结果
        if not result.empty:
            return {"success": True, "id": result.iloc[0]['id']}
        else:
            return {"success": False, "message": "创建实体失败: 未能获取返回结果"}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"创建实体失败: {str(e)}"}


@router.post("/entity/update")
def update_entity(entity_data: EntityUpdateData):
    """
    更新知识图谱中的实体
    
    该端点提供了灵活的实体更新功能，支持部分字段更新和类型更新。与简单更新不同，此接口允许选择性地更新实体的
    某些属性，而不是强制更新所有字段，更加高效和灵活。同时还支持修改实体的类型标签。
    
    Args:
        entity_data: 实体更新数据对象，所有字段均为可选（除ID外），包含以下字段：
            - id: 实体唯一标识符（必需，用于定位要更新的实体）
            - name: 可选，更新后的实体名称
            - description: 可选，更新后的实体描述信息
            - type: 可选，更新后的实体类型
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True}
            - 失败时: {"success": False, "message": 错误消息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查实体是否存在
    3. 如果实体不存在，返回失败结果
    4. 构建更新参数和SET子句，只包含非空字段
    5. 如果指定了类型更新，先获取当前标签，然后移除旧标签并添加新标签
    6. 执行属性更新查询（如果有属性需要更新）
    7. 返回成功或失败的操作结果
    8. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 提供灵活高效的实体更新机制，支持部分字段更新
        - 允许动态调整实体类型，适应知识结构的变化
        - 确保知识图谱数据的准确性和一致性
        - 为知识图谱的动态维护提供强大支持
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查实体是否存在 - 使用计数查询更高效
        check_query = """
        MATCH (e:__Entity__ {id: $id})
        RETURN count(e) AS count
        """
        
        check_result = db_manager.execute_query(check_query, {"id": entity_data.id})
        
        # 检查结果，判断实体是否存在
        if check_result.empty or check_result.iloc[0]['count'] == 0:
            return {"success": False, "message": f"实体ID '{entity_data.id}' 不存在"}
        
        # 构建更新参数 - 只包含非空字段，实现部分更新
        params = {"id": entity_data.id}
        set_clauses = []
        
        # 处理名称字段更新
        if entity_data.name is not None:
            set_clauses.append("e.name = $name")
            params["name"] = entity_data.name
        
        # 处理描述字段更新
        if entity_data.description is not None:
            set_clauses.append("e.description = $description")
            params["description"] = entity_data.description
        
        # 类型更新处理 - 如果需要更新类型，需要先移除旧标签，再添加新标签
        if entity_data.type is not None:
            # 获取当前实体的标签 - 用于确定需要移除哪些标签
            labels_query = """
            MATCH (e:__Entity__ {id: $id})
            RETURN labels(e) AS labels
            """
            
            labels_result = db_manager.execute_query(labels_query, {"id": entity_data.id})
            
            if not labels_result.empty:
                # 获取当前实体的所有标签
                current_labels = labels_result.iloc[0]['labels']
                
                # 提取需要移除的标签（保留__Entity__基标签）
                remove_labels = [label for label in current_labels if label != "__Entity__"]
                
                # 构建并执行类型更新查询 - 移除旧标签，添加新标签
                update_type_query = f"""
                MATCH (e:__Entity__ {{id: $id}})
                {' '.join(f'REMOVE e:{label}' for label in remove_labels)}
                SET e:{entity_data.type}
                RETURN e.id as id
                """
                
                db_manager.execute_query(update_type_query, {"id": entity_data.id})
        
        # 执行属性更新 - 仅当有属性需要更新时执行
        if set_clauses:
            update_query = f"""
            MATCH (e:__Entity__ {{id: $id}})
            SET {', '.join(set_clauses)}
            RETURN e.id as id
            """
            
            db_manager.execute_query(update_query, params)
        
        # 返回成功结果
        return {"success": True}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"更新实体失败: {str(e)}"}


@router.post("/entity/delete")
def delete_entity(entity_data: EntityDeleteData):
    """
    从知识图谱中删除实体
    
    该端点提供了从知识图谱中删除指定实体的功能。在删除实体前，会先删除与该实体相关的所有关系，
    以维护图数据库的完整性。同时实现了完善的错误检查和异常处理，确保操作的安全性和可靠性。
    
    Args:
        entity_data: 实体删除数据对象，包含以下字段：
            - id: 要删除的实体唯一标识符
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True}
            - 失败时: {"success": False, "message": 详细的错误信息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查实体是否存在（包含多层安全检查）
    3. 如果实体不存在，返回失败结果
    4. 如果实体存在，先删除实体的所有关系
    5. 然后删除实体本身
    6. 返回成功的操作结果
    7. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 支持知识图谱的内容维护和清理
        - 确保删除操作的安全性，避免产生孤立关系
        - 提供标准化的实体删除接口
        - 维护知识图谱的数据完整性和一致性
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查实体是否存在 - 使用计数查询
        check_query = """
        MATCH (e:__Entity__ {id: $id})
        RETURN count(e) AS count
        """
        
        check_result = db_manager.execute_query(check_query, {"id": entity_data.id})
        
        # 多重安全检查 - 第一层：检查结果是否为None
        if check_result is None:
            return {"success": False, "message": "检查实体存在性失败: 查询返回为空"}
            
        # 多重安全检查 - 第二层：检查是否存在计数结果
        if check_result.empty:
            return {"success": False, "message": f"实体ID '{entity_data.id}' 不存在: 查询结果为空"}
            
        # 多重安全检查 - 第三层：安全地访问count值
        count_value = 0
        if 'count' in check_result.columns:
            count_value = check_result.iloc[0]['count']
            
        # 判断实体是否存在
        if count_value == 0:
            return {"success": False, "message": f"实体ID '{entity_data.id}' 不存在"}
        
        # 维护数据完整性 - 先删除实体的所有关系
        delete_rels_query = """
        MATCH (e:__Entity__ {id: $id})-[r]-()
        DELETE r
        """
        
        db_manager.execute_query(delete_rels_query, {"id": entity_data.id})
        
        # 删除实体本身
        delete_query = """
        MATCH (e:__Entity__ {id: $id})
        DELETE e
        """
        
        db_manager.execute_query(delete_query, {"id": entity_data.id})
        
        # 返回成功结果
        return {"success": True}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"删除实体失败: {str(e)}"}


@router.post("/relation/create")
def create_relation(relation_data: RelationData):
    """
    创建新的知识图谱关系
    
    该端点提供了在知识图谱中两个实体之间创建关系的功能，支持设置关系的源实体、目标实体、类型、描述和权重。
    在创建前会先检查关系是否已存在，避免重复创建，确保知识图谱数据的唯一性和一致性。
    
    Args:
        relation_data: 关系数据对象，包含以下字段：
            - source: 源实体ID
            - target: 目标实体ID
            - type: 关系类型
            - description: 关系描述信息
            - weight: 关系权重（可选）
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True, "type": 关系类型}
            - 失败时: {"success": False, "message": 错误消息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查源实体和目标实体是否存在
    3. 检查关系是否已存在
    4. 如果实体存在且关系不存在，构建并执行创建关系的Cypher查询
    5. 返回创建结果，包含成功状态和关系类型
    6. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 支持知识图谱中实体间连接的建立
        - 确保关系的唯一性和数据一致性
        - 提供标准化的关系创建接口
        - 为知识图谱中的知识关联和推理提供基础
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查源实体和目标实体是否存在
        check_query = """
        MATCH (e1:__Entity__ {id: $source})
        MATCH (e2:__Entity__ {id: $target})
        RETURN count(e1) AS source_count, count(e2) AS target_count
        """
        
        check_result = db_manager.execute_query(check_query, {
            "source": relation_data.source,
            "target": relation_data.target
        })
        
        if not check_result.empty:
            if check_result.iloc[0]['source_count'] == 0:
                return {"success": False, "message": f"源实体 '{relation_data.source}' 不存在"}
            
            if check_result.iloc[0]['target_count'] == 0:
                return {"success": False, "message": f"目标实体 '{relation_data.target}' 不存在"}
        else:
            return {"success": False, "message": "无法验证实体存在性"}
        
        # 检查关系是否已存在 - 避免重复创建
        rel_check_query = """
        MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
        WHERE type(r) = $relType
        RETURN count(r) AS rel_count
        """
        
        rel_check_result = db_manager.execute_query(rel_check_query, {
            "source": relation_data.source,
            "target": relation_data.target,
            "relType": relation_data.type
        })
        
        if not rel_check_result.empty and rel_check_result.iloc[0]['rel_count'] > 0:
            return {"success": False, "message": f"关系 '{relation_data.source} -[{relation_data.type}]-> {relation_data.target}' 已存在"}
        
        # 创建关系 - 首先匹配源实体和目标实体，然后在它们之间创建关系
        create_query = f"""
        MATCH (e1:__Entity__ {{id: $source}})
        MATCH (e2:__Entity__ {{id: $target}})
        CREATE (e1)-[r:{relation_data.type} {{
            description: $description,
            weight: $weight
        }}]->(e2)
        RETURN type(r) AS type
        """
        
        params = {
            "source": relation_data.source,
            "target": relation_data.target,
            "description": relation_data.description,
            "weight": relation_data.weight if relation_data.weight is not None else 0.5
        }
        
        # 执行创建查询
        result = db_manager.execute_query(create_query, params)
        
        # 处理创建结果
        if not result.empty:
            return {"success": True, "type": result.iloc[0]['type']}
        else:
            return {"success": False, "message": "创建关系失败: 未能获取返回结果"}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"创建关系失败: {str(e)}"}


@router.post("/relation/update")
def update_relation(relation_data: RelationUpdateData):
    """
    更新知识图谱中的关系
    
    该端点提供了全面的关系更新功能，不仅支持部分字段属性更新，还支持关系类型的更改。
    由于Neo4j中关系类型无法直接修改，更改类型时采用"先删除后创建"的策略，确保数据的一致性和完整性。
    该功能设计灵活，能够满足知识图谱关系数据动态调整的需求。
    
    Args:
        relation_data: 关系更新数据对象，包含以下字段：
            - source: 源实体ID（必需）
            - target: 目标实体ID（必需）
            - original_type: 原始关系类型（必需，用于定位要更新的关系）
            - new_type: 可选，新的关系类型，如果需要更改关系类型时使用
            - description: 可选，更新后的关系描述
            - weight: 可选，更新后的关系权重
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True}
            - 失败时: {"success": False, "message": 错误消息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查指定的关系是否存在
    3. 如果关系不存在，返回失败结果
    4. 根据是否需要更改关系类型，分为两条处理路径：
       a. 更改关系类型路径：
          i. 获取当前关系的所有属性
          ii. 删除旧关系
          iii. 准备新关系属性（保留原属性或使用新提供的属性）
          iv. 创建新类型的关系
       b. 仅更新属性路径：
          i. 构建更新参数和SET子句，只包含非空字段
          ii. 执行关系属性更新查询
    5. 返回成功的操作结果
    6. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 提供全面的关系更新能力，包括属性更新和类型更改
        - 确保关系数据的准确性和时效性
        - 支持知识图谱的动态维护和结构调整
        - 为知识图谱中的关系语义演进提供技术支持
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查关系是否存在 - 通过源实体、目标实体和原始关系类型精确定位
        check_query = """
        MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
        WHERE type(r) = $relType
        RETURN count(r) AS count
        """
        
        check_result = db_manager.execute_query(check_query, {
            "source": relation_data.source,
            "target": relation_data.target,
            "relType": relation_data.original_type
        })
        
        # 检查结果，判断关系是否存在
        if check_result.empty or check_result.iloc[0]['count'] == 0:
            return {"success": False, "message": f"关系 '{relation_data.source} -[{relation_data.original_type}]-> {relation_data.target}' 不存在"}
        
        # 分支处理 - 更改关系类型路径
        # 由于Neo4j不支持直接修改关系类型，需要采用删除重建策略
        if relation_data.new_type and relation_data.new_type != relation_data.original_type:
            # 获取当前关系的所有属性 - 保留原有属性信息
            get_props_query = """
            MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
            WHERE type(r) = $relType
            RETURN r.description AS description,
                   r.weight AS weight
            """
            
            props_result = db_manager.execute_query(get_props_query, {
                "source": relation_data.source,
                "target": relation_data.target,
                "relType": relation_data.original_type
            })
            
            # 删除旧关系 - 准备创建新类型关系
            delete_query = """
            MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
            WHERE type(r) = $relType
            DELETE r
            """
            
            db_manager.execute_query(delete_query, {
                "source": relation_data.source,
                "target": relation_data.target,
                "relType": relation_data.original_type
            })
            
            # 处理新旧属性合并
            if not props_result.empty:
                props = props_result.iloc[0]
                
                # 属性合并逻辑：优先使用新提供的属性，否则保留原属性，最后使用默认值
                description = relation_data.description if relation_data.description is not None else props['description'] if 'description' in props else ''
                weight = relation_data.weight if relation_data.weight is not None else props['weight'] if 'weight' in props else 0.5
                
                # 创建新关系 - 使用新的关系类型和合并后的属性
                create_query = f"""
                MATCH (e1:__Entity__ {{id: $source}})
                MATCH (e2:__Entity__ {{id: $target}})
                CREATE (e1)-[r:{relation_data.new_type} {{
                    description: $description,
                    weight: $weight
                }}]->(e2)
                RETURN type(r) AS type
                """
                
                db_manager.execute_query(create_query, {
                    "source": relation_data.source,
                    "target": relation_data.target,
                    "description": description,
                    "weight": weight
                })
            else:
                # 后备方案 - 如果没有获取到原属性，使用新提供的属性或默认值
                create_query = f"""
                MATCH (e1:__Entity__ {{id: $source}})
                MATCH (e2:__Entity__ {{id: $target}})
                CREATE (e1)-[r:{relation_data.new_type} {{
                    description: $description,
                    weight: $weight
                }}]->(e2)
                RETURN type(r) AS type
                """
                
                db_manager.execute_query(create_query, {
                    "source": relation_data.source,
                    "target": relation_data.target,
                    "description": relation_data.description or '',
                    "weight": relation_data.weight or 0.5
                })
        else:
            # 分支处理 - 仅更新属性路径
            # 构建更新参数 - 只包含非空字段，实现部分更新
            params = {
                "source": relation_data.source,
                "target": relation_data.target,
                "relType": relation_data.original_type
            }
            set_clauses = []
            
            # 处理描述字段更新
            if relation_data.description is not None:
                set_clauses.append("r.description = $description")
                params["description"] = relation_data.description
            
            # 处理权重字段更新
            if relation_data.weight is not None:
                set_clauses.append("r.weight = $weight")
                params["weight"] = relation_data.weight
            
            # 执行属性更新 - 仅当有属性需要更新时执行
            if set_clauses:
                update_query = f"""
                MATCH (e1:__Entity__ {{id: $source}})-[r]->(e2:__Entity__ {{id: $target}})
                WHERE type(r) = $relType
                SET {', '.join(set_clauses)}
                RETURN type(r) as type
                """
                
                db_manager.execute_query(update_query, params)
        
        # 返回成功结果
        return {"success": True}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"更新关系失败: {str(e)}"}


@router.post("/relation/delete")
def delete_relation(relation_data: RelationDeleteData):
    """
    从知识图谱中删除指定关系
    
    该端点提供了精确删除知识图谱中特定关系的功能。与删除实体不同，删除关系不会影响相关的实体，
    只会移除实体之间的特定类型连接。这个操作对于调整知识图谱中的关系结构和维护数据准确性非常重要。
    
    Args:
        relation_data: 关系删除数据对象，包含以下字段：
            - source: 源实体ID（必需）
            - target: 目标实体ID（必需）
            - type: 关系类型（必需）
    
    Returns:
        Dict: 操作结果
            - 成功时: {"success": True}
            - 失败时: {"success": False, "message": 详细的错误信息}
    
    业务流程：
    1. 获取数据库连接管理器
    2. 检查指定的关系是否存在
    3. 如果关系不存在，返回失败结果
    4. 如果关系存在，执行删除操作
    5. 返回成功的操作结果
    6. 异常情况下捕获错误并返回友好的错误信息
    
    业务意义：
        - 支持知识图谱中关系结构的精确调整
        - 允许移除错误或过时的知识关联
        - 提供标准化的关系删除接口
        - 在保持实体完整性的同时维护关系网络的准确性
    """
    # 获取数据库连接管理器
    db_manager = get_db_manager()
    try:
        # 检查关系是否存在 - 通过源实体、目标实体和关系类型精确定位
        check_query = """
        MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
        WHERE type(r) = $relType
        RETURN count(r) AS count
        """
        
        check_result = db_manager.execute_query(check_query, {
            "source": relation_data.source,
            "target": relation_data.target,
            "relType": relation_data.type
        })
        
        # 检查结果，判断关系是否存在
        if check_result.empty or check_result.iloc[0]['count'] == 0:
            return {"success": False, "message": f"关系 '{relation_data.source} -[{relation_data.type}]-> {relation_data.target}' 不存在"}
        
        # 执行关系删除操作 - 仅删除指定的关系，不影响相关实体
        delete_query = """
        MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target})
        WHERE type(r) = $relType
        DELETE r
        """
        
        db_manager.execute_query(delete_query, {
            "source": relation_data.source,
            "target": relation_data.target,
            "relType": relation_data.type
        })
        
        # 返回成功结果
        return {"success": True}
    except Exception as e:
        # 记录错误信息并返回友好的错误消息
        print(e)
        traceback.print_exc()  # 打印完整堆栈用于调试
        return {"success": False, "message": f"删除关系失败: {str(e)}"}