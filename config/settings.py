from pathlib import Path

# ===== 基础配置 =====
# 
# 这部分包含系统运行的基础路径和核心参数设置，是整个系统的基础设施配置。
# 所有路径都是基于项目根目录构建的相对路径，确保系统在不同环境中都能正常工作。
# 这些配置构成了系统运行的基础框架，其他模块依赖于此部分配置。

# 基础路径设置
# BASE_DIR: 项目根目录，用于构建其他相对路径，确保路径引用的一致性
# FILES_DIR: 文件存储目录，存放待处理的文档，所有输入数据源的统一入口
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / 'files'

# 知识库主题设置，用于deepsearch（reasoning提示词）
# 定义了整个知识库的领域范围，指导系统的语义理解方向
KB_NAME = "华东理工大学"

# 系统运行参数
# workers: fastapi并发进程数，控制API服务器的并发处理能力
# 增加worker数量可以提高并发处理能力，但会增加系统资源消耗
workers = 2

# ===== 知识图谱配置 =====
# 
# 这部分配置定义了知识图谱的构建规则和内容范围，是整个图RAG系统的核心领域定义。
# 通过实体类型和关系类型的定义，指导系统如何从文本中提取结构化知识。
# 这些配置直接影响知识图谱的质量和图RAG的检索效果。

# 知识图谱主题设置，定义整个知识图谱的领域范围
# 明确的主题有助于指导实体和关系提取的准确性和相关性
theme = "华东理工大学学生管理"

# 知识图谱实体类型定义，用于从文本中提取实体
# 实体是知识图谱的基本节点，每种类型代表特定领域概念
# 精心设计的实体类型集能够有效捕捉领域知识的核心要素
entity_types = [
    "学生类型",       # 如：本科生、研究生等 - 学生身份分类
    "奖学金类型",     # 如：国家奖学金、校级奖学金等 - 奖励体系分类
    "处分类型",       # 如：警告、严重警告、记过等 - 纪律处分分类
    "部门",          # 如：学生处、教务处等 - 管理机构分类
    "学生职责",       # 如：按时上课、完成作业等 - 学生义务定义
    "管理规定"        # 如：考勤规定、奖学金评定规定等 - 规则制度分类
]

# 知识图谱关系类型定义，用于从文本中提取实体间关系
# 关系连接实体节点，形成图结构，表达概念间的关联
# 丰富的关系类型能够捕获实体间复杂的语义关联
relationship_types = [
    "申请",         # 实体间的申请关系 - 如学生申请奖学金
    "评选",         # 实体间的评选关系 - 如部门评选优秀学生
    "违纪",         # 实体间的违纪关系 - 如学生违反管理规定
    "资助",         # 实体间的资助关系 - 如奖学金资助学生
    "申诉",         # 实体间的申诉关系 - 如学生对处分进行申诉
    "管理",         # 实体间的管理关系 - 如部门管理学生
    "权利义务",      # 实体间的权利义务关系 - 如学生享有权利履行义务
    "互斥",         # 实体间的互斥关系 - 如有奖与处分互斥
]

# 冲突解决与更新策略
# 定义了当自动更新与手动编辑发生冲突时的处理方式
# manual_first: 优先保留手动编辑的内容 - 适用于需要人工校正的场景
# auto_first: 优先使用自动更新的内容 - 适用于频繁更新的内容
# merge: 尝试合并手动和自动更新的内容 - 适用于复杂的更新场景
conflict_strategy = "manual_first"

# 社区检测算法配置
# 社区检测用于发现知识图谱中的紧密相关实体群组
# 可选值：'leiden'或'sllpa'
# 注：sllpa如果发现不了社区，换成leiden效果可能会更好
# Leiden算法通常在大型图上表现更好，而SLLPA在特定结构的图上可能更高效
community_algorithm = 'leiden'

# ===== 文本处理配置 =====
# 
# 这部分配置控制文本的预处理、分块和相似度计算等关键参数，影响知识提取的质量。
# 文本分块大小和重叠设置需要根据文档特点和模型上下文窗口进行优化。
# 这些参数直接影响检索质量和系统性能，需要根据具体应用场景进行调优。

# 文本处理参数
# CHUNK_SIZE: 文本分块大小（字符数），控制每个文本片段的长度
# 太小会导致上下文不完整，太大则可能超出模型上下文窗口
CHUNK_SIZE = 500

# OVERLAP: 相邻块之间的重叠大小（字符数），确保上下文连续性
# 适当的重叠可以避免重要信息被分块边界切断
OVERLAP = 100

# MAX_TEXT_LENGTH: 最大文本长度限制，防止处理过大的文档
# 避免内存溢出和处理超时，提高系统稳定性
MAX_TEXT_LENGTH = 500000

# similarity_threshold: 相似度阈值，用于判断文本重复
# 阈值越高，认为重复的条件越严格；越低则越宽松
similarity_threshold = 0.9

# 回答生成配置
# response_type: 回答格式类型，指导输出格式
# 控制回答的组织结构和呈现方式
response_type = "多个段落"  # 回答格式类型，指导输出格式

# ===== Agent工具配置 =====
# 
# 这部分配置定义了系统中各种Agent工具的功能描述和使用场景，
# 用于向LLM介绍工具能力，使LLM能够正确选择和使用工具。
# 这些描述是LLM工具调用的重要依据，影响决策质量。

# Agent工具描述，用于向用户介绍各工具的功能和适用场景
# lc_description: 局部搜索工具描述，适用于具体细节查询
lc_description = "用于需要具体细节的查询。检索华东理工大学学生管理文件中的具体规定、条款、流程等详细内容。适用于'某个具体规定是什么'、'处理流程如何'等问题。"

# gl_description: 全局搜索工具描述，适用于总结归纳查询
gl_description = "用于需要总结归纳的查询。分析华东理工大学学生管理体系的整体框架、管理原则、学生权利义务等宏观内容。适用于'学校的学生管理总体思路'、'学生权益保护机制'等需要系统性分析的问题。"

# naive_description: 基础检索工具描述，适用于快速简单查询
naive_description = "基础检索工具，直接查找与问题最相关的文本片段，不做复杂分析。快速获取华东理工大学相关政策，返回最匹配的原文段落。"

# 前端示例问题，提供给用户参考的常见问题
# 这些示例帮助用户了解系统的查询能力和表达方式
# 同时也作为测试系统功能的基准问题
# 示例问题覆盖了不同类型的查询，包括事实性问题、关系问题等
examples = [
    "旷课多少学时会被退学？",      # 具体规则查询
    "国家奖学金和国家励志奖学金互斥吗？",  # 关系查询
    "优秀学生要怎么申请？",       # 流程查询
    "那上海市奖学金呢？",         # 上下文相关查询
]

# ===== 性能优化配置 =====
# 
# 这部分配置控制系统的并行处理能力和批处理大小，
# 是系统性能调优的关键参数，需要根据硬件资源进行合理配置。
# 合理的并行和批处理配置能够充分利用系统资源，提高处理效率。

# 并行处理配置
# MAX_WORKERS: 并行工作线程数，控制并发处理能力
# 增加线程数可以提高并行度，但会增加CPU和内存消耗
MAX_WORKERS = 4

# 批处理配置，控制一次处理的数据量
# 批处理大小需要平衡处理效率和内存消耗
BATCH_SIZE = 100               # 通用批处理大小
ENTITY_BATCH_SIZE = 50         # 实体处理批次大小
CHUNK_BATCH_SIZE = 100         # 文本块处理批次大小
EMBEDDING_BATCH_SIZE = 64      # 嵌入向量计算批次大小
LLM_BATCH_SIZE = 5             # 大语言模型处理批次大小

# 索引和社区检测配置
COMMUNITY_BATCH_SIZE = 50      # 社区处理批次大小

# GDS(Graph Data Science)相关配置，用于Neo4j图数据分析
# 这些配置影响图算法的执行效率和资源消耗
GDS_MEMORY_LIMIT = 6           # GDS内存限制(GB)，控制图算法使用的最大内存
GDS_CONCURRENCY = 4            # GDS并发度，控制图算法的并行执行度
GDS_NODE_COUNT_LIMIT = 50000   # GDS节点数量限制，防止处理过大的图
GDS_TIMEOUT_SECONDS = 300      # GDS超时时间(秒)，防止长时间运行的算法阻塞系统

# ===== 搜索模块配置 =====
# 
# 这部分配置定义了系统的搜索策略，包括局部搜索和全局搜索两种主要模式，
# 通过Cypher查询模板从Neo4j图数据库中检索相关信息。
# 这些配置直接影响检索结果的质量和覆盖范围。

# 本地搜索配置，用于基于图的局部搜索
# 主要针对与查询直接相关的实体和关系进行深入检索
LOCAL_SEARCH_CONFIG = {
    # 向量检索参数，控制检索结果的数量和质量
    "top_entities": 10,           # 返回的顶级实体数量 - 控制实体召回数量
    "top_chunks": 10,             # 返回的顶级文本块数量 - 控制文本召回数量
    "top_communities": 2,         # 返回的顶级社区数量 - 控制社区召回数量
    "top_outside_rels": 10,       # 返回的外部关系数量 - 控制跨社区关系召回
    "top_inside_rels": 10,        # 返回的内部关系数量 - 控制社区内关系召回

    # 索引配置，指定使用的向量索引
    "index_name": "vector",      # 向量索引名称，用于快速检索
    "response_type": response_type,  # 响应类型，保持与系统配置一致

    # 检索查询模板，使用Cypher查询语言从图数据库中检索相关信息
    # 这个Cypher查询是本地搜索的核心，它定义了如何从图中提取相关信息
    "retrieval_query": """
    WITH collect(node) as nodes
    WITH
    collect {  # 收集与实体相关的文本块
        UNWIND nodes as n
        MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
        WITH distinct c, count(distinct n) as freq
        RETURN {id:c.id, text: c.text} AS chunkText
        ORDER BY freq DESC
        LIMIT $topChunks
    } AS text_mapping,
    collect {  # 收集实体所在的社区信息
        UNWIND nodes as n
        MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
        WITH distinct c, c.community_rank as rank, c.weight AS weight
        RETURN c.summary
        ORDER BY rank, weight DESC
        LIMIT $topCommunities
    } AS report_mapping,
    collect {  # 收集实体与外部实体的关系
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__)
        WHERE NOT m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC
        LIMIT $topOutsideRels
    } as outsideRels,
    collect {  # 收集实体与内部实体的关系
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__)
        WHERE m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC
        LIMIT $topInsideRels
    } as insideRels,
    collect {  # 收集实体描述
        UNWIND nodes as n
        RETURN n.description AS descriptionText
    } as entities
    RETURN {
        Chunks: text_mapping,      # 相关文本块
        Reports: report_mapping,   # 社区报告
        Relationships: outsideRels + insideRels,  # 所有关系
        Entities: entities         # 实体信息
    } AS text, 1.0 AS score, {} AS metadata
    """,
}

# 全局搜索配置，用于基于图的全局搜索
# 主要针对整个知识图谱进行广度检索，发现潜在关联
GLOBAL_SEARCH_CONFIG = {
    # 社区层级配置，控制搜索的广度和深度
    "default_level": 0,  # 层级0，从最高层级开始 - 从全局到局部的搜索策略
    "response_type": response_type,

    # 批处理配置，控制搜索过程的资源消耗
    "batch_size": 10,          # 批处理大小 - 每次处理的社区数量
    "max_communities": 100,    # 最大社区数量 - 限制全局搜索的范围
}

# ===== 缓存配置 =====
# 
# 这部分配置控制系统的缓存策略，通过多级缓存机制优化搜索性能，
# 减少重复计算和数据库访问，提升响应速度。
# 缓存是系统性能优化的重要手段，合理配置缓存可以显著提升用户体验。

# 搜索缓存配置，用于优化搜索性能
SEARCH_CACHE_CONFIG = {
    # 缓存目录配置，定义缓存文件的存储位置
    "base_cache_dir": "./cache",                       # 基础缓存目录
    "local_search_cache_dir": "./cache/local_search",  # 本地搜索缓存目录 - 存储局部搜索结果
    "global_search_cache_dir": "./cache/global_search", # 全局搜索缓存目录 - 存储全局搜索结果
    "deep_research_cache_dir": "./cache/deep_research", # 深度研究缓存目录 - 存储复杂推理结果

    # 缓存策略配置，控制缓存的大小和有效期
    "max_cache_size": 200,     # 最大缓存项数量 - 防止缓存无限增长
    "cache_ttl": 3600,         # 缓存有效期（秒），1小时 - 平衡数据新鲜度和性能

    # 缓存类型配置，控制使用的缓存方式
    "memory_cache_enabled": True,  # 启用内存缓存 - 提供最快的访问速度
    "disk_cache_enabled": True,    # 启用磁盘缓存 - 提供持久化和更大的缓存容量
}

# ===== 推理配置 =====
# 
# 这部分配置控制系统的深度推理能力，通过迭代思考和证据收集，
# 实现复杂问题的分析和解决。是系统智能性的核心配置。
# 这些参数影响系统的推理深度、广度和质量。

# 推理引擎配置，控制深度搜索和推理过程
REASONING_CONFIG = {
    # 迭代配置，控制推理的次数和范围
    "max_iterations": 5,       # 最大迭代次数 - 控制推理的深度
    "max_search_limit": 10,    # 最大搜索限制 - 控制每次迭代的搜索范围

    # 思考引擎配置，控制思考的深度和广度
    "thinking_depth": 3,       # 思考深度 - 控制推理的深入程度
    "exploration_width": 3,    # 探索广度 - 控制并行考虑的思路数量
    "max_exploration_steps": 5, # 最大探索步骤数 - 控制单个思路的探索长度

    # 证据链配置，控制证据收集和评估
    "max_evidence_items": 50,  # 最大证据项数量 - 控制收集的证据数量上限
    "evidence_relevance_threshold": 0.7,  # 证据相关性阈值 - 过滤低相关性证据

    # 验证配置，控制系统对推理结果的自我验证
    "validation": {
        "enable_answer_validation": True,  # 启用答案验证 - 检查答案的合理性
        "validation_threshold": 0.8,       # 验证阈值 - 答案必须达到的置信度
        "enable_complexity_estimation": True,  # 启用复杂度估计 - 评估问题难度
        "consistency_threshold": 0.7       # 一致性阈值 - 检查推理过程的一致性
    },

    # 探索配置，控制知识空间的探索策略
    "exploration": {
        "max_exploration_steps": 5,       # 最大探索步骤数 - 控制单个路径的探索长度
        "exploration_depth": 3,           # 探索深度 - 控制知识网络的探索深度
        "exploration_breadth": 3,         # 探索广度 - 控制分支探索的数量
        "exploration_width": 3,           # 探索宽度 - 控制每个层次的探索范围
        "relevance_threshold": 0.5,       # 相关性阈值 - 指导探索方向选择
        "exploration_decay_factor": 0.8,  # 探索衰减因子 - 控制探索范围的收缩速度
        "enable_backtracking": True       # 启用回溯 - 允许在遇到死胡同后返回重试
    }
}