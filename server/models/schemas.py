"""
API数据模型定义模块

该模块使用Pydantic库定义了系统中所有API的请求和响应数据模型。
作为前后端数据交互的契约，这些模型确保了数据结构的一致性和类型安全。
同时，Pydantic还提供了自动的数据验证、转换和API文档生成功能。

主要功能：
- 定义所有API端点的数据结构
- 提供数据验证和类型检查
- 支持自动API文档生成
- 确保前后端数据交换的一致性

设计特点：
- 采用层次化模型设计，符合RESTful API最佳实践
- 使用Python类型注解，提高代码可读性
- 提供合理的默认值和可选字段设置
- 严格的数据验证，确保系统稳定性
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from config.settings import community_algorithm


class ChatRequest(BaseModel):
    """
    聊天请求模型
    
    定义了与聊天系统交互的请求数据结构，包含用户输入、会话标识和系统配置参数。
    该模型是用户与系统交互的主要入口，通过不同的配置参数可以控制响应方式和行为。
    
    字段说明：
        message: 用户发送的消息内容，系统将针对此内容生成响应
        session_id: 会话唯一标识，用于追踪和管理聊天历史
        debug: 是否开启调试模式，开启后会返回详细的执行日志
        agent_type: 使用的代理类型，默认为简单RAG代理
        use_deeper_tool: 是否使用深度搜索工具，默认为True启用
        show_thinking: 是否显示思考过程，默认为False
        
    业务意义：
        - 提供灵活的聊天配置选项，满足不同场景需求
        - 支持会话管理，实现多轮对话能力
        - 允许用户选择不同的代理和搜索策略
        - 提供调试功能，方便开发和问题排查
    """
    message: str
    session_id: str
    debug: bool = False
    agent_type: str = "naive_rag_agent"
    use_deeper_tool: Optional[bool] = True
    show_thinking: Optional[bool] = False


class ChatResponse(BaseModel):
    """
    聊天响应模型
    
    定义了聊天系统返回的响应数据结构，包含回答内容和各种附加信息。
    该模型提供了丰富的元数据和执行信息，便于前端展示和问题排查。
    
    字段说明：
        answer: 系统生成的回答内容，直接回应用户查询
        execution_log: 可选的执行日志，包含系统内部处理过程
        kg_data: 可选的知识图谱数据，包含与回答相关的实体和关系
        reference: 可选的引用信息，标明回答的来源和依据
        iterations: 可选的迭代信息，记录深度搜索的多轮迭代过程
        
    业务意义：
        - 提供结构化的响应数据，便于前端渲染
        - 支持调试和可解释性，展示系统内部工作原理
        - 包含引用和溯源信息，增强回答可信度
        - 支持知识图谱可视化，提供更丰富的信息展示
    """
    answer: str
    execution_log: Optional[List[Dict]] = None
    kg_data: Optional[Dict] = None
    reference: Optional[Dict] = None
    iterations: Optional[List[Dict]] = None


class SourceRequest(BaseModel):
    """
    源内容请求模型
    
    定义了获取源内容的请求数据结构，通过source_id标识需要获取的内容。
    该模型用于查询特定内容块的原始文本，支持引用查看和内容验证。
    
    字段说明：
        source_id: 源内容的唯一标识符，用于定位特定内容块
        
    业务意义：
        - 支持内容溯源，允许用户查看生成答案所引用的原始内容
        - 提供内容验证机制，增强系统透明度
        - 支持用户查看完整上下文，提高信息理解
    """
    source_id: str


class SourceResponse(BaseModel):
    """
    源内容响应模型
    
    定义了源内容查询的响应数据结构，返回请求的原始内容。
    该模型为用户提供透明的内容溯源机制，展示系统答案的依据。
    
    字段说明：
        content: 请求的源内容文本，包含完整的原始文本
        
    业务意义：
        - 提供内容溯源功能，增强系统透明度和可信度
        - 支持用户验证回答的准确性和可靠性
        - 帮助用户获取更完整的上下文信息
    """
    content: str


class SourceInfoResponse(BaseModel):
    """
    源文件信息响应模型
    
    定义了源文件信息的响应数据结构，返回源文件的基本信息。
    该模型用于向用户展示内容来源于哪个文件，提供更完整的溯源信息。
    
    字段说明：
        file_name: 源文件的名称，标识内容的原始来源
        
    业务意义：
        - 提供更详细的内容溯源信息
        - 帮助用户了解信息的来源和权威性
        - 支持知识库管理和内容维护
    """
    file_name: str


class ClearRequest(BaseModel):
    """
    清除聊天历史请求模型
    
    定义了清除特定会话聊天历史的请求数据结构。
    该模型用于支持用户管理会话数据，保护隐私和重置对话状态。
    
    字段说明：
        session_id: 需要清除聊天历史的会话ID
        
    业务意义：
        - 支持会话管理和隐私保护
        - 允许用户重置对话上下文
        - 提供灵活的会话控制能力
    """
    session_id: str


class ClearResponse(BaseModel):
    """
    清除聊天历史响应模型
    
    定义了清除聊天历史操作的响应数据结构，返回操作状态和剩余消息信息。
    该模型用于向用户确认清除操作的执行结果。
    
    字段说明：
        status: 操作状态，如"success"或"error"
        remaining_messages: 可选的剩余消息数信息
        
    业务意义：
        - 提供操作确认反馈
        - 支持会话状态管理
        - 帮助用户了解系统的操作结果
    """
    status: str
    remaining_messages: Optional[str] = None


class FeedbackRequest(BaseModel):
    """
    反馈请求模型
    
    定义了用户反馈的请求数据结构，包含用户对回答的评价和相关上下文信息。
    该模型用于收集用户反馈，支持系统改进和优化。
    
    字段说明：
        message_id: 消息的唯一标识符
        query: 用户的原始查询
        is_positive: 是否为正面反馈
        thread_id: 对话线程ID
        agent_type: 用于生成回答的代理类型
        
    业务意义：
        - 收集用户对系统回答的评价
        - 支持系统性能监控和质量改进
        - 提供用户参与系统优化的渠道
        - 为模型调优提供真实用户反馈数据
    """
    message_id: str
    query: str
    is_positive: bool
    thread_id: str
    agent_type: Optional[str] = "naive_rag_agent"


class FeedbackResponse(BaseModel):
    """
    反馈响应模型
    
    定义了用户反馈操作的响应数据结构，返回反馈处理状态和执行的操作。
    该模型用于向用户确认反馈已被接收和处理。
    
    字段说明：
        status: 反馈处理状态
        action: 系统执行的反馈处理操作
        
    业务意义：
        - 提供反馈确认，提升用户体验
        - 记录系统对反馈的处理方式
        - 支持反馈闭环管理
    """
    status: str
    action: str

class SourceInfoBatchRequest(BaseModel):
    """
    批量源信息请求模型
    
    定义了批量获取源文件信息的请求数据结构，通过多个source_ids批量查询。
    该模型用于高效获取多个内容块的源文件信息。
    
    字段说明：
        source_ids: 源内容ID列表
        
    业务意义：
        - 支持批量操作，提高系统效率
        - 减少API调用次数，优化网络性能
        - 便于前端一次性加载多个源文件信息
    """
    source_ids: List[str]

class ContentBatchRequest(BaseModel):
    """
    批量内容请求模型
    
    定义了批量获取内容块的请求数据结构，通过多个chunk_ids批量查询。
    该模型用于高效获取多个内容块的原始文本。
    
    字段说明：
        chunk_ids: 内容块ID列表
        
    业务意义：
        - 支持批量操作，提高系统效率
        - 减少API调用次数，优化网络性能
        - 便于前端一次性加载多个内容块
    """
    chunk_ids: List[str]

class ReasoningRequest(BaseModel):
    """
    推理请求模型
    
    定义了知识图谱推理的请求数据结构，包含推理类型、实体和算法参数。
    该模型用于支持基于知识图谱的复杂推理和关系发现。
    
    字段说明：
        reasoning_type: 推理类型，定义要执行的推理操作
        entity_a: 第一个实体
        entity_b: 可选的第二个实体
        max_depth: 最大搜索深度，默认为3
        algorithm: 使用的社区检测算法，默认为配置文件中的设置
        
    业务意义：
        - 支持基于知识图谱的复杂推理
        - 提供灵活的推理配置选项
        - 允许发现实体间的隐含关系
        - 支持多种算法选择，适应不同推理需求
    """
    reasoning_type: str
    entity_a: str
    entity_b: Optional[str] = None
    max_depth: Optional[int] = 3
    algorithm: Optional[str] = community_algorithm

class EntityData(BaseModel):
    """
    实体数据模型
    
    定义了知识图谱中实体的数据结构，包含实体的标识、名称、类型和属性。
    该模型用于知识图谱的实体表示和操作。
    
    字段说明：
        id: 实体的唯一标识符
        name: 实体名称
        type: 实体类型
        description: 可选的实体描述
        properties: 可选的实体属性字典
        
    业务意义：
        - 定义知识图谱的基本组成单元
        - 支持实体的创建和查询
        - 提供丰富的实体属性，增强语义表达
    """
    id: str
    name: str
    type: str
    description: Optional[str] = ""
    properties: Optional[Dict[str, Any]] = {}

class EntityUpdateData(BaseModel):
    """
    实体更新数据模型
    
    定义了知识图谱实体更新的数据结构，所有字段均为可选，只更新提供的字段。
    该模型用于灵活地更新实体的部分属性。
    
    字段说明：
        id: 实体的唯一标识符（必填，用于定位实体）
        name: 可选的新实体名称
        type: 可选的新实体类型
        description: 可选的新实体描述
        properties: 可选的新实体属性字典
        
    业务意义：
        - 支持实体的部分更新
        - 提供灵活的更新机制
        - 减少不必要的数据传输
        - 支持增量式的图谱维护
    """
    id: str
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class EntitySearchFilter(BaseModel):
    """
    实体搜索过滤模型
    
    定义了实体搜索的过滤条件，支持按关键词、类型和数量限制进行搜索。
    该模型用于支持知识图谱的高效实体查询。
    
    字段说明：
        term: 可选的搜索关键词
        type: 可选的实体类型过滤
        limit: 可选的结果数量限制，默认为100
        
    业务意义：
        - 支持灵活的实体搜索
        - 提供精确的查询条件
        - 防止结果过多导致的性能问题
        - 优化搜索结果的相关性和数量
    """
    term: Optional[str] = None
    type: Optional[str] = None
    limit: Optional[int] = 100

class RelationData(BaseModel):
    """
    关系数据模型
    
    定义了知识图谱中关系的数据结构，包含关系的源实体、关系类型和目标实体。
    该模型用于表示实体间的连接和关联。
    
    字段说明：
        source: 关系的源实体ID
        type: 关系类型
        target: 关系的目标实体ID
        
    业务意义：
        - 定义知识图谱中的实体连接
        - 支持关系的创建和查询
        - 表示实体间的语义关联
        - 为推理和路径分析提供基础
    """
    source: str
    type: str
    target: str
    description: Optional[str] = ""
    weight: Optional[float] = 0.5
    properties: Optional[Dict[str, Any]] = {}

class RelationUpdateData(BaseModel):
    """
    关系更新数据模型
    
    定义了知识图谱关系更新的数据结构，所有非ID字段均为可选，支持部分字段更新和关系类型更改。
    该模型设计灵活，既可以更新关系的简单属性，也可以更改关系的类型，满足不同的更新需求。
    
    字段说明：
        source: 源实体ID（必填，用于定位关系）
        original_type: 原始关系类型（必填，用于定位关系）
        target: 目标实体ID（必填，用于定位关系）
        new_type: 可选的新关系类型
        description: 可选的新关系描述
        weight: 可选的新关系权重
        properties: 可选的新关系属性字典
    
    业务意义：
        - 支持关系的部分属性更新
        - 允许更改关系类型，适应知识结构的变化
        - 提供灵活的关系维护机制
        - 支持增量式的图谱更新
    """
    source: str
    original_type: str
    target: str
    new_type: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None

class RelationSearchFilter(BaseModel):
    """
    关系搜索过滤模型
    
    定义了关系搜索的过滤条件，支持按源实体、目标实体和关系类型进行过滤。
    该模型用于灵活地在知识图谱中查询特定的关系连接。
    
    字段说明：
        source: 可选的源实体ID过滤条件
        target: 可选的目标实体ID过滤条件
        type: 可选的关系类型过滤条件
        limit: 可选的结果数量限制，默认为100
    
    业务意义：
        - 支持灵活的关系搜索和过滤
        - 允许基于实体和类型的精确查询
        - 优化搜索性能，避免结果过多
        - 支持知识图谱中的关系发现
    """
    source: Optional[str] = None
    target: Optional[str] = None
    type: Optional[str] = None
    limit: Optional[int] = 100

class EntityDeleteData(BaseModel):
    """
    实体删除数据模型
    
    定义了实体删除操作的请求数据结构，仅包含实体ID。
    该模型用于标识需要从知识图谱中删除的特定实体。
    
    字段说明：
        id: 要删除的实体的唯一标识符
    
    业务意义：
        - 提供简洁的实体删除接口
        - 确保删除操作的精确性
        - 支持知识图谱的内容维护和清理
    """
    id: str

class RelationDeleteData(BaseModel):
    """
    关系删除数据模型
    
    定义了关系删除操作的请求数据结构，包含源实体ID、关系类型和目标实体ID。
    该模型用于精确标识需要从知识图谱中删除的特定关系。
    
    字段说明：
        source: 关系的源实体ID
        type: 关系类型
        target: 关系的目标实体ID
    
    业务意义：
        - 提供精确的关系删除接口
        - 确保删除操作的准确性，避免误删除
        - 支持知识图谱的关系网络维护
        - 允许调整知识图谱中的连接结构
    """
    source: str
    type: str
    target: str