"""
数据模型模块

该模块定义了知识图谱问答系统中与数据库交互的核心数据模型，主要包括实体、时间事件及其相关的枚举类型。
这些模型构成了系统中知识表示的基础，支持知识图谱的数据结构和时间属性管理。

主要功能包括：
- 实体表示与解析（RawEntity和Entity类）
- 时间类型和语句类型的枚举定义
- 时间事件模型（TemporalEvent类）
- 数据序列化和验证功能

设计特点：
- 基于Pydantic构建，提供强类型检查和数据验证
- 支持UUID作为唯一标识符
- 实现时间事件的有效性管理
- 提供数据转换和辅助方法
"""
import json
import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

class RawEntity(BaseModel):
    """原始实体模型

    功能描述：
    表示待解析的原始实体数据，用于实体识别和解析过程

    字段说明：
    - entity_idx: 实体在文本中的索引位置
    - name: 实体名称
    - type: 实体类型，默认为空字符串
    - description: 实体描述，默认为空字符串

    业务意义：
    作为实体解析的中间表示，支持从文本中提取实体信息并进行后续规范化处理
    """

    entity_idx: int  # 实体在文本中的索引位置
    name: str  # 实体名称
    type: str = ""  # 实体类型
    description: str = ""  # 实体描述


class Entity(BaseModel):
    """实体模型

    功能描述：
    表示知识图谱中的实体，支持规范化实体和别名实体的表示

    字段说明：
    - id: 实体唯一标识符，如果是规范化实体，此ID为标准ID
    - event_id: 关联的事件ID，可选
    - name: 实体名称
    - type: 实体类型
    - description: 实体描述
    - resolved_id: 解析到的规范化实体ID，如果是别名实体则设置

    业务流程：
    1. 可以通过默认构造函数创建新实体
    2. 可以通过from_raw方法从原始实体转换
    3. 支持实体解析，将别名实体链接到规范化实体

    技术特点：
    - 使用UUID作为唯一标识符
    - 支持实体间的引用关系
    - 提供类型验证和默认值设置

    业务意义：
    作为知识图谱的核心节点，支持实体规范化和实体链接，确保知识表示的一致性
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)  # 实体唯一标识符
    event_id: uuid.UUID | None = None  # 关联的事件ID
    name: str  # 实体名称
    type: str  # 实体类型
    description: str  # 实体描述
    resolved_id: uuid.UUID | None = None  # 规范化实体ID

    @classmethod
    def from_raw(
        cls, raw_entity: "RawEntity", event_id: uuid.UUID | None = None
    ) -> "Entity":
        """从原始实体创建实体实例

        功能描述：
        将RawEntity对象转换为Entity对象，可选地关联到特定事件

        参数说明：
        - raw_entity: 原始实体对象
        - event_id: 关联的事件ID，可选

        返回值：
        - Entity: 创建的实体对象

        业务流程：
        1. 生成新的UUID作为实体ID
        2. 复制原始实体的名称、类型和描述
        3. 设置关联的事件ID（如果提供）
        4. 初始设置resolved_id为None

        技术特点：
        - 使用类方法实现对象转换
        - 保持数据一致性

        业务意义：
        支持从文本提取的原始实体到知识库实体的转换，是实体解析流程的重要组成部分
        """
        return cls(
            id=uuid.uuid4(),
            event_id=event_id,
            name=raw_entity.name,
            type=raw_entity.type,
            description=raw_entity.description,
            resolved_id=None,
        )

class TemporalType(StrEnum):
    """时间类型枚举

    功能描述：
    定义知识语句的时间属性类型，支持不同时间特性的知识表示

    枚举值说明：
    - ATEMPORAL: 非时间性知识，表示不受时间影响的普遍真理
    - STATIC: 静态知识，表示在较长时间内保持不变的知识
    - DYNAMIC: 动态知识，表示会随时间变化的知识

    业务意义：
    支持知识的时间特性分类，为基于时间的知识推理提供基础
    """

    ATEMPORAL = "ATEMPORAL"  # 非时间性知识
    STATIC = "STATIC"  # 静态知识
    DYNAMIC = "DYNAMIC"  # 动态知识

class StatementType(StrEnum):
    """语句类型枚举

    功能描述：
    定义知识语句的类型，区分不同性质的知识

    枚举值说明：
    - FACT: 事实性知识，表示已确认的客观事实
    - OPINION: 观点性知识，表示主观意见或评价
    - PREDICTION: 预测性知识，表示对未来的预测或估计

    业务意义：
    支持知识的性质分类，为不同类型知识的处理和推理提供依据
    """

    FACT = "FACT"  # 事实性知识
    OPINION = "OPINION"  # 观点性知识
    PREDICTION = "PREDICTION"  # 预测性知识

class TemporalEvent(BaseModel):
    """时间事件模型

    功能描述：
    表示具有时间属性的知识事件，包含语句内容、实体关系和有效性信息

    字段说明：
    - id: 事件唯一标识符
    - chunk_id: 关联的文本块ID
    - statement: 原始语句内容
    - embedding: 语句的向量表示，默认为256维零向量
    - triplets: 事件包含的三元组ID列表
    - valid_at: 事件有效的起始时间
    - invalid_at: 事件失效的时间
    - temporal_type: 时间类型（ATEMPORAL/STATIC/DYNAMIC）
    - statement_type: 语句类型（FACT/OPINION/PREDICTION）
    - created_at: 事件创建时间，默认为当前时间
    - expired_at: 事件过期时间
    - invalidated_by: 使此事件失效的其他事件ID

    业务流程：
    1. 创建事件时自动生成ID和创建时间
    2. 根据时间类型和失效时间自动设置过期时间
    3. 支持三元组关系的序列化和反序列化

    技术特点：
    - 使用Pydantic的model_validator进行数据验证和自动字段设置
    - 提供属性装饰器和类方法实现数据转换
    - 支持UUID和时间类型的序列化处理

    业务意义：
    作为时间知识图谱的核心单元，支持基于时间的知识表示和推理，允许知识随时间演变
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)  # 事件唯一标识符
    chunk_id: uuid.UUID  # 关联的文本块ID
    statement: str  # 原始语句内容
    embedding: list[float] = Field(default_factory=lambda: [0.0] * 256)  # 语句向量表示
    triplets: list[uuid.UUID]  # 三元组ID列表
    valid_at: datetime | None = None  # 有效起始时间
    invalid_at: datetime | None = None  # 失效时间
    temporal_type: TemporalType  # 时间类型
    statement_type: StatementType  # 语句类型
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间
    expired_at: datetime | None = None  # 过期时间
    invalidated_by: uuid.UUID | None = None  # 失效关联事件ID

    @property
    def triplets_json(self) -> str:
        """将三元组列表转换为JSON字符串

        功能描述：
        将UUID对象列表序列化为JSON字符串，便于存储和传输

        返回值：
        - str: 包含三元组ID的JSON字符串

        技术特点：
        - 使用json.dumps进行序列化
        - 处理空列表情况，确保返回有效JSON

        业务意义：
        支持三元组关系在数据库中的存储和恢复
        """
        return json.dumps([str(t) for t in self.triplets]) if self.triplets else "[]"

    @classmethod
    def parse_triplets_json(cls, triplets_str: str) -> list[uuid.UUID]:
        """解析JSON字符串为UUID列表

        功能描述：
        将JSON字符串反序列化为UUID对象列表，恢复三元组关系

        参数说明：
        - triplets_str: 包含三元组ID的JSON字符串

        返回值：
        - list[uuid.UUID]: 三元组ID的UUID对象列表

        业务流程：
        1. 检查输入字符串是否为空或空数组
        2. 解析JSON字符串为字符串列表
        3. 将每个字符串转换为UUID对象

        技术特点：
        - 处理边界情况，避免解析错误
        - 支持UUID类型转换

        业务意义：
        支持从数据库恢复三元组关系数据
        """
        if not triplets_str or triplets_str == "[]":
            return []
        return [uuid.UUID(t) for t in json.loads(triplets_str)]

    @model_validator(mode="after")
    def set_expired_at(self) -> "TemporalEvent":
        """自动设置过期时间

        功能描述：
        当事件被标记为失效且为动态类型时，自动设置过期时间为创建时间

        返回值：
        - TemporalEvent: 更新后的事件对象

        业务流程：
        1. 检查invalid_at是否已设置
        2. 检查temporal_type是否为DYNAMIC
        3. 如果条件满足，将expired_at设置为created_at

        技术特点：
        - 使用Pydantic的model_validator在实例化后自动执行
        - 实现业务规则的自动应用

        业务意义：
        确保动态知识的时间一致性，当动态知识被标记为失效时，自动将其视为立即过期
        """
        self.expired_at = (
            self.created_at
            if (self.invalid_at is not None)
            and (self.temporal_type == TemporalType.DYNAMIC)
            else None
        )
        return self
