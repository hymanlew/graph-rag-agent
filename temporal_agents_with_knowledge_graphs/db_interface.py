import os
import sqlite3
import uuid
from typing import Any
import pandas as pd
from models import Entity, TemporalEvent
from utils import safe_iso

'''
数据库接口模块

该模块提供了知识图谱问答系统中与SQLite数据库交互的核心功能，负责知识图谱数据的存储、查询和管理。
主要功能包括：
- 数据库连接管理（内存数据库或文件数据库）
- 表结构创建和维护
- 转录文本、文本块、实体、三元组和时间事件的数据操作
- 实体规范化和引用管理
- 时间事件的有效性管理

设计特点：
- 采用SQLite作为轻量级存储方案，便于本地开发和测试
- 支持内存数据库模式，适合临时数据处理
- 采用三元组模型存储实体关系，符合知识图谱的基本数据结构
- 实现了时间事件模型，支持事件的有效期、失效和更新管理
- 提供实体规范化机制，解决实体重复和引用一致性问题
'''


def make_connection(
    db_path: str = "my_database.db",
    memory: bool = False,
    refresh: bool = False,
) -> sqlite3.Connection:
    """创建数据库连接

    功能描述：
    创建或连接到SQLite数据库，支持文件数据库和内存数据库两种模式，并可选择是否重置数据库

    参数说明：
    - db_path: 数据库文件路径，默认为"my_database.db"
    - memory: 是否使用内存数据库，默认为False
    - refresh: 是否重置数据库（删除并重建），默认为False

    返回值：
    - sqlite3.Connection: 数据库连接对象

    业务流程：
    1. 如果选择重置且不是内存数据库，删除现有数据库文件
    2. 根据memory参数决定创建内存数据库还是文件数据库连接
    3. 如果是内存数据库且选择重置，删除所有表
    4. 创建必要的数据表结构
    5. 返回数据库连接

    技术特点：
    - 支持文件数据库和内存数据库的灵活切换
    - 提供重置机制，方便测试和初始化
    - 自动创建必要的表结构，确保数据模型完整性

    业务意义：
    为知识图谱问答系统提供数据持久化基础，支持临时开发环境和持久化生产环境的灵活切换
    """
    if not memory and refresh:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError as e:
                raise RuntimeError(
                    "Could not delete the database file. Please ensure all connections are closed."
                ) from e
    conn = sqlite3.connect(":memory:") if memory else sqlite3.connect(db_path)
    if memory and refresh:
        _drop_all_tables(conn)
    _create_lite_tables(conn)
    return conn


def _drop_all_tables(conn: sqlite3.Connection, tables: list[str] | None = None) -> None:
    """删除数据库中的所有表

    功能描述：
    删除数据库中的所有表或指定的表列表

    参数说明：
    - conn: 数据库连接对象
    - tables: 要删除的表列表，默认为None（删除所有表）

    业务流程：
    1. 如果未指定表列表，查询数据库中所有非系统表
    2. 遍历表列表，执行DROP TABLE语句删除每个表
    3. 提交事务

    技术特点：
    - 使用IF EXISTS确保即使表不存在也不会报错
    - 支持部分表删除和全部表删除

    业务意义：
    提供数据库重置机制，便于系统初始化和测试环境搭建
    """
    c = conn.cursor()
    if not tables:
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in c.fetchall()]
    for table in tables:
        c.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def _create_lite_tables(conn: sqlite3.Connection) -> None:
    """创建数据库所需的所有表

    功能描述：
    创建知识图谱问答系统所需的所有数据表，如果表已存在则不创建

    参数说明：
    - conn: 数据库连接对象

    业务流程：
    1. 创建transcripts表，存储原始转录文本
    2. 创建chunks表，存储文本块，与transcripts建立外键关系
    3. 为chunks表创建transcript_id索引，优化查询性能
    4. 创建events表，存储时间事件，与chunks建立外键关系
    5. 为events表创建chunk_id索引，优化查询性能
    6. 创建triplets表，存储实体关系三元组，与events建立外键关系
    7. 为triplets表创建event_id索引，优化查询性能
    8. 创建entities表，存储实体信息，与events建立外键关系
    9. 提交事务

    技术特点：
    - 使用IF NOT EXISTS确保表结构创建的幂等性
    - 建立适当的索引优化查询性能
    - 通过外键约束维护数据完整性

    业务意义：
    建立知识图谱的数据模型基础，支持文本、实体、关系和时间事件的结构化存储
    """
    c = conn.cursor()

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS transcripts (
        id BLOB PRIMARY KEY,
        text TEXT,
        company TEXT,
        date TEXT,
        quarter TEXT
    )
    """
    )

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        id BLOB PRIMARY KEY,
        transcript_id BLOB,
        text TEXT,
        metadata TEXT,
        FOREIGN KEY(transcript_id) REFERENCES transcripts(id)
    )
    """
    )
    c.execute(
        """CREATE INDEX IF NOT EXISTS idx_chunks_transcript_id ON chunks (transcript_id)"""
    )

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS events (
        id BLOB PRIMARY KEY,
        chunk_id BLOB,
        statement TEXT,
        triplets TEXT,
        statement_type TEXT,
        temporal_type TEXT,
        created_at TEXT,
        valid_at TEXT,
        expired_at TEXT,
        invalid_at TEXT,
        invalidated_by BLOB,
        embedding BLOB,
        FOREIGN KEY(chunk_id) REFERENCES chunks(id),
        FOREIGN KEY(invalidated_by) REFERENCES events(id)
    )
    """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_chunk_id ON events (chunk_id)")

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS triplets (
        id BLOB PRIMARY KEY,
        event_id BLOB,
        subject_name TEXT,
        subject_id BLOB,
        predicate TEXT,
        object_name TEXT,
        object_id BLOB,
        value TEXT,
        FOREIGN KEY(event_id) REFERENCES events(id)
    )
    """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_triplets_event_id ON triplets (event_id)")

    c.execute(
        """
    CREATE TABLE IF NOT EXISTS entities (
        id BLOB PRIMARY KEY,
        event_id BLOB,
        name TEXT,
        type TEXT,
        description TEXT,
        resolved_id BLOB,
        FOREIGN KEY(event_id) REFERENCES events(id),
        FOREIGN KEY(resolved_id) REFERENCES entities(id)
    )
    """
    )

    conn.commit()


def view_db_table(
    conn: sqlite3.Connection, table_name: str, max_rows: int | None = None
) -> pd.DataFrame:
    """查看数据库表内容

    功能描述：
    将数据库表内容以pandas DataFrame形式返回，方便数据分析和调试

    参数说明：
    - conn: 数据库连接对象
    - table_name: 要查看的表名
    - max_rows: 最大返回行数，默认为None（返回所有行）

    返回值：
    - pd.DataFrame: 表数据的DataFrame表示

    业务流程：
    1. 根据max_rows参数构造SQL查询语句
    2. 执行查询并将结果转换为DataFrame
    3. 返回DataFrame对象

    技术特点：
    - 支持限制返回行数，避免大数据集造成的性能问题
    - 利用pandas提供强大的数据处理能力

    业务意义：
    提供便捷的数据查看和分析工具，便于开发调试和数据验证
    """
    if max_rows:
        query = f"SELECT * FROM {table_name} LIMIT {max_rows}"
    else:
        query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(query, conn)


def insert_transcript(conn: sqlite3.Connection, transcript: dict[str, Any]) -> None:
    """插入转录文本到数据库

    功能描述：
    将原始转录文本数据插入到transcripts表中

    参数说明：
    - conn: 数据库连接对象
    - transcript: 转录文本数据字典，包含id、text、company、date、quarter等字段

    业务流程：
    1. 获取数据库游标
    2. 执行INSERT语句将转录文本插入表中
    3. 将date字段转换为ISO格式字符串存储

    技术特点：
    - 使用参数化查询防止SQL注入
    - 使用get方法获取可选字段，提高代码健壮性

    业务意义：
    存储原始转录文本数据，作为知识图谱构建的数据源
    """
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO transcripts
        (id, text, company, date, quarter)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            transcript["id"],
            transcript["text"],
            transcript["company"],
            transcript["date"].isoformat(),
            transcript.get("quarter"),
        ),
    )


def insert_chunk(conn: sqlite3.Connection, chunk: dict[str, Any]) -> None:
    """插入文本块到数据库

    功能描述：
    将转录文本的分块数据插入到chunks表中

    参数说明：
    - conn: 数据库连接对象
    - chunk: 文本块数据字典，包含id、transcript_id、text、metadata等字段

    业务流程：
    1. 获取数据库游标
    2. 执行INSERT语句将文本块插入表中
    3. 通过transcript_id与原始转录文本建立关联

    技术特点：
    - 使用参数化查询防止SQL注入
    - 使用get方法获取可选字段

    业务意义：
    将长文本分解为可管理的小块，便于后续实体和关系提取处理
    """
    c = conn.cursor()
    c.execute(
        "INSERT INTO chunks (id, transcript_id, text, metadata) VALUES (?, ?, ?, ?)",
        (chunk["id"], chunk["transcript_id"], chunk["text"], chunk.get("metadata")),
    )


# ======================
# TRIPLET INTERACTIONS
# ======================


def insert_triplet(conn: sqlite3.Connection, triplet: dict[str, Any]) -> None:
    """插入实体关系三元组到数据库

    功能描述：
    将实体关系三元组（主体-谓词-客体）插入到triplets表中，同时存储实体名称和解析后的ID

    参数说明：
    - conn: 数据库连接对象
    - triplet: 三元组数据字典，包含id、event_id、subject_name、subject_id、predicate、object_name、object_id、value等字段

    业务流程：
    1. 直接使用数据库连接执行INSERT语句
    2. 将三元组的各个组成部分插入表中
    3. 与事件建立关联（通过event_id）

    技术特点：
    - 使用参数化查询防止SQL注入
    - 同时存储实体名称和ID，兼顾可读性和引用完整性
    - 使用get方法获取可选字段

    业务意义：
    存储知识图谱的核心关系数据，是知识表示和推理的基础
    """
    conn.execute(
        """
        INSERT INTO triplets
        (id, event_id, subject_name, subject_id, predicate, object_name, object_id, value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            triplet["id"],
            triplet["event_id"],
            triplet["subject_name"],
            triplet.get("subject_id"),
            triplet["predicate"],
            triplet["object_name"],
            triplet.get("object_id"),
            triplet.get("value"),
        ),
    )


def get_all_triplets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """获取数据库中所有三元组

    功能描述：
    查询数据库中所有实体关系三元组，返回包含实体名称和解析ID的完整信息

    参数说明：
    - conn: 数据库连接对象

    返回值：
    - list[dict]: 三元组列表，每个元素为包含完整三元组信息的字典

    业务流程：
    1. 获取数据库游标
    2. 执行SELECT语句查询所有三元组数据
    3. 将查询结果转换为字典列表返回

    技术特点：
    - 返回结构化的字典数据，便于后续处理
    - 包含完整的三元组信息，包括ID和名称

    业务意义：
    提供完整的知识图谱关系数据，支持图谱构建和查询
    """
    c = conn.cursor()
    c.execute(
        """
        SELECT
            id, event_id,
            subject_name, subject_id,
            predicate,
            object_name, object_id,
            value
        FROM triplets
    """
    )
    return [
        {
            "id": row[0],
            "event_id": row[1],
            "subject_name": row[2],
            "subject_id": row[3],
            "predicate": row[4],
            "object_name": row[5],
            "object_id": row[6],
            "value": row[7],
        }
        for row in c.fetchall()
    ]


def get_all_unique_predicates(conn: sqlite3.Connection) -> list[str]:
    """获取所有唯一的谓词

    功能描述：
    查询数据库中所有不同的谓词类型，用于了解知识图谱中存在的关系类型

    参数说明：
    - conn: 数据库连接对象

    返回值：
    - list[str]: 唯一谓词类型的列表

    业务流程：
    1. 获取数据库游标
    2. 执行DISTINCT查询获取唯一谓词
    3. 将结果转换为字符串列表返回

    技术特点：
    - 使用DISTINCT关键字高效获取唯一值

    业务意义：
    提供关系类型统计，支持知识图谱的关系分析和可视化
    """
    c = conn.cursor()
    c.execute("SELECT DISTINCT predicate FROM triplets")
    rows = c.fetchall()
    return [row[0] for row in rows]


# =====================
# ENTITY INTERACTIONS
# =====================


def insert_entity(conn: sqlite3.Connection, entity: dict[str, Any]) -> None:
    """插入实体到数据库

    功能描述：
    将实体信息插入到entities表中，使用INSERT OR IGNORE确保不会重复插入

    参数说明：
    - conn: 数据库连接对象
    - entity: 实体数据字典，包含id、name、type、description等字段

    业务流程：
    1. 获取数据库游标
    2. 执行INSERT OR IGNORE语句插入实体
    3. 使用get方法获取可选字段

    技术特点：
    - 使用INSERT OR IGNORE防止实体重复插入
    - 使用参数化查询防止SQL注入

    业务意义：
    存储知识图谱中的实体数据，支持实体识别和链接
    """
    c = conn.cursor()
    c.execute(
        """
              INSERT OR IGNORE INTO entities (id, name, type, description)
              VALUES (?, ?, ?, ?)""",
        (entity["id"], entity["name"], entity.get("type"), entity.get("description")),
    )


def get_all_canonical_entities(conn: sqlite3.Connection) -> list[Entity]:
    """获取所有规范化实体

    功能描述：
    查询数据库中所有规范化实体，返回Entity对象列表

    参数说明：
    - conn: 数据库连接对象

    返回值：
    - list[Entity]: Entity对象列表，包含实体的id、name、type和description

    业务流程：
    1. 获取数据库游标
    2. 执行SELECT语句查询所有实体
    3. 将查询结果转换为Entity对象列表返回
    4. 将UUID字符串转换为UUID对象
    5. 处理空值情况，确保返回有效数据

    技术特点：
    - 返回强类型的Entity对象，便于类型检查和代码提示
    - 处理UUID类型转换和空值情况

    业务意义：
    提供规范化实体列表，支持实体统一管理和引用
    """
    c = conn.cursor()
    c.execute("SELECT id, name, type, description FROM entities")
    rows = c.fetchall()
    return [
        Entity(
            id=uuid.UUID(row[0]),
            name=row[1],
            type=row[2] or "",
            description=row[3] or "",
        )
        for row in rows
    ]


def insert_canonical_entity(conn: sqlite3.Connection, entity: dict[str, Any]) -> None:
    """插入规范化实体

    功能描述：
    将规范化实体插入到entities表中，作为实体的标准表示

    参数说明：
    - conn: 数据库连接对象
    - entity: 规范化实体数据字典，必须包含id、name字段，可选type和description字段

    业务流程：
    1. 获取数据库游标
    2. 执行INSERT OR IGNORE语句插入规范化实体
    3. 使用get方法获取可选字段

    技术特点：
    - 使用INSERT OR IGNORE防止重复插入
    - 使用参数化查询防止SQL注入

    业务意义：
    建立实体的标准表示，支持实体链接和规范化管理
    """
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO entities (id, name, type, description) VALUES (?, ?, ?, ?)",
        (entity["id"], entity["name"], entity.get("type"), entity.get("description")),
    )


def update_entity_references(
    conn: sqlite3.Connection, old_id: str, new_id: str
) -> None:
    """更新实体引用

    功能描述：
    在数据库中更新所有从old_id到new_id的引用，用于实体合并或规范化

    参数说明：
    - conn: 数据库连接对象
    - old_id: 要替换的旧实体ID
    - new_id: 替换为的新实体ID

    业务流程：
    1. 更新entities表中的resolved_id字段
    2. 更新triplets表中的subject_id字段
    3. 更新triplets表中的object_id字段
    4. 提交事务

    技术特点：
    - 批量更新实体引用，确保数据一致性
    - 使用参数化查询防止SQL注入

    业务意义：
    支持实体规范化和合并操作，确保知识图谱中实体引用的一致性
    """
    conn.execute(
        "UPDATE entities SET resolved_id = ? WHERE resolved_id = ?", (new_id, old_id)
    )
    conn.execute(
        "UPDATE triplets SET subject_id = ? WHERE subject_id = ?", (new_id, old_id)
    )
    conn.execute(
        "UPDATE triplets SET object_id = ? WHERE object_id = ?", (new_id, old_id)
    )
    conn.commit()


def remove_entity(conn: sqlite3.Connection, entity_id: str) -> None:
    """删除实体

    功能描述：
    从entities表中删除指定ID的实体

    参数说明：
    - conn: 数据库连接对象
    - entity_id: 要删除的实体ID

    业务流程：
    1. 执行DELETE语句删除实体
    2. 提交事务

    技术特点：
    - 使用参数化查询防止SQL注入

    业务意义：
    支持实体管理，允许删除不需要的或错误的实体
    """
    conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
    conn.commit()


# ====================
# EVENT INTERACTIONS
# ====================


def insert_event(conn: sqlite3.Connection, event_dict: dict[str, Any]) -> None:
    """插入事件到数据库

    功能描述：
    将时间事件数据插入到events表中，记录实体关系的时间属性

    参数说明：
    - conn: 数据库连接对象
    - event_dict: 事件数据字典，包含id、chunk_id、statement、embedding、triplets、statement_type、temporal_type、created_at、valid_at、expired_at、invalid_at、invalidated_by等字段

    业务流程：
    1. 获取数据库游标
    2. 执行INSERT语句插入事件数据
    3. 与文本块建立关联（通过chunk_id）

    技术特点：
    - 使用参数化查询防止SQL注入
    - 支持时间事件的完整属性存储

    业务意义：
    存储带有时间属性的知识事件，支持基于时间的知识推理和问答
    """
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO events
        (id, chunk_id, statement, embedding, triplets, statement_type, temporal_type,
         created_at, valid_at, expired_at, invalid_at, invalidated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (event_dict["id"]),
            event_dict["chunk_id"],
            event_dict["statement"],
            event_dict["embedding"],
            event_dict["triplets"],
            event_dict["statement_type"],
            event_dict["temporal_type"],
            event_dict["created_at"],
            event_dict["valid_at"],
            event_dict["expired_at"],
            event_dict["invalid_at"],
            event_dict.get("invalidated_by"),
        ),
    )


def has_events(conn: sqlite3.Connection) -> bool:
    """检查是否存在FACT类型的事件

    功能描述：
    检查数据库中是否存在FACT类型的事件，用于验证系统是否有足够的事实数据

    参数说明：
    - conn: 数据库连接对象

    返回值：
    - bool: 如果存在FACT事件则返回True，否则返回False

    业务流程：
    1. 获取数据库游标
    2. 执行COUNT查询统计FACT类型事件数量
    3. 返回是否存在事件的布尔值

    技术特点：
    - 使用COUNT聚合函数高效检查数据存在性

    业务意义：
    提供数据可用性检查，确保系统有足够的事实数据进行验证和推理
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM events WHERE statement_type = ?", ("FACT",))
    count = cursor.fetchone()[0]
    return count > 0  # type: ignore


def update_events_batch(conn: sqlite3.Connection, events: list[TemporalEvent]) -> None:
    """批量更新多个事件

    功能描述：
    批量更新事件的时间属性，包括失效时间、过期时间和失效关联

    参数说明：
    - conn: 数据库连接对象
    - events: TemporalEvent对象列表，包含要更新的事件

    业务流程：
    1. 检查事件列表是否为空，如果为空则直接返回
    2. 构建更新数据列表，包含invalid_at、expired_at、invalidated_by和id字段
    3. 使用safe_iso函数安全处理时间字段
    4. 执行批量更新语句
    5. 提交事务

    技术特点：
    - 使用executemany进行批量操作，提高性能
    - 使用hasattr检查对象属性，增强代码健壮性
    - 使用safe_iso函数安全处理时间格式转换

    业务意义：
    支持知识事件的时间管理，允许批量更新事件的有效性状态，是时间推理的基础
    """
    if not events:
        return

    c = conn.cursor()
    update_data = [
        (
            safe_iso(event.invalid_at) if hasattr(event, "invalid_at") else None,
            safe_iso(event.expired_at) if hasattr(event, "expired_at") else None,
            (
                str(event.invalidated_by)
                if hasattr(event, "invalidated_by") and event.invalidated_by
                else None
            ),
            str(event.id) if hasattr(event, "id") else event.id,
        )
        for event in events
    ]

    c.executemany(
        """UPDATE events SET
           invalid_at = ?,
           expired_at = ?,
           invalidated_by = ?
           WHERE id = ?""",
        update_data,
    )
    conn.commit()
