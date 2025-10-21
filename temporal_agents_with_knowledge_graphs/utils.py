import re
from datetime import UTC, datetime

from dateutil.parser import parse

'''
工具函数模块

该模块提供了知识图谱问答系统中的通用工具函数，主要集中在日期时间处理方面。
这些函数支持系统中的时间属性管理，确保时间数据的正确解析和格式化。

主要功能包括：
- 智能日期字符串解析
- 安全的ISO格式转换

设计特点：
- 处理多种日期格式，包括年份、标准日期格式等
- 自动处理时区信息，默认为UTC
- 提供容错机制，优雅处理解析错误
- 支持不同输入类型的统一处理
'''


def parse_date_str(value: str | datetime | None) -> datetime | None:
    """解析日期字符串为datetime对象

    功能描述：
    智能解析各种格式的日期字符串，支持年份格式和通用日期格式，并确保正确处理时区信息

    参数说明：
    - value: 日期值，可以是字符串、datetime对象或None

    返回值：
    - datetime | None: 解析后的datetime对象，如果无法解析则返回None

    业务流程：
    1. 检查值是否为None，如果是则直接返回None
    2. 检查值是否已经是datetime对象，如果是则直接返回
    3. 尝试解析字符串值：
       a. 如果是4位数字，则视为年份，返回该年1月1日
       b. 否则使用dateutil.parser.parse解析通用日期格式
    4. 确保返回的datetime对象有时区信息，默认为UTC
    5. 捕获所有异常，确保解析失败时返回None

    技术特点：
    - 使用正则表达式识别年份格式
    - 使用dateutil.parser处理各种日期格式
    - 自动添加UTC时区信息
    - 实现容错机制，不抛出异常

    业务意义：
    支持系统中各种时间属性的正确解析，确保时间事件的时间属性一致性，是时间知识图谱的基础设施
    """
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        # Year Handling
        if re.fullmatch(r"\d{4}", value.strip()):
            year = int(value.strip())
            return datetime(year, 1, 1, tzinfo=UTC)

        #  General Handing
        dt: datetime = parse(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    except Exception:
        return None


def safe_iso(dt: datetime | None) -> str | None:
    """安全地将datetime对象转换为ISO格式字符串

    功能描述：
    将datetime对象或日期字符串安全地转换为ISO 8601格式的字符串表示

    参数说明：
    - dt: 日期时间值，可以是datetime对象、日期字符串或None

    返回值：
    - str | None: ISO格式的日期字符串，如果输入无效则返回None

    业务流程：
    1. 检查输入是否为字符串，如果是则先使用parse_date_str解析
    2. 检查输入是否为datetime对象，如果是则调用isoformat()方法
    3. 如果输入为None或无法转换，则返回None

    技术特点：
    - 支持多种输入类型
    - 与parse_date_str函数协同工作
    - 实现容错机制，不抛出异常

    业务意义：
    确保日期时间数据在数据库存储和系统交互中的标准化表示，支持时间知识的正确存储和检索
    """
    if isinstance(dt, str):
        dt = parse_date_str(dt)

    if isinstance(dt, datetime):
        return dt.isoformat()

    return None
