import re
import json
from typing import Dict, Any, List, Optional

"""
引用数据提取模块

此模块提供了从AI生成的回答中提取引用数据的功能，包括实体、关系、文本块和报告等引用信息。
在GraphRAG系统中，这些引用数据对于评估检索质量和答案生成质量至关重要。

主要功能包括：
- 从各种格式的引用数据部分中提取结构化信息
- 处理不同格式的JSON数据和文本数据
- 验证和格式化提取的ID
- 支持多种引用数据的格式，提高提取的鲁棒性

提取的数据将用于评估检索的准确性、利用率和覆盖范围等指标。
"""

def extract_references_from_answer(answer: str) -> Dict[str, Any]:
    """
    从回答中提取引用数据，支持多种格式，提高提取的鲁棒性
    
    作为引用数据提取的入口函数，首先检查回答是否包含引用数据部分，
    然后尝试多种方式解析不同格式的引用数据，最后对提取的ID进行验证和格式化。
    
    在GraphRAG评估系统中，提取的引用数据用于评估检索的准确性和利用率，
    是计算检索精确度、实体覆盖率等指标的基础。
    
    Args:
        answer: AI生成的回答，可能包含引用数据部分
        
    Returns:
        Dict: 包含entities, relationships, chunks和reports等引用信息的字典
    """
    # 初始化结果
    result = {
        "entities": [],
        "relationships": [],
        "chunks": [],
        "reports": []
    }
    
    # 如果没有回答或引用数据部分，直接返回空结果
    if not answer or "引用数据" not in answer:
        return result
    
    try:
        # 先尝试提取完整的引用数据部分
        reference_section = extract_reference_section(answer)
        if not reference_section:
            return result
            
        # 尝试多种方式解析JSON格式的引用数据
        parsed_data = parse_json_data(reference_section)
        if parsed_data:
            # 处理实体
            entities = extract_entities_from_parsed(parsed_data)
            result["entities"].extend(entities)
            
            # 处理关系
            relationships = extract_relationships_from_parsed(parsed_data)
            result["relationships"].extend(relationships)
            
            # 处理文本块
            chunks = extract_chunks_from_parsed(parsed_data)
            result["chunks"].extend(chunks)

            # 处理报告
            reports = extract_reports_from_parsed(parsed_data)
            result["reports"].extend(reports)
        else:
            # 如果无法解析JSON，尝试直接从文本中提取
            result["entities"] = extract_entities_from_text(reference_section)
            result["relationships"] = extract_relationships_from_text(reference_section)
            result["chunks"] = extract_chunks_from_text(reference_section)
            result["reports"] = extract_reports_from_text(reference_section)
        
        # 验证和格式化提取的ID
        result["entities"] = validate_and_format_ids(result["entities"])
        result["relationships"] = validate_and_format_ids(result["relationships"])
        
        # 去重处理
        result["entities"] = list(set(result["entities"]))
        result["relationships"] = list(set(result["relationships"]))
        result["chunks"] = list(set(result["chunks"]))
        result["reports"] = list(set(result["reports"]))
        
        return result
    except Exception as e:
        print(f"提取引用数据时出错: {e}")
        return result

def validate_and_format_ids(ids_list: List) -> List[str]:
    """
    验证并格式化ID列表，处理不同格式的ID
    
    对提取的ID进行规范化处理，确保所有ID都是有效的字符串格式，
    同时过滤掉空值和无效的ID。这一步对于确保后续评估指标计算的准确性至关重要。
    
    支持处理以下类型的ID：
    - 数字类型（整数或浮点数）
    - 数字字符串
    - UUID或其他长字符串格式的ID
    - 其他非空字符串ID
    
    Args:
        ids_list: 包含各种格式ID的列表
        
    Returns:
        List[str]: 格式化后的有效ID列表，所有ID都被转换为字符串格式
    """
    valid_ids = []
    for id_value in ids_list:
        # 跳过None和空值
        if id_value is None or id_value == "":
            continue
            
        # 尝试处理不同格式的ID
        if isinstance(id_value, (int, float)):
            valid_ids.append(str(int(id_value)))
        elif isinstance(id_value, str):
            # 如果是数字字符串，直接添加
            if id_value.isdigit() or id_value.lstrip('-').isdigit():
                valid_ids.append(id_value)
            # 如果看起来像是UUID或其他特殊ID格式(长字符串)，也添加
            elif len(id_value) > 10:
                valid_ids.append(id_value)
            # 其他非空字符串也添加
            elif id_value.strip():
                valid_ids.append(id_value)
    return valid_ids

def extract_reference_section(answer: str) -> str:
    """
    从回答文本中提取引用数据部分
    
    使用多种正则表达式模式尝试匹配不同格式的引用数据标记，
    增强对各种AI输出格式的兼容性。这是引用数据提取过程的第一步，
    为后续的解析和数据提取奠定基础。
    
    支持的引用数据格式包括：
    - Markdown标题格式：#### 引用数据 {...}
    - 冒号分隔格式：引用数据: {...}
    - XML标签格式：<引用数据> {...} </引用数据>
    - 简化格式：引用: {...}, 参考: {...}, 数据: {...}
    - 直接JSON格式：{...data...}
    
    Args:
        answer: AI生成的回答文本，可能包含引用数据标记
        
    Returns:
        str: 提取的引用数据部分，若未找到则返回空字符串
    """
    # 尝试多种引用数据标记格式
    patterns = [
        r'#{1,4}\s*引用数据[\s\S]*?(\{[\s\S]*?\})\s*$',  # #### 引用数据 {...}
        r'引用数据[：:]\s*(\{[\s\S]*?\})\s*$',           # 引用数据: {...}
        r'<引用数据>\s*(\{[\s\S]*?\})\s*</引用数据>',    # <引用数据> {...} </引用数据>
        r'引用[：:]\s*(\{[\s\S]*?\})\s*$',               # 引用: {...}
        r'参考[：:]\s*(\{[\s\S]*?\})\s*$',               # 参考: {...}
        r'数据[：:]\s*(\{[\s\S]*?\})\s*$',               # 数据: {...}
        r'(\{[\s\S]*?[\'"]*data[\'"]*[\s\S]*?\})\s*$'    # {...data...}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def parse_json_data(data_text: str) -> Optional[Dict]:
    """
    尝试多种方式解析JSON数据
    
    实现了多种回退解析策略，增强对各种非标准JSON格式的处理能力。
    在AI生成的输出中，JSON格式可能不严格遵循标准，需要灵活的解析方法。
    
    解析策略按优先级排序：
    1. 直接尝试标准JSON解析
    2. 修复单引号问题后解析
    3. 提取data字段后解析
    4. 进行全面清理和格式修复后解析
    
    这种多策略解析方法确保了即使面对格式不规范的输出，也能尽可能地提取有用的数据。
    
    Args:
        data_text: 可能包含JSON数据的文本
        
    Returns:
        Optional[Dict]: 解析后的字典对象，若无法解析则返回None
    """
    # 直接尝试解析
    try:
        parsed = json.loads(data_text)
        return parsed
    except:
        pass
    
    # 尝试修复常见JSON格式问题
    try:
        # 修复单引号问题
        fixed_text = data_text.replace("'", '"')
        parsed = json.loads(fixed_text)
        return parsed
    except:
        pass
    
    # 尝试提取data字段
    try:
        data_match = re.search(r'\{\s*["\']*data["\']*\s*:\s*(\{[\s\S]*?\})\s*\}', data_text, re.DOTALL)
        if data_match:
            data_content = data_match.group(1)
            # 修复单引号
            fixed_text = "{\"data\":" + data_content.replace("'", '"') + "}"
            parsed = json.loads(fixed_text)
            return parsed
    except:
        pass
    
    # 尝试将text包装成合法JSON
    try:
        # 去除非ASCII字符
        cleaned_text = ''.join(c for c in data_text if ord(c) < 128)
        # 替换所有单引号为双引号
        cleaned_text = cleaned_text.replace("'", '"')
        # 确保键名有双引号
        cleaned_text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned_text)
        parsed = json.loads(cleaned_text)
        return parsed
    except:
        return None

def extract_entities_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取实体ID
    
    从已成功解析的JSON数据中提取实体引用信息，处理多种可能的数据结构，
    包括嵌套的data结构、列表格式、字典格式和逗号分隔的字符串格式等。
    
    在GraphRAG评估中，提取的实体ID用于计算实体覆盖率、检索精确度等指标，
    是评估图检索效果的重要数据来源。
    
    支持的实体数据格式包括：
    - 直接的实体ID列表
    - 包含id字段的实体对象列表
    - 逗号分隔的实体ID字符串
    - 实体字典映射
    
    Args:
        parsed_data: 已解析的JSON数据字典
    
    Returns:
        List[str]: 提取的实体ID列表，所有ID已转换为字符串格式
    """
    entities = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Entities字段的值
    entity_keys = ["Entities", "entities", "Entity", "entity"]
    for key in entity_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理列表格式
                for item in parsed_data[key]:
                    if isinstance(item, (int, float)):
                        entities.append(str(int(item)))
                    elif isinstance(item, str):
                        entities.append(item)
                    elif isinstance(item, dict) and "id" in item:
                        # 处理{id: 123}格式
                        entities.append(str(item["id"]))
            elif isinstance(parsed_data[key], str):
                # 处理逗号分隔的字符串
                parts = parsed_data[key].split(",")
                for part in parts:
                    clean_part = part.strip()
                    if clean_part:
                        entities.append(clean_part)
            elif isinstance(parsed_data[key], dict):
                # 处理字典格式
                for k, v in parsed_data[key].items():
                    if isinstance(v, (int, str)):
                        entities.append(str(v))
    
    return entities

def extract_relationships_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取关系ID
    
    从已成功解析的JSON数据中提取关系引用信息，处理多种可能的数据结构和命名方式，
    包括嵌套的data结构、三元组格式和其他各种关系表示形式。
    
    在GraphRAG评估中，提取的关系ID用于计算关系利用率、图覆盖率等指标，
    是评估图检索中关系信息利用效果的重要依据。
    
    支持的关系数据格式包括：
    - 直接的关系ID列表
    - 包含id字段的关系对象列表
    - 三元组格式 (source, relation, target)
    - 逗号分隔的关系ID字符串
    - 关系字典映射
    
    Args:
        parsed_data: 已解析的JSON数据字典
    
    Returns:
        List[str]: 提取的关系ID列表，所有ID已转换为字符串格式
    """
    relationships = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取关系ID的所有可能键
    rel_keys = [
        "Relationships", "relationships", "Relations", "relations", 
        "Relation", "relation", "Reports", "reports", "Report", "report"
    ]
    
    for key in rel_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理列表格式
                for item in parsed_data[key]:
                    if isinstance(item, (int, float)):
                        relationships.append(str(int(item)))
                    elif isinstance(item, str):
                        relationships.append(item)
                    elif isinstance(item, dict) and "id" in item:
                        # 处理{id: 123}格式
                        relationships.append(str(item["id"]))
                    elif isinstance(item, tuple) or (isinstance(item, list) and len(item) >= 3):
                        # 处理三元组格式 (source, relation, target)
                        # 在这种情况下，我们可以提取关系ID或使用整个三元组
                        relationships.append(str(item))
            elif isinstance(parsed_data[key], str):
                # 处理逗号分隔的字符串
                parts = parsed_data[key].split(",")
                for part in parts:
                    clean_part = part.strip()
                    if clean_part:
                        relationships.append(clean_part)
            elif isinstance(parsed_data[key], dict):
                # 处理字典格式
                for k, v in parsed_data[key].items():
                    if isinstance(v, (int, str)):
                        relationships.append(str(v))
    
    return relationships

def extract_chunks_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取文本块ID
    
    从已成功解析的JSON数据中提取文本块引用信息，处理多种可能的数据结构和命名方式，
    专注于提取字符串格式的文本块ID。
    
    在RAG评估中，文本块ID用于计算文本块利用率、检索精确度等指标，
    特别是在评估传统向量检索效果时非常重要。
    
    支持的文本块数据格式包括：
    - 直接的文本块ID字符串列表
    - 逗号分隔的文本块ID字符串
    
    Args:
        parsed_data: 已解析的JSON数据字典
    
    Returns:
        List[str]: 提取的文本块ID列表
    """
    chunks = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Chunks字段的值
    chunk_keys = ["Chunks", "chunks", "Chunk", "chunk", "TextChunks", "textchunks"]
    for key in chunk_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理字符串列表
                for item in parsed_data[key]:
                    if isinstance(item, str):
                        chunks.append(item)
            elif isinstance(parsed_data[key], str):
                # 如果是逗号分隔的字符串
                chunks.extend([c.strip() for c in parsed_data[key].split(",") if c.strip()])
    
    return chunks

def extract_reports_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取报告ID
    
    从已成功解析的JSON数据中提取报告引用信息，处理多种可能的数据结构和命名方式，
    支持处理数字和字符串格式的报告ID。
    
    在GraphRAG评估中，特别是使用深度研究工具时，报告ID用于追踪和评估
    深度研究的结果利用情况。
    
    支持的报告数据格式包括：
    - 直接的报告ID列表（数字或字符串）
    - 逗号分隔的报告ID字符串
    
    Args:
        parsed_data: 已解析的JSON数据字典
    
    Returns:
        List[str]: 提取的报告ID列表，所有ID已转换为字符串格式
    """
    reports = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Reports字段的值
    report_keys = ["Reports", "reports", "Report", "report"]
    for key in report_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                for item in parsed_data[key]:
                    if isinstance(item, (int, str)):
                        reports.append(str(item))
            elif isinstance(parsed_data[key], str):
                reports.extend([r.strip() for r in parsed_data[key].split(",") if r.strip()])
    
    return reports

def extract_entities_from_text(text: str) -> List[str]:
    """
    直接从文本中提取实体ID
    
    在无法通过JSON解析获取实体信息时的备用提取方法，使用正则表达式
    直接从文本中匹配实体ID的模式。这是一种回退策略，增强了提取功能的鲁棒性。
    
    支持匹配以下文本模式：
    - Entities = [1, 2, 3] 格式
    - entities: 1, 2, 3 格式
    
    Args:
        text: 可能包含实体ID信息的文本
        
    Returns:
        List[str]: 从文本中提取的实体ID列表
    """
    # 尝试匹配实体ID部分
    entity_matches = re.search(r'[Ee]ntities\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                    re.search(r'[Ee]ntities\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if entity_matches:
        entity_str = entity_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', entity_str)
    
    return []

def extract_relationships_from_text(text: str) -> List[str]:
    """
    直接从文本中提取关系ID
    
    在无法通过JSON解析获取关系信息时的备用提取方法，使用正则表达式
    直接从文本中匹配关系ID的模式。这是一种回退策略，增强了提取功能的鲁棒性。
    
    同时支持提取关系和报告ID，因为在某些输出格式中，这两种类型可能使用相似的表示方式。
    
    支持匹配以下文本模式：
    - Relationships = [1, 2, 3] 格式
    - relationships: 1, 2, 3 格式
    - Reports = [1, 2, 3] 格式
    - reports: 1, 2, 3 格式
    
    Args:
        text: 可能包含关系或报告ID信息的文本
        
    Returns:
        List[str]: 从文本中提取的关系/报告ID列表
    """
    # 尝试匹配关系ID部分
    rel_matches = re.search(r'[Rr]elationships\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                re.search(r'[Rr]elationships\s*[=:]\s*([\d\s,]+)', text, re.DOTALL) or \
                re.search(r'[Rr]eports\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                re.search(r'[Rr]eports\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if rel_matches:
        rel_str = rel_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', rel_str)
    
    return []

def extract_chunks_from_text(text: str) -> List[str]:
    """
    直接从文本中提取文本块ID
    
    在无法通过JSON解析获取文本块信息时的备用提取方法，使用正则表达式
    直接从文本中匹配文本块ID的模式。这是一种回退策略，增强了提取功能的鲁棒性。
    
    与实体和关系提取不同，文本块ID通常是字符串格式，需要从引号中提取。
    
    支持匹配以下文本模式：
    - Chunks = ["chunk1", "chunk2"] 格式
    - chunks: ["chunk1", "chunk2"] 格式
    
    Args:
        text: 可能包含文本块ID信息的文本
        
    Returns:
        List[str]: 从文本中提取的文本块ID列表
    """
    # 尝试匹配文本块ID部分
    chunk_matches = re.search(r'[Cc]hunks\s*[=:]\s*\[(.*?)\]', text, re.DOTALL)
    
    if chunk_matches:
        chunk_str = chunk_matches.group(1).strip()
        # 提取引号中的内容
        return re.findall(r'[\'"]([^\'"]*)[\'"]', chunk_str)
    
    return []

def extract_reports_from_text(text: str) -> List[str]:
    """
    直接从文本中提取报告ID
    
    在无法通过JSON解析获取报告信息时的备用提取方法，使用正则表达式
    直接从文本中匹配报告ID的模式。这是一种回退策略，增强了提取功能的鲁棒性。
    
    报告ID通常是数字格式，用于追踪和评估深度研究工具的输出结果。
    
    支持匹配以下文本模式：
    - Reports = [1, 2, 3] 格式
    - reports: 1, 2, 3 格式
    
    Args:
        text: 可能包含报告ID信息的文本
        
    Returns:
        List[str]: 从文本中提取的报告ID列表
    """
    # 尝试匹配报告ID部分
    report_matches = re.search(r'[Rr]eports\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                    re.search(r'[Rr]eports\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if report_matches:
        report_str = report_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', report_str)
    
    return []