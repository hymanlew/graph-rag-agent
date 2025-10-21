"""
Neo4j批量处理工具模块

该模块提供了与Neo4j图数据库交互的批量处理功能，主要用于高效查询和获取
知识图谱中的源文件信息和内容块数据。通过批量操作，显著提高了系统性能，
减少了数据库连接开销，特别适用于需要同时获取多个资源信息的场景。

主要功能：
- 批量获取源文件元信息（文件名、类型等）
- 批量获取内容块的完整文本内容
- 支持不同类型ID的智能分类和处理
- 提供统一的错误处理和默认值填充

设计特点：
- 静态方法设计，无需实例化即可使用
- 高效的ID分类处理，识别不同类型的标识符
- 健壮的错误处理机制
- 返回标准化的数据结构
"""
from typing import List, Dict
import os
from neo4j import Result

class BatchProcessor:
    """
    Neo4j批量处理类
    
    提供一系列静态方法，用于高效地批量查询和处理Neo4j图数据库中的数据。
    该类专注于批量操作，通过减少数据库交互次数来提高系统性能。
    
    主要功能：
    - 批量获取源文件元信息
    - 批量获取内容块详细内容
    - 智能分类和处理不同类型的标识符
    """
    
    @staticmethod
    def get_source_info_batch(source_ids: List[str], driver) -> Dict[str, Dict]:
        """
        批量获取源文件元信息
        
        该方法接收多个源文件ID，通过一次数据库操作批量获取它们的元信息，
        如文件名、类型等。它能够智能识别不同类型的ID（Chunk ID和Community ID），
        并对每种类型执行专门的查询，最后将结果整合为统一的映射结构。
        
        Args:
            source_ids: 需要获取信息的源文件ID列表，支持复合ID格式
            driver: 已配置的Neo4j驱动实例，用于执行Cypher查询
            
        Returns:
            Dict[str, Dict]: 以源ID为键，源文件信息为值的映射字典
                每个源文件信息包含：
                - file_name: 文件名或描述信息
                
        业务流程：
        1. 验证输入，处理空列表情况
        2. 根据ID特征（长度、格式）将ID分类为Chunk ID和Community ID
        3. 分别对不同类型的ID执行批量查询
        4. 处理查询结果，建立原始ID到信息的映射
        5. 为未找到的ID提供默认信息
        6. 异常处理，确保即使出错也能返回有意义的结果
        
        技术特点：
        - 使用参数化查询防止SQL注入
        - 利用DataFrame格式高效处理查询结果
        - 支持复合ID格式的解析（如"2,chunk_id"格式）
        - 自动处理边缘情况（空ID、未找到的ID）
        """
        if not source_ids:
            return {}
            
        # 创建结果容器
        source_info = {}
        
        try:
            # 分批处理不同类型的ID
            chunk_ids = []
            community_ids = []
            
            # 分类ID
            for source_id in source_ids:
                if not source_id:
                    source_info[source_id] = {"file_name": "未知文件"}
                    continue
                    
                # 检查ID类型
                if len(source_id) == 40:  # SHA1哈希的长度，是Chunk ID
                    chunk_ids.append(source_id)
                else:
                    # 尝试解析复合ID
                    id_parts = source_id.split(",")
                    
                    if len(id_parts) >= 2 and id_parts[0] == "2":
                        chunk_ids.append(id_parts[-1])
                    else:
                        community_id = id_parts[1] if len(id_parts) > 1 else source_id
                        community_ids.append(community_id)
            
            # 如果有Chunk IDs，批量查询
            if chunk_ids:
                chunk_query = """
                MATCH (n:__Chunk__) 
                WHERE n.id IN $ids 
                RETURN n.id AS id, n.fileName AS fileName
                """
                
                chunk_results = driver.execute_query(
                    chunk_query,
                    {"ids": chunk_ids},
                    result_transformer_=Result.to_df
                )
                
                if not chunk_results.empty:
                    for _, row in chunk_results.iterrows():
                        chunk_id = row['id']
                        file_name = row['fileName']
                        base_name = os.path.basename(file_name) if file_name else "未知文件"
                        
                        # 找出原始请求中对应的IDs
                        for src_id in source_ids:
                            if chunk_id == src_id or (len(src_id.split(",")) >= 2 and src_id.split(",")[-1] == chunk_id):
                                source_info[src_id] = {"file_name": base_name}
            
            # 如果有Community IDs，批量查询
            if community_ids:
                community_query = """
                MATCH (n:__Community__) 
                WHERE n.id IN $ids 
                RETURN n.id AS id
                """
                
                community_results = driver.execute_query(
                    community_query,
                    {"ids": community_ids},
                    result_transformer_=Result.to_df
                )
                
                if not community_results.empty:
                    for _, row in community_results.iterrows():
                        community_id = row['id']
                        
                        # 找出原始请求中对应的IDs
                        for src_id in source_ids:
                            id_parts = src_id.split(",")
                            if (len(id_parts) > 1 and id_parts[1] == community_id) or src_id == community_id:
                                source_info[src_id] = {"file_name": "社区摘要"}
            
            # 为未找到的ID添加默认信息
            for source_id in source_ids:
                if source_id not in source_info:
                    source_info[source_id] = {"file_name": f"源文本 {source_id}"}
            
            return source_info
            
        except Exception as e:
            print(f"批量获取源信息失败: {e}")
            # 返回默认值
            return {sid: {"file_name": f"源文本 {sid}"} for sid in source_ids}
    
    @staticmethod
    def get_content_batch(chunk_ids: List[str], driver) -> Dict[str, Dict]:
        """
        批量获取内容块的详细文本内容
        
        该方法接收多个内容块ID，通过一次数据库操作批量获取它们的完整文本内容。
        它能够识别不同类型的ID，并为每种类型获取相应的内容信息，
        包括文件名、文本内容、摘要等，最后整合为统一的映射结构返回。
        
        Args:
            chunk_ids: 需要获取内容的ID列表，支持Chunk ID和Community ID
            driver: 已配置的Neo4j驱动实例，用于执行Cypher查询
            
        Returns:
            Dict[str, Dict]: 以内容ID为键，内容信息为值的映射字典
                每个内容信息包含：
                - content: 格式化的内容文本，包括文件名/类型和实际内容
                
        业务流程：
        1. 验证输入，处理空列表情况
        2. 根据ID特征将ID分类为直接Chunk ID和Community ID
        3. 分别对不同类型的ID执行专门的批量查询
        4. 处理查询结果，构建格式化的内容文本
        5. 建立原始ID到内容的映射关系
        6. 为未找到的ID提供默认内容
        7. 异常处理，确保即使出错也能返回用户可读的错误信息
        
        技术特点：
        - 使用高效的Cypher查询批量获取数据
        - 灵活处理不同类型内容的格式化展示
        - 支持复杂ID格式解析和匹配
        - 健壮的错误处理和默认值机制
        
        业务意义：
        - 显著提高多内容获取的性能和效率
        - 减少数据库连接和查询次数
        - 支持前端高效展示多个相关内容片段
        - 为用户提供统一的内容访问接口
        """
        if not chunk_ids:
            return {}
            
        # 创建结果容器
        chunk_content = {}
        
        try:
            # 分批处理不同类型的ID
            direct_chunk_ids = []
            community_ids = []
            
            # 分类ID
            for chunk_id in chunk_ids:
                if not chunk_id:
                    chunk_content[chunk_id] = {"content": "未提供有效的源ID"}
                    continue
                    
                # 检查ID类型
                if len(chunk_id) == 40:  # SHA1哈希的长度，是Chunk ID
                    direct_chunk_ids.append(chunk_id)
                else:
                    # 尝试解析复合ID
                    id_parts = chunk_id.split(",")
                    
                    if len(id_parts) >= 2 and id_parts[0] == "2":
                        direct_chunk_ids.append(id_parts[-1])
                    else:
                        community_id = id_parts[1] if len(id_parts) > 1 else chunk_id
                        community_ids.append(community_id)
            
            # 如果有直接Chunk IDs，批量查询
            if direct_chunk_ids:
                chunk_query = """
                MATCH (n:__Chunk__) 
                WHERE n.id IN $ids 
                RETURN n.id AS id, n.fileName AS fileName, n.text AS text
                """
                
                chunk_results = driver.execute_query(
                    chunk_query,
                    {"ids": direct_chunk_ids},
                    result_transformer_=Result.to_df
                )
                
                if not chunk_results.empty:
                    for _, row in chunk_results.iterrows():
                        chunk_id = row['id']
                        file_name = row.get('fileName', '未知文件')
                        text = row.get('text', '')
                        content = f"文件名: {file_name}\n\n{text}"
                        
                        # 找出原始请求中对应的IDs
                        for original_id in chunk_ids:
                            if chunk_id == original_id or (len(original_id.split(",")) >= 2 and original_id.split(",")[-1] == chunk_id):
                                chunk_content[original_id] = {"content": content}
            
            # 如果有Community IDs，批量查询
            if community_ids:
                community_query = """
                MATCH (n:__Community__) 
                WHERE n.id IN $ids 
                RETURN n.id AS id, n.summary AS summary, n.full_content AS full_content
                """
                
                community_results = driver.execute_query(
                    community_query,
                    {"ids": community_ids},
                    result_transformer_=Result.to_df
                )
                
                if not community_results.empty:
                    for _, row in community_results.iterrows():
                        comm_id = row['id']
                        summary = row.get('summary', '')
                        full_content = row.get('full_content', '')
                        content = f"摘要:\n{summary}\n\n全文:\n{full_content}"
                        
                        # 找出原始请求中对应的IDs
                        for original_id in chunk_ids:
                            id_parts = original_id.split(",")
                            if (len(id_parts) > 1 and id_parts[1] == comm_id) or original_id == comm_id:
                                chunk_content[original_id] = {"content": content}
            
            # 为未找到的ID添加默认信息
            for chunk_id in chunk_ids:
                if chunk_id not in chunk_content:
                    chunk_content[chunk_id] = {"content": f"未找到相关内容: 源ID {chunk_id}"}
            
            return chunk_content
            
        except Exception as e:
            print(f"批量获取内容失败: {e}")
            # 返回默认值
            return {cid: {"content": f"检索源内容时发生错误: {str(e)}", "chunk_id": cid} for cid in chunk_ids}