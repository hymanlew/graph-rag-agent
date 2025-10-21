"""
搜索工具模块

该模块提供了向量搜索和相似度计算的核心工具类，是图检索增强生成（Graph-RAG）
系统中的基础组件。主要实现了向量化搜索所需的各种向量操作，包括余弦相似度计算、
批量相似度处理、文档相关性排序等功能。
"""
import numpy as np
from typing import List, Dict, Any, Union

class VectorUtils:
    """
    向量操作工具类
    
    该类封装了向量相似度计算、排序和过滤的核心功能，为图检索系统提供基础支持。
    设计为静态方法集合，便于跨模块调用，同时确保向量操作的一致性和高性能。
    
    核心功能：
    - 余弦相似度计算（单个和批量）
    - 基于相似度的文档排序
    - 相关性过滤
    - 高性能向量操作优化
    
    算法特点：
    - 实现了数学上精确的余弦相似度公式
    - 考虑了边界情况处理（如零向量）
    - 支持批量计算，提高性能
    - 兼容多种向量表示格式（列表和numpy数组）
    """
    
    @staticmethod
    def cosine_similarity(vec1: Union[List[float], np.ndarray], 
                         vec2: Union[List[float], np.ndarray]) -> float:
        """
        计算两个向量的余弦相似度
        
        参数:
            vec1: 第一个向量，可以是列表或numpy数组
            vec2: 第二个向量，可以是列表或numpy数组
            
        返回:
            float: 相似度值 (0-1)，值越大表示相似度越高
            
        实现思路:
        1. 统一向量格式，转换为numpy数组以提高计算效率
        2. 计算点积（dot product）
        3. 计算两个向量的模长（L2范数）
        4. 实现零向量检查，避免除零错误
        5. 根据余弦相似度公式计算结果: cos(θ) = (A·B) / (||A|| × ||B||)
        
        数学原理:
        - 余弦相似度衡量两个向量方向的相似性，不考虑长度
        - 返回值范围为[-1,1]，但在嵌入向量中通常为[0,1]
        - 值为1表示完全相同的方向，0表示正交，-1表示相反方向
        
        业务意义:
        - 作为语义相似度的主要度量标准
        - 用于判断查询和文档之间的语义相关性
        - 是检索排序的核心依据
        """
        # 确保向量是numpy数组，统一处理格式
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        # 计算向量的模长
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # 避免被零除的边界情况处理
        if norm_a == 0 or norm_b == 0:
            return 0
            
        # 计算余弦相似度
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def rank_by_similarity(query_embedding: List[float], 
                          candidates: List[Dict[str, Any]], 
                          embedding_field: str = "embedding",
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        对候选项按与查询向量的相似度排序
        
        参数:
            query_embedding: 查询向量
            candidates: 候选项列表，每项都包含embedding_field指定的字段
            embedding_field: 包含嵌入向量的字段名，默认为"embedding"
            top_k: 返回的最大结果数，None表示返回所有结果
            
        返回:
            按相似度排序的候选项列表，每项增加"score"字段表示相似度
            
        实现思路:
        1. 创建一个新的列表来存储带有分数的候选项
        2. 遍历每个候选项，检查是否有嵌入向量
        3. 对每个有向量的候选项计算相似度
        4. 创建候选项的副本并添加分数字段
        5. 按相似度分数降序排序
        6. 根据需要限制返回结果数量
        
        设计考虑:
        - 创建候选项副本，避免修改原始数据
        - 只处理有嵌入向量的候选项
        - 提供灵活的top_k参数，控制返回数量
        - 保持原始候选项的所有属性，仅添加新的分数字段
        
        业务意义:
        - 实现语义搜索的核心排序逻辑
        - 将无结构化的候选项转换为有序的结果集
        - 为检索系统提供最相关的内容
        - 是Graph-RAG查询处理的基础组件
        """
        # 创建带有分数的候选项列表
        scored_items = []
        
        for item in candidates:
            # 检查候选项是否有嵌入向量
            if embedding_field in item and item[embedding_field]:
                # 计算相似度
                similarity = VectorUtils.cosine_similarity(query_embedding, item[embedding_field])
                # 复制item并添加分数，避免修改原始数据
                scored_item = item.copy()
                scored_item["score"] = similarity
                scored_items.append(scored_item)
        
        # 按相似度降序排序
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # 如果指定了top_k，则返回前top_k个结果
        if top_k is not None:
            return scored_items[:top_k]
            
        return scored_items
    
    @staticmethod
    def filter_documents_by_relevance(query_embedding: List[float],
                                     docs: List, 
                                     embedding_attr: str = "embedding",
                                     threshold: float = 0.0,
                                     top_k: int = None) -> List:
        """
        基于相似度过滤文档
        
        参数:
            query_embedding: 查询向量
            docs: 文档列表，可以是具有embedding属性的对象
            embedding_attr: 嵌入向量的属性名称
            threshold: 最小相似度阈值，低于此值的文档将被过滤
            top_k: 返回的最大结果数
            
        返回:
            按相似度排序的文档列表
            
        实现思路:
        1. 创建一个新的列表来存储带有分数的文档
        2. 遍历每个文档，尝试获取其嵌入向量
        3. 使用hasattr()处理不同类型的文档对象
        4. 对每个有向量的文档计算相似度
        5. 根据阈值过滤文档
        6. 为没有向量的文档设置默认分数
        7. 按相似度降序排序
        8. 提取排序后的文档对象
        9. 根据需要限制返回结果数量
        
        特殊处理:
        - 支持具有属性的文档对象，而不仅限于字典
        - 为无向量文档提供默认分数，确保它们也能被包含在结果中
        - 使用阈值过滤，提高检索质量
        
        业务意义:
        - 实现相关性过滤，提高检索精度
        - 支持不同类型的文档表示
        - 提供灵活的阈值控制，适应不同的检索需求
        - 是文档检索系统的核心功能之一
        """
        # 创建带有分数的文档列表
        scored_docs = []
        
        for doc in docs:
            # 获取文档的向量表示，支持对象属性形式
            doc_embedding = getattr(doc, embedding_attr, None) if hasattr(doc, embedding_attr) else None
            
            if doc_embedding:
                # 计算相似度
                similarity = VectorUtils.cosine_similarity(query_embedding, doc_embedding)
                # 只添加超过阈值的文档
                if similarity >= threshold:
                    scored_docs.append({
                        'document': doc,
                        'score': similarity
                    })
            else:
                # 如果没有向量，给一个基础分数
                scored_docs.append({
                    'document': doc,
                    'score': 0.0
                })
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 提取排序后的文档对象
        if top_k is not None:
            top_docs = [item['document'] for item in scored_docs[:top_k]]
        else:
            top_docs = [item['document'] for item in scored_docs]
            
        return top_docs
    
    @staticmethod
    def batch_cosine_similarity(query_embedding: np.ndarray, 
                            embeddings: List[np.ndarray]) -> np.ndarray:
        """
        批量计算余弦相似度，提高效率
        
        参数:
            query_embedding: 查询向量
            embeddings: 多个向量的列表
            
        返回:
            包含每个向量相似度的numpy数组
            
        实现思路:
        1. 将向量列表转换为二维numpy数组，实现并行计算
        2. 规范化查询向量（单位向量）
        3. 规范化所有候选向量（单位向量）
        4. 使用矩阵乘法一次性计算所有相似度
        5. 处理零向量的边界情况
        
        性能优化:
        - 使用numpy的向量化操作，避免循环
        - 利用矩阵乘法的并行计算能力
        - 一次性规范化所有向量，减少计算量
        - 使用keepdims参数保持数组维度一致性
        
        业务意义:
        - 大幅提升批量检索的性能
        - 支持高效的大规模向量搜索
        - 是图检索系统的性能优化关键
        - 适用于需要处理大量候选结果的场景
        """
        # 将向量列表转换为二维数组，便于批量计算
        matrix = np.vstack(embeddings)
        
        # 规范化查询向量
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            # 查询向量为零向量，返回全零相似度
            return np.zeros(len(embeddings))
        query_normalized = query_embedding / query_norm
        
        # 规范化所有候选向量（按行）
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)  # 保持维度，便于广播
        # 避免除以零，将零向量的模长设为1
        matrix_norm[matrix_norm == 0] = 1.0
        matrix_normalized = matrix / matrix_norm
        
        # 一次性计算所有相似度（矩阵乘法提高效率）
        similarities = np.dot(matrix_normalized, query_normalized)
        
        return similarities