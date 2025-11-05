import numpy as np
from typing import List, Dict, Any, Union

class VectorUtils:
    """
    向量操作工具类

    核心功能：
    - 余弦相似度计算（单个和批量）
    - 基于相似度的文档排序
    - 相关性过滤
    - 高性能向量操作优化
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

        数学原理:
        - 余弦相似度衡量两个向量方向的相似性，不考虑长度
        - 返回值范围为[-1,1]，但在嵌入向量中通常为[0,1]
        - 值为1表示完全相同的方向，0表示正交，-1表示相反方向
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
        
        设计考虑:
        - 创建候选项副本，避免修改原始数据
        - 只处理有嵌入向量的候选项
        - 提供灵活的top_k参数，控制返回数量
        - 保持原始候选项的所有属性，仅添加新的分数字段
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
        
        特殊处理:
        - 支持具有属性的文档对象，而不仅限于字典
        - 为无向量文档提供默认分数，确保它们也能被包含在结果中
        - 使用阈值过滤，提高检索质量
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
        
        业务意义:
        - 大幅提升批量检索的性能
        - 支持高效的大规模向量搜索
        - 是图检索系统的性能优化关键
        - 适用于需要处理大量候选结果的场景
        """
        # 将向量列表转换为二维numpy数组，便于批量并行计算
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
        
        # 使用矩阵乘法一次性计算所有相似度（矩阵乘法提高效率）
        similarities = np.dot(matrix_normalized, query_normalized)
        return similarities