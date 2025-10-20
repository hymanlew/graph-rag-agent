import faiss
import pickle
import os
import threading
from typing import List, Tuple, Dict, Any
from .embeddings import EmbeddingProvider, get_cache_embedding_provider

from config.settings import similarity_threshold as st

"""
向量相似性匹配模块

本模块实现了一个基于FAISS(Facebook AI Similarity Search)的向量相似性匹配器，
用于支持基于语义相似度的缓存匹配功能。它是缓存系统中实现"模糊匹配"和"语义搜索"
的核心组件，能够根据查询的语义含义找到相关的缓存项，而不仅仅是精确匹配。

核心功能：
1. 将查询文本编码为向量表示
2. 使用FAISS库实现高效的向量相似度搜索
3. 维护键到向量的映射关系
4. 支持上下文感知的相似度搜索
5. 提供索引持久化和加载功能

算法原理：
- 使用余弦相似度计算向量之间的语义相似度
- 采用FAISS的IndexFlatIP(内积索引)进行高效向量搜索
- 结合上下文信息进行更精确的匹配过滤
- 支持相似度阈值过滤，控制匹配质量
"""


class VectorSimilarityMatcher:
    """向量相似性匹配器，支持基于向量相似度的缓存匹配
    
    这个类是缓存系统实现语义匹配的核心组件，通过将文本查询转换为向量表示，
    并使用高效的向量搜索算法找到语义相似的缓存项。它支持：
    
    1. 精确的余弦相似度计算
    2. 高效的近似最近邻搜索
    3. 上下文感知的匹配过滤
    4. 索引的持久化和恢复
    5. 线程安全的并发操作
    
    与缓存系统的集成：
    - 为CacheManager提供语义搜索能力
    - 支持基于向量相似度的缓存查找
    - 增强缓存命中率，特别是对于相似但不完全相同的查询
    """

    def __init__(self,
                 embedding_provider: EmbeddingProvider = None,
                 similarity_threshold: float = st,
                 max_vectors: int = 10000,
                 index_file: str = None):
        """
        初始化向量相似性匹配器

        参数:
            embedding_provider: 嵌入向量提供者，如果为None则根据配置自动选择
            similarity_threshold: 相似度阈值，用于过滤不相关的匹配结果
            max_vectors: 最大向量数量，防止内存溢出
            index_file: 索引文件路径，用于持久化索引
            
        初始化流程:
        1. 设置嵌入提供者或使用默认配置的提供者
        2. 配置相似度阈值和容量限制
        3. 初始化FAISS索引（使用内积索引，适用于归一化向量的余弦相似度计算）
        4. 创建各种映射字典，维护键、索引、上下文和原始查询之间的关系
        5. 如果提供了索引文件路径，尝试加载已持久化的索引
        """
        self.embedding_provider = embedding_provider or get_cache_embedding_provider()
        self.similarity_threshold = similarity_threshold
        self.max_vectors = max_vectors
        self.index_file = index_file
        
        # 初始化FAISS索引
        self.dimension = self.embedding_provider.get_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 存储键到向量的映射
        self.key_to_index = {}
        self.index_to_key = {}
        self.key_to_context = {}
        self.key_to_query = {}  # 存储原始查询
        
        self._lock = threading.RLock()
        self._next_index = 0
        
        # 如果指定了索引文件，尝试加载
        if self.index_file and os.path.exists(f"{self.index_file}.pkl"):
            self._load_index()
    
    def add_vector(self, cache_key: str, query: str, context_info: Dict[str, Any] = None):
        """添加向量到索引
        
        将缓存键对应的查询文本编码为向量，并添加到FAISS索引中。
        同时维护键到向量索引、上下文和原始查询的映射关系。
        
        参数:
            cache_key: 缓存键
            query: 查询文本，用于生成向量
            context_info: 上下文信息，用于匹配过滤
            
        实现步骤:
        1. 如果键已存在，先移除旧的向量（确保更新时的一致性）
        2. 使用嵌入提供者将查询文本编码为向量
        3. 将向量添加到FAISS索引
        4. 更新各种映射字典
        5. 检查是否超出最大容量，如果超出则触发清理
        
        线程安全保证:
        - 使用可重入锁保证并发安全
        - 确保添加操作的原子性
        """
        with self._lock:
            # 如果已存在，先删除
            if cache_key in self.key_to_index:
                self.remove_vector(cache_key)

            # 生成嵌入向量
            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            # 添加到FAISS索引
            faiss_index = self._next_index
            self.index.add(embedding)

            # 更新映射
            self.key_to_index[cache_key] = faiss_index
            self.index_to_key[faiss_index] = cache_key
            self.key_to_context[cache_key] = context_info or {}
            self.key_to_query[cache_key] = query

            self._next_index += 1

            # 检查是否超出最大容量
            if self._next_index > self.max_vectors:
                self._cleanup_old_vectors()
    
    def find_similar(self, query: str, context_info: Dict[str, Any] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相似的缓存键
        
        这是向量匹配的核心方法，根据查询文本和上下文信息找到语义相似的缓存键。
        
        参数:
            query: 查询文本
            context_info: 上下文信息，用于过滤匹配结果
            top_k: 返回的最大结果数量
            
        返回:
            包含(cache_key, similarity_score)元组的列表，按相似度降序排列
            
        算法流程:
        1. 检查索引是否为空，为空则直接返回空列表
        2. 将查询文本编码为向量
        3. 使用FAISS搜索最相似的向量（搜索top_k*2个结果以留出过滤空间）
        4. 遍历搜索结果，应用上下文匹配过滤
        5. 应用相似度阈值过滤
        6. 对结果按相似度降序排序，并限制返回数量为top_k
        
        性能优化:
        - 搜索时获取两倍于top_k的结果，确保过滤后仍有足够结果
        - 使用FAISS的批处理能力进行高效搜索
        - 只返回符合相似度阈值的高质量匹配
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []

            # 生成查询向量
            query_embedding = self.embedding_provider.encode(query)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # 搜索相似向量
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.index_to_key):
                    continue

                if idx in self.index_to_key:
                    cache_key = self.index_to_key[idx]

                    # 检查上下文匹配
                    if self._context_matches(context_info, self.key_to_context.get(cache_key, {})):
                        if score >= self.similarity_threshold:
                            results.append((cache_key, float(score)))

            # 按相似度排序
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def remove_vector(self, cache_key: str):
        """从索引中移除向量
        
        从FAISS索引和所有映射字典中移除指定缓存键对应的向量。
        注意：由于FAISS索引不直接支持删除操作，此实现主要是从映射关系中删除，
        实际的向量仍保留在索引中，但由于映射被删除，不会被检索到。
        
        参数:
            cache_key: 要移除的缓存键
            
        实现思路:
        1. 检查键是否存在于映射中
        2. 从key_to_index映射中获取FAISS索引ID
        3. 从所有映射字典中删除对应的条目
        4. 注意：FAISS索引本身不直接支持删除，旧向量仍保留在索引中但无法访问
        """
        with self._lock:
            if cache_key not in self.key_to_index:
                return
            
            faiss_index = self.key_to_index[cache_key]
            
            # 从映射中删除
            del self.key_to_index[cache_key]
            if faiss_index in self.index_to_key:
                del self.index_to_key[faiss_index]
            if cache_key in self.key_to_context:
                del self.key_to_context[cache_key]
            if cache_key in self.key_to_query:
                del self.key_to_query[cache_key]
    
    def clear(self):
        """清空所有向量
        
        完全重置向量索引和所有映射关系。
        
        实现步骤:
        1. 重置FAISS索引
        2. 清空所有映射字典
        3. 重置索引计数器
        
        线程安全保证:
        - 使用锁确保清空操作的原子性
        - 防止在清空过程中的读写操作导致不一致状态
        """
        with self._lock:
            self.index.reset()
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0
    
    def _context_matches(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> bool:
        """检查两个上下文是否匹配
        
        这是一个辅助方法，用于确保只有在相同上下文中的查询才能匹配，
        特别是在会话感知的缓存场景中非常重要。
        
        参数:
            context1: 第一个上下文信息
            context2: 第二个上下文信息
            
        返回:
            如果上下文匹配，则返回True
            
        实现逻辑:
        1. 如果两个上下文都为空，则认为匹配
        2. 如果只有一个上下文为空，则认为不匹配
        3. 检查两个上下文的thread_id是否相同
        
        设计考虑:
        - 当前实现主要检查thread_id，但可以扩展以支持更多上下文匹配规则
        - 上下文匹配确保了会话隔离，避免跨会话的不相关匹配
        """
        if not context1 and not context2:
            return True
        
        if not context1 or not context2:
            return False
        
        # 检查线程ID是否匹配
        thread_id1 = context1.get('thread_id', 'default')
        thread_id2 = context2.get('thread_id', 'default')
        
        return thread_id1 == thread_id2
    
    def _cleanup_old_vectors(self):
        """清理旧向量以保持在最大容量内
        
        当向量数量超过max_vectors限制时，此方法负责清理最旧的向量。
        注意：当前实现只是一个占位符，实际的清理逻辑需要根据具体策略实现。
        
        可能的实现策略:
        1. LRU(最近最少使用)：移除最久未被访问的向量
        2. FIFO(先进先出)：移除最早添加的向量
        3. 基于质量：保留高质量向量，优先移除低质量向量
        4. 混合策略：结合访问时间、质量和使用频率
        """
        # 重建索引，保留最近的向量
        pass
    
    def save_index(self, file_path: str = None):
        """保存索引到文件
        
        将FAISS索引和所有映射关系持久化到磁盘，以便在系统重启后恢复。
        
        参数:
            file_path: 保存路径，如果为None则使用初始化时设置的路径
            
        实现步骤:
        1. 确定最终的文件路径
        2. 准备要保存的数据字典，包含所有映射关系
        3. 将FAISS索引保存为单独的文件
        4. 使用pickle序列化并保存映射关系
        5. 捕获并记录可能的异常，确保程序不会崩溃
        
        保存格式:
        - {file_path}.faiss: FAISS索引文件
        - {file_path}.pkl: 映射关系的pickle文件
        """
        if file_path is None:
            file_path = self.index_file
        
        if file_path is None:
            return
        
        with self._lock:
            try:
                data = {
                    'key_to_index': self.key_to_index,
                    'index_to_key': self.index_to_key,
                    'key_to_context': self.key_to_context,
                    'key_to_query': self.key_to_query,
                    'next_index': self._next_index
                }
                
                # 保存FAISS索引
                if self.index.ntotal > 0:
                    faiss.write_index(self.index, f"{file_path}.faiss")
                
                # 保存映射关系
                with open(f"{file_path}.pkl", 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"保存向量索引失败: {e}")
    
    def _load_index(self):
        """从文件加载索引
        
        从磁盘加载之前保存的FAISS索引和映射关系，恢复向量匹配器的状态。
        
        实现步骤:
        1. 加载pickle文件，恢复所有映射关系
        2. 尝试加载FAISS索引文件
        3. 如果FAISS文件不存在，调用_rebuild_index重建索引
        4. 捕获并记录可能的异常，如果加载失败则重置为初始状态
        
        错误处理:
        - 加载失败时提供优雅降级，重置为新的空索引
        - 记录详细错误信息，便于调试
        """
        try:
            # 加载映射关系
            with open(f"{self.index_file}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.key_to_index = data.get('key_to_index', {})
                self.index_to_key = data.get('index_to_key', {})
                self.key_to_context = data.get('key_to_context', {})
                self.key_to_query = data.get('key_to_query', {})
                self._next_index = data.get('next_index', 0)
            
            # 加载FAISS索引
            faiss_file = f"{self.index_file}.faiss"
            if os.path.exists(faiss_file):
                self.index = faiss.read_index(faiss_file)
            else:
                # 如果FAISS文件不存在，重建索引
                self._rebuild_index()
                
        except Exception as e:
            print(f"加载向量索引失败: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.key_to_index.clear()
            self.index_to_key.clear()
            self.key_to_context.clear()
            self.key_to_query.clear()
            self._next_index = 0
    
    def _rebuild_index(self):
        """重建FAISS索引
        
        当FAISS索引文件丢失或损坏时，使用保存的查询文本重建索引。
        
        实现步骤:
        1. 检查是否有保存的查询文本
        2. 创建新的FAISS索引
        3. 遍历所有保存的查询，重新编码为向量并添加到索引
        
        设计目的:
        - 提供索引恢复机制，增强系统的鲁棒性
        - 允许在只有映射关系的情况下重建功能完整的索引
        - 处理FAISS索引文件损坏或丢失的情况
        """
        if not self.key_to_query:
            return
        
        # 重新创建索引
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 重新添加所有向量
        for cache_key, query in self.key_to_query.items():
            embedding = self.embedding_provider.encode(query)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            self.index.add(embedding)