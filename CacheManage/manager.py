import time
import os
from typing import Any, Dict, Optional, Callable
from pathlib import Path

# 导入缓存策略接口和实现
from .strategies import CacheKeyStrategy, SimpleCacheKeyStrategy, ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy
# 导入存储后端接口和实现
from .backends import CacheStorageBackend, MemoryCacheBackend, HybridCacheBackend, ThreadSafeCacheBackend
# 导入缓存项模型
from .models import CacheItem
# 导入向量相似性相关组件
from .vector_similarity import VectorSimilarityMatcher, get_cache_embedding_provider

# 导入配置和环境变量处理
from config.settings import similarity_threshold as st
from dotenv import load_dotenv

# 加载环境变量，支持从.env文件读取配置
load_dotenv()


class CacheManager:
    """
    统一缓存管理器，提供高级缓存功能和向量相似性匹配
    
    这是整个缓存系统的核心组件，负责协调缓存键策略、存储后端和向量相似性匹配器。
    
    核心功能：
    1. 多级缓存查找（精确匹配 + 语义相似匹配）
    2. 可插拔的缓存键生成策略
    3. 可配置的存储后端（内存、磁盘、混合）
    4. 缓存质量控制和验证机制
    5. 性能指标收集
    6. 向量相似性匹配，支持语义级别的缓存查找
    
    设计模式：
    - 策略模式：用于缓存键生成策略
    - 适配器模式：用于不同存储后端的适配
    - 装饰器模式：用于线程安全和性能监控
    """
    
    def __init__(self, 
                 key_strategy: CacheKeyStrategy = None, 
                 storage_backend: CacheStorageBackend = None,
                 cache_dir: str = "./cache",
                 memory_only: bool = False,
                 max_memory_size: int = 100,
                 max_disk_size: int = 1000,
                 thread_safe: bool = True,
                 enable_vector_similarity: bool = True,
                 similarity_threshold: float = st,
                 max_vectors: int = 10000):
        """
        初始化缓存管理器
        
        此初始化方法配置缓存系统的核心组件，允许灵活的缓存策略定制。
        
        参数:
            key_strategy: 缓存键策略，决定如何从查询生成缓存键
                          默认为SimpleCacheKeyStrategy，可根据需求替换为其他策略
                          （如ContextAwareCacheKeyStrategy或ContextAndKeywordAwareCacheKeyStrategy）
            
            storage_backend: 自定义存储后端实现
                          如果为None，则根据memory_only参数自动创建合适的后端
                          允许完全自定义缓存存储逻辑
            
            cache_dir: 缓存目录，磁盘缓存的存储位置
                      对于混合后端和纯磁盘后端，此参数决定数据持久化位置
            
            memory_only: 是否仅使用内存缓存
                      True - 仅内存缓存，速度快但不持久化
                      False - 使用混合缓存（内存+磁盘），提供持久化
            
            max_memory_size: 最大内存缓存数量
                      限制内存中保留的缓存项数量，防止内存溢出
            
            max_disk_size: 最大磁盘缓存数量
                      限制磁盘上的缓存项总数，避免磁盘空间耗尽
            
            thread_safe: 是否线程安全
                      在多线程环境中必须设为True
                      通过ThreadSafeCacheBackend装饰器实现线程安全
            
            enable_vector_similarity: 是否启用向量相似性匹配
                      True - 允许语义相似的查询命中缓存
                      False - 仅支持精确匹配
            
            similarity_threshold: 向量相似度阈值
                      决定两个查询多相似才视为匹配（0-1之间，越高越严格）
                      从配置中导入默认值
            
            max_vectors: 最大向量数量
                      限制向量索引的大小，防止索引过大
                      影响查询性能和内存使用
        """
        # 设置缓存键策略 - 负责将查询转换为唯一的缓存键
        # 如果未提供，则使用简单哈希策略
        self.key_strategy = key_strategy or SimpleCacheKeyStrategy()
        
        # 设置存储后端 - 处理缓存数据的实际存储和检索
        backend = self._create_storage_backend(
            storage_backend, memory_only, cache_dir, 
            max_memory_size, max_disk_size
        )
        
        # 如果需要线程安全，添加线程安全包装器
        # 这确保在多线程环境中安全使用缓存
        self.storage = ThreadSafeCacheBackend(backend) if thread_safe else backend
        
        # 向量相似性匹配器 - 提供语义相似性搜索功能
        self.enable_vector_similarity = enable_vector_similarity
        if enable_vector_similarity:
            # 确保缓存目录存在，用于存储向量索引
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

            # 创建向量索引文件路径 - 用于持久化向量索引
            vector_index_file = f"{cache_dir}/vector_index" if not memory_only else None

            # 获取配置的嵌入提供者 - 用于文本向量化
            embedding_provider = get_cache_embedding_provider()

            self.vector_matcher = VectorSimilarityMatcher(
                embedding_provider=embedding_provider,
                similarity_threshold=similarity_threshold,
                max_vectors=max_vectors,
                index_file=vector_index_file
            )
        else:
            self.vector_matcher = None
        
        # 性能指标收集 - 跟踪缓存操作的性能
        self.performance_metrics = {
            'exact_hits': 0,       # 精确匹配命中次数
            'vector_hits': 0,      # 向量相似匹配命中次数
            'misses': 0,           # 缓存未命中次数
            'total_queries': 0     # 总查询次数
        }
    
    def _create_storage_backend(self, storage_backend, memory_only, cache_dir, 
                              max_memory_size, max_disk_size) -> CacheStorageBackend:
        """
        创建适合需求的存储后端
        
        根据配置选择或创建合适的存储后端实现
        支持自定义后端、纯内存后端和混合后端
        
        参数:
            storage_backend: 自定义存储后端
            memory_only: 是否仅使用内存
            cache_dir: 缓存目录
            max_memory_size: 最大内存缓存数量
            max_disk_size: 最大磁盘缓存数量
            
        返回:
            CacheStorageBackend: 初始化好的存储后端
        """
        if storage_backend:
            # 使用提供的自定义后端
            return storage_backend
        elif memory_only:
            # 使用纯内存后端 - 速度快但不持久化
            return MemoryCacheBackend(max_size=max_memory_size)
        else:
            # 使用混合后端 - 结合内存和磁盘的优势
            return HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=max_memory_size,
                disk_max_size=max_disk_size
            )
    
    def _get_consistent_key(self, query: str, **kwargs) -> str:
        """
        生成一致的缓存键
        
        使用配置的缓存键策略为查询生成唯一标识
        确保相同查询在相同上下文中生成相同的键
        
        参数:
            query: 查询字符串
            **kwargs: 额外的上下文参数
            
        返回:
            str: 生成的缓存键
        """
        return self.key_strategy.generate_key(query, **kwargs)
    
    def _extract_context_info(self, **kwargs) -> Dict[str, Any]:
        """
        提取上下文信息用于向量匹配
        
        从参数中提取会话ID、关键词等上下文信息，
        用于增强向量相似性匹配的相关性
        
        参数:
            **kwargs: 可能包含上下文信息的关键字参数
            
        返回:
            Dict[str, Any]: 提取的上下文信息字典
        """
        return {
            'thread_id': kwargs.get('thread_id', 'default'),           # 会话标识
            'keywords': kwargs.get('keywords', []),                    # 通用关键词
            'low_level_keywords': kwargs.get('low_level_keywords', []), # 低级关键词
            'high_level_keywords': kwargs.get('high_level_keywords', []) # 高级关键词
        }
    
    def get(self, query: str, skip_validation: bool = False, **kwargs) -> Optional[Any]:
        """
        获取缓存内容，支持精确匹配和向量相似性匹配
        
        实现了智能多级缓存查找策略，首先尝试精确匹配，失败后尝试语义相似匹配。
        整个过程记录性能指标，并支持缓存质量验证。
        
        工作流程：
        1. 生成缓存键（使用配置的缓存键策略）
        2. 尝试精确匹配（速度最快，优先级最高）
        3. 如果启用向量相似性且精确匹配失败，尝试语义相似匹配
        4. 对找到的缓存进行质量验证
        5. 返回缓存内容或None
        
        参数:
            query: 查询字符串，用户输入或系统请求
            skip_validation: 是否跳过缓存质量验证
                             True - 直接返回缓存内容，不验证质量
                             False - 仅返回高质量缓存或经过验证的缓存
            **kwargs: 额外参数，包含上下文信息
                      例如：thread_id（会话标识）、keywords（关键词）等
                      这些参数会影响缓存键的生成和向量匹配
            
        返回:
            Optional[Any]: 缓存的内容，如果未找到或质量不符合要求则返回None
        """
        # 记录操作开始时间，用于性能监控
        start_time = time.time()
        # 更新总查询计数
        self.performance_metrics['total_queries'] += 1
        
        # 阶段1：精确匹配
        # 生成一致的缓存键 - 使用配置的缓存键策略
        key = self._get_consistent_key(query, **kwargs)
        
        # 首先尝试精确匹配 - 速度最快，是首选策略
        cached_data = self.storage.get(key)
        if cached_data is not None:
            # 更新精确匹配命中计数
            self.performance_metrics['exact_hits'] += 1
            
            # 将缓存数据转换为CacheItem对象，便于处理元数据
            cache_item = CacheItem.from_any(cached_data)
            # 更新访问统计信息（用于缓存质量评估和淘汰策略）
            cache_item.update_access_stats()
            
            # 质量验证逻辑
            # 如果跳过验证或缓存质量高，则直接返回
            if skip_validation or cache_item.is_high_quality():
                content = cache_item.get_content()
                # 记录操作耗时
                self.performance_metrics["get_time"] = time.time() - start_time
                return content
            
            # 即使不是高质量缓存也返回，但可能后续会被验证
            # 这样设计允许系统收集更多反馈，逐步提高缓存质量
            content = cache_item.get_content()
            self.performance_metrics["get_time"] = time.time() - start_time
            return content
        
        # 阶段2：向量相似性匹配（语义匹配）
        # 如果精确匹配失败且启用了向量相似性，则尝试语义相似匹配
        if self.enable_vector_similarity and self.vector_matcher:
            # 提取上下文信息用于增强向量匹配
            context_info = self._extract_context_info(**kwargs)
            
            # 查找最相似的前3个查询 - 尝试找到语义接近的缓存
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=3)
            
            # 遍历相似查询的缓存键
            for similar_key, similarity_score in similar_keys:
                # 尝试获取相似查询对应的缓存
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    # 更新向量匹配命中计数
                    self.performance_metrics['vector_hits'] += 1
                    
                    # 转换为CacheItem对象
                    cache_item = CacheItem.from_any(cached_data)
                    # 更新访问统计
                    cache_item.update_access_stats()
                    
                    # 添加相似性信息到元数据，便于后续分析和优化
                    cache_item.metadata['similarity_score'] = similarity_score
                    cache_item.metadata['original_query'] = query
                    cache_item.metadata['matched_via_vector'] = True
                    
                    # 同样进行质量验证
                    if skip_validation or cache_item.is_high_quality():
                        content = cache_item.get_content()
                        self.performance_metrics["get_time"] = time.time() - start_time
                        return content
                    
                    # 非高质量缓存也返回，允许系统学习和改进
                    content = cache_item.get_content()
                    self.performance_metrics["get_time"] = time.time() - start_time
                    return content
        
        # 缓存未命中情况
        self.performance_metrics['misses'] += 1
        self.performance_metrics["get_time"] = time.time() - start_time
        return None
    
    def get_fast(self, query: str, **kwargs) -> Optional[Any]:
        """
        快速获取高质量缓存内容
        
        提供优化的快速路径，只返回高质量缓存，
        跳过一些验证步骤以提高性能
        
        参数:
            query: 查询字符串
            **kwargs: 额外参数
            
        返回:
            Optional[Any]: 高质量缓存内容，如果没有则返回None
        """
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is not None:
            cache_item = CacheItem.from_any(cached_data)
            
            # 只返回高质量缓存 - 快速路径的核心优化
            if cache_item.is_high_quality():
                cache_item.update_access_stats()
                
                # 更新上下文历史 - 用于上下文感知策略
                self._update_strategy_history(query, **kwargs)
                
                content = cache_item.get_content()
                self.performance_metrics["fast_get_time"] = time.time() - start_time
                return content
        
        # 尝试向量相似性匹配高质量缓存
        # 只查找最相似的1个查询，提高速度
        if self.enable_vector_similarity and self.vector_matcher:
            context_info = self._extract_context_info(**kwargs)
            similar_keys = self.vector_matcher.find_similar(query, context_info, top_k=1)
            
            for similar_key, similarity_score in similar_keys:
                cached_data = self.storage.get(similar_key)
                if cached_data is not None:
                    cache_item = CacheItem.from_any(cached_data)
                    
                    if cache_item.is_high_quality():
                        cache_item.update_access_stats()
                        cache_item.metadata['similarity_score'] = similarity_score
                        cache_item.metadata['matched_via_vector'] = True
                        
                        content = cache_item.get_content()
                        self.performance_metrics["fast_get_time"] = time.time() - start_time
                        return content
        
        self.performance_metrics["fast_get_time"] = time.time() - start_time
        return None
    
    def set(self, query: str, result: Any, **kwargs) -> None:
        """
        设置缓存内容
        
        将查询和结果保存到缓存中，支持元数据跟踪和向量索引更新。
        这个方法是缓存写入的核心，确保数据被正确存储并可通过精确匹配或语义匹配检索。
        
        工作流程：
        1. 更新缓存策略历史（如上下文感知策略需要）
        2. 生成缓存键
        3. 将结果包装为缓存项对象，添加元数据
        4. 存储到配置的后端
        5. 如果启用向量相似性，更新向量索引
        6. 记录操作性能指标
        
        参数:
            query: 查询字符串，用户输入或系统请求
            result: 要缓存的结果内容
                   可以是任意可序列化的对象或值
            **kwargs: 额外参数
                      - thread_id: 会话标识，用于上下文感知策略
                      - keywords: 查询关键词，用于增强向量匹配
                      - 其他可能影响缓存键生成的参数
        """
        # 记录操作开始时间，用于性能监控
        start_time = time.time()
        
        # 步骤1: 更新策略历史 - 对上下文感知策略至关重要
        # 这允许缓存系统跟踪会话历史，正确处理上下文相关查询
        self._update_strategy_history(query, **kwargs)
        
        # 步骤2: 生成缓存键 - 使用配置的缓存键策略
        # 确保查询和结果之间的一致映射
        key = self._get_consistent_key(query, **kwargs)
        
        # 步骤3: 包装缓存项 - 将结果封装为带元数据的标准格式
        # 添加创建时间、初始质量分数等元数据
        cache_item = self._wrap_cache_item(result)
        
        # 步骤4: 存储缓存项 - 通过配置的存储后端保存
        # 可能是内存、磁盘或混合后端
        self.storage.set(key, cache_item.to_dict())
        
        # 步骤5: 向量索引更新 - 如果启用了向量相似性
        # 确保查询可以通过语义相似性被找到
        if self.enable_vector_similarity and self.vector_matcher:
            # 提取上下文信息增强向量匹配
            context_info = self._extract_context_info(**kwargs)
            # 将查询添加到向量索引
            self.vector_matcher.add_vector(key, query, context_info)
        
        # 记录操作耗时
        self.performance_metrics["set_time"] = time.time() - start_time
    
    def _update_strategy_history(self, query: str, **kwargs):
        """
        更新策略历史
        
        为上下文感知的缓存策略更新查询历史记录
        这对于处理会话中的上下文相关查询很重要
        
        参数:
            query: 查询字符串
            **kwargs: 包含线程ID等信息的参数
        """
        # 检查是否使用上下文感知策略
        if isinstance(self.key_strategy, (ContextAwareCacheKeyStrategy, ContextAndKeywordAwareCacheKeyStrategy)):
            thread_id = kwargs.get("thread_id", "default")
            self.key_strategy.update_history(query, thread_id)
    
    def _wrap_cache_item(self, result: Any) -> CacheItem:
        """
        包装缓存项
        
        将结果封装为标准的CacheItem对象，
        支持从字典或原始值创建
        
        参数:
            result: 要包装的结果，可以是字典或其他值
            
        返回:
            CacheItem: 包装后的缓存项
        """
        if isinstance(result, dict) and "content" in result and "metadata" in result:
            # 如果已经是标准格式的字典，直接解析
            return CacheItem.from_dict(result)
        else:
            # 否则创建新的缓存项
            return CacheItem(result)
    
    def mark_quality(self, query: str, is_positive: bool, **kwargs) -> bool:
        """
        标记缓存质量
        
        这是缓存系统的质量反馈机制，允许用户或系统对缓存内容进行质量评估。
        通过累积反馈，系统能够识别高质量缓存和低质量缓存，优化缓存使用。
        
        工作流程：
        1. 生成缓存键
        2. 获取对应的缓存项
        3. 更新质量标记和分数
        4. 保存更新后的缓存
        5. 记录操作性能指标
        
        质量更新规则：
        - 正面反馈: 质量分数+1，标记为已验证，启用快速路径
        - 负面反馈: 质量分数-2，但不低于-5，禁用快速路径
        
        参数:
            query: 查询字符串，与缓存项关联的原始查询
            is_positive: 是否为正面反馈
                        True - 用户确认答案准确或有用
                        False - 用户认为答案不准确或无用
            **kwargs: 额外参数，影响缓存键生成
            
        返回:
            bool: 是否成功标记缓存质量
                 True - 缓存存在且成功更新
                 False - 找不到对应的缓存项
        """
        start_time = time.time()
        
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            self.performance_metrics["mark_time"] = time.time() - start_time
            return False
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 标记质量 - 更新质量分数和验证状态
        cache_item.mark_quality(is_positive)
        
        # 更新缓存
        item_dict = cache_item.to_dict()
        # 如果是高质量缓存，标记为快速路径可用
        if is_positive and cache_item.is_high_quality():
            item_dict["metadata"]["fast_path_eligible"] = True
        
        self.storage.set(key, item_dict)
        
        self.performance_metrics["mark_time"] = time.time() - start_time
        return True
    
    def delete(self, query: str, **kwargs) -> bool:
        """
        删除缓存项
        
        从缓存和向量索引中删除指定的缓存项
        
        参数:
            query: 查询字符串
            **kwargs: 额外参数
            
        返回:
            bool: 是否成功删除
        """
        # 生成缓存键
        key = self._get_consistent_key(query, **kwargs)
        
        # 从向量索引中删除 - 确保数据一致性
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.remove_vector(key)
        
        # 从存储后端删除缓存项
        return self.storage.delete(key)
    
    def clear(self) -> None:
        """
        清空缓存
        
        清除所有缓存内容和向量索引
        谨慎使用，会删除所有缓存数据
        """
        self.storage.clear()
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.clear()
    
    def flush(self) -> None:
        """
        强制刷新所有待写入的数据到磁盘
        
        确保内存中的缓存数据持久化到磁盘，
        特别是对于混合缓存和向量索引
        """
        # 刷新存储后端
        if hasattr(self.storage, 'backend') and hasattr(self.storage.backend, 'flush'):
            self.storage.backend.flush()
        elif hasattr(self.storage, 'flush'):
            self.storage.flush()
        
        # 如果是混合缓存，需要刷新磁盘缓存部分
        if hasattr(self.storage, 'backend'):
            backend = self.storage.backend
            if hasattr(backend, 'disk_cache') and hasattr(backend.disk_cache, 'flush'):
                backend.disk_cache.flush()
        elif hasattr(self.storage, 'disk_cache') and hasattr(self.storage.disk_cache, 'flush'):
            self.storage.disk_cache.flush()
        
        # 保存向量索引 - 确保向量数据持久化
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        提供缓存系统的性能统计信息，
        包括命中率、未命中率等关键指标
        
        返回:
            Dict[str, Any]: 性能指标字典
        """
        metrics = self.performance_metrics.copy()
        # 计算各种命中率统计
        if metrics['total_queries'] > 0:
            metrics['exact_hit_rate'] = metrics['exact_hits'] / metrics['total_queries']
            metrics['vector_hit_rate'] = metrics['vector_hits'] / metrics['total_queries']
            metrics['total_hit_rate'] = (metrics['exact_hits'] + metrics['vector_hits']) / metrics['total_queries']
            metrics['miss_rate'] = metrics['misses'] / metrics['total_queries']
        return metrics
    
    def validate_answer(self, query: str, answer: str, validator: Callable[[str, str], bool] = None, **kwargs) -> bool:
        """
        验证答案质量
        
        确保返回给用户的答案满足基本质量要求，防止低质量缓存被使用。
        这个方法实现了多级验证策略，优先使用缓存元数据，然后是自定义验证器，最后是默认验证逻辑。
        
        验证流程：
        1. 生成缓存键并查找对应的缓存项
        2. 如果缓存存在且已被用户验证，直接通过验证
        3. 检查缓存质量分数，负分缓存直接不通过
        4. 如果提供了自定义验证器，使用它进行验证
        5. 否则使用默认验证逻辑（长度检查+关键词匹配）
        6. 如果缓存不存在，直接验证提供的答案
        
        参数:
            query: 查询字符串，原始用户查询
            answer: 要验证的答案内容
            validator: 自定义验证函数，签名为 (query, answer) -> bool
                      如果提供，将优先使用此验证器
            **kwargs: 额外参数，影响缓存键生成
            
        返回:
            bool: 答案是否通过验证
                 True - 答案质量符合要求
                 False - 答案质量不满足要求
        """
        # 生成缓存键
        key = self.key_strategy.generate_key(query, **kwargs)
        
        # 获取缓存项
        cached_data = self.storage.get(key)
        if cached_data is None:
            # 如果缓存不存在，直接使用验证函数
            if validator:
                return validator(query, answer)
            return self._default_validation(query, answer)
        
        # 包装为缓存项
        cache_item = CacheItem.from_any(cached_data)
        
        # 检查用户验证状态 - 已验证的缓存直接通过
        if cache_item.metadata.get("user_verified", False):
            return True
        
        # 检查质量分数 - 质量分数为负的缓存不通过
        quality_score = cache_item.metadata.get("quality_score", 0)
        if quality_score < 0:
            return False
        
        # 如果提供了自定义验证函数，使用它
        if validator:
            return validator(query, answer)
        
        # 使用默认验证逻辑
        return self._default_validation(query, answer)
    
    def _default_validation(self, query: str, answer: str) -> bool:
        """
        默认验证逻辑
        
        实现简单但有效的答案质量验证，包含两个关键检查：
        
        1. 长度验证：确保答案有足够的内容，不是简单的短语或空响应
        2. 相关性验证：确保答案包含查询中的关键词，避免返回完全不相关的内容
        
        这些检查虽然简单，但能有效过滤掉低质量或不相关的缓存结果。
        
        参数:
            query: 查询字符串，包含用户想要知道的信息
            answer: 答案文本，需要验证的响应内容
            
        返回:
            bool: 是否通过验证
                 True - 答案通过长度和相关性检查
                 False - 答案太短或与查询不相关
        """
        # 基本验证：长度检查
        if len(answer.strip()) < 10:
            return False
        
        # 检查答案是否与查询相关
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # 至少要有一些共同词汇 - 简单的相关性检查
        common_words = query_words.intersection(answer_words)
        if len(common_words) == 0 and len(query_words) > 2:
            return False
        
        return True
    
    def save_vector_index(self):
        """
        保存向量索引
        
        将向量索引持久化到磁盘，
        确保在程序重启后仍能使用向量相似性功能
        """
        if self.enable_vector_similarity and self.vector_matcher:
            self.vector_matcher.save_index()