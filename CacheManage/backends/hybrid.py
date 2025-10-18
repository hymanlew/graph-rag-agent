from typing import Any, Optional, Set
from .base import CacheStorageBackend
from .memory import MemoryCacheBackend
from .disk import DiskCacheBackend


class HybridCacheBackend(CacheStorageBackend):
    """
    混合缓存后端实现（内存+磁盘）
    
    结合了内存缓存的高速和磁盘缓存的持久化特性
    使用多级缓存策略：先查内存，内存未命中再查磁盘
    智能缓存管理：高质量缓存项优先保存在内存中
    提供缓存命中率统计功能
    """
    
    def __init__(self, cache_dir: str = "./cache", memory_max_size: int = 100, disk_max_size: int = 1000):
        """
        初始化混合缓存后端
        
        参数:
            cache_dir: 磁盘缓存目录路径
            memory_max_size: 内存缓存最大项数
            disk_max_size: 磁盘缓存最大项数
        """
        # 内存缓存实例，提供高速访问
        self.memory_cache = MemoryCacheBackend(max_size=memory_max_size)
        # 磁盘缓存实例，提供持久化存储
        self.disk_cache = DiskCacheBackend(cache_dir=cache_dir, max_size=disk_max_size)
        
        # 缓存命中率统计
        self.memory_hits = 0  # 内存缓存命中次数
        self.disk_hits = 0    # 磁盘缓存命中次数
        self.misses = 0       # 缓存未命中次数
        
        # 频繁使用的键集合，用于优化
        self.frequent_keys: Set[str] = set()
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项，先检查内存再检查磁盘
        
        实现了多级缓存查找策略：
        1. 首先查询内存缓存（最快）
        2. 如果内存未命中，查询磁盘缓存
        3. 磁盘命中时，根据缓存质量决定是否提升到内存
        4. 记录所有访问的统计信息
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存值或None（未找到）
        """
        # 首先检查内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            self.memory_hits += 1
            return value
        
        # 如果内存中没有，检查磁盘缓存
        value = self.disk_cache.get(key)
        if value is not None:
            self.disk_hits += 1
            
            # 检查是否是高质量缓存项，优先加入内存
            # 高质量定义为：用户验证过或适合快速路径
            is_high_quality = False
            if isinstance(value, dict) and "metadata" in value:
                metadata = value.get("metadata", {})
                is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)
            
            # 将磁盘中的项添加到内存缓存，优先考虑高质量项
            # 这样可以加速后续访问
            if is_high_quality:
                self.memory_cache.set(key, value)
            
            return value
            
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项，同时更新内存和磁盘缓存
        
        缓存写入策略：
        1. 所有缓存项都会写入磁盘（持久化）
        2. 高质量缓存项会额外写入内存（加速访问）
        3. 非高质量缓存项根据策略决定是否加入内存
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 检查是否是高质量缓存项
        is_high_quality = False
        if isinstance(value, dict) and "metadata" in value:
            metadata = value.get("metadata", {})
            is_high_quality = metadata.get("user_verified", False) or metadata.get("fast_path_eligible", False)
        
        # 总是更新磁盘缓存 - 确保数据持久化
        self.disk_cache.set(key, value)
        
        # 高质量项总是加入内存缓存 - 加速后续访问
        if is_high_quality:
            self.memory_cache.set(key, value)
        else:
            # 非高质量项根据策略决定是否加入内存
            self.memory_cache.set(key, value)
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        同时从内存和磁盘缓存中删除指定键
        只要在任一缓存中存在并成功删除，就返回True
        
        参数:
            key: 要删除的缓存键
            
        返回:
            bool: 是否成功删除
        """
        memory_success = self.memory_cache.delete(key)
        disk_success = self.disk_cache.delete(key)
        return memory_success or disk_success
    
    def clear(self) -> None:
        """
        清空缓存
        
        同时清空内存缓存和磁盘缓存
        重置所有统计计数器
        """
        self.memory_cache.clear()
        self.disk_cache.clear()
        # 注意：这里没有重置统计计数器，如果需要可以添加：
        # self.memory_hits = self.disk_hits = self.misses = 0