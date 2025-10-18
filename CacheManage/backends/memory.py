import time
from typing import Any, Optional
from .base import CacheStorageBackend


class MemoryCacheBackend(CacheStorageBackend):
    """
    内存缓存后端实现
    
    基于Python字典实现的内存缓存，使用LRU（Least Recently Used）淘汰策略
    适用于对速度要求高、不需要持久化存储的场景
    限制最大项数以防止内存溢出
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化内存缓存后端
        
        参数:
            max_size: 缓存最大项数，超过这个数量将触发LRU淘汰
        """
        # 主缓存字典，存储键值对
        self.cache = {}
        # 缓存最大容量限制
        self.max_size = max_size
        # 存储每个键的访问时间，用于LRU策略
        self.access_times = {}  # 用于LRU淘汰策略
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存项值，不存在则返回None
        """
        value = self.cache.get(key)
        if value is not None:
            # 更新访问时间（LRU策略）
            # 这是LRU策略的核心：每次访问都更新时间戳
            self.access_times[key] = time.time()
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        如果缓存已满且要添加的键不存在，则会删除最久未使用的项
        每次设置都会更新访问时间，确保最近使用的项不会被淘汰
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存已满且不是更新现有键，则需要淘汰最旧的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        # 存储或更新缓存项
        self.cache[key] = value
        # 更新访问时间
        self.access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        从缓存和访问时间记录中同时移除键
        
        参数:
            key: 缓存键
            
        返回:
            bool: 是否成功删除
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """
        清空缓存
        
        删除所有缓存项和访问时间记录
        """
        self.cache.clear()
        self.access_times.clear()  # 确保同时清空访问时间字典
    
    def _evict_lru(self) -> None:
        """
        淘汰最久未使用的缓存项(LRU算法)
        
        内部方法，通过查找最早的访问时间来确定要淘汰的项
        使用Python的min函数和lambda表达式高效找到最旧的键
        """
        if not self.access_times:
            return
        
        # 找出访问时间最早的键
        # lambda x: x[1] 告诉min函数使用字典项的第二个元素(时间戳)作为比较依据
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        # 使用delete方法确保同时清理cache和access_times
        self.delete(oldest_key)
        
    def cleanup_unused(self) -> None:
        """
        清理access_times中未使用的键
        
        防止access_times字典无限增长，处理缓存和访问时间不同步的情况
        例如：当通过其他方式修改缓存字典后
        """
        # 找出那些在access_times中存在但在cache中不存在的键
        # 这种情况可能发生在直接操作cache字典时
        unused_keys = [k for k in self.access_times if k not in self.cache]
        for key in unused_keys:
            del self.access_times[key]