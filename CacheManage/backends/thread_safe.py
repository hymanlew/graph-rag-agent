import threading
from typing import Any, Optional
from .base import CacheStorageBackend

"""
线程安全缓存后端模块

本模块实现了一个线程安全的缓存后端装饰器，通过装饰器模式为任何缓存存储后端
添加线程安全特性，确保在多线程环境下的并发访问安全性。

设计思路：
- 采用装饰器模式，避免修改原有缓存后端代码
- 使用可重入锁(RLock)，支持在同一线程内多次获取锁
- 对所有缓存操作进行同步，保证原子性
- 保持与CacheStorageBackend接口完全兼容

在整个缓存系统中的作用：
- 为memory、disk、hybrid等各种缓存后端提供线程安全保障
- 简化并发环境下的缓存使用，无需上层应用关心线程安全问题
- 支持高并发场景下的缓存操作
"""


class ThreadSafeCacheBackend(CacheStorageBackend):
    """线程安全的缓存后端装饰器
    
    这是一个装饰器类，通过组合方式包装任何现有的缓存存储后端，
    为其添加线程安全特性。实现了CacheStorageBackend接口，
    可以无缝替换任何非线程安全的缓存后端。
    
    使用可重入锁(RLock)而不是普通锁(Lock)的原因：
    1. 支持在同一线程内多次获取锁，避免死锁
    2. 允许被装饰后端在内部调用自身的方法
    3. 更灵活地支持复杂的缓存操作链
    """

    def __init__(self, backend: CacheStorageBackend):
        """
        初始化线程安全缓存后端
        
        参数:
            backend: 被装饰的缓存后端，必须实现CacheStorageBackend接口
            
        实现思路：
        - 保存对原始缓存后端的引用
        - 创建可重入锁用于同步访问
        - 无需修改原始后端，采用组合方式实现装饰器模式
        """
        self.backend = backend
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        在线程安全的上下文中执行get操作，确保读取操作的原子性。
        使用with语句管理锁的获取和释放，确保即使发生异常也能正确释放锁。
        
        参数:
            key: 要获取的缓存键
            
        返回:
            缓存的值，如果不存在则返回None
            
        线程安全保证：
        - 所有读取操作都是原子的，不会被其他线程中断
        - 避免了读取到部分更新的数据
        """
        with self.lock:
            return self.backend.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存
        
        在线程安全的上下文中执行set操作，确保写入操作的原子性。
        
        参数:
            key: 缓存键
            value: 要存储的值
            
        线程安全保证：
        - 确保缓存项的完整写入，不会被其他线程中断
        - 避免了数据竞争和不一致状态
        - 在多线程同时设置同一键时，保证最终结果是最后一次写入的值
        """
        with self.lock:
            self.backend.set(key, value)
    
    def delete(self, key: str) -> bool:
        """删除缓存项
        
        在线程安全的上下文中执行delete操作，确保删除操作的原子性。
        
        参数:
            key: 要删除的缓存键
            
        返回:
            如果项被删除则为True，否则为False
            
        线程安全保证：
        - 确保删除操作要么完全执行，要么完全不执行
        - 避免了删除过程中的数据不一致
        - 正确返回操作结果，反映实际执行状态
        """
        with self.lock:
            return self.backend.delete(key)
    
    def clear(self) -> None:
        """清空缓存
        
        在线程安全的上下文中执行clear操作，确保清空操作的原子性。
        
        线程安全保证：
        - 确保缓存的清空操作是原子的，不会被其他线程中断
        - 避免了在清空过程中其他线程看到部分清空的缓存状态
        - 防止在清空期间的读写操作导致数据不一致
        """
        with self.lock:
            self.backend.clear()