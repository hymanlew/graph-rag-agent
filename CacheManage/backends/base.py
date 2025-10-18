from abc import ABC, abstractmethod
from typing import Any, Optional


class CacheStorageBackend(ABC):
    """
    缓存存储后端的抽象基类
    
    定义了缓存存储的统一接口，允许实现不同的存储策略
    遵循开闭原则，便于扩展新的存储后端类型
    所有具体缓存实现必须继承此类并实现所有抽象方法
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        参数:
            key: 缓存键，用于唯一标识一个缓存项
            
        返回:
            Optional[Any]: 找到的缓存值，如果键不存在则返回None
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        参数:
            key: 缓存键，用于唯一标识一个缓存项
            value: 要存储的缓存值
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        参数:
            key: 要删除的缓存键
            
        返回:
            bool: 删除操作是否成功
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        清空缓存
        
        移除所有缓存项，恢复到初始状态
        """
        pass