import hashlib
from .base import CacheKeyStrategy


class SimpleCacheKeyStrategy(CacheKeyStrategy):
    """
    简单的MD5哈希缓存键策略
    
    基于查询字符串内容生成缓存键，不考虑上下文
    优点是实现简单、速度快，适用于简单场景
    缺点是无法区分上下文相关的相同查询
    """
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        使用查询字符串的MD5哈希生成缓存键
        
        参数:
            query: 查询字符串
            **kwargs: 额外参数（在此策略中被忽略）
            
        返回:
            str: 查询的MD5哈希值，作为缓存键
        """
        # 移除查询前后空白并转换为UTF-8字节
        # 计算MD5哈希值并返回十六进制表示
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()