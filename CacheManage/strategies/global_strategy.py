import hashlib
from .base import CacheKeyStrategy

class GlobalCacheKeyStrategy(CacheKeyStrategy):
    """
    全局缓存键策略
    
    此策略仅使用查询内容生成缓存键，完全忽略会话ID，线程ID和其他上下文信息。
    适用于希望在所有用户/会话间共享缓存结果的场景。
    
    特点:
    - 最大化缓存命中率，相同查询无论来自哪个会话都返回相同结果
    - 节省存储空间，避免缓存冗余数据
    - 适合无状态的查询处理
    
    注意事项:
    - 不适合需要会话隔离的场景
    - 不考虑查询的上下文差异
    - 对于依赖用户状态或会话上下文的查询可能产生错误结果
    """
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        仅使用查询内容生成缓存键，忽略会话ID和其他上下文参数
        
        实现了查询标准化预处理和哈希计算
        
        参数:
            query: 查询字符串，用户输入或系统请求
            **kwargs: 其他上下文参数（在此策略中完全被忽略）
            
        返回:
            str: 基于查询内容的MD5哈希值作为缓存键
        """
        # 预处理：移除可能的前缀（如"generate:"）
        # 这一步可以处理一些特殊格式的查询字符串，提取核心查询内容
        if ":" in query:
            parts = query.split(":", 1)
            if len(parts) > 1:
                query = parts[1]
                
        # 标准化：去除首尾空白字符，直接使用查询内容的MD5哈希作为缓存键
        # 然后计算MD5哈希作为缓存键
        # 使用UTF-8编码确保多语言支持
        return hashlib.md5(query.strip().encode('utf-8')).hexdigest()