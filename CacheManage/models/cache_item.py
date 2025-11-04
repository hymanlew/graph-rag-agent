import time
import json
from typing import Any, Dict, Optional


class CacheItem:
    """缓存项包装类，支持元数据和序列化
    
    CacheItem是缓存系统的核心数据结构，使用标准包装格式，封装了缓存的实际内容和各种元数据。
    它不仅存储数据，还维护了缓存的质量、访问频率、生命周期等信息，为智能缓存策略提供基础支持。
    支持序列化/反序列化，以及缓存质量评估、标记等机制功能。

    在整个缓存系统中的作用：
    - 为所有缓存后端提供统一的数据结构
    - 支持缓存质量评估和优先级排序
    - 便于缓存内容的持久化和恢复
    - 提供访问统计，支持缓存策略优化

    元数据设计：
    - created_at: 创建时间，用于计算缓存年龄
    - quality_score: 质量评分，反映缓存内容的可靠性
    - user_verified: 用户是否已验证，高置信度标记
    - access_count: 访问次数，反映缓存的受欢迎程度
    - fast_path_eligible: 是否符合快速路径条件
    - last_accessed: 最后访问时间，用于LRU策略
    - similarity_score: 相似度评分，用于向量匹配
    - matched_via_vector: 是否通过向量匹配找到
    - original_query: 原始查询，用于上下文理解
    """
    
    def __init__(self, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """初始化缓存项
        
        参数:
            content: 缓存的实际内容
            metadata: 可选的元数据字典
            
        实现思路:
        - 存储实际缓存内容
        - 调用_initialize_metadata确保元数据完整
        - 即使没有提供metadata，也会创建默认的元数据集合
        """
        self.content = content
        self.metadata = self._initialize_metadata(metadata)
    
    def _initialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """初始化元数据，确保包含必要字段
        此方法确保缓存项始终具有完整的元数据集合，即使创建时未提供。
        
        参数:
            metadata: 用户提供的元数据字典
            
        返回:
            合并了默认值和用户提供值的完整元数据字典
        """
        meta = metadata or {}
        
        defaults = {
            "created_at": time.time(),
            "quality_score": 0,
            "user_verified": False,
            "access_count": 0,
            "fast_path_eligible": False,
            "last_accessed": None,
            "similarity_score": None,
            "matched_via_vector": False,
            "original_query": None
        }
        
        # 合并默认值和提供的元数据
        for key, default_value in defaults.items():
            if key not in meta:
                meta[key] = default_value
        
        return meta
    
    def get_content(self) -> Any:
        """获取内容
        
        返回缓存的实际内容，提供统一的访问接口。
        
        返回:
            缓存的原始内容
            
        设计目的:
        - 提供封装的访问方法，而非直接访问属性
        - 保持接口一致性，便于未来扩展
        """
        return self.content
    
    def is_high_quality(self) -> bool:
        """判断是否为高质量缓存

        返回:
            如果缓存项被认为是高质量的，则返回True
            
        质量判断标准:
        - 用户已验证(user_verified = True)
        - 质量评分超过阈值(quality_score > 2)
        - 符合快速路径条件(fast_path_eligible = True)
        
        在缓存策略中的应用:
        - 高质量缓存项可能被优先保留在内存中
        - 高质量缓存项可能有更长的过期时间
        - 高质量缓存项可能获得更高的搜索优先级
        """
        return (self.metadata.get("user_verified", False) or 
                self.metadata.get("quality_score", 0) > 2 or
                self.metadata.get("fast_path_eligible", False))
    
    def mark_quality(self, is_positive: bool) -> None:
        """标记缓存质量
        
        根据用户反馈或系统评估更新缓存项的质量相关元数据。
        
        参数:
            is_positive: 如果为True，表示正面反馈；如果为False，表示负面反馈
            
        实现逻辑:
        - 正面反馈: 增加质量评分，标记为用户验证，设置快速路径资格
        - 负面反馈: 降低质量评分(有下限-5)，取消快速路径资格
        
        设计思路:
        - 负面反馈的影响比正面反馈更强，防止低质量内容积累
        - 设置质量评分下限，避免过度惩罚
        - 结合多种质量指标，提供更全面的质量评估
        """
        if is_positive:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = current_score + 1
            self.metadata["user_verified"] = True
            self.metadata["fast_path_eligible"] = True
        else:
            current_score = self.metadata.get("quality_score", 0)
            self.metadata["quality_score"] = max(-5, current_score - 2)  # 允许负分，但有下限
            self.metadata["fast_path_eligible"] = False
    
    def update_access_stats(self) -> None:
        """更新访问统计
        
        在每次访问缓存项时调用，更新访问次数和最后访问时间。
        这些信息对于LRU策略和访问频率分析至关重要。
        
        实现思路:
        - 原子性地增加访问计数
        - 更新最后访问时间为当前时间
        - 这些数据用于:
          1. LRU(最近最少使用)缓存淘汰策略
          2. 访问频率分析和热点检测
          3. 缓存项价值评估
        """
        self.metadata["access_count"] = self.metadata.get("access_count", 0) + 1
        self.metadata["last_accessed"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        将缓存项转换为字典格式，便于序列化和存储。
        
        返回:
            包含content和metadata的字典
            
        设计目的:
        - 提供统一的序列化格式
        - 便于在不同存储后端之间转换
        - 为JSON序列化提供基础
        """
        return {
            "content": self.content,
            "metadata": self.metadata
        }
    
    def to_json(self, ensure_ascii: bool = False) -> str:
        """转换为JSON字符串
        
        将缓存项序列化为JSON格式，支持持久化存储。
        
        参数:
            ensure_ascii: 是否确保ASCII编码，默认为False(支持Unicode)
            
        返回:
            序列化后的JSON字符串
            
        错误处理:
        - 捕获序列化异常，返回包含错误信息的有效JSON
        - 确保即使序列化失败也不会导致程序崩溃
        - 使用default=str参数处理不可直接序列化的对象
        """
        try:
            return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, default=str)
        except (TypeError, ValueError) as e:
            # 如果序列化失败，返回错误信息
            return json.dumps({
                "content": f"Serialization failed: {str(e)}",
                "metadata": self.metadata
            })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheItem':
        """从字典创建缓存项
        
        从字典格式反序列化缓存项，支持从存储中恢复。
        
        参数:
            data: 包含缓存项数据的字典
            
        返回:
            重建的CacheItem实例
            
        处理策略:
        - 支持标准格式(包含content和metadata字段)
        - 支持简单格式(直接将整个字典作为content)
        - 健壮的错误处理，确保即使数据格式不正确也能返回有效对象
        - 对错误情况创建特殊的错误缓存项，便于调试
        """
        try:
            if isinstance(data, dict):
                if "content" in data and "metadata" in data:
                    metadata = data["metadata"]
                    if not isinstance(metadata, dict):
                        metadata = {}
                    return cls(data["content"], metadata)
                else:
                    # 处理简单格式
                    return cls(data)
            else:
                return cls(data)
        except Exception as e:
            # 返回错误缓存项，确保程序不会崩溃
            return cls(f"Error deserializing cache item: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,  # 标记为低质量
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CacheItem':
        """从JSON字符串创建缓存项
        
        从JSON字符串反序列化缓存项，支持从持久化存储恢复。
        
        参数:
            json_str: 包含缓存项数据的JSON字符串
            
        返回:
            重建的CacheItem实例
            
        错误处理:
        - 捕获JSON解析异常
        - 返回包含错误信息的缓存项
        - 确保反序列化失败不会导致程序崩溃
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            return cls(f"JSON decode error: {str(e)}", {
                "created_at": time.time(),
                "quality_score": -10,
                "user_verified": False,
                "access_count": 0,
                "error": str(e)
            })
    
    @classmethod
    def from_any(cls, data: Any) -> 'CacheItem':
        """从任意数据创建缓存项，具有自动类型检测
        智能工厂方法，能够从多种数据类型创建缓存项，提供最大的灵活性。
        
        参数:
            data: 任意类型的数据
            
        返回:
            根据输入数据类型创建的CacheItem实例
        
        设计目的:
        - 提供统一的创建接口，简化客户端代码
        - 自动处理多种输入格式，增强灵活性
        - 减少类型检查和转换的样板代码
        """
        if isinstance(data, cls):
            return data
        elif isinstance(data, str):
            # 尝试解析JSON
            try:
                parsed_data = json.loads(data)
                return cls.from_dict(parsed_data)
            except json.JSONDecodeError:
                # 如果不是JSON，直接作为内容
                return cls(data)
        elif isinstance(data, dict) and "content" in data:
            return cls.from_dict(data)
        else:
            return cls(data)
    
    def get_age(self) -> float:
        """获取缓存项的年龄（秒）
        
        计算缓存项自创建以来经过的时间，用于TTL(生存时间)策略。
        
        返回:
            缓存项的年龄，以秒为单位
            
        实现思路:
        - 使用当前时间减去创建时间
        - 提供标准化的年龄计算方法
        - 为缓存过期策略提供基础
        """
        created_at = self.metadata.get("created_at", time.time())
        return time.time() - created_at
    
    def is_expired(self, max_age: float) -> bool:
        """检查缓存项是否过期
        
        根据最大允许年龄判断缓存项是否过期。
        
        参数:
            max_age: 最大允许年龄（秒）
            
        返回:
            如果缓存项年龄超过max_age，则返回True
            
        应用场景:
        - TTL(生存时间)缓存策略
        - 定期清理过期缓存
        - 确保缓存内容的时效性
        """
        return self.get_age() > max_age
    
    def __repr__(self) -> str:
        """字符串表示
        
        提供缓存项的可读字符串表示，便于调试和日志记录。
        
        返回:
            包含内容预览、质量评分和访问计数的字符串表示
            
        设计特点:
        - 限制内容预览长度，避免过长输出
        - 显示关键元数据，便于快速评估缓存项
        - 提供简洁而信息丰富的表示
        """
        content_preview = str(self.content)[:50]
        if len(str(self.content)) > 50:
            content_preview += "..."
        
        return (f"CacheItem(content='{content_preview}', "
                f"quality_score={self.metadata.get('quality_score', 0)}, "
                f"access_count={self.metadata.get('access_count', 0)})")