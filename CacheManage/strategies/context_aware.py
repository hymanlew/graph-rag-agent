import hashlib
from .base import CacheKeyStrategy


class ContextAwareCacheKeyStrategy(CacheKeyStrategy):
    """
    上下文感知的缓存键策略，考虑会话历史
    
    这种策略通过将查询与其会话上下文结合来生成缓存键
    可以区分相同查询在不同上下文中的不同含义
    适用于需要上下文理解的对话场景
    """
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文感知缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
                           这个窗口大小决定了上下文的广度
        """
        # 上下文窗口大小
        self.context_window = context_window
        # 存储每个线程的会话历史
        self.conversation_history = {}
        # 存储每个线程的历史版本号
        self.history_versions = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """
        更新会话历史
        
        将新的查询添加到会话历史中，维护上下文信息
        
        参数:
            query: 当前查询
            thread_id: 会话线程ID，用于区分不同会话
            max_history: 最大历史记录条数，防止内存溢出
        """
        # 如果是新会话，初始化历史记录
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小 - 只保留最新的记录
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        # 这是一个重要的机制，确保不同上下文的相同查询生成不同的键
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        生成上下文感知的缓存键
        
        通过结合会话ID、上下文历史、版本号和查询内容生成唯一键
        
        参数:
            query: 查询字符串
            **kwargs: 额外参数，必须包含thread_id以区分会话
            
        返回:
            str: 生成的缓存键
        """
        # 获取会话ID，默认为"default"
        thread_id = kwargs.get("thread_id", "default")
        
        # 获取当前会话的历史记录
        history = self.conversation_history.get(thread_id, [])
        
        # 获取历史版本号
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 只包含最近的n条消息
        # 这样可以在保持上下文的同时避免键过长
        context = " ".join(history[-self.context_window:] if self.context_window > 0 else [])
        
        # 组合上下文、线程ID、版本和查询生成缓存键
        # 这种组合方式确保:
        # 1. 不同会话的相同查询生成不同的键
        # 2. 相同会话但不同上下文的相同查询生成不同的键
        # 3. 历史变化时键也会变化
        combined = f"thread:{thread_id}|ctx:{context}|v{version}|{query}".strip()
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


class ContextAndKeywordAwareCacheKeyStrategy(CacheKeyStrategy):
    """
    结合上下文和关键词的缓存键策略，同时考虑会话历史和关键词
    
    这是上下文感知策略的增强版，除了会话历史外，还考虑了关键词信息
    关键词可以分为低级关键词和高级关键词，提供更精细的缓存控制
    适用于需要基于概念和主题组织缓存的复杂系统
    """
    
    def __init__(self, context_window: int = 3):
        """
        初始化上下文与关键词感知的缓存键策略
        
        参数:
            context_window: 要考虑的前几条会话历史记录
        """
        self.context_window = context_window
        self.conversation_history = {}
        self.history_versions = {}
    
    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """
        更新会话历史
        
        与基本的上下文感知策略相同的历史管理机制
        
        参数:
            query: 当前查询
            thread_id: 会话线程ID
            max_history: 最大历史记录条数
        """
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0
        
        # 添加新查询到历史
        self.conversation_history[thread_id].append(query)
        
        # 保持历史记录在可管理的大小
        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]
        
        # 增加版本号，确保上下文变化时键也会变化
        self.history_versions[thread_id] += 1
    
    def generate_key(self, query: str, **kwargs) -> str:
        """
        生成同时考虑上下文和关键词的缓存键
        
        这是一个更复杂的键生成策略，结合了多种因素:
        1. 会话线程ID
        2. 查询内容
        3. 上下文历史(哈希处理)
        4. 历史版本号
        5. 低级关键词
        6. 高级关键词
        
        参数:
            query: 查询字符串
            **kwargs: 额外参数，包含thread_id、low_level_keywords和high_level_keywords
            
        返回:
            str: 生成的缓存键
        """
        # 基础键部分：会话ID和查询内容
        thread_id = kwargs.get("thread_id", "default")
        key_parts = [f"thread:{thread_id}", query.strip()]
        
        # 添加上下文信息 - 使用哈希处理减少长度
        history = self.conversation_history.get(thread_id, [])
        version = self.history_versions.get(thread_id, 0)
        
        # 构建上下文字符串 - 包含最近的n条消息
        if self.context_window > 0 and history:
            context = " ".join(history[-self.context_window:])
            # 使用哈希处理上下文，避免键过长
            key_parts.append(f"ctx:{hashlib.md5(context.encode('utf-8')).hexdigest()}")
        
        # 添加版本号
        key_parts.append(f"v:{version}")
        
        # 添加低级关键词 - 排序确保一致性
        low_level_keywords = kwargs.get("low_level_keywords", [])
        if low_level_keywords:
            # 排序关键词，确保同样的关键词集合生成相同的部分
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))
        
        # 添加高级关键词 - 排序确保一致性
        high_level_keywords = kwargs.get("high_level_keywords", [])
        if high_level_keywords:
            key_parts.append("high:" + ",".join(sorted(high_level_keywords)))
        
        # 使用双竖线连接各部分，生成最终的键
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()