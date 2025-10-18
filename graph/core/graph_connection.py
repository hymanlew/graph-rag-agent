from typing import Any, Optional
from config.neo4jdb import get_db_manager

class GraphConnectionManager:
    """
    图数据库连接管理器。
    
    负责创建和管理Neo4j图数据库连接，确保连接的复用和高效管理。
    实现了单例模式，确保整个应用中只有一个连接管理器实例，
    从而避免创建过多的数据库连接，提高资源利用效率。
    """
    
    # 类变量，用于存储单例实例
    _instance = None
    
    def __new__(cls):
        """
        单例模式实现，确保只创建一个连接管理器实例
        
        在Python中，通过重写__new__方法实现单例模式，
        检查类变量_instance是否为None，如果是则创建新实例。
        这种模式确保整个应用程序生命周期中只有一个连接管理器实例。
        
        Returns:
            GraphConnectionManager: 类的唯一实例
        """
        # 检查实例是否已存在
        if cls._instance is None:
            # 如果实例不存在，则创建新实例
            cls._instance = super(GraphConnectionManager, cls).__new__(cls)
            # 标记实例尚未初始化
            cls._instance._initialized = False
        # 返回已有实例（或新创建的实例）
        return cls._instance
    
    def __init__(self):
        """
        初始化连接管理器，只在第一次创建时执行
        
        使用_initialized标志确保初始化代码只执行一次，
        获取数据库管理器并从中获取图数据库连接。
        这种设计确保单例模式下的初始化代码不会重复执行。
        """
        # 检查是否已初始化
        if not getattr(self, "_initialized", False):
            # 获取数据库管理器 - 从配置模块获取
            db_manager = get_db_manager()
            # 从数据库管理器获取图数据库连接
            self.graph = db_manager.graph
            # 标记为已初始化
            self._initialized = True
    
    def get_connection(self):
        """
        获取图数据库连接
        
        提供统一的获取图数据库连接的接口，
        确保应用中的其他组件能够方便地访问图数据库。
        这种方式封装了连接获取逻辑，便于管理和维护。
        
        Returns:
            langchain_neo4j.Neo4jGraph: 连接到Neo4j数据库的LangChain图对象
        """
        return self.graph
    
    def refresh_schema(self):
        """
        刷新图数据库模式
        
        当数据库结构发生变化时，需要调用此方法更新内存中的模式缓存，
        以确保后续查询能够正确识别新的节点和关系类型。
        例如，在创建新的节点标签或关系类型后，应调用此方法。
        """
        self.graph.refresh_schema()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> Any:
        """
        执行图数据库查询
        
        提供一个统一的接口来执行Cypher查询，
        支持参数化查询以提高安全性和性能，避免SQL注入攻击。
        这是与图数据库交互的主要方法。
        
        Args:
            query: Cypher查询语句，Neo4j的查询语言
            params: 查询参数字典，用于参数化查询，避免SQL注入风险
            
        Returns:
            Any: 查询结果，具体类型取决于查询内容
                 可能是节点列表、关系列表或其他查询结果形式
        """
        # 如果没有提供参数，使用空字典
        return self.graph.query(query, params or {})
    
    def create_index(self, index_query: str) -> None:
        """
        创建索引
        
        执行索引创建查询，为图数据库中的特定属性创建索引，
        从而提高查询性能。索引对于大型图数据库的查询效率至关重要。
        
        Args:
            index_query: 索引创建查询语句，例如：
                        "CREATE INDEX FOR (n:Entity) ON (n.name)"
                        这个例子为Entity节点的name属性创建索引
        """
        self.graph.query(index_query)
        
    def create_multiple_indexes(self, index_queries: list) -> None:
        """
        创建多个索引
        
        批量创建多个索引，适用于初始化数据库或需要创建多个索引的场景。
        例如在应用启动时，为常用查询路径上的属性创建索引。
        
        Args:
            index_queries: 索引创建查询列表，每个元素为一个索引创建语句
        """
        # 逐个执行索引创建查询
        for query in index_queries:
            self.create_index(query)
            
    def drop_index(self, index_name: str) -> None:
        """
        删除索引
        
        安全地删除指定名称的索引，使用IF EXISTS确保即使索引不存在也不会报错。
        当索引不再需要或需要重建时使用此方法。
        
        Args:
            index_name: 要删除的索引名称
        """
        try:
            # 执行索引删除，使用IF EXISTS确保安全
            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")
            print(f"已删除索引 {index_name}（如果存在）")
        except Exception as e:
            # 捕获并打印错误，但不中断执行
            print(f"删除索引 {index_name} 时出错 (可忽略): {e}")

# 创建全局连接管理器实例，应用中可直接导入使用
# 这种方式允许其他模块通过简单导入使用单例实例
connection_manager = GraphConnectionManager()