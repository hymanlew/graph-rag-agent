import logging
import os
from typing import Dict, Any, List, Optional, cast
import pandas as pd
from neo4j import Query
from neo4j import GraphDatabase, Result, Session, Transaction, Driver  # 导入Neo4j官方驱动
from neo4j.exceptions import Neo4jError, ClientError, DatabaseError  # 导入具体异常类
from langchain_neo4j import Neo4jGraph  # 导入LangChain的Neo4j集成
from dotenv import load_dotenv  # 用于加载环境变量
from pandas import DataFrame

logger = logging.getLogger(__name__)

class DBConnectionManager:
    """
    Neo4j图数据库连接管理器，实现单例模式，负责管理所有数据库连接资源。
    
    核心设计原则：
    1. 单例模式：确保整个应用中只有一个数据库连接管理器实例
    2. 会话池管理：通过复用会话减少连接创建和销毁的开销
    3. 双重接口：同时提供原生Neo4j驱动和LangChain Neo4jGraph接口
    4. 资源自动管理：支持上下文管理器模式，确保资源正确释放
    5. 健壮的错误处理：提供全面的异常捕获和日志记录
    """
    
    # 单例实例存储 - 类变量，用于保存唯一的实例引用
    _instance = None
    
    def __new__(cls):
        """
        通过重写__new__方法实现单例模式，确保只创建一个连接管理器实例。
        单例模式确保系统中只存在一个数据库连接管理器，
        __new__方法在__init__之前被调用。
        """
        if cls._instance is None:
            # 创建新实例，获取父类（默认是object）的__new__方法，然后调用它并传入当前类cls来创建一个实例。这个实例就是一个实例对象
            # cls._instance = super(DBConnectionManager, cls).__new__(cls) 这个是旧写法，会有继承性问题
            cls._instance = super().__new__(cls)
            # 标记实例尚未初始化 - 延迟初始化模式
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        初始化数据库连接管理器，只在第一次创建时执行
        
        实现了延迟初始化模式，确保数据库连接只在首次需要时创建。
        初始化过程遵循配置读取、连接建立、资源准备的流程。
        
        初始化步骤：
        1. 检查初始化状态，避免重复初始化
        2. 加载环境变量配置
        3. 初始化Neo4j原生驱动（提供低级访问）
        4. 初始化LangChain Neo4jGraph接口（提供高级集成）
        5. 配置会话池参数
        6. 标记初始化完成
        """
        # 避免重复初始化 - 确保初始化代码只执行一次
        # 这是单例模式实现的重要部分，与__new__方法配合
        if self._initialized:
            return
            
        try:
            # 加载环境变量，获取数据库连接信息
            load_dotenv()
            
            # 从环境变量获取Neo4j连接信息
            self.neo4j_uri = os.getenv('NEO4J_URI')
            self.neo4j_username = os.getenv('NEO4J_USERNAME')
            self.neo4j_password = os.getenv('NEO4J_PASSWORD')
            
            # 验证必要的配置信息
            if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
                raise ValueError("Missing required Neo4j configuration in environment variables")
            
            # 初始化Neo4j官方驱动，配置内置连接池参数
            # Neo4j Python驱动确实内置了连接池管理机制，Driver对象负责管理连接池
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password),
                max_connection_lifetime=60 * 60 * 1000,  # 连接最大生命周期(毫秒)
                max_connection_pool_size=50,             # 每个主机的最大连接数
                connection_timeout=30 * 1000,            # 获取连接的超时时间(毫秒)
            )
            
            # 测试连接，确保数据库可达
            with self.driver.session() as test_session:
                test_session.run("RETURN 1 AS connection_test")
                logger.info("Successfully connected to Neo4j database")
            
            # 初始化LangChain Neo4j图实例，其内部是初始化Neo4j官方驱动，所以也内置了连接池管理机制
            self.graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                refresh_schema=False,  # 按需刷新schema
            )
            
            # 自定义会话池配置 - 这是在驱动连接池之上的会话复用机制
            # 注意：Neo4j驱动已内置连接池，但我们在此之上实现会话级别的复用，
            # 以减少创建/销毁会话对象的开销
            self.session_pool = []  # 存储可复用的会话对象
            self.max_pool_size = 10  # 最大会话池大小
            self._closed = False     # 连接状态标志
            
            # 标记为已初始化
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection manager: {str(e)}")
            # 清理部分初始化的资源
            if hasattr(self, 'driver') and self.driver:
                try:
                    self.driver.close()
                except:
                    pass
            raise  # 重新抛出异常，让调用者知道初始化失败
    
    def get_driver(self) -> Driver:
        """
        获取Neo4j官方驱动实例
        
        Returns:
            Driver: Neo4j官方驱动实例
            
        Raises:
            RuntimeError: 如果连接管理器已关闭
        """
        if self._closed:
            raise RuntimeError("Cannot access driver: Connection manager is closed")
        return self.driver
    
    def get_graph(self) -> Neo4jGraph:
        """
        获取LangChain Neo4jGraph实例
        
        Returns:
            Neo4jGraph: LangChain Neo4j图实例
            
        Raises:
            RuntimeError: 如果连接管理器已关闭
        """
        if self._closed:
            raise RuntimeError("Cannot access graph: Connection manager is closed")
        return self.graph
    
    def execute_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> DataFrame:
        """
        执行Cypher查询并返回结果
        1. 参数化查询：通过params参数支持参数化查询，防止SQL注入
        2. 结果转换：自动将Neo4j结果转换为DataFrame，简化数据处理
        3. 异常传递：保持底层异常传递，便于上层处理特定错误
        4. 会话复用：使用会话池管理会话资源，提高性能
        
        Args:
            cypher: Cypher查询语句，Neo4j的图数据库查询语言
            params: 查询参数字典（cypher 中的 params），用于参数化查询，避免注入风险
            
        Returns:
            pd.DataFrame: 查询结果DataFrame，便于后续数据处理和分析
            
        Raises:
            Neo4jError: 如果查询执行失败
        """
        if self._closed:
            raise RuntimeError("Cannot execute query: Connection manager is closed")
            
        session = None
        try:
            # 从会话池获取会话
            session = self.get_session()

            # data, _, _ = self.driver.execute_query(
            #     Query(text=query, timeout=self.timeout),
            #     database_=self._database,
            #     parameters_=params,
            # )
            # 使用会话池中的会话执行查询，而非直接调用driver.execute_query
            result = session.run(Query(text=cypher, timeout=10), parameters=params or {})
            
            # 将结果转换为DataFrame
            return self._result_to_dataframe(result)
            
        except ClientError as e:
            # 客户端错误，通常是查询语法问题或权限问题
            logger.error(f"Client error executing query: {str(e)}")
            raise
        except DatabaseError as e:
            # 数据库错误，通常是数据库内部问题
            logger.error(f"Database error executing query: {str(e)}")
            raise
        except Exception as e:
            # 其他未预期的错误
            logger.error(f"Unexpected error executing query: {str(e)}")
            raise
        finally:
            # 确保会话被释放回池中
            if session:
                self.release_session(session)
    
    def _result_to_dataframe(self, result: Result) -> DataFrame:
        """
        将Neo4j Result对象转换为pandas DataFrame
        
        Args:
            result: Neo4j查询结果对象
            
        Returns:
            DataFrame: 转换后的DataFrame对象
        """
        # 获取所有记录
        records = list(result)
        if not records:
            return pd.DataFrame()
        
        # 提取列名
        columns = list(records[0].keys())
        
        # 构建数据 - 处理可能的复杂类型
        data = []
        for record in records:
            row = []
            for column in columns:
                value = record[column]
                # 处理某些特殊类型的转换
                if hasattr(value, 'properties') and hasattr(value, 'labels'):
                    # 处理节点对象，转换为字典
                    row.append(value._properties)
                elif hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                    # 处理路径对象
                    row.append(str(value))
                else:
                    row.append(value)
            data.append(row)
        
        return pd.DataFrame(data, columns=columns)
    
    def get_session(self) -> Session:
        """
        从连接池获取原生Neo4j驱动会话
        
        Returns:
            neo4j.Session: Neo4j会话对象，用于执行事务和查询
        """
        if self._closed:
            raise RuntimeError("Cannot get session: Connection manager is closed")
            
        # 优先从池中获取会话
        while self.session_pool:
            try:
                session = self.session_pool.pop()
                # 验证会话是否仍然有效
                if not session.closed():
                    return session
            except Exception:
                # 会话无效，继续尝试下一个
                continue
        
        # 池为空或所有会话都无效，创建新会话
        return self.driver.session()
    
    def release_session(self, session: Session) -> None:
        """
        释放原生Neo4j驱动会话回连接池
        
        Args:
            session: 要释放的Neo4j会话对象
        """
        # 检查会话是否有效
        if session is None or session.closed():
            return
            
        try:
            if len(self.session_pool) < self.max_pool_size and not self._closed:
                # 池未满且管理器未关闭，将会话添加回池中
                self.session_pool.append(session)
            else:
                # 池已满或管理器已关闭，关闭会话释放资源
                session.close()
        except Exception as e:
            logger.warning(f"Error releasing session: {str(e)}")
            # 即使出错也尝试确保会话被关闭
            try:
                if not session.closed():
                    session.close()
            except:
                # 忽略关闭会话时可能出现的错误
                pass
    
    def close(self) -> None:
        """
        关闭所有资源（彻底清理），实现完整的资源生命周期管理

        资源清理策略：
        1. 会话池清理：关闭并释放所有池化会话（包括原生驱动和LangChain图实例）
        2. 错误处理：捕获并忽略关闭过程中的异常，确保清理过程继续
        3. 驱动关闭：释放底层数据库连接资源
        4. 状态重置：清空会话池，为可能的重新初始化做准备
        
        注意事项：
        - 此方法通常在应用程序关闭时调用
        - 调用后，除非重新初始化，否则实例将无法使用
        - 实现了健壮的错误处理，确保即使部分资源已不可用也能继续执行
        """
        # 关闭所有池中的原生驱动会话
        closed_count = 0
        error_count = 0
        
        while self.session_pool:
            session = self.session_pool.pop()
            try:
                if not session.closed():
                    session.close()
                    closed_count += 1
            except Exception as e:
                logger.error(f"Error closing pooled session: {str(e)}")
                error_count += 1
        
        # 清空池
        # 重置会话池状态，释放对会话对象的引用
        self.session_pool = []
        
        # 关闭驱动 - 释放与数据库的所有连接
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j driver successfully closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {str(e)}")
        
        # 注意：LangChain Neo4jGraph实例没有显式的close方法
        # 它内部会管理自己的连接资源，当实例被垃圾回收时会释放资源
        
        # 重置状态
        self._initialized = False
    
    def __enter__(self):
        """
        上下文管理器入口方法，实现Python的上下文协议
        
        使DBConnectionManager支持with语句，提供优雅的资源管理方式。
        返回管理器自身，允许在with块内直接访问其所有方法和属性。
        
        上下文管理器设计特点：
        - 返回self，而非单个会话，提供更灵活的使用方式
        - 支持在with块内调用所有管理器方法
        - 与__exit__配合，确保资源自动释放
        
        使用示例：
            with DBConnectionManager() as db:
                results = db.execute_query("MATCH (n) RETURN n LIMIT 10")
                # 使用results...
        
        Returns:
            DBConnectionManager: 返回自身实例，支持链式调用
        """
        if self._closed:
            # 如果已关闭，重新初始化
            self._initialized = False
            self.__init__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口方法，处理资源清理
        
        当with语句块执行完成或发生异常时自动调用，
        确保无论执行结果如何，资源都能被正确释放。
        
        异常处理行为：
        - 调用close()方法释放所有资源
        - 默认不捕获或抑制异常，让异常正常传播
        - 异常信息(exc_type, exc_val, exc_tb)可用于日志记录或特殊处理
        
        Args:
            exc_type: 异常类型，如果无异常则为None
            exc_val: 异常值，包含异常具体信息
            exc_tb: 异常追踪对象，包含堆栈信息
            
        Returns:
            bool: False表示不抑制异常传播
        """
        # 记录可能的异常信息
        if exc_type:
            logger.warning(f"Exception occurred in context manager: {exc_type.__name__}: {exc_val}")
            
        # 调用close方法清理所有资源
        self.close()

        # 不抑制异常传播
        return False
    
    def begin_transaction(self, session: Session) -> Transaction:
        """
        在指定会话中开始事务
        
        Args:
            session: Neo4j会话对象
            
        Returns:
            Transaction: Neo4j事务对象
            
        Raises:
            RuntimeError: 如果连接管理器已关闭
        """
        if self._closed:
            raise RuntimeError("Cannot begin transaction: Connection manager is closed")
            
        return session.begin_transaction()


# 创建并导出全局数据库连接管理器 单实例（包含上下文管理），提供整个应用的统一访问点
# 单例模式（__new__）+ 工厂函数（get_db_manager()），确保整个应用中只有一个数据库连接管理器实例。
# 两种方式都可以正常使用到上下文管理，但 with get_db_manager() as db 比 with DBConnectionManager() as db 有以下优势：
# - 更好的封装，工厂函数隐藏了实例化细节，客户端代码不需要知道具体的实现方式。
# - 便于未来扩展，并保持单例一致性
# 是软件架构设计中的常见模式。并且要注意，不使用 with 语句则 __enter__ 和 __exit__ 方法不会被自动调用，无法确保连接资源在使用后被正确关闭。
db_manager = DBConnectionManager()


def get_db_manager() -> DBConnectionManager:
    """
    获取数据库连接管理器实例的工厂函数
    
    实现工厂方法设计模式，提供获取单例实例的统一接口。
    这种设计有多重优势：
    
    1. 解耦：客户端代码不直接依赖具体实例，降低耦合度
    2. 灵活性：未来可轻松替换或修改实例创建逻辑
    3. 统一管理：提供获取实例的唯一入口点
    4. 向后兼容：即使内部实现变化，接口保持稳定
    5. 文档化：通过函数注释明确说明返回对象的用途
    
    该函数在应用中广泛使用，确保所有组件使用同一个连接管理器实例，
    避免资源浪费并确保一致性。
    
    Returns:
        DBConnectionManager: 数据库连接管理器的单例实例
    """
    # 简单返回预定义的全局实例
    # 这种实现简洁高效，同时提供了良好的抽象层
    return db_manager


# 推荐用法示例（文档用）
"""
# 1. 使用上下文管理器（推荐用于所有场景）
with get_db_manager() as db:
    # 执行查询
    results = db.execute_query("MATCH (n) RETURN n LIMIT 10")
    # 或获取驱动/图实例
    driver = db.get_driver()
    graph = db.get_graph()
    # 注意：在上下文管理器结束时，所有资源会自动清理

# 2. 直接获取实例（适合长时间运行的应用）
db = get_db_manager()
try:
    # 执行操作
    results = db.execute_query("MATCH (n) RETURN n LIMIT 10")
finally:
    # 在应用退出或不再需要时手动关闭
    db.close()

# 3. 使用事务
with get_db_manager() as db:
    session = db.get_session()
    try:
        tx = db.begin_transaction(session)
        try:
            tx.run("CREATE (n:Person {name: 'Alice'})")
            tx.run("CREATE (n:Person {name: 'Bob'})")
            tx.commit()
        except Exception as e:
            tx.rollback()
            raise
    finally:
        db.release_session(session)
"""