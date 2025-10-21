"""
数据库配置与管理模块

该模块提供了数据库连接管理和访问的抽象层，主要负责：
1. 从配置系统导入数据库管理器
2. 提供统一的数据库访问接口给应用其他模块使用
3. 定义数据库管理器的基本接口规范

这种设计使得数据库配置与应用逻辑解耦，便于更换数据库实现或修改连接配置。
"""
# 从配置模块导入原始的数据库管理器工厂函数
from config.neo4jdb import get_db_manager as original_get_db_manager

class DatabaseManager:
    """
    Neo4j 数据库管理类
    
    定义了数据库管理器的基本接口和功能，负责数据库连接的生命周期管理。
    这是一个基础类，实际使用的是从config模块导入的实现。
    
    属性：
        driver: Neo4j驱动实例，用于执行查询和管理事务
    
    方法：
        close(): 关闭数据库连接，释放资源
    """
    def __init__(self):
        """初始化数据库管理器，创建驱动但不立即连接"""
        self.driver = None

    def close(self):
        """关闭数据库连接
        
        安全地关闭数据库驱动实例，释放所有相关资源。
        该方法应该在应用关闭时被调用，确保不会有连接泄漏。
        """
        if self.driver:
            self.driver.close()


def get_db_manager():
    """
    获取数据库管理器实例
    
    工厂函数，返回一个配置好的数据库管理器实例。
    实际返回的是从config.neo4jdb模块导入的实现，这里只作为统一的访问入口。
    
    Returns:
        DatabaseManager: 配置好的数据库管理器实例，提供执行查询和管理连接的方法
        
    业务意义：
        - 提供统一的数据库访问入口
        - 实现数据库配置与业务逻辑的分离
        - 便于替换或升级数据库实现
    """
    return original_get_db_manager()