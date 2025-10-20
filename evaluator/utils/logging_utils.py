import os
import logging
from typing import Optional

"""
日志工具模块

此模块提供了评估系统中日志记录的核心功能，通过封装Python标准库的logging模块，
提供了更便捷的日志配置和管理能力。主要功能包括：
- 创建和配置具有控制台和文件输出的日志记录器
- 维护全局日志记录器字典，避免重复创建
- 提供统一的日志格式和级别控制

在评估系统中，日志用于记录评估过程、结果和可能出现的错误，便于问题排查和性能分析。
"""

# 全局日志字典，用于存储已创建的日志记录器，避免重复创建
_loggers = {}

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    创建并配置日志记录器
    
    此函数实现了创建和配置日志记录器的核心逻辑，支持：
    1. 控制台输出：默认将日志输出到控制台
    2. 文件输出：可选将日志同时写入指定文件
    3. 自动目录创建：如果日志文件目录不存在，自动创建
    4. 单例模式：通过全局_loggers字典确保每个名称只创建一个记录器实例
    
    日志格式统一采用 "时间戳 - 记录器名称 - 日志级别 - 消息内容" 的标准格式，
    确保所有组件的日志风格一致，便于后期分析和处理。
    
    Args:
        name: 日志记录器名称，用于标识不同组件的日志来源
        log_file: 可选的日志文件路径，如果提供则同时写入文件
        level: 日志记录级别，默认为INFO
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 检查是否已经有此名称的记录器
    if name in _loggers:
        return _loggers[name]
    
    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 保存记录器
    _loggers[name] = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的日志记录器或创建默认记录器
    
    此函数实现了获取日志记录器的便捷方法，避免了重复配置。
    如果指定名称的记录器已存在，则直接返回；否则，自动创建一个默认配置的记录器。
    
    Args:
        name: 要获取的日志记录器名称
        
    Returns:
        logging.Logger: 对应名称的日志记录器实例
    """
    if name not in _loggers:
        # 如果没有找到记录器，创建一个默认的
        return setup_logger(name)
    
    return _loggers[name]