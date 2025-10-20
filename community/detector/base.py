from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from graphdatascience import GraphDataScience
from langchain_community.graphs import Neo4jGraph
import psutil
import os
import time
from contextlib import contextmanager

from config.settings import MAX_WORKERS, GDS_CONCURRENCY, GDS_MEMORY_LIMIT

"""
社区检测器基类模块

本模块定义了所有社区检测器必须实现的抽象基类，提供了通用的功能和统一的接口。
基类实现了社区检测的通用流程控制，包括资源管理、性能监控和结果处理，
同时定义了具体检测器需要实现的抽象方法。

设计理念：
- 采用模板方法模式，定义算法骨架
- 分离通用逻辑和算法特定逻辑
- 提供统一的性能监控和错误处理
- 实现资源管理和自动清理

核心功能：
1. 系统资源检测和参数自动调整
2. 图投影管理和资源清理
3. 完整的社区检测流程控制
4. 性能指标收集和报告
5. 异常处理和错误报告
"""

class BaseCommunityDetector(ABC):
    """社区检测基类
    
    所有社区检测算法的抽象基类，定义了社区检测的通用框架和必须实现的接口。
    实现了模板方法模式，其中process()方法定义了算法骨架，而具体算法细节由子类实现。
    
    主要职责：
    - 管理图数据库连接和图投影
    - 监控和报告性能指标
    - 根据系统资源自动调整参数
    - 提供统一的异常处理机制
    - 确保资源的正确释放
    """
    
    def __init__(self, gds: GraphDataScience, graph: Neo4jGraph):
        """初始化社区检测器
        
        参数:
            gds: GraphDataScience实例，用于执行图算法
            graph: Neo4jGraph实例，表示要分析的图
            
        初始化步骤:
        1. 保存图数据库连接和GDS客户端
        2. 设置投影名称
        3. 初始化性能统计计数器
        4. 检测系统资源并调整运行参数
        """
        # 保存图数据库连接
        self.gds = gds
        self.graph = graph
        # 设置默认投影名称
        self.projection_name = "communities"
        # 投影图对象引用，初始为None
        self.G = None
        
        # 性能统计指标
        self.projection_time = 0    # 图投影耗时
        self.detection_time = 0     # 社区检测耗时
        self.save_time = 0          # 保存结果耗时
        
        # 初始化系统资源和运行参数
        self._init_system_resources()
        
    def _init_system_resources(self):
        """初始化系统资源参数
        
        检测系统资源情况，为算法执行提供资源限制信息。
        
        实现步骤:
        1. 获取系统总内存（MB）
        2. 获取CPU核心数
        3. 调用_adjust_parameters()根据资源情况调整运行参数
        """
        # 获取系统内存总量（MB）
        self.memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        # 获取CPU核心数，默认为4
        self.cpu_count = os.cpu_count() or 4
        # 根据系统资源调整算法参数
        self._adjust_parameters()
        
    def _adjust_parameters(self):
        """调整运行参数
        
        根据系统资源和配置文件设置算法的运行参数，优化性能和资源使用。
        
        调整策略:
        - 设置并发度为配置文件中指定的值
        - 根据内存大小设置节点数量限制和超时时间
        - 打印参数信息，便于调试和监控
        
        参数说明:
        - max_concurrency: 算法执行的最大并发度
        - node_count_limit: 处理的最大节点数量
        - timeout_seconds: 算法执行的超时时间（秒）
        """
        # 设置并发度
        self.max_concurrency = GDS_CONCURRENCY
        
        # 使用配置的内存限制
        memory_gb = GDS_MEMORY_LIMIT
        
        # 根据配置的内存大小调整限制
        # 大内存环境可以处理更大规模的图和更长的超时时间
        if memory_gb > 32:
            self.node_count_limit = 100000  # 32GB以上内存支持10万个节点
            self.timeout_seconds = 600      # 超时时间10分钟
        elif memory_gb > 16:
            self.node_count_limit = 50000   # 16-32GB内存支持5万个节点
            self.timeout_seconds = 300      # 超时时间5分钟
        else:
            self.node_count_limit = 20000   # 16GB以下内存支持2万个节点
            self.timeout_seconds = 180      # 超时时间3分钟
            
        # 打印参数信息，便于调试和监控
        print(f"社区检测参数: CPU={MAX_WORKERS}, 内存={memory_gb:.1f}GB, "
            f"并发度={self.max_concurrency}, 节点限制={self.node_count_limit}")

    @contextmanager
    def _graph_projection_context(self):
        """图投影的上下文管理器
        
        提供一个上下文管理器，负责图投影的创建和自动清理。
        确保无论执行是否成功，都能正确清理资源，避免内存泄漏。
        
        实现步骤:
        1. 记录开始时间
        2. 创建图投影
        3. 计算投影耗时
        4. 执行上下文内容
        5. 无论是否发生异常，都执行清理操作
        6. 记录并打印清理耗时
        """
        try:
            # 记录投影开始时间
            projection_start = time.time()
            # 创建图投影
            self.create_projection()
            # 计算投影耗时
            self.projection_time = time.time() - projection_start
            # 执行上下文内容
            yield
        finally:
            # 无论如何都执行清理
            cleanup_start = time.time()
            self.cleanup()
            print(f"图投影清理完成，耗时: {time.time() - cleanup_start:.2f}秒")

    def process(self) -> Dict[str, Any]:
        """执行完整的社区检测流程
        
        这是模板方法，定义了社区检测的整体流程骨架，调用子类实现的具体方法。
        负责协调投影创建、社区检测和结果保存，并收集性能指标。
        
        返回:
            包含检测结果、状态和性能指标的字典
            
        异常:
            任何异常都会被捕获并记录，但会重新抛出以便上层处理
            
        流程:
        1. 记录开始时间并打印日志
        2. 初始化结果字典
        3. 在图投影上下文中执行检测:
           a. 调用detect_communities()执行检测
           b. 调用save_communities()保存结果
        4. 收集并记录性能指标
        5. 异常处理和错误报告
        """
        # 记录开始时间
        start_time = time.time()
        # 打印开始日志
        print(f"开始执行{self.__class__.__name__}社区检测...")
        
        # 初始化结果字典
        results = {
            'status': 'success',     # 初始状态设为成功
            'algorithm': self.__class__.__name__,  # 记录使用的算法
            'details': {}            # 存储详细结果
        }
        
        try:
            # 在图投影上下文中执行检测流程
            with self._graph_projection_context():
                # 执行社区检测
                detection_start = time.time()
                detection_result = self.detect_communities()
                self.detection_time = time.time() - detection_start
                results['details']['detection'] = detection_result
                
                # 保存社区结果
                save_start = time.time()
                save_result = self.save_communities()
                self.save_time = time.time() - save_start
                results['details']['save'] = save_result
                
            # 添加性能统计信息
            total_time = time.time() - start_time
            results['performance'] = {
                'totalTime': total_time,         # 总耗时
                'projectionTime': self.projection_time,  # 投影耗时
                'detectionTime': self.detection_time,    # 检测耗时
                'saveTime': self.save_time               # 保存耗时
            }
            
            return results
            
        except Exception as e:
            # 异常处理，更新结果状态
            results.update({
                'status': 'error',          # 将状态更新为错误
                'error': str(e),            # 记录错误信息
                'elapsed': time.time() - start_time  # 记录已用时间
            })
            raise  # 重新抛出异常，让上层处理

    def create_projection(self) -> Tuple[Any, Dict]:
        """创建图投影
        
        在内存中创建图的投影，以便执行社区检测算法。
        默认实现为空，子类可以根据需要重写。
        
        返回:
            投影图对象和配置信息的元组
        """
        pass

    @abstractmethod
    def detect_communities(self) -> Dict[str, Any]:
        """检测社区
        
        所有子类必须实现的抽象方法，执行实际的社区检测算法。
        
        返回:
            包含检测结果的字典，具体格式由算法实现定义
        """
        pass

    @abstractmethod
    def save_communities(self) -> Dict[str, int]:
        """保存社区结果
        
        所有子类必须实现的抽象方法，将检测到的社区保存到图数据库。
        
        返回:
            包含保存结果统计信息的字典
        """
        pass

    def cleanup(self):
        """清理资源
        
        清理图投影和其他临时资源，避免内存泄漏。
        调用时机：在图投影上下文管理器的finally块中自动调用。
        
        实现步骤:
        1. 检查投影图对象是否存在
        2. 尝试删除投影图
        3. 捕获并记录可能的异常
        4. 重置投影图引用为None
        """
        if self.G:
            try:
                self.G.drop()
                print("社区投影图已清理")
            except Exception as e:
                print(f"清理投影图出错: {e}")
            finally:
                self.G = None