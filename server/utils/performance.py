"""
性能监控工具模块

该模块提供API端点性能监控功能，通过装饰器模式实现对异步API函数
执行时间的精确测量、性能日志记录和异常监控。这些性能数据对于系统
调优、问题排查和用户体验优化至关重要。

主要功能：
- 测量异步API函数执行时间
- 记录端点性能日志
- 监控API异常并记录异常性能数据
- 保留原始函数的元数据和文档

设计特点：
- 使用装饰器模式，非侵入式集成
- 支持异步函数的性能测量
- 异常透明传递，不影响原有错误处理
- 轻量级实现，对系统性能影响小
"""
import time
import functools


def measure_performance(endpoint_name):
    """
    API端点性能测量装饰器
    
    该装饰器用于测量异步API函数的执行时间，记录性能日志，并在发生异常时
    记录异常信息和性能数据。它采用装饰器模式，能够在不修改原始函数代码的
    情况下为API端点添加性能监控能力。
    
    Args:
        endpoint_name: API端点名称，用于在性能日志中标识被监控的端点
        
    Returns:
        function: 装饰后的异步函数，保持原始函数的接口和行为不变
        
    业务流程：
    1. 在函数执行前记录开始时间
    2. 执行原始函数，捕获可能的异常
    3. 计算函数执行耗时
    4. 根据执行结果，记录正常性能日志或异常性能日志
    5. 正常情况下返回原始函数的结果，异常情况下重新抛出异常
    
    技术特点：
    - 使用functools.wraps保留原始函数的元数据
    - 支持异步函数的性能测量
    - 异常透明传递，不影响原有错误处理逻辑
    - 精确到毫秒级的性能测量
    
    业务意义：
    - 提供API性能监控数据，用于系统性能评估和优化
    - 快速识别性能瓶颈和异常端点
    - 建立API性能基准，便于持续改进
    - 帮助开发人员了解系统各组件的响应时间
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # 记录性能
                duration = time.time() - start_time
                print(f"API性能 - {endpoint_name}: {duration:.4f}s")
                
                return result
            except Exception as e:
                # 记录异常和性能
                duration = time.time() - start_time
                print(f"API异常 - {endpoint_name}: {str(e)} ({duration:.4f}s)")
                raise
                
        return wrapper
    return decorator