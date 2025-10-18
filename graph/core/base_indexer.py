import time
import concurrent.futures
from typing import List, Any, Optional

from config.settings import MAX_WORKERS as CONFIG_MAX_WORKERS

class BaseIndexer:
    """
    基础索引器类，为各种索引器提供通用功能。
    包含批处理、并行计算和性能监控逻辑，是所有具体索引器的基类。
    
    主要功能：
    - 提供通用的批处理框架
    - 支持并行处理以提高效率
    - 包含性能监控工具
    - 定义索引器的基本接口
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        初始化基础索引器
        
        Args:
            batch_size: 批处理大小，决定每次处理的数据量
            max_workers: 并行工作线程数，控制并行处理的并发度
        """
        # 批处理和并行参数
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # 性能监控参数
        self.embedding_time = 0  # 嵌入计算耗时
        self.db_time = 0         # 数据库操作耗时
        
    def _create_indexes(self) -> None:
        """
        创建必要的索引以优化查询性能 - 由子类实现
        
        子类必须实现此方法以创建适合特定索引类型的数据库索引。
        这些索引对于提高大规模数据检索的性能至关重要。
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_optimal_batch_size(self, total_items: int) -> int:
        """
        计算最优批处理大小
        
        根据总项目数量动态计算最适合的批处理大小，
        确保批次既不过大也不过小，以平衡内存使用和处理效率。
        
        Args:
            total_items: 待处理的总项目数量
            
        Returns:
            int: 计算得到的最优批处理大小
        """
        # 使用配置的批处理大小作为上限，同时确保最小为20
        optimal_size = min(self.batch_size, max(20, total_items // 10))
        return optimal_size
    
    def batch_process_with_progress(self, 
                                   items: List[Any], 
                                   process_func, 
                                   batch_size: Optional[int] = None, 
                                   desc: str = "处理中") -> None:
        """
        通用批处理逻辑，带进度跟踪
        
        将大量项目分成较小的批次进行处理，并显示处理进度和性能指标，
        适用于需要处理大量数据且希望实时了解进度的场景。
        
        Args:
            items: 待处理项目列表
            process_func: 处理单个批次的函数，接收批次数据和批次索引
            batch_size: 批处理大小，如果不提供则使用最优值
            desc: 进度描述文本
        """
        if not items:
            print(f"没有找到需要处理的项目")
            return
            
        # 计算批处理参数
        item_count = len(items)
        optimal_batch_size = batch_size or self.get_optimal_batch_size(item_count)
        total_batches = (item_count + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"{desc}: 共{item_count}项，批次大小: {optimal_batch_size}, 总批次: {total_batches}")
        
        # 保存每个批次的处理时间，用于计算平均处理速度
        batch_times = []
        
        # 批量处理循环
        for batch_index in range(total_batches):
            batch_start = time.time()
            
            # 计算当前批次的起始和结束索引
            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, item_count)
            batch = items[start_idx:end_idx]
            
            # 处理当前批次
            process_func(batch, batch_index)
            
            # 计算和显示进度
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # 计算平均时间和剩余时间，提供实时进度反馈
            avg_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - (batch_index + 1)
            estimated_remaining = avg_time * remaining_batches
            
            print(f"已处理批次 {batch_index+1}/{total_batches}, "
                  f"批次耗时: {batch_time:.2f}秒, "
                  f"平均: {avg_time:.2f}秒/批, "
                  f"预计剩余: {estimated_remaining:.2f}秒")
    
    def process_in_parallel(self, items: List[Any], process_func) -> List[Any]:
        """
        并行处理项目
        
        使用线程池并行处理多个项目，提高处理效率，
        适用于独立且计算密集型的任务。
        
        Args:
            items: 待处理项目列表  
            process_func: 处理单个项目的函数
            
        Returns:
            List[Any]: 处理结果列表，与输入顺序可能不同
        """
        # 确保线程数不超过全局配置限制
        max_workers = min(self.max_workers, CONFIG_MAX_WORKERS)
        results = []
        
        # 创建线程池并提交任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_item = {
                executor.submit(process_func, item): i 
                for i, item in enumerate(items)
            }
            
            # 收集并处理完成的任务结果
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"并行处理出错: {e}")
        
        return results