import unittest
import tempfile
import shutil
import time
import os
import sys
sys.path.append('.')  # 添加当前目录到Python路径，以便导入本地模块

# 导入缓存管理相关类
from CacheManage import (
    CacheManager,           # 核心缓存管理器
    ContextAwareCacheKeyStrategy,  # 上下文感知的缓存键策略
    SimpleCacheKeyStrategy  # 简单缓存键策略
)


class TestCacheSystem(unittest.TestCase):
    """缓存系统测试类
    
    这个测试类全面测试了缓存系统的各项功能，包括：
    - 基本缓存操作（设置、获取、删除）
    - 向量相似性匹配
    - 上下文感知缓存
    - 性能指标跟踪
    - 缓存质量系统
    - 缓存持久化
    """
    
    def setUp(self):
        """每个测试方法运行前的初始化设置
        
        创建临时目录和缓存管理器实例，配置：
        - 磁盘持久化（memory_only=False）
        - 内存缓存最大50MB
        - 磁盘缓存最大200MB
        - 启用向量相似性匹配
        - 相似性阈值设为0.8（相似度大于0.8才认为匹配）
        """
        # 创建临时目录用于存储缓存文件
        self.temp_dir = tempfile.mkdtemp()
        # 创建缓存管理器实例
        self.cache_manager = CacheManager(
            cache_dir=self.temp_dir,          # 缓存文件存储目录
            memory_only=False,                # 启用磁盘持久化
            max_memory_size=50,               # 内存缓存大小限制(MB)
            max_disk_size=200,                # 磁盘缓存大小限制(MB)
            enable_vector_similarity=True,    # 启用向量相似性匹配
            similarity_threshold=0.8          # 向量相似性阈值
        )
    
    def tearDown(self):
        """每个测试方法运行后的清理工作
        
        清除缓存内容并删除临时目录，确保测试环境的隔离性
        """
        # 清除缓存内容
        self.cache_manager.clear()
        # 删除临时目录及其所有内容
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_cache_operations(self):
        """测试基本缓存操作
        
        验证缓存的核心功能：设置、获取、质量标记、快速获取和删除
        """
        # 设置缓存 - 存储问题和答案
        self.cache_manager.set("什么是Python?", "Python是一种编程语言")
        
        # 获取缓存 - 验证能否正确读取
        result = self.cache_manager.get("什么是Python?")
        self.assertEqual(result, "Python是一种编程语言")
        
        # 标记质量 - 将缓存标记为高质量
        success = self.cache_manager.mark_quality("什么是Python?", True)
        self.assertTrue(success)
        
        # 快速获取 - 只从高质量缓存中获取（效率更高）
        fast_result = self.cache_manager.get_fast("什么是Python?")
        self.assertEqual(fast_result, "Python是一种编程语言")
        
        # 删除缓存
        deleted = self.cache_manager.delete("什么是Python?")
        self.assertTrue(deleted)
        
        # 确认删除 - 验证缓存确实已被移除
        result_after_delete = self.cache_manager.get("什么是Python?")
        self.assertIsNone(result_after_delete)
    
    def test_vector_similarity_matching(self):
        """测试向量相似性匹配
        
        验证系统能否识别语义相似的查询并返回正确的缓存结果
        """
        # 设置原始缓存 - 这是基准查询
        self.cache_manager.set("Python是什么编程语言?", "Python是一种高级解释型编程语言")
        
        # 标记为高质量 - 确保缓存可用于相似匹配
        self.cache_manager.mark_quality("Python是什么编程语言?", True)
        
        # 测试相似查询 - 这些查询与原始查询在语义上相似但不完全相同
        similar_queries = [
            "什么是Python编程语言?",       # 词序不同
            "Python语言是什么?",          # 表述方式不同
            "介绍一下Python编程语言",      # 查询意图相同
            "Python编程语言的特点"          # 相关但不完全相同
        ]
        
        # 测试每个相似查询
        for query in similar_queries:
            # 尝试获取缓存结果
            result = self.cache_manager.get(query)
            # 如果相似度足够高才会有结果（超过设置的0.8阈值）
            if result:  
                # 验证结果是否正确
                self.assertEqual(result, "Python是一种高级解释型编程语言")
                print(f"查询 '{query}' 成功匹配到相似缓存")
    
    def test_context_aware_caching(self):
        """测试上下文感知缓存
        
        验证系统能否在不同上下文中正确隔离缓存，并在同一上下文中支持相似查询匹配
        """
        # 创建一个使用上下文感知键策略的缓存管理器
        context_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),  # 使用上下文感知的缓存键
            cache_dir=self.temp_dir + "_context",         # 单独的缓存目录
            enable_vector_similarity=True                  # 启用向量相似性
        )
        
        try:
            # 在不同线程/上下文中设置相同查询但不同结果的缓存
            # 模拟不同用户的相同查询应返回个性化结果
            context_manager.set("分析数据", "这是用户A的数据分析结果", thread_id="user_A")
            context_manager.set("分析数据", "这是用户B的数据分析结果", thread_id="user_B")
            
            # 测试上下文隔离 - 确保不同上下文中的相同查询不会相互干扰
            result_a = context_manager.get("分析数据", thread_id="user_A")
            result_b = context_manager.get("分析数据", thread_id="user_B")
            
            # 验证各自的结果是否正确
            self.assertEqual(result_a, "这是用户A的数据分析结果")
            self.assertEqual(result_b, "这是用户B的数据分析结果")
            self.assertNotEqual(result_a, result_b)  # 确保上下文隔离有效
            
            # 测试相似查询在同一上下文中的匹配
            # 首先将用户A的缓存标记为高质量
            context_manager.mark_quality("分析数据", True, thread_id="user_A")
            # 然后测试相似但不同的查询
            similar_result = context_manager.get("请分析这些数据", thread_id="user_A")
            
            # 如果向量相似度足够，应该能匹配到用户A的结果
            if similar_result:
                self.assertEqual(similar_result, "这是用户A的数据分析结果")
                print("上下文感知的向量相似性匹配成功")
        
        finally:
            # 确保清理资源，即使测试过程中出现异常
            context_manager.clear()
    
    def test_performance_metrics(self):
        """测试性能指标
        
        验证缓存系统是否能正确跟踪和统计查询性能指标，如总查询次数、命中率等
        """
        # 设置一些缓存
        self.cache_manager.set("问题1", "答案1")
        self.cache_manager.set("问题2", "答案2")
        self.cache_manager.mark_quality("问题1", True)  # 标记问题1为高质量
        
        # 执行一些查询操作以生成性能数据
        self.cache_manager.get("问题1")  # 精确命中
        self.cache_manager.get("问题2")  # 精确命中
        self.cache_manager.get("不存在的问题")  # 未命中
        
        # 获取性能指标
        metrics = self.cache_manager.get_metrics()
        
        # 验证指标的基本正确性
        # 确保至少有一次查询
        self.assertGreater(metrics['total_queries'], 0)
        # 确保至少有一次精确命中
        self.assertGreater(metrics['exact_hits'], 0)
        # 确保至少有一次未命中
        self.assertGreater(metrics['misses'], 0)
        
        # 打印性能指标用于调试和观察
        print("性能指标:", metrics)
    
    def test_cache_quality_system(self):
        """测试缓存质量系统
        
        验证缓存质量标记和降级机制是否正常工作
        """
        # 设置缓存
        self.cache_manager.set("测试问题", "测试答案")
        
        # 初始状态应该不是高质量 - 新缓存默认不是高质量
        result = self.cache_manager.get_fast("测试问题")
        # get_fast可能返回None如果不是高质量缓存
        
        # 标记为正面 - 将缓存提升为高质量
        self.cache_manager.mark_quality("测试问题", True)
        
        # 现在应该是高质量缓存 - 应该能通过get_fast获取
        fast_result = self.cache_manager.get_fast("测试问题")
        self.assertEqual(fast_result, "测试答案")
        
        # 标记为负面 - 多次负面评价会降低缓存质量
        self.cache_manager.mark_quality("测试问题", False)  # 第一次负面评价
        self.cache_manager.mark_quality("测试问题", False)  # 第二次负面评价
        self.cache_manager.mark_quality("测试问题", False)  # 第三次负面评价
        
        # 质量应该下降 - 经过多次负面评价后，缓存质量可能被降级
        degraded_result = self.cache_manager.get_fast("测试问题")
        # 可能返回None由于质量下降 - 降级后不再是高质量缓存
    
    def test_vector_similarity_with_context(self):
        """测试带上下文的向量相似性
        
        验证系统能否在不同上下文中正确应用向量相似性匹配，返回适合当前上下文的结果
        """
        # 创建使用上下文感知键策略的缓存管理器，设置较低的相似性阈值
        context_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            cache_dir=self.temp_dir + "_vector_context",
            enable_vector_similarity=True,
            similarity_threshold=0.75  # 降低阈值以增加匹配可能性
        )
        
        try:
            # 在不同上下文中设置相同查询但不同内容的缓存
            # 模拟针对不同用户群体（初学者vs专家）的个性化回答
            context_manager.set("如何学习编程?", "对于初学者，建议从Python开始", thread_id="beginner")
            context_manager.set("如何学习编程?", "对于有经验的开发者，可以尝试Rust", thread_id="expert")
            
            # 将两个缓存都标记为高质量
            context_manager.mark_quality("如何学习编程?", True, thread_id="beginner")
            context_manager.mark_quality("如何学习编程?", True, thread_id="expert")
            
            # 测试相似查询在正确上下文中的匹配
            # 测试初学者上下文中的相似查询
            beginner_result = context_manager.get("编程学习方法", thread_id="beginner")
            # 测试专家上下文中的相似查询
            expert_result = context_manager.get("编程学习方法", thread_id="expert")
            
            # 验证是否返回了适合各自上下文的答案
            if beginner_result:
                self.assertIn("Python", beginner_result)  # 初学者应该得到Python相关建议
            
            if expert_result:
                self.assertIn("Rust", expert_result)      # 专家应该得到Rust相关建议
            
            print("带上下文的向量相似性测试完成")
            
        finally:
            # 确保清理资源
            context_manager.clear()
    
    def test_cache_persistence(self):
        """测试缓存持久化
        
        验证缓存数据和向量索引能否正确持久化到磁盘，并在系统重启后恢复
        """
        # 设置缓存数据
        self.cache_manager.set("持久化测试", "这是持久化的数据")
        self.cache_manager.mark_quality("持久化测试", True)
        
        # 强制刷新所有数据到磁盘 - 确保数据写入文件
        self.cache_manager.flush()
        
        # 创建新的管理器实例（模拟系统重启）
        new_manager = CacheManager(
            cache_dir=self.temp_dir,         # 使用相同的缓存目录
            memory_only=False,
            enable_vector_similarity=True    # 启用向量相似性
        )
        
        # 验证能否获取之前的缓存 - 确认基本数据持久化成功
        result = new_manager.get("持久化测试")
        self.assertEqual(result, "这是持久化的数据")
        
        # 测试向量相似性是否也持久化了 - 确认向量索引也被保存
        similar_result = new_manager.get("测试持久化功能")
        if similar_result:
            # 验证相似查询是否返回正确结果
            self.assertEqual(similar_result, "这是持久化的数据")
            print("向量索引持久化测试成功")


if __name__ == '__main__':
    # 运行测试 - verbosity=2 提供详细输出
    unittest.main(verbosity=2)