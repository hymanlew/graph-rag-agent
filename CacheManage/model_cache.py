"""
模型缓存管理模块，用于预加载和管理SentenceTransformer和其他嵌入模型

本模块是CacheManage系统的重要组成部分，主要负责：
1. 管理模型缓存目录，确保缓存路径结构正确
2. 预加载SentenceTransformer模型到本地缓存
3. 根据配置选择并初始化适当的嵌入模型
4. 提供统一的模型加载接口，简化模型使用流程

在整个缓存系统中的作用：
- 为vector_similarity模块提供嵌入模型支持
- 优化模型加载性能，避免重复下载和加载
- 支持多种嵌入模型提供商，包括OpenAI和SentenceTransformer
- 通过预加载机制提高缓存匹配的响应速度

设计思路：
- 采用懒加载和预加载结合的策略，平衡启动时间和运行时性能
- 支持通过环境变量灵活配置模型类型和路径
- 提供统一的初始化接口，简化集成流程
- 实现完整的错误处理和日志记录，便于调试和监控
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelCache")

# 加载环境变量
load_dotenv()


def ensure_model_cache_dir() -> str:
    """确保模型缓存目录存在，并返回路径
    
    此函数负责：
    1. 从环境变量获取缓存根目录，如未设置则使用默认路径'./cache'
    2. 构建模型缓存子目录路径
    3. 递归创建目录结构（如不存在）
    4. 返回完整的模型缓存目录路径
    
    返回:
        str: 模型缓存目录的绝对路径
    """
    cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
    model_cache_dir = os.path.join(cache_root, 'model')
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    return model_cache_dir


def preload_sentence_transformer_models(models: Optional[List[str]] = None) -> None:
    """预加载SentenceTransformer模型到缓存目录
    
    此函数实现了模型的预加载机制，通过提前下载和初始化模型来优化运行时性能。
    它支持两种模型列表获取方式：
    1. 直接通过参数传入模型名称列表
    2. 从环境变量SENTENCE_TRANSFORMER_MODELS获取逗号分隔的模型列表
    
    实现思路：
    - 使用try-except捕获可能的导入错误，提供优雅降级
    - 对每个模型进行单独的加载和错误处理，确保一个模型失败不会影响其他模型
    - 通过指定缓存目录，确保模型文件被下载到正确位置
    - 采用详细的日志记录，便于跟踪模型加载状态
    
    参数:
        models: 要预加载的模型名称列表，如为None则从环境变量获取
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # 获取要预加载的模型列表
        if models is None:
            models_str = os.getenv('SENTENCE_TRANSFORMER_MODELS', '')
            if not models_str:
                return
            models = [m.strip() for m in models_str.split(',') if m.strip()]
        
        if not models:
            return
            
        # 获取缓存目录
        cache_dir = ensure_model_cache_dir()
        logger.info(f"预加载SentenceTransformer模型到 {cache_dir}")
        
        # 加载每个模型
        for model_name in models:
            try:
                logger.info(f"加载模型: {model_name}")
                # 加载模型，指定缓存目录
                _ = SentenceTransformer(model_name, cache_folder=cache_dir)
                logger.info(f"模型 {model_name} 加载成功")
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {e}")
                
    except ImportError:
        logger.warning("未安装sentence_transformers，跳过预加载")


def preload_cache_embedding_model() -> None:
    """预加载缓存使用的嵌入模型
    
    此函数根据系统配置选择并预加载适当的嵌入模型，支持以下功能：
    1. 从环境变量CACHE_EMBEDDING_PROVIDER判断使用哪种嵌入提供者
    2. 对于OpenAI提供者，不需要预加载（API调用时才会使用）
    3. 对于SentenceTransformer提供者，预加载配置的模型
    
    设计目的：
    - 实现不同嵌入模型提供者的统一接口
    - 确保缓存系统使用的嵌入模型已准备就绪
    - 优化向量相似度计算的性能
    
    与vector_similarity模块的关系：
    - 为VectorSimilarityMatcher提供预加载的嵌入模型
    - 确保get_cache_embedding_provider()返回的模型已就绪
    """
    provider_type = os.getenv('CACHE_EMBEDDING_PROVIDER', 'sentence_transformer').lower()
    
    if provider_type == 'openai':
        # OpenAI模型不需要预加载
        logger.info("使用OpenAI作为缓存嵌入提供者，无需预加载模型")
        return
    
    # 预加载SentenceTransformer模型
    model_name = os.getenv('CACHE_SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    preload_sentence_transformer_models([model_name])


def initialize_model_cache() -> None:
    """初始化模型缓存，预加载配置的模型
    
    此函数是模型缓存系统的主入口，负责协调整个模型缓存的初始化过程：
    1. 确保模型缓存目录存在
    2. 预加载缓存系统使用的嵌入模型
    3. 提供详细的初始化日志
    
    调用时机：
    - 系统启动时可以调用此函数进行预热
    - 也可以作为独立脚本运行，专门用于预加载模型
    
    性能考量：
    - 预加载会增加启动时间，但能显著提升运行时的首次模型访问速度
    - 适合在服务启动阶段或离线环境中执行
    """
    logger.info("初始化模型缓存...")
    
    # 确保缓存目录存在
    cache_dir = ensure_model_cache_dir()
    logger.info(f"模型缓存目录: {cache_dir}")
    
    # 预加载缓存使用的嵌入模型
    preload_cache_embedding_model()
    
    logger.info("模型缓存初始化完成")


if __name__ == "__main__":
    # 直接运行此脚本可以预加载模型
    initialize_model_cache()
