import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Union
from sentence_transformers import SentenceTransformer
import threading
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

"""
嵌入向量提供者模块

本模块实现了多种嵌入向量提供者，用于将文本转换为向量表示，
这是向量相似性匹配的基础。模块采用了抽象基类和具体实现的设计模式，
支持多种嵌入模型提供商，包括OpenAI和SentenceTransformer。

核心功能：
1. 提供统一的嵌入向量生成接口
2. 支持多种嵌入模型的无缝切换
3. 实现单例模式，避免重复加载模型
4. 支持模型缓存，提高性能
5. 提供向量归一化，优化相似度计算

设计思路：
- 使用抽象基类定义统一接口
- 实现具体的提供者类，支持不同的嵌入模型
- 采用单例模式管理模型实例，节省资源
- 提供工厂函数，根据配置自动选择合适的提供者
- 支持向量归一化，提高相似度计算的准确性
"""


class EmbeddingProvider(ABC):
    """嵌入向量提供者抽象基类
    
    定义了所有嵌入向量提供者必须实现的接口，确保系统可以无缝切换
    不同的嵌入模型实现。这是策略模式的应用，将算法族（不同的嵌入模型）封装
    在各自的类中，使它们可以互相替换。
    
    核心接口：
    - encode: 将文本编码为向量表示
    - get_dimension: 获取向量的维度
    
    设计目的：
    - 解耦具体的嵌入模型实现与使用代码
    - 支持运行时切换不同的嵌入模型
    - 便于扩展新的嵌入提供者
    - 统一接口，简化调用代码
    """

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """将文本编码为向量
        
        所有嵌入提供者必须实现的核心方法，负责将文本或文本列表转换为向量表示。
        
        参数:
            texts: 单个文本字符串或文本列表
            
        返回:
            numpy数组，形状为 [num_texts, dimension] 的向量表示
            
        设计考虑:
        - 支持单文本和多文本批处理，优化性能
        - 返回numpy数组，便于后续处理和相似度计算
        - 通常实现会包括向量归一化步骤
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度
        
        返回嵌入向量的维度，这对于初始化FAISS索引等操作非常重要。
        
        返回:
            嵌入向量的维度（整数）
            
        实现思路:
        - 通常会缓存维度值，避免重复计算
        - 首次调用时可能通过编码一个测试文本来获取维度
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """基于OpenAI API的嵌入向量提供者，复用RAG的向量模型
    
    这个类实现了基于OpenAI API的嵌入向量生成功能，通过复用RAG系统中已有的
    嵌入模型，确保整个系统使用一致的向量表示。采用单例模式避免重复创建实例。
    
    主要特点：
    - 复用系统中已配置的OpenAI嵌入模型
    - 实现单例模式，节省API调用资源
    - 支持向量归一化，优化余弦相似度计算
    - 懒加载初始化，只在需要时创建模型
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式，避免重复创建
        
        实现线程安全的单例模式，确保整个应用中只有一个OpenAIEmbeddingProvider实例。
        
        实现思路：
        1. 使用类变量_instance保存单例实例
        2. 使用线程锁确保线程安全
        3. 检查_instance是否存在，不存在则创建
        4. 返回唯一实例
        
        设计目的：
        - 避免重复初始化，节省资源
        - 确保系统中使用一致的嵌入模型配置
        - 优化API调用，避免不必要的请求
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """初始化OpenAI嵌入提供者
        
        实现懒加载初始化，只在第一次使用时创建模型实例。
        检查是否已经初始化，避免重复初始化。
        
        初始化步骤：
        1. 检查是否已初始化，已初始化则直接返回
        2. 尝试导入model.get_models模块中的get_embeddings_model函数
        3. 获取并保存嵌入模型实例
        4. 初始化维度缓存为None
        5. 标记为已初始化
        
        错误处理：
        - 如果导入失败，抛出ImportError异常
        - 提供清晰的错误信息，便于调试
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 导入并复用现有的embedding模型
        try:
            from model.get_models import get_embeddings_model
            self.model = get_embeddings_model()
            self._dimension = None
            self._initialized = True
        except ImportError as e:
            raise ImportError(f"无法导入embedding模型: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量
        
        使用OpenAI模型将文本转换为向量表示，并进行归一化处理。
        
        参数:
            texts: 单个文本字符串或文本列表
            
        返回:
            归一化后的向量表示
            
        实现步骤:
        1. 处理输入，确保是列表格式
        2. 使用OpenAI模型生成嵌入向量
        3. 将结果转换为numpy数组
        4. 对向量进行L2归一化，使余弦相似度计算等同于内积计算
        5. 返回归一化后的向量
        
        性能优化:
        - 支持批处理，优化多个文本的编码效率
        - 向量归一化，提高后续相似度计算效率
        """
        if isinstance(texts, str):
            texts = [texts]

        # 使用OpenAI embedding模型
        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # 归一化向量
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度
        
        返回OpenAI嵌入向量的维度，使用缓存避免重复计算。
        
        返回:
            嵌入向量的维度
            
        实现思路:
        1. 检查维度是否已缓存
        2. 如果未缓存，编码一个测试文本来获取维度
        3. 缓存并返回维度值
        
        设计考虑:
        - 使用懒计算模式，只在需要时确定维度
        - 缓存结果，提高性能
        - 使用简单文本作为测试，减少计算量
        """
        if self._dimension is None:
            # 使用一个简单文本获取维度
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


class SentenceTransformerEmbedding(EmbeddingProvider):
    """基于SentenceTransformer的嵌入向量提供者，支持模型缓存
    
    这个类实现了基于SentenceTransformer库的嵌入向量生成功能，适用于本地运行的
    嵌入模型。支持多种预训练模型，并且实现了按模型名称的单例模式，确保相同模型
    只被加载一次。
    
    主要特点：
    - 支持多种预训练的SentenceTransformer模型
    - 实现基于模型名称的单例模式
    - 支持自定义模型缓存目录
    - 自动处理模型下载和缓存
    - 提供向量归一化功能
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        """单例模式，避免重复加载模型
        
        实现基于模型名称的线程安全单例模式，确保相同模型只被加载一次。
        
        参数:
            model_name: SentenceTransformer模型名称
            cache_dir: 模型缓存目录
            
        返回:
            SentenceTransformerEmbedding实例
            
        实现思路:
        1. 使用字典保存不同模型名称的实例
        2. 使用线程锁确保线程安全
        3. 检查指定模型名称的实例是否存在，不存在则创建
        4. 返回对应的实例
        
        设计优势:
        - 避免相同模型的重复加载，节省内存
        - 支持同时使用多个不同的模型
        - 线程安全，适用于并发环境
        """
        with cls._lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = super().__new__(cls)
                cls._instances[model_name]._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        """初始化SentenceTransformer嵌入提供者
        
        实现懒加载初始化，设置模型名称和缓存目录，并加载模型。
        
        参数:
            model_name: SentenceTransformer模型名称，默认为'all-MiniLM-L6-v2'
            cache_dir: 模型缓存目录，如为None则使用环境变量或默认路径
            
        初始化步骤：
        1. 检查是否已初始化，已初始化则直接返回
        2. 设置模型名称
        3. 确定模型缓存目录（优先使用参数，其次环境变量，最后默认值）
        4. 确保缓存目录存在（创建必要的父目录）
        5. 加载SentenceTransformer模型，指定缓存目录
        6. 初始化维度缓存为None
        7. 标记为已初始化
        
        设计考虑：
        - 支持灵活的缓存目录配置
        - 自动创建必要的目录结构
        - 与model_cache模块配合，支持模型预加载
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.model_name = model_name

        # 设置模型缓存目录
        if cache_dir is None:
            cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
            cache_dir = os.path.join(cache_root, 'model')

        # 确保缓存目录存在
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型，指定缓存目录
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self._dimension = None
        self._initialized = True

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量
        
        使用SentenceTransformer模型将文本转换为向量表示。
        
        参数:
            texts: 单个文本字符串或文本列表
            
        返回:
            归一化后的向量表示
            
        实现特点:
        - 利用SentenceTransformer的批处理能力
        - 设置convert_to_numpy=True，直接返回numpy数组
        - 设置normalize_embeddings=True，自动进行向量归一化
        - 支持单个文本输入的便捷处理
        
        性能优化:
        - 批处理多个文本，提高吞吐量
        - 自动归一化，简化后续处理
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def get_dimension(self) -> int:
        """获取向量维度
        
        返回SentenceTransformer嵌入向量的维度，使用缓存避免重复计算。
        
        返回:
            嵌入向量的维度
            
        实现思路:
        1. 检查维度是否已缓存
        2. 如果未缓存，编码一个测试文本来获取维度
        3. 缓存并返回维度值
        
        与OpenAI提供者的区别：
        - 实现逻辑相似，但使用不同的底层模型
        - 保持接口一致性，便于切换
        """
        if self._dimension is None:
            # 使用一个简单文本获取维度
            test_embedding = self.encode("test")
            self._dimension = test_embedding.shape[-1]
        return self._dimension


def get_cache_embedding_provider() -> EmbeddingProvider:
    """根据配置获取缓存向量提供者
    
    工厂函数，根据环境变量配置创建并返回适当的嵌入向量提供者实例。
    这是整个系统获取嵌入提供者的统一入口点。
    
    返回:
        EmbeddingProvider实例，根据配置选择OpenAI或SentenceTransformer
        
    实现逻辑:
    1. 从环境变量CACHE_EMBEDDING_PROVIDER获取提供者类型
    2. 如果指定为'openai'，返回OpenAIEmbeddingProvider实例
    3. 否则，返回SentenceTransformerEmbedding实例
    4. 对于SentenceTransformer，从环境变量读取模型名称和缓存目录
    
    设计目的:
    - 提供统一的嵌入提供者获取方式
    - 支持通过配置文件或环境变量灵活切换嵌入模型
    - 隐藏具体实现细节，简化客户端代码
    - 与model_cache模块的预加载机制配合
    """
    provider_type = os.getenv('CACHE_EMBEDDING_PROVIDER', 'sentence_transformer').lower()

    if provider_type == 'openai':
        return OpenAIEmbeddingProvider()
    else:
        # 使用sentence transformer
        model_name = os.getenv('CACHE_SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        cache_root = os.getenv('MODEL_CACHE_ROOT', './cache')
        cache_dir = os.path.join(cache_root, 'model')
        return SentenceTransformerEmbedding(model_name=model_name, cache_dir=cache_dir)