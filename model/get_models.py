"""
模型管理模块

该模块提供统一的模型加载和获取接口，支持文本嵌入、语言模型和流式输出等功能。
主要用于集中管理项目中使用的各类AI模型，简化模型调用并提供一致的接口。
"""
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


import os
from pathlib import Path
from dotenv import load_dotenv


def setup_cache():
    """
    设置tiktoken缓存目录以避免网络问题
    
    实现思路：
    - 在用户主目录下创建专用缓存文件夹
    - 设置环境变量指向该缓存目录
    - 使用exist_ok=True确保不会因目录已存在而报错
    
    功能意义：
    - 本地缓存tokenizer可以提高性能并减少网络请求
    - 避免重复下载tokenizer数据
    - 即使在网络不稳定情况下也能正常使用token计数功能
    """
    # 创建缓存目录路径
    cache_dir = Path.home() / "cache" / "tiktoken"
    # 递归创建目录，忽略已存在错误
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 设置环境变量，告诉tiktoken使用指定缓存目录
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)


# 初始化调用缓存设置函数
setup_cache()

# 加载环境变量，从.env文件中读取配置信息
load_dotenv()

def get_embeddings_model():
    """
    获取OpenAI嵌入模型实例
    
    返回：
        OpenAIEmbeddings: 配置好的嵌入模型实例
        
    实现思路：
    - 从环境变量中读取模型配置参数
    - 初始化并返回OpenAIEmbeddings实例
    - 使用环境变量配置实现灵活的模型切换
    
    功能说明：
    - 此模型用于将文本转换为向量表示
    - 主要用于构建向量索引和相似度搜索
    - 支持通过环境变量自定义模型类型、API密钥和基础URL
    """
    # 初始化嵌入模型实例，从环境变量读取配置
    model = OpenAIEmbeddings(
        model=os.getenv('OPENAI_EMBEDDINGS_MODEL'),  # 嵌入模型名称
        api_key=os.getenv('OPENAI_API_KEY'),  # API密钥
        base_url=os.getenv('OPENAI_BASE_URL'),  # API基础URL，支持自定义端点
    )
    return model


def get_llm_model():
    """
    获取OpenAI聊天语言模型实例
    
    返回：
        ChatOpenAI: 配置好的聊天语言模型实例
        
    实现思路：
    - 从环境变量读取模型配置和生成参数
    - 初始化并返回ChatOpenAI实例
    - 提供统一的模型访问接口
    
    功能说明：
    - 此模型用于生成文本回复和处理复杂查询
    - 用于图知识库问答、实体关系提取等核心功能
    - 通过环境变量配置实现模型参数的灵活调整
    """
    # 初始化聊天语言模型实例，从环境变量读取配置
    model = ChatOpenAI(
        model=os.getenv('OPENAI_LLM_MODEL'),  # 语言模型名称
        temperature=os.getenv('TEMPERATURE'),  # 生成温度，控制输出随机性
        max_tokens=os.getenv('MAX_TOKENS'),  # 最大生成token数
        api_key=os.getenv('OPENAI_API_KEY'),  # API密钥
        base_url=os.getenv('OPENAI_BASE_URL'),  # API基础URL
    )
    return model

def get_stream_llm_model():
    """
    获取支持流式输出的OpenAI聊天语言模型实例
    
    返回：
        ChatOpenAI: 配置好的流式输出聊天语言模型实例
        
    实现思路：
    - 创建异步迭代器回调处理器
    - 将回调处理器注册到异步回调管理器中
    - 初始化支持流式输出的ChatOpenAI实例
    - 设置streaming=True启用流式输出功能
    
    功能说明：
    - 提供流式文本生成，用于实现实时响应
    - 增强用户体验，避免长时间等待完整回复
    - 适用于需要实时显示生成结果的场景
    """
    # 创建异步迭代器回调处理器，用于处理流式输出
    callback_handler = AsyncIteratorCallbackHandler()
    # 将回调处理器添加到异步回调管理器中
    manager = AsyncCallbackManager(handlers=[callback_handler])

    # 初始化支持流式输出的聊天语言模型实例
    model = ChatOpenAI(
        model=os.getenv('OPENAI_LLM_MODEL'),  # 语言模型名称
        temperature=os.getenv('TEMPERATURE'),  # 生成温度
        max_tokens=os.getenv('MAX_TOKENS'),  # 最大生成token数
        api_key=os.getenv('OPENAI_API_KEY'),  # API密钥
        base_url=os.getenv('OPENAI_BASE_URL'),  # API基础URL
        streaming=True,  # 启用流式输出
        callbacks=manager,  # 注册回调管理器
    )
    return model

def count_tokens(text):
    """
    计算文本的token数量
    
    参数：
        text: 要计算token数量的文本字符串
        
    返回：
        int: 文本的token数量
        
    实现思路：
    1. 优先根据模型类型选择专用tokenizer进行精确计数
    2. 对于DeepSeek模型使用transformers库的AutoTokenizer
    3. 对于GPT模型使用tiktoken库
    4. 提供基于字符统计的备用方案
    
    性能优化：
    - 使用try-except结构确保代码健壮性
    - 按优先级尝试不同的计数方法
    - 空文本直接返回0，避免不必要计算
    
    业务意义：
    - 确保输入文本符合模型token限制
    - 帮助控制API调用成本
    - 为文本分块和处理提供依据
    """
    # 空文本检查
    if not text:
        return 0
    
    # 获取当前使用的模型名称
    model_name = os.getenv('OPENAI_LLM_MODEL', '').lower()
    
    # 针对DeepSeek模型使用transformers库
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
            return len(tokenizer.encode(text))
        except:
            # 出错时静默失败，尝试下一种方法
            pass
    
    # 针对GPT模型使用tiktoken库
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # 出错时静默失败，尝试下一种方法
            pass
    
    # 备用方案：基于字符的简单估算
    # 中文每个字符算1个token，英文每4个字符算1个token
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4

if __name__ == '__main__':
    """
    主函数：测试模型功能
    
    功能：
    - 测试语言模型的基本功能
    - 测试嵌入模型的向量生成
    - 测试token计数功能
    
    注意事项：
    - 流式模型测试因langchain版本问题暂时注释
    """
    # 测试语言模型功能
    print("测试语言模型...")
    llm = get_llm_model()
    print(llm.invoke("你好"))
    print()

    # 由于langchain版本问题，流式模型测试暂时禁用
    # print("测试流式语言模型...")
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("你好"))
    # print()

    # 测试嵌入模型功能
    print("测试嵌入模型...")
    test_text = "你好，这是一个测试。"
    embeddings = get_embeddings_model()
    result = embeddings.embed_query(test_text)
    print(f"嵌入向量维度: {len(result)}")
    print(f"嵌入向量前5个值: {result[:5]}")
    print()

    # 测试token计数功能
    print("测试Token计数...")
    test_text = "Hello 你好世界"
    tokens = count_tokens(test_text)
    print(f"Token计数: '{test_text}' = {tokens} tokens")
