"""
模型管理模块

该模块提供统一的模型加载和获取接口，支持文本嵌入、语言模型和流式输出等功能。
主要用于集中管理项目中使用的各类AI模型，简化模型调用并提供一致的接口。
"""
from langchain_huggingface import HuggingFaceEmbeddings
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
    """获取OpenAI嵌入模型实例"""
    model = HuggingFaceEmbeddings(
        model_name="models/text/bge-small-zh-v1.5",
        model_kwargs={"device": 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={
            "normalize_embeddings": True,  # 归一化
            "batch_size": 32  # 根据内存调整
        }
    )
    # model = OpenAIEmbeddings(
    #     model=os.getenv('OPENAI_EMBEDDINGS_MODEL'),  # 嵌入模型名称
    #     api_key=os.getenv('OPENAI_API_KEY'),  # API密钥
    #     base_url=os.getenv('OPENAI_BASE_URL'),  # API基础URL，支持自定义端点
    # )
    return model

def get_llm_model():
    """获取OpenAI聊天语言模型实例"""
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

    创建异步迭代器回调处理器，并将回调处理器注册到异步回调管理器中
    回调管理器允许你在LLM生成过程中的不同阶段（如开始、新token生成、结束等）插入自定义的逻辑。用于监控和控制模型调用过程。
    例如可以使用回调函数来记录日志、计算token使用量、实时将生成的token发送到前端等。

    - 流式输出与回调处理器，在处理流式响应时经常一起使用。例如当流式输出时，可以通过回调函数来实时处理每一个新生成的token。
      - on_llm_start：当LLM开始生成时触发。
      - on_llm_new_token：当每个新token生成时触发（在流式模式下特别有用）。
      - on_llm_end：当LLM生成结束时触发。

    虽然使用 model.astream 确实是更直接和简洁的方式。但回调器提供了更细粒度的控制，可以在不同的阶段（如开始、令牌生成、结束）插入自定义逻辑。
    async def stream_example():
        messages = [HumanMessage(content="请介绍一下人工智能")]
        async for chunk in model.astream(messages):
            print(chunk.content, end="", flush=True)

    # 创建一个任务来运行模型生成
    task = asyncio.create_task(model.agenerate([messages]))
    # 从回调处理器中读取令牌
    async for token in callback_handler.aiter():
        print(token, end="", flush=True)
    # 等待任务完成
    await task

    # 多层级监控
    class AnalysisCallbackHandler(AsyncIteratorCallbackHandler):
        async def on_llm_start(self, serialized, prompts, **kwargs):
            print(f"开始处理请求，提示词: {prompts[0][:50]}...")

        async def on_llm_new_token(self, token: str, **kwargs):
            # 不只是输出，还可以分析
            if len(token.strip()) > 0:
                print(f"生成token: '{token}' (长度: {len(token)})")

        async def on_llm_end(self, response, **kwargs):
            usage = response.llm_output.get('token_usage', {})
            print(f"生成完成，总token数: {usage.get('total_tokens', '未知')}")
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
