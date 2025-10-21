"""
流式模型测试模块

该模块提供了一个简单的异步测试环境，用于验证流式语言模型的功能。
主要用于测试get_models.py中定义的流式输出模型是否正常工作，
并展示如何正确处理流式文本生成结果。
"""
import asyncio
from langchain_core.messages import HumanMessage
from model.get_models import get_stream_llm_model


async def main():
    """
    主异步函数：测试流式语言模型的功能
    
    实现思路：
    1. 获取配置好的流式语言模型实例
    2. 构造测试消息列表
    3. 使用异步迭代方式处理流式输出
    4. 实时打印生成的文本片段
    5. 包含异常处理机制
    
    功能说明：
    - 展示如何使用异步API调用流式模型
    - 演示如何逐块处理和显示生成的文本
    - 提供错误捕获和调试信息输出
    """
    # 获取支持流式输出的语言模型实例
    chat = get_stream_llm_model() 
    
    # 构造测试消息，使用HumanMessage表示用户输入
    messages = [HumanMessage(content="Tell me a short joke.")]
    
    # 异常处理机制
    try:
        # 异步迭代流式输出结果
        async for chunk in chat.astream(messages):
            # 实时打印生成的文本内容，不换行并立即刷新缓冲区
            print(chunk.content, end="", flush=True)
        
        # 所有内容生成完毕后换行并显示完成信息
        print("\nStream finished.")
    
    except Exception as e:
        # 捕获并显示异常信息
        print(f"\nError during basic stream: {e}")
        # 导入并使用traceback模块打印详细的堆栈跟踪信息
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    程序入口点
    
    功能：
    - 调用asyncio.run()来执行异步主函数
    - 启动整个流式模型测试流程
    """
    # 使用asyncio.run()运行异步主函数
    asyncio.run(main())