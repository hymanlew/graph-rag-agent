"""
聊天API路由模块

该模块定义了与聊天功能相关的所有API端点，是系统与前端交互的核心接口。
主要提供普通聊天响应、流式聊天响应和聊天历史管理功能，是用户与
知识图谱问答系统交互的主要入口点。

主要功能：
- 处理标准聊天请求，返回完整响应
- 提供流式聊天响应，实现实时交互体验
- 支持聊天历史清除，维护会话状态
- 包含执行日志处理和序列化功能

架构设计：
- 基于FastAPI的路由系统，提供RESTful API接口
- 采用装饰器模式实现性能监控
- 使用异步处理提升并发能力
- 支持SSE(Server-Sent Events)实现流式响应
"""
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
from models.schemas import ChatRequest, ChatResponse, ClearRequest, ClearResponse
from services.chat_service import process_chat, process_chat_stream
from services.agent_service import agent_manager, format_execution_log
from utils.performance import measure_performance

# 创建路由器
# 初始化聊天相关的API路由组，用于管理所有聊天功能的端点
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@measure_performance("chat")
async def chat(request: ChatRequest):
    """
    处理聊天请求
    
    该端点接收用户发送的聊天消息，通过调用相应的聊天服务处理请求，
    并返回结构化的聊天响应。它是系统处理标准聊天交互的主要入口，
    支持各种配置选项以控制响应行为。
    
    Args:
        request: 聊天请求对象，包含用户消息、会话标识和配置参数
        
    Returns:
        ChatResponse: 聊天响应对象，包含生成的回答和相关元数据
        
    业务流程：
    1. 接收并验证用户请求数据
    2. 调用chat_service中的process_chat处理聊天内容
    3. 根据debug标志决定是否格式化执行日志
    4. 将结果封装为ChatResponse格式并返回
    5. 通过性能监控装饰器记录API调用性能数据
    
    技术特点：
    - 异步处理，提高系统并发能力
    - 使用Pydantic模型确保数据验证和类型安全
    - 集成性能监控，便于系统优化
    - 支持丰富的配置选项，满足不同场景需求
    """
    # 调用聊天处理服务，传入用户消息和配置参数
    # process_chat是系统的核心处理函数，实现了消息理解和回答生成
    result = await process_chat(
        message=request.message,  # 用户输入的消息内容
        session_id=request.session_id,  # 会话唯一标识，用于上下文管理
        debug=request.debug,  # 是否开启调试模式
        agent_type=request.agent_type,  # 使用的代理类型
        use_deeper_tool=request.use_deeper_tool,  # 是否使用深度搜索工具
        show_thinking=request.show_thinking  # 是否显示思考过程
    )
    
    # 如果开启调试模式且结果包含执行日志，则格式化日志
    # 格式化后的日志更易于阅读和调试
    if request.debug and "execution_log" in result:
        # 格式化执行日志，使其更易读
        result["execution_log"] = format_execution_log(result["execution_log"])
    
    # 将处理结果转换为预定义的响应模型
    # ChatResponse确保响应数据符合预期的结构和类型
    return ChatResponse(**result)

def serialize_log_entry(log_entry):
    """
    将日志条目转换为可序列化的格式
    
    该函数负责将复杂的日志条目转换为可JSON序列化的格式，
    处理了各种可能的数据类型，确保流式响应过程中不会出现序列化错误。
    它特别处理了Message对象和嵌套结构，确保日志能够正确传输。
    
    Args:
        log_entry: 需要序列化的日志条目，可以是字典、对象或其他类型
        
    Returns:
        可JSON序列化的对象，通常是字典或字符串
        
    实现思路：
    1. 对于字典类型，递归处理每个键值对
    2. 特别处理"input"和"output"字段，识别Message对象
    3. 对嵌套结构进行深度遍历和转换
    4. 使用json.dumps验证序列化可行性，对不可序列化对象转为字符串
    5. 对于非字典类型，直接转换为字符串
    
    业务意义：
        - 确保日志数据能够正确序列化并通过SSE传输
        - 提高系统稳定性，避免因序列化错误导致的请求失败
        - 保留关键信息的同时处理各种边缘情况
    """
    # 处理字典类型的日志条目
    if isinstance(log_entry, dict):
        result = {}
        for key, value in log_entry.items():
            # 处理输入字段，通常包含模型输入或用户输入
            if key == "input":
                # 检查是否为Message对象（有content属性）
                if hasattr(value, "content"):
                    # 提取Message对象的content属性
                    result[key] = {"content": value.content}
                # 处理嵌套字典结构
                elif isinstance(value, dict):
                    result[key] = {}
                    for k, v in value.items():
                        # 递归处理嵌套Message对象
                        if hasattr(v, "content"):
                            result[key][k] = {"content": v.content}
                        else:
                            try:
                                # 验证值是否可JSON序列化
                                json.dumps(v)
                                result[key][k] = v
                            except:
                                # 不可序列化的值转换为字符串
                                result[key][k] = str(v)
                else:
                    # 其他情况，统一转换为字符串
                    result[key] = str(value)
            # 处理输出字段，通常包含模型生成的输出
            elif key == "output":
                # 处理Message对象
                if hasattr(value, "content"):
                    result[key] = {"content": value.content}
                else:
                    # 转换为字符串
                    result[key] = str(value)
            # 其他字段直接保留原值
            else:
                result[key] = value
        return result
    # 非字典类型直接转为字符串
    return str(log_entry)

@router.post("/chat/stream")
async def chat_stream(request: Request):
    """
    流式响应聊天请求
    
    该端点提供实时的流式聊天响应，使用Server-Sent Events (SSE)技术，
    允许系统在生成回复的过程中逐步向客户端推送内容，提供更好的用户体验。
    它支持流式文本生成、实时执行日志传输和错误处理。
    
    Args:
        request: FastAPI请求对象，包含JSON格式的聊天参数
        
    Returns:
        StreamingResponse: 流式HTTP响应，使用text/event-stream格式
        
    业务流程：
    1. 从请求体解析JSON数据，提取聊天参数
    2. 创建异步事件生成器函数，用于产生流式响应
    3. 初始化流式处理，发送开始事件
    4. 通过process_chat_stream异步生成器处理聊天内容
    5. 根据数据类型处理并序列化不同类型的响应块
    6. 在debug模式下收集和处理执行日志
    7. 完成后发送最终日志和完成事件
    8. 处理所有可能的异常，确保稳定的错误响应
    
    技术特点：
    - 使用SSE技术实现服务器到客户端的单向实时通信
    - 异步生成器模式，实现高效的非阻塞流式处理
    - 全面的错误处理和日志记录
    - 针对JSON序列化的健壮性保护
    - 优化的HTTP头配置，防止代理缓冲问题
    """
    # 解析请求数据，从JSON中提取聊天参数
    data = await request.json()
    message = data.get("message")  # 用户消息内容
    session_id = data.get("session_id")  # 会话标识
    debug = data.get("debug", False)  # 调试模式标志
    agent_type = data.get("agent_type", "hybrid_agent")  # 代理类型
    use_deeper_tool = data.get("use_deeper_tool", True)  # 是否使用深度搜索
    show_thinking = data.get("show_thinking", False)  # 是否显示思考过程
    
    # 定义异步事件生成器函数，负责生成SSE事件流
    async def event_generator():
        try:
            # 发送开始事件，通知客户端处理已开始
            yield "data: " + json.dumps({"status": "start"}) + "\n\n"
            
            # 初始化执行日志列表，用于收集处理过程中的日志
            execution_log = []
            
            # 使用异步for循环从process_chat_stream接收流式响应块
            async for chunk in process_chat_stream(
                message=message,
                session_id=session_id,
                debug=debug,
                agent_type=agent_type,
                use_deeper_tool=use_deeper_tool,
                show_thinking=show_thinking
            ):
                # 根据响应块类型进行不同处理
                if isinstance(chunk, dict):
                    # 处理包含执行日志的响应块
                    if "execution_log" in chunk and debug:
                        log_entry = chunk["execution_log"]
                        execution_log.append(log_entry)  # 保存日志条目
                        
                        # 序列化日志条目，确保可JSON化
                        serialized_log = serialize_log_entry(log_entry)
                        try:
                            # 发送格式化的执行日志事件
                            yield "data: " + json.dumps({
                                "status": "execution_log",
                                "content": serialized_log
                            }) + "\n\n"
                        except Exception as json_error:
                            # 处理序列化错误，提供简化版本
                            print(f"执行日志序列化错误: {json_error}")
                            yield "data: " + json.dumps({
                                "status": "execution_log",
                                "content": {"simplified": str(log_entry)}
                            }) + "\n\n"
                    # 处理包含状态信息的响应块
                    elif "status" in chunk:
                        try:
                            yield "data: " + json.dumps(chunk) + "\n\n"
                        except Exception as json_error:
                            # 处理状态序列化错误
                            print(f"状态序列化错误: {json_error}")
                            yield "data: " + json.dumps({
                                "status": "error", 
                                "message": "状态序列化错误"
                            }) + "\n\n"
                    # 处理其他字典类型响应块，作为文本令牌
                    else:
                        try:
                            yield "data: " + json.dumps({
                                "status": "token", 
                                "content": str(chunk)
                            }) + "\n\n"
                        except Exception as json_error:
                            print(f"令牌序列化错误: {json_error}")
                else:
                    # 处理普通文本响应块
                    try:
                        yield "data: " + json.dumps({
                            "status": "token", 
                            "content": chunk
                        }) + "\n\n"
                    except Exception as json_error:
                        print(f"普通文本序列化错误: {json_error}")
                
            # 在处理完成后，发送完整的执行日志集合
            if debug and execution_log:
                try:
                    # 序列化所有收集的日志
                    serialized_logs = [serialize_log_entry(log) for log in execution_log]
                    yield "data: " + json.dumps({
                        "status": "execution_logs",
                        "content": serialized_logs
                    }) + "\n\n"
                except Exception as json_error:
                    # 处理日志组序列化错误
                    print(f"执行日志组序列化错误: {json_error}")
                    yield "data: " + json.dumps({
                        "status": "execution_logs",
                        "content": [{"simplified": "日志序列化失败"}]
                    }) + "\n\n"
                
            # 发送完成事件，通知客户端处理已完成
            yield "data: " + json.dumps({"status": "done"}) + "\n\n"
        except Exception as e:
            # 处理所有可能的异常，发送错误事件
            print(f"事件生成器错误: {e}")
            yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"
    
    # 返回流式HTTP响应
    # 配置适当的媒体类型和HTTP头，确保正确的流式传输行为
    return StreamingResponse(
        event_generator(),  # 使用上面定义的事件生成器
        media_type="text/event-stream",  # SSE标准媒体类型
        headers={
            "Cache-Control": "no-cache",  # 防止客户端缓存响应
            "Connection": "keep-alive",  # 保持长连接
            "X-Accel-Buffering": "no"  # 阻止Nginx等代理服务器缓冲响应
        }
    )

@router.post("/clear", response_model=ClearResponse)
async def clear_chat(request: ClearRequest):
    """
    清除聊天历史
    
    该端点用于清除指定会话的聊天历史记录，允许用户重置对话上下文，
    保护隐私并重新开始新的对话。它通过agent_manager来管理会话状态。
    
    Args:
        request: 包含会话ID的清除请求对象
        
    Returns:
        ClearResponse: 包含操作状态和剩余消息信息的响应对象
        
    业务流程：
    1. 接收包含session_id的清除请求
    2. 调用agent_manager的clear_history方法执行清除操作
    3. 将结果封装为ClearResponse格式并返回
    
    业务意义：
        - 提供会话管理功能，支持用户重置对话
        - 保护用户隐私，允许清除历史记录
        - 支持多轮对话的灵活控制
    """
    # 调用agent_manager清除指定会话的历史记录
    # agent_manager是一个全局单例，负责管理所有会话和代理实例
    result = agent_manager.clear_history(request.session_id)
    
    # 将清除操作结果转换为预定义的响应模型
    return ClearResponse(**result)