"""
Graph-RAG Agent 前端API工具模块

此模块提供与FastAPI后端通信的所有接口函数，是前端与后端交互的核心层。
主要功能包括：

1. 聊天消息处理
   - 流式响应和非流式响应两种模式
   - 支持不同Agent类型的消息处理
   - 实现基于SSE的实时数据流传输

2. 知识图谱相关功能
   - 获取完整知识图谱数据
   - 从消息中提取相关知识图谱
   - 执行知识图谱推理操作

3. 源文档管理
   - 获取源文档内容
   - 查询源文档元信息
   - 批量处理源文档请求

4. 反馈机制
   - 收集用户反馈（好评/差评）
   - 记录反馈相关上下文

5. 性能优化
   - 实现请求缓存机制，减少重复请求
   - 提供请求批处理功能，合并短时间内的同类请求
   - 集成性能监控装饰器

模块结构：
- API配置常量
- 性能监控装饰器
- 消息处理函数
- 知识图谱相关函数
- 源文档相关函数
- 请求批处理类
- 批处理器实例和管理
"""

import time
import uuid
import requests
import queue
import json
import threading
import streamlit as st
from typing import Dict, Callable
# 导入API配置常量
# 后端FastAPI服务的基础URL
from frontend_config.settings import API_URL

# 导入性能监控装饰器
from utils.performance import monitor_performance

# 导入社区检测算法配置
# 用于知识图谱的社区检测分析，默认使用leiden算法
# leiden算法是一种高效的图聚类算法，适用于大规模知识图谱
from config.settings import community_algorithm

"""API工具模块

本模块提供了前端与后端交互的所有API函数，主要包括：
- 聊天消息发送与接收（支持流式和非流式）
- 知识图谱查询和操作
- 源文件内容获取
- 用户反馈提交
- API请求批处理优化

所有网络请求都包含错误处理和性能监控，部分数据支持本地缓存以提高性能。
"""

@monitor_performance(endpoint="send_message")
def send_message(message: str) -> Dict:
    """
    发送聊天消息到FastAPI后端，带性能监控
    
    功能：
    - 构建包含用户消息和当前会话状态的API请求
    - 根据不同Agent类型添加特定参数
    - 发送HTTP POST请求到后端聊天接口
    - 记录API调用性能指标
    - 处理可能的网络异常
    
    参数：
        message: str - 用户发送的聊天消息内容
    
    返回值：
        Dict - 后端返回的JSON响应，包含AI回答和可能的调试信息
        None - 当请求失败时返回
    
    实现思路：
    - 使用装饰器实现性能监控
    - 动态构建参数，根据Agent类型添加特定配置
    - 记录详细的性能数据到session_state中，用于调试面板显示
    - 完善的错误处理机制，确保UI不会因网络问题崩溃
    """
    start_time = time.time()
    try:
        # 构建请求参数 - 基础参数包含消息内容、会话ID、调试状态和Agent类型
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "debug": st.session_state.debug_mode,
            "agent_type": st.session_state.agent_type
        }
        
        # 如果是深度研究Agent，添加特定配置参数
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)
        
        # 如果是融合Agent，添加特定配置参数
        if st.session_state.agent_type == "fusion_agent":
            params["use_chain_exploration"] = st.session_state.get("use_chain_exploration", True)
        
        # 发送POST请求到后端聊天接口
        response = requests.post(
            f"{API_URL}/chat",
            json=params,
            # timeout=120  # 增加超时时间（可根据需求启用）
        )
        
        # 记录性能数据
        duration = time.time() - start_time
        print(f"前端API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据，用于调试面板显示
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_message",  # 操作类型
            "duration": duration,         # 执行时间（秒）
            "timestamp": time.time(),     # 时间戳
            "message_length": len(message) # 消息长度，用于分析
        })
        
        # 返回后端JSON响应
        return response.json()
    except requests.exceptions.RequestException as e:
        # 记录错误性能数据
        duration = time.time() - start_time
        print(f"前端API调用错误: {str(e)} ({duration:.4f}s)")
        
        # 显示错误信息到UI
        st.error(f"服务器连接错误: {str(e)}")
        return None

def send_message_stream(message: str, on_token: Callable[[str, bool], None]) -> str:
    """
    向FastAPI后端发送聊天消息，获取流式响应
    
    功能：
    - 建立与后端的Server-Sent Events (SSE)连接
    - 实时接收并处理模型输出的token流
    - 支持接收并记录AI思考过程
    - 支持实时接收执行日志（在调试模式下）
    - 在调试模式下自动降级为非流式API
    
    参数：
        message: str - 用户发送的聊天消息内容
        on_token: Callable[[str, bool], None] - 处理token的回调函数，第一个参数是token内容，
                                              第二个参数是布尔值，表示是否为思考内容
    
    返回值：
        str: 收集的思考内容（如果有），用于后续存储
        None: 当请求失败时返回
    
    实现思路：
    - 使用SSE协议实现服务器推送，实现流式输出
    - 对不同类型的事件进行分类处理（模型输出、思考过程、日志、状态等）
    - 实现多级异常捕获，确保UI不会因网络或解析问题而崩溃
    - 在调试模式下切换到非流式API，确保调试信息完整显示
    """
    # 如果调试模式启用，直接回退到非流式API，确保调试信息完整
    if st.session_state.debug_mode:
        print("调试模式已启用，使用非流式API")
        response = send_message(message)
        if response and "answer" in response:
            on_token(response["answer"])
            # 如果有思考内容，返回它用于存储
            return response.get("raw_thinking", "")
        return ""
        
    try:
        # 构建请求参数 - 与非流式API保持一致
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "debug": st.session_state.debug_mode,
            "agent_type": st.session_state.agent_type
        }
        
        # 根据Agent类型添加特定参数
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)
        
        if st.session_state.agent_type == "fusion_agent":
            params["use_chain_exploration"] = st.session_state.get("use_chain_exploration", True)
        
        # 设置SSE连接 - 按需导入依赖
        import sseclient
        
        # 非阻塞模式发起请求，设置stream=True和Accept头
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=params,
            stream=True,  # 启用流式响应
            headers={"Accept": "text/event-stream"}  # 告诉服务器使用SSE格式
        )
        
        # 初始化SSE客户端
        client = sseclient.SSEClient(response)
        
        # 用于收集思考内容的变量
        thinking_content = ""
        
        # 处理每个SSE事件
        for event in client.events():
            try:
                # 确保解析JSON时捕获所有可能的异常
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {str(e)}, 原始数据: {event.data[:100]}")
                    continue
                
                # 根据事件类型进行不同处理
                if data.get("status") == "token":
                    # 正常模型输出的token
                    on_token(data.get("content", ""))
                elif data.get("status") == "thinking":
                    # AI思考过程内容块
                    chunk = data.get("content", "")
                    thinking_content += chunk  # 累计思考内容
                    on_token(chunk, is_thinking=True)  # 调用回调函数，标记为思考内容
                elif data.get("status") == "execution_log" and st.session_state.debug_mode:
                    # 执行日志，只在调试模式下处理
                    if "execution_log" not in st.session_state:
                        st.session_state.execution_log = []
                    st.session_state.execution_log.append(data.get("content", {}))
                elif data.get("status") == "done":
                    # 完成通知，结束循环
                    break
                elif data.get("status") == "error":
                    # 错误通知，显示错误信息并结束
                    on_token(f"\n\n错误: {data.get('message', '未知错误')}")
                    break
                else:
                    # 其他未定义的状态类型，跳过
                    pass
            except Exception as e:
                # 处理任何未捕获的异常，确保流处理不会中断
                print(f"处理SSE事件时出错: {str(e)}")
                continue
        
        # 返回收集的思考内容用于存储
        return thinking_content
    except Exception as e:
        # 处理连接错误，向UI显示错误信息
        on_token(f"\n\n连接错误: {str(e)}")
        print(f"流式API连接错误: {str(e)}")
        return None

@monitor_performance(endpoint="send_feedback")
def send_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "graph_agent"):
    """
    向后端发送用户反馈
    
    功能：
    - 收集用户对AI回答的反馈（好评/差评）
    - 记录反馈相关的上下文信息（查询内容、消息ID等）
    - 发送反馈数据到后端进行存储和分析
    - 监控反馈API的性能
    
    参数：
        message_id: str - 反馈对应的消息ID
        query: str - 用户的原始查询内容
        is_positive: bool - 是否为正面反馈（True为好评，False为差评）
        thread_id: str - 对话线程ID
        agent_type: str - 使用的Agent类型，默认为"graph_agent"
    
    返回值：
        Dict - 后端返回的响应，包含处理状态
        Dict - 当发生错误时，返回包含错误信息的字典
    
    实现思路：
    - 确保agent_type参数有默认值，防止空值错误
    - 构建包含完整上下文的反馈请求
    - 记录反馈操作的性能指标，用于监控
    - 处理可能的网络异常和响应解析异常
    """
    start_time = time.time()
    try:
        # 确保agent_type有默认值，防止空值错误
        if not agent_type:
            agent_type = "graph_agent"
            
        # 发送POST请求到反馈接口
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "message_id": message_id,     # 反馈对应的消息ID
                "query": query,              # 用户的原始查询
                "is_positive": is_positive,  # 是否为正面反馈
                "thread_id": thread_id,      # 对话线程ID
                "agent_type": agent_type     # 使用的Agent类型
            },
            timeout=10  # 设置超时时间为10秒
        )
        
        # 记录性能数据
        duration = time.time() - start_time
        print(f"前端反馈API调用耗时: {duration:.4f}s")
        
        # 在会话中保存性能数据，用于调试面板显示
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_feedback",  # 操作类型
            "duration": duration,         # 执行时间（秒）
            "timestamp": time.time(),     # 时间戳
            "is_positive": is_positive    # 反馈类型标记
        })
        
        # 尝试解析并返回响应
        try:
            return response.json()
        except:
            # 当响应无法解析为JSON时的错误处理
            return {"status": "error", "action": "解析响应失败"}
    except requests.exceptions.RequestException as e:
        # 记录错误性能数据
        duration = time.time() - start_time
        print(f"前端反馈API调用错误: {str(e)} ({duration:.4f}s)")
        
        # 显示错误信息到UI
        st.error(f"发送反馈时出错: {str(e)}")
        return {"status": "error", "action": str(e)}

@monitor_performance(endpoint="get_knowledge_graph")
def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    """
    获取知识图谱数据
    
    功能：
    - 从后端获取知识图谱的节点和边数据
    - 支持限制返回节点数量和基于查询过滤
    - 实现本地缓存机制，避免重复请求
    - 监控API调用性能
    
    参数：
        limit: int - 返回的节点数量上限，默认100
        query: str - 可选的过滤查询条件，用于筛选相关节点
    
    返回值：
        Dict - 包含nodes和links字段的知识图谱数据
        Dict - 当发生错误时，返回空的图谱数据
    
    实现思路：
    - 使用基于参数的缓存键，确保相同请求得到相同缓存
    - 优先检查本地缓存，缓存命中时直接返回
    - 缓存未命中时才发起网络请求
    - 请求成功后更新缓存，提高后续访问性能
    - 错误处理确保即使网络请求失败，UI也能正常显示空图谱
    """
    # 生成缓存键 - 基于请求参数构建，确保缓存的唯一性
    cache_key = f"kg:limit={limit}:query={query}"
    
    # 检查缓存 - 优先使用本地缓存避免重复请求
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        # 构建查询参数
        params = {"limit": limit}
        if query:
            params["query"] = query
            
        # 发送GET请求获取知识图谱数据
        response = requests.get(
            f"{API_URL}/knowledge_graph",
            params=params,
            timeout=30  # 设置较长的超时时间，因为图谱数据可能较大
        )
        result = response.json()
        
        # 缓存结果 - 更新本地缓存，提高后续访问性能
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        # 错误处理 - 显示错误信息并返回空图谱数据
        st.error(f"获取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}  # 返回空图谱，确保UI不会崩溃

def get_knowledge_graph_from_message(message: str, query: str = None):
    """
    从AI响应中提取相关的知识图谱数据
    
    功能：
    - 根据AI的响应消息获取相关的知识图谱节点和边
    - 实现基于消息内容的智能图谱提取
    - 支持可选的过滤查询
    - 使用基于消息哈希的缓存机制，避免重复处理相同内容
    
    参数：
        message: str - AI的响应消息内容
        query: str - 可选的过滤查询条件
    
    返回值：
        Dict - 包含nodes和links字段的相关知识图谱数据
        Dict - 当发生错误时，返回空的图谱数据
    
    实现思路：
    - 使用消息内容的MD5哈希作为缓存键的一部分，确保相同内容只处理一次
    - 结合查询参数生成完整缓存键，支持不同查询条件的差异化缓存
    - 优先使用本地缓存，提高性能
    - 当缓存未命中时，调用专门的后端API从消息中提取相关图谱
    - 完善的错误处理机制确保UI不会崩溃
    """
    # 生成缓存键 - 使用消息哈希和查询组合，确保缓存的唯一性
    import hashlib
    message_hash = hashlib.md5(message.encode()).hexdigest()  # 计算消息的MD5哈希
    cache_key = f"kg_msg:{message_hash}:query={query}"
    
    # 检查缓存 - 优先使用本地缓存避免重复处理
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        # 构建请求参数
        params = {"message": message}
        if query:
            params["query"] = query
            
        # 发送GET请求到专门的API端点，提取与消息相关的知识图谱
        response = requests.get(
            f"{API_URL}/knowledge_graph_from_message",
            params=params,
            timeout=30  # 设置较长的超时时间
        )
        result = response.json()
        
        # 缓存结果 - 更新本地缓存
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        # 错误处理 - 显示错误信息并返回空图谱数据
        st.error(f"从响应提取知识图谱时出错: {str(e)}")
        return {"nodes": [], "links": []}  # 返回空图谱，确保UI不会崩溃

@monitor_performance(endpoint="get_source_content")
def get_source_content(source_id: str) -> Dict:
    """
    获取源内容（原始文本片段）
    
    功能：
    - 根据源ID获取文档片段的原始内容
    - 实现缓存机制避免重复请求
    - 监控API调用性能
    - 处理网络异常
    
    参数：
        source_id: str - 源内容的唯一标识符
    
    返回值：
        Dict - 包含原始文本内容的响应数据
        None - 当发生错误时返回
    
    实现思路：
    - 使用源ID作为缓存键，确保相同ID的内容只请求一次
    - 优先检查本地缓存，提高性能和响应速度
    - 缓存未命中时，向后端发送POST请求获取内容
    - 请求成功后更新缓存，供后续使用
    - 完善的错误处理确保UI体验流畅
    """
    # 构建缓存键 - 使用源ID作为唯一标识
    cache_key = f"content:{source_id}"
    
    # 检查缓存 - 优先使用本地缓存避免重复请求
    if cache_key in st.session_state.cache.get('api_responses', {}):
        return st.session_state.cache['api_responses'][cache_key]
    
    try:
        # 发送POST请求获取源内容
        response = requests.post(
            f"{API_URL}/source",  # 源内容API端点
            json={"source_id": source_id},  # 请求体中包含源ID
            timeout=30  # 设置超时时间
        )
        result = response.json()
        
        # 缓存结果 - 更新本地缓存
        if 'api_responses' not in st.session_state.cache:
            st.session_state.cache['api_responses'] = {}
        st.session_state.cache['api_responses'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        # 错误处理 - 显示错误信息并返回None
        st.error(f"获取源内容时出错: {str(e)}")
        return None

def get_source_file_info(source_id: str) -> dict:
    """
    获取源ID对应的文件信息
    
    功能：
    - 根据源ID获取文档片段所在的文件信息（如文件名、路径等）
    - 实现缓存机制避免重复请求
    - 当请求失败时返回默认信息，确保UI正常显示
    
    参数：
        source_id: str - 源内容的唯一标识符
    
    返回值：
        dict - 包含文件信息的字典，至少包含file_name字段
    
    实现思路：
    - 使用源ID直接作为缓存键，简化缓存逻辑
    - 优先从本地缓存获取信息，提高性能
    - 缓存未命中时发送API请求
    - 请求失败时返回并缓存默认信息，确保用户体验连续性
    - 即使网络错误也能显示基本的文件标识
    """
    # 检查缓存 - 优先使用本地缓存避免重复请求
    if source_id in st.session_state.cache.get('source_info', {}):
        return st.session_state.cache['source_info'][source_id]
    
    try:
        # 发送POST请求获取源文件信息
        response = requests.post(
            f"{API_URL}/source_info",  # 源文件信息API端点
            json={"source_id": source_id},  # 请求体中包含源ID
            timeout=10  # 设置较短的超时时间，因为文件信息通常较小
        )
        result = response.json()
        
        # 缓存结果 - 更新本地缓存
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = result
        
        return result
    except requests.exceptions.RequestException as e:
        # 错误处理 - 显示错误信息并创建默认文件信息
        st.error(f"获取源文件信息时出错: {str(e)}")
        default_info = {"file_name": f"源文本 {source_id}"}  # 创建默认文件名
        
        # 缓存默认结果 - 确保相同ID下次不会重复报错
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = default_info
        
        return default_info  # 返回默认信息，确保UI不会崩溃

def get_source_file_info_batch(source_ids: list) -> dict:
    """
    批量获取多个源ID对应的文件信息
    
    功能：
    - 一次性获取多个源ID的文件信息，减少API调用次数
    - 提高批量加载场景下的性能
    - 处理异常情况并返回默认信息
    
    参数：
        source_ids: list - 源ID字符串列表
    
    返回值：
        Dict - 源ID到文件信息的映射字典
    
    实现思路：
    - 使用专门的批量API端点，一次请求获取多个信息
    - 大幅减少网络请求次数，提高性能和响应速度
    - 错误处理时，为每个源ID生成默认文件信息
    - 确保即使在错误情况下，UI也能获取到基本的显示数据
    """
    try:
        # 发送POST请求到批量API端点
        response = requests.post(
            f"{API_URL}/source_info_batch",  # 批量文件信息API端点
            json={"source_ids": source_ids},  # 请求体中包含ID列表
            timeout=10  # 设置适当的超时时间
        )
        return response.json()  # 返回ID到文件信息的映射字典
    except requests.exceptions.RequestException as e:
        # 错误处理 - 显示错误信息并为每个ID生成默认信息
        st.error(f"批量获取源文件信息时出错: {str(e)}")
        # 使用字典推导式为每个源ID创建默认文件信息
        return {sid: {"file_name": f"源文本 {sid}"} for sid in source_ids}

@monitor_performance(endpoint="kg_reasoning")
def get_kg_reasoning(reasoning_type, entity_a, entity_b=None, max_depth=3, algorithm=community_algorithm):
    """
    知识图谱推理API调用
    
    功能：
    - 执行知识图谱上的各类推理操作（如路径查找、社区检测等）
    - 支持单实体和双实体两种推理模式
    - 控制推理深度，确保性能和结果质量平衡
    - 监控推理操作的性能
    - 处理各种可能的错误情况
    
    参数：
        reasoning_type: str - 推理类型，如路径查找、社区检测等
        entity_a: str - 起始实体
        entity_b: str, optional - 目标实体，某些推理类型需要
        max_depth: int - 推理最大深度，默认3，会被限制在1-5之间
        algorithm: str - 社区检测算法，使用全局配置的算法
    
    返回值：
        Dict - 包含推理结果（nodes和links）的字典
        Dict - 发生错误时，返回包含错误信息的字典
    
    实现思路：
    - 对输入参数进行验证和清理（去除首尾空格，限制深度范围）
    - 构建包含完整推理配置的请求
    - 发送POST请求到专门的推理API端点
    - 处理HTTP错误状态码和JSON解析错误
    - 完善的异常捕获，确保UI体验
    """
    try:
        # 构建请求参数，确保参数有效性
        params = {
            "reasoning_type": reasoning_type,  # 推理类型
            "entity_a": entity_a.strip() if entity_a else "",  # 清理起始实体
            "max_depth": min(max(1, max_depth), 5),  # 确保深度在合理范围内
            "algorithm": algorithm  # 使用全局配置的社区检测算法
        }
        
        # 如果提供了目标实体，添加到参数中
        if entity_b:
            params["entity_b"] = entity_b.strip()  # 清理目标实体
        
        # 使用JSON格式发送POST请求到推理API
        response = requests.post(
            f"{API_URL}/kg_reasoning",  # 知识图谱推理API端点
            json=params,  # 请求体中包含完整的推理配置
            timeout=60  # 设置较长的超时时间，因为推理操作可能较耗时
        )
        
        # 处理HTTP错误状态码
        if response.status_code != 200:
            st.error(f"API请求失败: HTTP {response.status_code}")
            try:
                # 尝试获取详细错误信息
                error_details = response.json()
                return {"error": f"API错误: {error_details.get('detail', '未知错误')}", "nodes": [], "links": []}
            except:
                # JSON解析失败时返回基本错误信息
                return {"error": f"API错误: HTTP {response.status_code}", "nodes": [], "links": []}
        
        # 返回推理结果
        return response.json()
    except requests.exceptions.RequestException as e:
        # 处理网络请求异常
        st.error(f"知识图谱推理请求失败: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}

def get_entity_types():
    """获取所有实体类型"""
    try:
        response = requests.get(
            f"{API_URL}/entity_types",
            timeout=10
        )
        result = response.json()
        return result.get("entity_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取实体类型失败: {str(e)}")
        return []

def get_relation_types():
    """获取所有关系类型"""
    try:
        response = requests.get(
            f"{API_URL}/relation_types",
            timeout=10
        )
        result = response.json()
        return result.get("relation_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取关系类型失败: {str(e)}")
        return []

def get_entities(filters=None):
    """获取实体列表，支持筛选"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/entities/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("entities", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取实体列表失败: {str(e)}")
        return []

def get_relations(filters=None):
    """获取关系列表，支持筛选"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/relations/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("relations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"获取关系列表失败: {str(e)}")
        return []

def create_entity(entity_data):
    """创建新实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/create",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"创建实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def update_entity(entity_data):
    """更新实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/update",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"更新实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_entity(entity_id):
    """删除实体"""
    try:
        response = requests.post(
            f"{API_URL}/entity/delete",
            json={"id": entity_id},
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"删除实体失败: {str(e)}")
        return {"success": False, "message": str(e)}

def create_relation(relation_data):
    """创建新关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/create",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"创建关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def update_relation(relation_data):
    """更新关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/update",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"更新关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_relation(relation_data):
    """删除关系"""
    try:
        response = requests.post(
            f"{API_URL}/relation/delete",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"删除关系失败: {str(e)}")
        return {"success": False, "message": str(e)}

def clear_chat():
    """清除聊天历史"""
    try:
        # 清除前端状态
        st.session_state.processing_lock = False
        st.session_state.messages = []
        st.session_state.execution_log = None
        st.session_state.kg_data = None
        st.session_state.source_content = None
        
        # 重要：也要清除current_kg_message
        if 'current_kg_message' in st.session_state:
            del st.session_state.current_kg_message
        
        # 清除后端状态
        response = requests.post(
            f"{API_URL}/clear",
            json={"session_id": st.session_state.session_id}
        )
        
        if response.status_code != 200:
            st.error("清除后端对话历史失败")
            return
            
        # 重新生成会话ID
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    except Exception as e:
        st.session_state.processing_lock = False
        st.error(f"清除对话时发生错误: {str(e)}")

def clear_cache(cache_type=None):
    """清除指定类型或所有缓存"""
    if cache_type and cache_type in st.session_state.cache:
        st.session_state.cache[cache_type] = {}
    elif not cache_type:
        st.session_state.cache = {
            'source_info': {},
            'knowledge_graphs': {},
            'vector_search_results': {},
            'api_responses': {},
        }



class ApiBatchProcessor:
    """
    API请求批处理器，合并短时间内的相似请求
    
    功能：
    - 收集短时间内的多个API请求，根据请求类型进行分类批处理
    - 减少网络调用次数，提高应用性能
    - 支持多种请求类型的并行处理
    - 线程安全设计，可在多线程环境中使用
    - 自动管理处理线程生命周期
    
    实现思路：
    - 为每种请求类型创建单独的队列和处理线程
    - 使用批处理窗口收集短时间内的请求
    - 批量发送请求到服务器，减少网络开销
    - 自动处理异常情况，确保服务稳定性
    """
    
    def __init__(self, batch_window=0.5, max_batch_size=10):
        """
        初始化批处理器
        
        参数：
            batch_window: 批处理窗口时间(秒)，默认0.5秒
            max_batch_size: 最大批量大小，默认10个请求
        """
        self.batch_window = batch_window  # 批处理窗口时间，用于收集请求
        self.max_batch_size = max_batch_size  # 每批最大请求数
        self.queues = {}  # 每种请求类型的队列
        self.locks = {}   # 每种队列的锁，确保线程安全
        self.threads = {} # 处理线程，每种请求类型一个
        self.running = True  # 运行状态标志
    
    def add_request(self, request_type, request_data, callback):
        """
        添加请求到对应类型的队列
        
        功能：
        - 根据请求类型将请求添加到对应的队列中
        - 自动创建新的队列和锁（如果不存在）
        - 自动启动处理线程（如果尚未启动）
        
        参数：
            request_type: str - 请求类型，用于分类请求
            request_data: Any - 请求数据，将被传递给批处理函数
            callback: Callable - 请求完成后的回调函数
        """
        # 如果是第一次使用这种请求类型，初始化
        if request_type not in self.queues:
            self.queues[request_type] = queue.Queue()
            self.locks[request_type] = threading.Lock()
            # 启动处理线程
            self.threads[request_type] = threading.Thread(
                target=self._process_queue,
                args=(request_type,),
                daemon=True
            )
            self.threads[request_type].start()
        
        # 添加到队列
        self.queues[request_type].put((request_data, callback))
    
    def _process_queue(self, request_type):
        """
        处理特定类型的请求队列
        
        功能：
        - 持续监控并处理指定类型的请求队列
        - 获取队列中的第一个请求作为阻塞点
        - 在批处理窗口内收集更多请求进行批量处理
        - 根据请求数量选择批量处理或单个处理
        - 完整的错误处理确保线程不会意外终止
        
        参数：
            request_type: str - 请求类型，用于识别要处理的队列
        """
        while self.running:
            batch = []
            callbacks = []
            
            # 尝试在窗口时间内收集请求
            try:
                # 获取第一个请求，阻塞等待
                first_request, first_callback = self.queues[request_type].get(block=True)
                batch.append(first_request)
                callbacks.append(first_callback)
                
                # 设置批处理结束时间
                end_time = time.time() + self.batch_window
                
                # 收集更多请求直到窗口结束或达到最大批量
                while time.time() < end_time and len(batch) < self.max_batch_size:
                    try:
                        request, callback = self.queues[request_type].get(block=False)
                        batch.append(request)
                        callbacks.append(callback)
                    except queue.Empty:
                        break
                
                # 处理批量请求
                if len(batch) > 1:
                    # 执行批量处理
                    self._execute_batch(request_type, batch, callbacks)
                else:
                    # 单个请求，直接处理
                    self._execute_single(request_type, batch[0], callbacks[0])
                    
            except Exception as e:
                print(f"批处理错误({request_type}): {e}")
                time.sleep(0.1)  # 避免CPU占用过高
    
    def _execute_batch(self, request_type, batch, callbacks):
        """执行批量请求"""
        try:
            if request_type == 'source_info':
                # 批量获取源信息
                source_ids = batch
                results = self._batch_get_source_info(source_ids)
                
                # 处理回调
                for i, callback in enumerate(callbacks):
                    source_id = source_ids[i]
                    if source_id in results:
                        callback(results[source_id])
                    else:
                        # 默认结果
                        callback({"file_name": f"源文本 {source_id}"})
                        
            elif request_type == 'content':
                # 批量获取内容
                chunk_ids = batch
                results = self._batch_get_content(chunk_ids)
                
                # 处理回调
                for i, callback in enumerate(callbacks):
                    chunk_id = chunk_ids[i]
                    if chunk_id in results:
                        callback(results[chunk_id])
                    else:
                        callback(None)
                        
            # 可以添加其他批处理类型...
            
        except Exception as e:
            print(f"执行批量请求错误({request_type}): {e}")
            # 出错时单独执行每个请求
            for i, request in enumerate(batch):
                try:
                    self._execute_single(request_type, request, callbacks[i])
                except Exception as single_err:
                    print(f"单个请求错误({request_type}): {single_err}")
    
    def _execute_single(self, request_type, request, callback):
        """执行单个请求"""
        try:
            if request_type == 'source_info':
                result = get_source_file_info(request)
                callback(result)
            elif request_type == 'content':
                result = get_source_content(request)
                callback(result)
            # 可以添加其他请求类型...
        except Exception as e:
            print(f"执行单个请求错误({request_type}): {e}")
            callback(None)
    
    def _batch_get_source_info(self, source_ids):
        """批量获取源信息"""
        try:
            response = requests.post(
                f"{API_URL}/source_info_batch",
                json={"source_ids": source_ids},
                timeout=10
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批量获取源信息错误: {e}")
            return {}
    
    def _batch_get_content(self, chunk_ids):
        """批量获取内容"""
        try:
            response = requests.post(
                f"{API_URL}/content_batch",
                json={"chunk_ids": chunk_ids},
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"批量获取内容错误: {e}")
            return {}
    
    def shutdown(self):
        """关闭批处理器"""
        self.running = False
        # 等待所有线程完成
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)

# 初始化批处理器
def get_batch_processor():
    if 'api_batch_processor' not in st.session_state:
        # 创建API批处理器实例
        # 这个处理器将0.5秒内的同类请求合并成一个批处理请求，减少API调用次数
        # 适用于频繁的小请求，如获取源文件信息、获取知识图谱数据等场景
        st.session_state.api_batch_processor = ApiBatchProcessor(batch_window=0.5, max_batch_size=10)
    return st.session_state.api_batch_processor

# 使用批处理器的API函数示例
def get_source_info_async(source_id, callback):
    """异步获取源信息，使用批处理器"""
    processor = get_batch_processor()
    processor.add_request('source_info', source_id, callback)

def get_content_async(chunk_id, callback):
    """异步获取内容，使用批处理器"""
    processor = get_batch_processor()
    processor.add_request('content', chunk_id, callback)

# 在应用退出时关闭批处理器
def shutdown_batch_processor():
    if 'api_batch_processor' in st.session_state:
        st.session_state.api_batch_processor.shutdown()