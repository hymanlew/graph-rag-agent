"""
聊天界面组件

该模块实现了GraphRAG对话系统的主聊天界面，提供完整的用户交互体验，
包括多Agent选择、流式响应、反馈系统、源内容引用和知识图谱提取功能。

主要功能：
1. 支持多种Agent类型切换，提供不同的检索和回答策略
2. 实现流式响应，提升用户等待体验
3. 提供反馈系统，允许用户对回答进行评价
4. 支持查看和提取知识图谱，增强信息可视化
5. 处理各种错误和异常情况，确保系统稳定性
"""

import streamlit as st
import uuid
import re
import json
import traceback

# 导入API相关函数 - 实现与后端的通信
from utils.api import (
    send_message,               # 发送消息获取回答
    send_feedback,              # 发送用户反馈
    get_source_content,         # 获取源文档内容
    get_knowledge_graph_from_message,  # 从消息中提取知识图谱
    get_source_file_info_batch,  # 批量获取源文件信息
    clear_chat,                 # 清除聊天历史
    send_message_stream         # 流式发送消息获取回答
)

# 导入辅助函数
from utils.helpers import extract_source_ids  # 从回答中提取源文档ID

def reset_processing_lock():
    """
    重置处理锁状态
    
    该函数用于重置会话状态中的处理锁，确保用户可以继续与系统交互。
    处理锁用于防止用户在系统处理当前请求时发送新请求，避免并发错误。
    """
    st.session_state.processing_lock = False

def display_chat_interface():
    """
    显示主聊天界面
    
    该函数是聊天组件的核心，负责渲染完整的聊天界面，包括：
    1. 页面标题和配置选项
    2. Agent类型选择
    3. 消息历史显示
    4. 用户输入框和发送功能
    5. 反馈按钮和源内容引用
    6. 知识图谱提取功能
    
    实现思路：
    - 使用Streamlit会话状态管理用户交互和系统状态
    - 支持多种Agent类型和模式切换
    - 处理流式和非流式响应两种模式
    - 实现反馈系统和错误处理机制
    """
    # 设置页面标题
    st.title("GraphRAG 对话系统")
    
    # 初始化处理锁 - 防止并发请求导致的错误
    if "processing_lock" not in st.session_state:
        st.session_state.processing_lock = False
    
    # 创建设置栏 - 包含Agent类型选择、响应模式设置和清除聊天按钮
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])  # 创建三列布局
        
        with col1:  # 第一列：Agent类型选择
            # 保存当前Agent类型，用于检测类型变化
            previous_agent = st.session_state.agent_type
            
            # 创建Agent类型下拉选择框
            agent_type = st.selectbox(
                "选择 Agent 类型",
                # 提供多种Agent选项，每个Agent使用不同的检索和回答策略
                options=["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent"],
                key="header_agent_type",
                help="选择不同的Agent以体验不同的检索策略",
                # 根据当前会话状态设置默认选择的索引
                index=0 if st.session_state.agent_type == "graph_agent" 
                        else (1 if st.session_state.agent_type == "hybrid_agent" 
                             else (2 if st.session_state.agent_type == "naive_rag_agent"
                                  else (3 if st.session_state.agent_type == "deep_research_agent"
                                       else 4))),
                on_change=reset_processing_lock  # 类型变化时重置处理锁
            )
            
            # 检测Agent类型是否发生变化
            if previous_agent != agent_type:
                # 切换agent类型时重置处理锁，确保新的Agent可以正常工作
                st.session_state.processing_lock = False
                
            # 更新会话状态中的Agent类型
            st.session_state.agent_type = agent_type
            
            # 为deep_research_agent添加特定选项 - 仅在选择该Agent时显示
            if agent_type == "deep_research_agent":
                # 显示/隐藏推理过程的选项 - deep_research_agent特有功能
                show_thinking = st.checkbox("显示推理过程", 
                          value=st.session_state.get("show_thinking", False),
                          key="header_show_thinking",
                          help="显示AI的思考过程",
                          on_change=reset_processing_lock)
                # 保存显示思考过程的设置
                st.session_state.show_thinking = show_thinking

                # 启用/禁用增强版研究工具的选项 - 控制是否使用高级研究功能
                use_deeper = st.checkbox("使用增强版研究工具", 
                                        value=st.session_state.get("use_deeper_tool", True),
                                        key="header_use_deeper",
                                        help="启用社区感知和知识图谱增强",
                                        on_change=reset_processing_lock)
                # 更新会话状态中的研究工具设置
                st.session_state.use_deeper_tool = use_deeper
    
        with col2:  # 第二列：响应模式设置
            # 添加流式响应选项 - 仅当调试模式未启用时显示
            if not st.session_state.debug_mode:
                use_stream = st.checkbox("使用流式响应", 
                                        value=st.session_state.get("use_stream", True),
                                        key="header_use_stream",
                                        help="启用流式响应，实时显示生成结果",
                                        on_change=reset_processing_lock)
                # 保存流式响应设置
                st.session_state.use_stream = use_stream
            else:
                # 在调试模式下自动禁用流式响应，确保完整的错误信息可见
                st.session_state.use_stream = False
                st.info("调试模式下已禁用流式响应")
            
        with col3:  # 第三列：清除聊天按钮
            # 添加清除聊天按钮，点击时同时清除聊天历史和重置处理锁
            st.button("🗑️ 清除聊天", key="header_clear_chat", on_click=clear_chat_with_lock_reset)
    
    # 添加分隔线，清晰区分设置区域和聊天内容区域
    st.markdown("---")
    
    # 处理正在进行的请求 - 防止用户发送重复请求
    if st.session_state.processing_lock:
        st.warning("请等待当前操作完成...")
        # 提供强制重置按钮，允许用户在必要时中断当前操作
        if st.button("强制重置处理状态", key="force_reset_lock"):
            st.session_state.processing_lock = False
            # 重新运行应用以更新UI
            st.rerun()
    
    # 创建聊天容器 - 用于显示消息历史
    chat_container = st.container()
    with chat_container:
        # 循环显示聊天历史中的每条消息
        for i, msg in enumerate(st.session_state.messages):
            # 使用Streamlit的chat_message组件显示消息，自动处理用户和助手消息的不同样式
            with st.chat_message(msg["role"]):
                # 获取消息内容
                content = msg["content"]
                
                # 特殊处理助手消息 - 可能包含思考过程和引用源
                if msg["role"] == "assistant":
                    # 判断是否需要显示思考过程（针对deep_research_agent）
                    show_thinking = (st.session_state.agent_type == "deep_research_agent" and 
                                    st.session_state.get("show_thinking", False))
                    
                    # 优先使用raw_thinking字段
                    if "raw_thinking" in msg and show_thinking:
                        # 提取思考过程
                        thinking_process = msg["raw_thinking"]
                        answer_content = msg.get("processed_content", content)
                        
                        # 格式化思考过程，使用引用格式
                        thinking_lines = thinking_process.split('\n')
                        quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                        
                        # 显示思考过程
                        st.markdown(quoted_thinking)
                        
                        # 添加两行空行间隔
                        st.markdown("\n\n")
                        
                        # 显示答案
                        st.markdown(answer_content)
                    # 检查是否有<think>标签
                    elif "<think>" in content and "</think>" in content:
                        # 提取<think>标签中的内容
                        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        
                        if thinking_match:
                            thinking_process = thinking_match.group(1)
                            # 移除思考过程，保留答案
                            answer_content = content.replace(f"<think>{thinking_process}</think>", "").strip()
                            
                            if show_thinking:
                                # 显示思考过程（仅当show_thinking为True时）
                                # 格式化思考过程，使用引用格式
                                thinking_lines = thinking_process.split('\n')
                                quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                
                                # 显示思考过程
                                st.markdown(quoted_thinking)
                                
                                # 添加两行空行间隔
                                st.markdown("\n\n")
                                
                                # 显示答案
                                st.markdown(answer_content)
                            else:
                                # 只显示答案部分（不显示思考过程）
                                st.markdown(answer_content)
                        else:
                            # 如果提取失败，显示完整内容但移除可能的<think>标签
                            cleaned_content = re.sub(r'<think>|</think>', '', content)
                            st.markdown(cleaned_content)
                    else:
                        # 普通回答，无思考过程
                        st.markdown(content)
                else:
                    # 普通消息直接显示
                    st.markdown(content)
                
                # 为助手回答添加额外功能：反馈按钮和源内容引用
                if msg["role"] == "assistant":
                    # 确保消息有唯一ID - 用于标识和反馈关联
                    if "message_id" not in msg:
                        msg["message_id"] = str(uuid.uuid4())
                        
                    # 获取对应的用户问题 - 用于反馈和上下文
                    user_query = ""
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_query = st.session_state.messages[i-1]["content"]
                        
                    # 准备反馈相关的键名 - 用于跟踪反馈状态
                    feedback_key = f"{msg['message_id']}"
                    feedback_type_key = f"feedback_type_{feedback_key}"
                    
                    # 创建空容器用于显示反馈结果
                    feedback_container = st.empty()
                    
                    # 只在未提供过反馈时显示反馈按钮
                    if feedback_key not in st.session_state.feedback_given:
                        # 创建列布局放置反馈按钮
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                        
                        with col1:  # 点赞按钮
                            thumbs_up_key = f"thumbs_up_{msg['message_id']}_{i}"
                            if st.button("👍", key=thumbs_up_key):
                                # 处理点赞反馈 - 提交正面评价
                                # 初始化反馈处理锁
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                # 检查是否有正在进行的反馈操作
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("请等待当前操作完成...")
                                else:
                                    # 标记反馈处理开始
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        # 显示提交中状态
                                        with feedback_container:
                                            with st.spinner("正在提交反馈..."):
                                                # 调用API提交正面反馈
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    True,  # True表示正面反馈
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        # 更新反馈状态
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "positive"
                                        
                                        # 根据响应显示不同的确认消息
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "高质量" in response["action"]:
                                                    st.success("感谢您的肯定！此回答已被标记为高质量。", icon="🙂")
                                                else:
                                                    st.success("感谢您的反馈！", icon="👍")
                                            else:
                                                st.info("已收到您的反馈。", icon="ℹ️")
                                    except Exception as e:
                                        # 显示错误信息
                                        st.error(f"提交反馈时出错: {str(e)}")
                                    finally:                                            
                                        # 确保标记反馈处理结束
                                        st.session_state.feedback_in_progress = False
                                    
                        with col2:  # 点踩按钮
                            thumbs_down_key = f"thumbs_down_{msg['message_id']}_{i}"
                            if st.button("👎", key=thumbs_down_key):
                                # 处理点踩反馈 - 提交负面评价
                                # 初始化反馈处理锁
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False
                                
                                # 检查是否有正在进行的反馈操作
                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("请等待当前操作完成...")
                                else:
                                    # 标记反馈处理开始
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        # 显示提交中状态
                                        with feedback_container:
                                            with st.spinner("正在提交反馈..."):
                                                # 调用API提交负面反馈
                                                response = send_feedback(
                                                    msg["message_id"], 
                                                    user_query, 
                                                    False,  # False表示负面反馈
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )
                                        
                                        # 更新反馈状态
                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "negative"
                                        
                                        # 根据响应显示不同的确认消息
                                        with feedback_container:
                                            if response and "action" in response:
                                                if "清除" in response["action"]:
                                                    st.error("已收到您的反馈，此回答将不再使用。", icon="🔄")
                                                else:
                                                    st.error("已收到您的反馈，我们会改进。", icon="👎")
                                            else:
                                                st.info("已收到您的反馈。", icon="ℹ️")
                                    except Exception as e:
                                        # 显示错误信息
                                        st.error(f"提交反馈时出错: {str(e)}")
                                    finally:
                                        # 确保标记反馈处理结束
                                        st.session_state.feedback_in_progress = False
                    else:
                        # 已经提供过反馈，显示反馈状态
                        feedback_type = st.session_state.get(feedback_type_key, None)
                        with feedback_container:
                            # 根据反馈类型显示不同的状态提示
                            if feedback_type == "positive":
                                st.success("您已对此回答给予肯定！", icon="👍")
                            elif feedback_type == "negative":
                                st.error("您已对此回答提出改进建议。", icon="👎")
                            else:
                                st.info("已收到您的反馈。", icon="ℹ️")
                
                    # 源内容引用功能 - 仅在调试模式下且非deep_research_agent时显示
                    if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                        # 从回答中提取引用的源文件ID
                        source_ids = extract_source_ids(msg["content"])
                        if source_ids:  # 如果有引用的源文件
                            # 创建可折叠区域显示源文本引用选项
                            with st.expander("查看引用源文本", expanded=False):
                                # 批量获取所有引用源文件的元数据
                                source_infos = get_source_file_info_batch(source_ids)
                                
                                # 为每个源文件创建加载按钮
                                for s_idx, source_id in enumerate(source_ids):
                                    # 获取文件名作为显示名称
                                    display_name = source_infos.get(source_id, {}).get("file_name", f"源文本 {source_id}")
                                    source_btn_key = f"src_{source_id}_{i}_{s_idx}"
                                    
                                    # 点击按钮加载完整源内容
                                    if st.button(f"加载 {display_name}", key=source_btn_key):
                                        with st.spinner(f"加载源文本 {display_name}..."):
                                            # 获取源文件内容
                                            source_data = get_source_content(source_id)
                                            if source_data and "content" in source_data:
                                                # 保存源内容到会话状态
                                                st.session_state.source_content = source_data["content"]
                                                # 切换到源内容标签页
                                                st.session_state.current_tab = "源内容"
                                                st.rerun()
                        
                        # 知识图谱提取功能 - 手动提取按钮，deep_research_agent禁用此功能
                        if st.session_state.agent_type != "deep_research_agent":
                            # 创建唯一的按钮ID
                            extract_kg_key = f"extract_kg_{i}"
                            
                            # 提取知识图谱按钮
                            if st.button("提取知识图谱", key=extract_kg_key):
                                with st.spinner("提取知识图谱数据..."):
                                    # 获取对应的用户查询作为上下文
                                    user_query = ""
                                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                        user_query = st.session_state.messages[i-1]["content"]
                                        
                                    # 从回答中提取知识图谱，并使用用户查询过滤结果
                                    kg_data = get_knowledge_graph_from_message(msg["content"], user_query)
                                    
                                    # 如果成功提取到图谱数据
                                    if kg_data and len(kg_data.get("nodes", [])) > 0:
                                        # 将图谱数据保存到消息对象中
                                        st.session_state.messages[i]["kg_data"] = kg_data
                                        # 更新当前处理的图谱消息索引
                                        st.session_state.current_kg_message = i
                                        # 自动切换到知识图谱标签页显示结果
                                        st.session_state.current_tab = "知识图谱"
                                        st.rerun()
        
        # 处理用户新输入的问题
        if prompt := st.chat_input("请输入您的问题...", key="chat_input"):
            # 初始化处理锁（防止并发请求）
            if "processing_lock" not in st.session_state:
                st.session_state.processing_lock = False
                
            # 检查是否有正在处理的请求
            if st.session_state.processing_lock:
                st.warning("请等待当前操作完成...")
                return
                
            # 锁定处理状态，防止并发请求
            st.session_state.processing_lock = True
            
            # 显示用户消息
            with st.chat_message("user"):
                st.write(prompt)
            # 将用户消息添加到会话历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 处理并显示助手响应
            with st.chat_message("assistant"):
                try:
                    # 创建响应占位符，用于动态更新内容
                    message_placeholder = st.empty()
                    full_response = ""  # 存储完整回答内容
                    thinking_content = ""  # 存储思考过程内容
                    
                    # 判断是否使用流式响应模式（调试模式下禁用流式）
                    use_stream = st.session_state.get("use_stream", True) and not st.session_state.debug_mode
                    
                    # 使用流式响应模式
                    if use_stream:
                        # 定义令牌处理函数 - 处理每个从服务器返回的文本片段
                        def handle_token(token, is_thinking=False):
                            nonlocal full_response, thinking_content
                            try:
                                # 处理JSON格式的令牌（某些API可能返回JSON格式的token）
                                if isinstance(token, str) and token.startswith("{") and token.endswith("}"):
                                    try:
                                        import json
                                        # 尝试解析JSON数据
                                        json_data = json.loads(token)
                                        if "content" in json_data:
                                            token = json_data["content"]
                                        elif "status" in json_data:
                                            # 跳过状态消息
                                            return
                                    except json.JSONDecodeError as json_error:
                                        # 不是有效的JSON，保持原样处理
                                        print(f"JSON解析错误: {str(json_error)}")
                                        pass
                                
                                # 处理思考内容
                                if is_thinking:
                                    # 累加思考内容
                                    thinking_content += token
                                    # 将思考内容格式化为引用文本样式
                                    thinking_lines = thinking_content.split('\n')
                                    quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                    # 在UI占位符中更新显示
                                    message_placeholder.markdown(quoted_thinking)
                                else:
                                    # 处理回答内容
                                    # 累加完整响应内容
                                    full_response += token
                                    # 在UI占位符中更新显示，添加光标模拟打字效果
                                    message_placeholder.markdown(full_response + "▌")
                            except Exception as e:
                                # 捕获并记录处理令牌时的错误
                                print(f"处理令牌出错: {str(e)}")
                        
                        # 调用流式API获取响应
                        with st.spinner("思考中..."):
                            try:
                                # 调用流式消息API，处理每个返回的token
                                raw_thinking = send_message_stream(prompt, handle_token)
                                
                                # 验证响应格式是否正确
                                if not full_response or (full_response.startswith("{") and full_response.endswith("}")):
                                    print("流式响应格式不正确，切换到非流式API")
                                    # 格式不正确时切换到非流式API
                                    response = send_message(prompt)
                                    if response:
                                        full_response = response.get("answer", "")
                                        message_placeholder.markdown(full_response)
                            except Exception as e:
                                # 捕获流式API调用失败的异常
                                print(f"流式API失败: {str(e)}")
                                # 回退到非流式API
                                response = send_message(prompt)
                                if response:
                                    full_response = response.get("answer", "")
                                    message_placeholder.markdown(full_response)
                        
                        # 完成流式传输后，移除打字光标效果
                        message_placeholder.markdown(full_response)
                        
                        # 构建助手消息对象
                        message_obj = {
                            "role": "assistant",
                            "content": full_response,
                            "message_id": str(uuid.uuid4())
                        }
                        
                        # 如果存在思考过程内容，添加到消息对象中
                        if thinking_content:
                            message_obj["raw_thinking"] = thinking_content
                            message_obj["processed_content"] = full_response
                    else:
                        # 使用非流式API（调试模式下使用）
                        with st.spinner("思考中..."):
                            # 调用非流式消息API
                            response = send_message(prompt)
                        
                        # 处理API响应
                        if response:
                            # 获取回答内容，设置默认值以防字段不存在
                            answer = response.get("answer", "抱歉，我无法处理您的请求。")
                            
                            # 在UI中显示回答内容
                            message_placeholder.markdown(answer)
                            
                            # 构建助手消息对象
                            message_obj = {
                                "role": "assistant", 
                                "content": answer,
                                "message_id": str(uuid.uuid4())
                            }
                            
                            # 如果API返回了思考内容，添加到消息对象中
                            if "raw_thinking" in response:
                                message_obj["raw_thinking"] = response["raw_thinking"]
                                message_obj["processed_content"] = answer
                                
                            # 调试模式下，保存执行日志用于分析
                            if "execution_log" in response and st.session_state.debug_mode:
                                st.session_state.execution_log = response["execution_log"]
                        else:
                            # 处理响应为空的错误情况
                            error_message = "抱歉，服务器没有返回有效响应。"
                            message_placeholder.markdown(error_message)
                            # 创建错误消息对象
                            message_obj = {
                                "role": "assistant", 
                                "content": error_message,
                                "message_id": str(uuid.uuid4())
                            }
                    
                    # 将助手消息添加到会话历史
                    st.session_state.messages.append(message_obj)
                        
                    # 自动提取知识图谱数据 - 仅在调试模式下且非deep_research_agent时执行
                    if st.session_state.debug_mode and st.session_state.agent_type != "deep_research_agent":
                        with st.spinner("提取知识图谱数据..."):
                            # 获取当前新消息的索引（最后一条）
                            current_msg_index = len(st.session_state.messages) - 1
                            
                            # 优先使用后端直接返回的kg_data（非流式模式）
                            kg_data = response.get("kg_data") if not use_stream else None
                            
                            # 如果后端没有返回kg_data或数据为空，尝试从回答中提取
                            if not kg_data or len(kg_data.get("nodes", [])) == 0:
                                answer_content = message_obj["content"]
                                kg_data = get_knowledge_graph_from_message(answer_content, prompt)
                            
                            # 如果成功提取到知识图谱数据
                            if kg_data and len(kg_data.get("nodes", [])) > 0:
                                # 更新消息对象中的图谱数据
                                st.session_state.messages[current_msg_index]["kg_data"] = kg_data
                                
                                # 更新当前处理的图谱消息索引
                                st.session_state.current_kg_message = current_msg_index
                                
                                # 自动切换到知识图谱标签页显示
                                st.session_state.current_tab = "知识图谱"
                                st.rerun()
                            else:
                                # 提取失败时显示警告
                                if st.session_state.agent_type != "deep_research_agent":
                                    st.warning("无法提取知识图谱数据")
                except Exception as e:
                    # 捕获并显示处理消息过程中的错误
                    st.error(f"处理消息时出错: {str(e)}")
                    traceback.print_exc()
                finally:
                    # 确保在所有情况下都释放处理锁
                    st.session_state.processing_lock = False
                    
            st.rerun()

def clear_chat_with_lock_reset():
    """
    清除聊天并重置处理锁
    
    该函数在清除聊天历史的同时重置处理锁，确保用户可以在清除聊天后
    立即发送新消息，而不会被之前可能锁定的状态阻止。
    """
    # 重置处理锁
    st.session_state.processing_lock = False
    # 调用原始清除函数清除聊天历史
    clear_chat()