"""
Graph-RAG Agent 会话状态管理模块

此模块负责初始化和管理Streamlit应用的会话状态（session_state），是整个应用的状态管理核心。
主要功能包括：

1. 会话基础状态初始化
   - 生成唯一会话ID
   - 初始化消息历史记录
   - 配置调试模式和日志系统

2. Agent配置状态
   - 设置默认Agent类型
   - 管理思考过程显示选项
   - 配置工具使用偏好

3. 响应模式控制
   - 管理流式/非流式响应切换
   - 确保调试模式下的特殊配置

4. 数据缓存管理
   - 初始化多种数据缓存结构
   - 支持源文件信息、知识图谱、搜索结果和API响应缓存

5. 知识图谱相关状态
   - 管理图谱显示设置
   - 跟踪当前操作的实体和关系
   - 控制图谱探索模式

该模块中的函数通常在应用启动时或每个会话的开始调用，确保所有必要的状态变量都已正确初始化。
"""

import streamlit as st
import uuid
from frontend_config.settings import DEFAULT_KG_SETTINGS

def init_session_state():
    """
    初始化会话状态变量
    
    功能：
    - 为Streamlit应用初始化所有必要的会话状态变量
    - 设置默认值以确保应用正常运行
    - 配置缓存结构以优化性能
    - 管理Agent和UI显示偏好设置
    
    实现思路：
    - 逐一检查并初始化所有状态变量
    - 确保变量仅在不存在时才被初始化（避免覆盖用户设置）
    - 建立依赖关系（如调试模式与流式响应的关系）
    - 预设合理的默认值
    """
    # 会话标识和基础信息
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # 生成唯一会话ID
    if 'messages' not in st.session_state:
        st.session_state.messages = []  # 初始化消息历史列表
    
    # 调试和日志
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False  # 默认禁用调试模式
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []  # 初始化执行日志
    
    # Agent配置
    if 'agent_type' not in st.session_state:
        st.session_state.agent_type = "naive_rag_agent"  # 默认使用naive_rag_agent
    if 'show_thinking' not in st.session_state:
        st.session_state.show_thinking = True  # 默认显示思考过程
    if 'use_deeper_tool' not in st.session_state:
        st.session_state.use_deeper_tool = True  # 默认使用增强版研究工具
    
    # 流式响应设置 - 默认启用，但调试模式下自动禁用
    if 'use_stream' not in st.session_state:
        st.session_state.use_stream = True
    elif st.session_state.debug_mode:
        # 确保调试模式下禁用流式响应，便于观察完整响应
        st.session_state.use_stream = False
        
    # 数据显示状态
    if 'kg_data' not in st.session_state:
        st.session_state.kg_data = None  # 知识图谱数据
    if 'source_content' not in st.session_state:
        st.session_state.source_content = None  # 源文档内容
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "执行轨迹"  # 当前活动标签
    if 'kg_display_settings' not in st.session_state:
        st.session_state.kg_display_settings = DEFAULT_KG_SETTINGS  # 知识图谱显示设置
    
    # 反馈和处理状态
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = set()  # 已给出反馈的消息ID集合
    if 'feedback_in_progress' not in st.session_state:
        st.session_state.feedback_in_progress = False  # 反馈处理状态
    if 'processing_lock' not in st.session_state:
        st.session_state.processing_lock = False  # 全局处理锁，防止并发操作
    
    # 当前上下文状态
    if 'current_kg_message' not in st.session_state:
        st.session_state.current_kg_message = None  # 当前关联的消息
    
    # 知识图谱管理相关状态
    if 'entity_to_update' not in st.session_state:
        st.session_state.entity_to_update = None  # 当前要更新的实体
    if 'found_relations' not in st.session_state:
        st.session_state.found_relations = None  # 找到的关系
    if 'relation_to_update' not in st.session_state:
        st.session_state.relation_to_update = None  # 当前要更新的关系
    if 'use_chain_exploration' not in st.session_state:
        st.session_state.use_chain_exploration = True  # 是否使用链式探索

    # 缓存系统初始化
    if 'cache' not in st.session_state:
        st.session_state.cache = {
            'source_info': {},  # 源文件信息缓存
            'knowledge_graphs': {},  # 知识图谱缓存
            'vector_search_results': {},  # 向量搜索结果缓存
            'api_responses': {},  # API响应缓存
        }