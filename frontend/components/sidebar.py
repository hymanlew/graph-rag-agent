import streamlit as st
from utils.api import clear_chat
from frontend_config.settings import examples

def display_sidebar():
    """
    显示应用侧边栏函数
    
    功能：
    - 创建应用侧边栏，包含应用标题、Agent选择、系统设置和示例问题等功能区域
    - 提供不同检索策略的选择界面
    - 支持调试模式和流式响应的设置
    - 显示示例问题和项目信息
    """
    with st.sidebar:
        # 应用标题和分隔线
        st.title("📚 GraphRAG")
        st.markdown("---")
        
        # Agent选择部分 - 提供不同的检索策略选项
        st.header("Agent 选择")
        agent_type = st.radio(
            "选择检索策略:",
            ["graph_agent", "hybrid_agent", "naive_rag_agent", "deep_research_agent", "fusion_agent"],
            # 根据当前会话状态自动选择对应索引，保持用户选择的连续性
            index=0 if st.session_state.agent_type == "graph_agent" 
                    else (1 if st.session_state.agent_type == "hybrid_agent" 
                         else (2 if st.session_state.agent_type == "naive_rag_agent" 
                              else (3 if st.session_state.agent_type == "deep_research_agent"
                                   else 4))),
            # 为每个Agent类型提供详细说明
            help="graph_agent：使用知识图谱的局部与全局搜索；hybrid_agent：使用混合搜索方式；naive_rag_agent：使用朴素RAG；deep_research_agent：私域深度研究；fusion_agent：融合式图谱Agent",
            key="sidebar_agent_type"
        )
        # 更新全局agent_type状态，使选择在整个应用中生效
        st.session_state.agent_type = agent_type

        # 思考过程选项 - 仅当选择 deep_research_agent 时显示
        if agent_type == "deep_research_agent":
            # 思考过程选项 - 控制是否显示AI的推理步骤
            show_thinking = st.checkbox("显示推理过程", 
                                    value=st.session_state.get("show_thinking", False), 
                                    key="sidebar_show_thinking",
                                    help="显示AI的思考过程")
            # 更新全局 show_thinking 状态
            st.session_state.show_thinking = show_thinking
            
            # 增强版工具选择 - 控制是否启用高级研究功能
            use_deeper = st.checkbox("使用增强版研究工具", 
                                value=st.session_state.get("use_deeper_tool", True), 
                                key="sidebar_use_deeper",
                                help="启用社区感知和知识图谱增强功能")
            # 更新全局 use_deeper_tool 状态
            st.session_state.use_deeper_tool = use_deeper
            
            # 添加工具说明 - 根据用户选择显示对应的工具说明
            if use_deeper:
                st.info("增强版研究工具：整合社区感知和知识图谱增强，实现更深度的多级推理")
            else:
                st.info("标准版研究工具：实现基础的多轮推理和搜索")
                
        elif "show_thinking" in st.session_state:
            # 如果切换到其他Agent类型，重置show_thinking为False，保持界面一致性
            st.session_state.show_thinking = False
        
        st.markdown("---")
        
        # 系统设置部分 - 组合调试模式和响应设置
        st.header("系统设置")
        
        # 调试选项 - 控制是否显示调试信息面板
        debug_mode = st.checkbox("启用调试模式", 
                               value=st.session_state.debug_mode, 
                               key="sidebar_debug_mode",
                               help="显示执行轨迹、知识图谱和源内容")
        
        # 当调试模式切换时，处理流式响应状态 - 调试模式与流式响应互斥
        previous_debug_mode = st.session_state.debug_mode
        if debug_mode != previous_debug_mode:
            if debug_mode:
                # 启用调试模式时，自动禁用流式响应以确保调试信息完整显示
                st.session_state.use_stream = False
        
        # 更新全局debug_mode状态
        st.session_state.debug_mode = debug_mode
        
        # 添加流式响应选项（仅当调试模式未启用时显示）
        if not debug_mode:
            use_stream = st.checkbox("使用流式响应", 
                                   value=st.session_state.get("use_stream", True), 
                                   key="sidebar_use_stream",
                                   help="启用流式响应，实时显示生成结果")
            # 更新全局 use_stream 状态
            st.session_state.use_stream = use_stream
        else:
            # 在调试模式下显示提示，告知用户流式响应已自动禁用
            st.info("调试模式下已禁用流式响应")
        
        st.markdown("---")
        
        # 示例问题部分 - 提供预设问题供用户快速体验
        st.header("示例问题")
        example_questions = examples
        
        # 显示样式美化的示例问题列表
        for question in example_questions:
            st.markdown(f"""
            <div style="background-color: #f7f7f7; padding: 8px; 
                 border-radius: 4px; margin: 5px 0; font-size: 14px; cursor: pointer;">
                {question}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 项目信息 - 显示应用简介和功能说明
        st.markdown("""
        ### 关于
        这个 GraphRAG 演示基于本地文档建立的知识图谱，可以使用不同的Agent策略回答问题。
        
        **调试模式**可查看:
        - 执行轨迹
        - 知识图谱可视化
        - 原始文本内容
        - 性能监控
        """)
        
        # 重置按钮 - 提供清除对话历史的功能
        if st.button("🗑️ 清除对话历史", key="clear_chat"):
            # 调用API函数清除对话历史
            clear_chat()