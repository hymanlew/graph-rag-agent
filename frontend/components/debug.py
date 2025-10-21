import streamlit as st
import json
import re
from utils.helpers import display_source_content
from utils.performance import display_performance_stats, clear_performance_data
from components.knowledge_graph import display_knowledge_graph_tab
from components.knowledge_graph.management import display_kg_management_tab
from components.styles import KG_MANAGEMENT_CSS

def display_source_content_tab(tabs):
    """
    显示源内容标签页内容
    
    功能：
    - 显示从源文本提取的内容，帮助用户验证AI回答的来源依据
    - 根据不同的Agent类型显示不同的提示信息
    - 对Deep Research Agent提供特殊说明，引导用户查看执行轨迹
    
    参数：
    - tabs: Streamlit标签页对象，用于显示源内容的容器
    """
    with tabs[2]:
        # 检查是否有源内容可显示
        if st.session_state.source_content:
            # 使用自定义样式容器包裹源内容
            st.markdown('<div class="source-content-container">', unsafe_allow_html=True)
            # 调用辅助函数显示格式化的源内容
            display_source_content(st.session_state.source_content)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 根据Agent类型显示不同的提示信息
            if st.session_state.agent_type == "deep_research_agent":
                # Deep Research Agent特有提示
                st.info("Deep Research Agent 不提供源内容查看功能。请查看执行轨迹标签页了解详细推理过程。")
            else:
                # 其他Agent的通用提示
                st.info("点击AI回答中的'查看源内容'按钮查看源文本")

def display_execution_trace_tab(tabs):
    """
    显示执行轨迹标签页内容
    
    功能：
    - 根据当前选择的Agent类型显示不同格式的执行轨迹
    - 对于Deep Research Agent，提供详细的迭代过程和增强功能分析
    - 对于其他Agent，显示节点输入输出的JSON格式详情
    
    参数：
    - tabs: Streamlit标签页对象，用于显示执行轨迹的容器
    """
    with tabs[0]:
        # 显示DeepResearchAgent的执行轨迹 - 使用特殊格式展示深度研究过程
        if st.session_state.agent_type == "deep_research_agent":
            # 创建标题和样式 - 使用HTML自定义样式使标题更醒目
            st.markdown("""
            <div style="padding:10px 0px; margin:15px 0; border-bottom:1px solid #eee;">
                <h2 style="margin:0; color:#333333;">深度研究执行过程</h2>
            </div>
            """, unsafe_allow_html=True)

            # 显示当前使用的工具类型 - 增强版或标准版
            tool_type = "增强版(DeeperResearch)" if st.session_state.get("use_deeper_tool", True) else "标准版(DeepResearch)"
            # 使用信息框样式突出显示工具类型
            st.markdown(f"""
            <div style="background-color:#f0f7ff; padding:8px 15px; border-radius:5px; margin-bottom:15px; border-left:4px solid #4285F4;">
                <span style="font-weight:500;">当前工具：</span>{tool_type}
            </div>
            """, unsafe_allow_html=True)
            
            # 增强版工具特有功能说明 - 仅在使用增强版时显示
            if st.session_state.get("use_deeper_tool", True):
                # 使用可折叠面板显示增强功能详情
                with st.expander("增强功能详情", expanded=False):
                    st.markdown("""
                    #### 社区感知增强
                    智能识别相关知识社区，自动提取有价值的背景知识和关联信息。
                    
                    #### 知识图谱增强
                    实时构建查询相关的知识图谱，提供结构化推理和关系发现。
                    
                    #### 证据链追踪
                    记录完整的推理路径和证据来源，提供可解释的结论过程。
                    """)

            # 多来源获取执行日志 - 按优先级尝试不同的日志来源
            execution_logs = []
            
            # 第一优先级：直接从execution_logs获取
            if hasattr(st.session_state, 'execution_logs') and st.session_state.execution_logs:
                execution_logs = st.session_state.execution_logs
            
            # 第二优先级：从execution_log中的deep_research节点提取
            elif hasattr(st.session_state, 'execution_log') and st.session_state.execution_log:
                for entry in st.session_state.execution_log:
                    if entry.get("node") == "deep_research" and entry.get("output"):
                        output = entry.get("output")
                        if isinstance(output, str):
                            # 将输出分割为行
                            execution_logs = output.strip().split('\n')
            
            # 第三优先级：从最近的助手消息的raw_thinking字段提取
            if not execution_logs and len(st.session_state.messages) > 0:
                for msg in reversed(st.session_state.messages):  # 从最新消息开始查找
                    if msg.get("role") == "assistant" and "raw_thinking" in msg:
                        thinking_text = msg["raw_thinking"]
                        # 检查是否包含深度研究标记
                        if "[深度研究]" in thinking_text or "[KB检索]" in thinking_text:
                            execution_logs = thinking_text.strip().split('\n')
                            break
            
            # 最后检查：从session_state.raw_thinking获取
            if not execution_logs and 'raw_thinking' in st.session_state:
                thinking_text = st.session_state.raw_thinking
                if thinking_text and ("[深度研究]" in thinking_text or "[KB检索]" in thinking_text):
                    execution_logs = thinking_text.strip().split('\n')
            
            # 增强版特有功能：社区分析和知识图谱信息展示
            if st.session_state.get("use_deeper_tool", True) and "reasoning_chain" in st.session_state:
                reasoning_chain = st.session_state.reasoning_chain
                
                # 显示社区分析和知识图谱统计 - 使用两列布局同时展示
                if reasoning_chain:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 社区分析部分
                        st.markdown("#### 社区分析")
                        steps = reasoning_chain.get("steps", [])
                        # 从推理链中提取社区分析步骤
                        community_step = next((s for s in steps if "knowledge_community_analysis" in s.get("search_query", "")), None)
                        
                        if community_step:
                            # 显示社区识别结果
                            st.success(f"✓ 识别到相关社区")
                            evidence = community_step.get("evidence", [])
                            
                            # 显示社区知识证据
                            for ev in evidence:
                                if ev.get("source_type") == "community_knowledge":
                                    with st.expander(f"社区知识 {ev.get('evidence_id', '')}"):
                                        st.write(ev.get("content", ""))
                        else:
                            st.info("未执行社区分析")
                    
                    with col2:
                        # 知识图谱统计部分
                        st.markdown("#### 知识图谱")
                        # 检查是否有知识图谱数据
                        if "knowledge_graph" in st.session_state:
                            kg = st.session_state.knowledge_graph
                            # 显示实体和关系数量指标
                            st.metric("实体数量", kg.get("entity_count", 0))
                            st.metric("关系数量", kg.get("relation_count", 0))
                            
                            # 显示核心实体列表（最多5个）
                            central_entities = kg.get("central_entities", [])
                            if central_entities:
                                st.write("**核心实体:**")
                                for entity in central_entities[:5]:
                                    entity_id = entity.get("id", "")
                                    entity_type = entity.get("type", "未知")
                                    st.markdown(f"- **{entity_id}** ({entity_type})")
                        else:
                            st.info("暂无知识图谱数据")
            
            # 如果所有尝试都没有找到执行日志，显示提示信息
            if not execution_logs:
                st.info("正在等待执行日志。请发送新的查询生成执行轨迹，如果看到此消息但已发送查询，请再试一次。")
            else:
                # 调用格式化日志函数进行展示
                display_formatted_logs(execution_logs)
        else:
            # 其他Agent类型的执行轨迹显示逻辑 - 以JSON格式展示节点的输入输出
            if st.session_state.execution_log:
                for entry in st.session_state.execution_log:
                    with st.expander(f"节点: {entry['node']}", expanded=False):
                        st.markdown("**输入:**")
                        st.code(json.dumps(entry["input"], ensure_ascii=False, indent=2), language="json")
                        st.markdown("**输出:**")
                        st.code(json.dumps(entry["output"], ensure_ascii=False, indent=2), language="json")
            else:
                st.info("发送查询后将在此显示执行轨迹。")

def display_formatted_logs(log_lines):
    """
    格式化显示日志行
    
    功能：
    - 识别日志类型（深度研究或知识库检索）
    - 根据日志类型使用不同的显示方式
    - 对深度研究日志，按迭代轮次组织并提供选择器
    - 对知识库检索日志，提供分类展示
    - 提供美化的日志显示样式
    
    参数：
    - log_lines: 日志行列表
    """
    if not log_lines:
        st.warning("没有执行日志")
        return
        
    # 识别日志类型 - 检查是否包含深度研究或知识库检索标记
    has_deep_research_markers = any("[深度研究]" in line for line in log_lines)
    has_kb_search_markers = any("[KB检索]" in line for line in log_lines)
    
    # 针对深度研究和知识库检索类型日志的特殊格式化处理
    if has_deep_research_markers or has_kb_search_markers:
        # 初始化变量用于跟踪迭代轮次和内容
        current_round = None
        in_search_results = False
        
        # 用于存储按轮次组织的日志数据
        current_iteration = None
        current_iteration_content = []
        iterations = []
        current_round = None

        # 遍历日志行，按轮次组织日志内容
        for line in log_lines:
            # 检测新的迭代轮次开始
            if "[深度研究] 开始第" in line and "轮迭代" in line:
                # 如果已有内容，保存前一轮迭代
                if current_iteration_content:
                    iterations.append({
                        "round": current_round,
                        "content": current_iteration_content
                    })
                
                # 使用正则表达式提取轮次数字
                round_match = re.search(r'开始第(\d+)轮迭代', line)
                if round_match:
                    current_round = int(round_match.group(1))
                    current_iteration_content = [line]
            # 如果已经在某一轮次中，将行添加到当前内容
            elif current_round is not None:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 检测查询执行日志      
            elif "[深度研究] 执行查询:" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 检测KB检索开始
            elif "[KB检索] 开始搜索:" in line:
                in_search_results = True
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 检测KB检索结果日志
            elif "[KB检索]" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 检测发现有用信息日志
            elif "[深度研究] 发现有用信息:" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 检测结束迭代日志
            elif "[深度研究] 没有生成新查询且已有信息，结束迭代" in line:
                if current_iteration_content is not None:
                    current_iteration_content.append(line)
            
            # 其他类型的日志行
            elif current_iteration_content is not None:
                current_iteration_content.append(line)
        
        # 添加最后一轮迭代到列表
        if current_iteration_content:
            iterations.append({
                "round": current_round,
                "content": current_iteration_content
            })
        
        # 如果识别到了迭代轮次，提供轮次选择器
        if iterations:
            # 添加轮次选择器标题
            st.markdown("#### 选择迭代轮次")
            
            # 过滤出有效的迭代轮次（排除None轮次）
            valid_iterations = [it for it in iterations if it["round"] is not None]
            if not valid_iterations:
                st.warning("未找到有效的迭代轮次")
                return
                
            # 创建轮次选择字典，确保round是整数
            iteration_options = {f"第 {it['round']} 轮迭代": it for it in valid_iterations}
            
            # 优先将第1轮设置为默认选项
            default_key = next((k for k in iteration_options.keys() if "1 轮" in k), list(iteration_options.keys())[0])
            
            # 创建选择器控件
            selected_round_key = st.selectbox(
                "选择迭代轮次", 
                list(iteration_options.keys()),
                index=list(iteration_options.keys()).index(default_key)
            )
            
            # 获取用户选择的迭代数据
            iteration = iteration_options[selected_round_key]
            
            # 显示所选迭代的详细内容 - 使用自定义标题样式
            st.markdown("""
            <div style="padding:10px 0; margin:10px 0; border-bottom:1px solid #eee;">
                <h4 style="margin:0;">迭代详情</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 分类处理不同类型的日志行 - 提取查询、搜索、结果和有用信息
            queries = []         # 存储执行的查询
            kb_searches = []     # 存储知识库搜索内容
            kb_results = []      # 存储知识库检索结果
            useful_info = None   # 存储发现的有用信息
            other_lines = []     # 存储其他类型的日志
            
            # 遍历迭代内容，按类型分类
            for line in iteration.get("content", []):
                if "[深度研究] 执行查询:" in line:
                    # 提取执行查询内容
                    query = re.sub(r'\[深度研究\] 执行查询:', '', line).strip()
                    queries.append(query)
                elif "[KB检索] 开始搜索:" in line:
                    # 提取知识库搜索内容
                    search = re.sub(r'\[KB检索\] 开始搜索:', '', line).strip()
                    kb_searches.append(search)
                elif "[KB检索] 结果:" in line:
                    # 存储知识库检索结果
                    result = line
                    kb_results.append(result)
                elif "[深度研究] 发现有用信息:" in line:
                    # 提取有用信息内容
                    useful_info = re.sub(r'\[深度研究\] 发现有用信息:', '', line).strip()
                else:
                    # 其他日志行
                    other_lines.append(line)
            
            # 显示查询 - 如果有查询内容
            if queries:
                st.markdown("##### 执行的查询")
                for query in queries:
                    # 使用绿色边框样式展示查询内容
                    st.markdown(f"""
                    <div style="background-color:#f5f5f5; padding:8px; border-left:4px solid #4CAF50; margin:8px 0; border-radius:3px;">
                        {query}
                    </div>
                    """, unsafe_allow_html=True)
            
            # 显示有用信息 - 如果有发现的有用信息
            if useful_info:
                st.markdown("##### 发现的有用信息")
                # 使用绿色背景样式突出显示有用信息
                st.markdown(f"""
                <div style="background-color:#E8F5E9; padding:10px; border-left:4px solid #4CAF50; margin:10px 0; border-radius:4px;">
                    {useful_info}
                </div>
                """, unsafe_allow_html=True)
            
            # 显示知识库检索 - 如果有搜索内容或检索结果
            if kb_searches or kb_results:
                st.markdown("##### 知识库检索")
                # 使用两列布局同时显示搜索内容和结果
                col1, col2 = st.columns(2)
                
                with col1:
                    if kb_searches:
                        st.markdown("**搜索内容**")
                        for search in kb_searches:
                            # 使用橙色边框样式展示搜索内容
                            st.markdown(f"""
                            <div style="background-color:#FFF8E1; padding:8px; border-left:4px solid #FFA000; margin:8px 0; border-radius:3px;">
                                {search}
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    if kb_results:
                        st.markdown("**检索结果**")
                        # 使用代码块显示检索结果
                        st.code("\n".join(kb_results), language="text")
            
            # 显示其他日志行 - 如果有未分类的日志
            if other_lines:
                with st.expander("详细日志", expanded=False):
                    # 创建美化的日志显示容器
                    st.markdown("""
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin:10px 0; font-family:monospace;">
                    """, unsafe_allow_html=True)
                    
                    # 按类型显示不同颜色的日志行
                    for line in other_lines:
                        if "[KB检索]" in line:
                            # 知识库检索日志使用橙色
                            st.markdown(f'<div style="padding:2px 0; color:#f57c00;">{line}</div>', unsafe_allow_html=True)
                        elif "[深度研究]" in line:
                            # 深度研究日志使用蓝色
                            st.markdown(f'<div style="padding:2px 0; color:#1976d2;">{line}</div>', unsafe_allow_html=True)
                        elif "[双路径搜索]" in line:
                            # 双路径搜索日志使用紫色
                            st.markdown(f'<div style="padding:2px 0; color:#7b1fa2;">{line}</div>', unsafe_allow_html=True)
                        else:
                            # 其他日志使用灰色
                            st.markdown(f'<div style="padding:2px 0; color:#666;">{line}</div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            # 没有识别到迭代轮次的情况 - 直接按日志类型分类显示
            # 按类型过滤日志
            deep_research_logs = [line for line in log_lines if "[深度研究]" in line]
            kb_search_logs = [line for line in log_lines if "[KB检索]" in line]
            other_logs = [line for line in log_lines if "[深度研究]" not in line and "[KB检索]" not in line]
            
            # 使用标签页分类展示不同类型的日志
            log_tabs = st.tabs(["深度研究日志", "知识库检索日志", "其他日志"])
            
            # 深度研究日志标签页
            with log_tabs[0]:
                for line in deep_research_logs:
                    if "发现有用信息" in line:
                        # 美化显示有用信息
                        useful_info = re.sub(r'\[深度研究\] 发现有用信息:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#E8F5E9; padding:10px; border-left:4px solid #4CAF50; margin:10px 0; border-radius:4px;">
                            <span style="color:#4CAF50; font-weight:bold;">发现有用信息:</span><br>{useful_info}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "执行查询" in line:
                        # 美化显示执行查询
                        query = re.sub(r'\[深度研究\] 执行查询:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#f5f5f5; padding:8px; border-left:4px solid #4CAF50; margin:8px 0; border-radius:3px;">
                            <span style="color:#4CAF50; font-weight:bold;">执行查询:</span> {query}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # 其他深度研究日志使用蓝色显示
                        st.markdown(f"<span style='color:#1976d2;'>{line}</span>", unsafe_allow_html=True)
            
            # 知识库检索日志标签页
            with log_tabs[1]:
                for line in kb_search_logs:
                    if "开始搜索" in line:
                        # 美化显示搜索内容
                        search = re.sub(r'\[KB检索\] 开始搜索:', '', line).strip()
                        st.markdown(f"""
                        <div style="background-color:#FFF8E1; padding:8px; border-left:4px solid #FFA000; margin:8px 0; border-radius:3px;">
                            <span style="color:#FFA000; font-weight:bold;">开始搜索:</span> {search}
                        </div>
                        """, unsafe_allow_html=True)
                    elif "结果" in line:
                        # 结果日志使用橙色粗体显示
                        st.markdown(f"<span style='color:#f57c00; font-weight:bold;'>{line}</span>", unsafe_allow_html=True)
                    else:
                        # 其他知识库检索日志使用橙色显示
                        st.markdown(f"<span style='color:#f57c00;'>{line}</span>", unsafe_allow_html=True)
            
            # 其他日志标签页
            with log_tabs[2]:
                if other_logs:
                    # 美化显示其他类型的日志
                    st.markdown("""
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; font-family:monospace;">
                    """, unsafe_allow_html=True)
                    
                    for line in other_logs:
                        if "[双路径搜索]" in line:
                            # 双路径搜索日志使用紫色显示
                            st.markdown(f'<div style="padding:2px 0; color:#7b1fa2;">{line}</div>', unsafe_allow_html=True)
                        else:
                            # 其他类型日志使用灰色显示
                            st.markdown(f'<div style="padding:2px 0; color:#666;">{line}</div>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("没有其他日志")
    else:
        # 没有识别到特殊标记的普通日志 - 使用简单代码块格式显示
        st.code("\n".join(log_lines), language="text")

def add_performance_tab(tabs):
    """
    添加性能监控标签页
    
    功能：
    - 在指定标签页中显示应用性能统计信息
    - 提供清除性能数据的功能
    
    参数：
    - tabs: Streamlit标签页对象，用于添加性能监控标签
    """
    with tabs[4]:  # 第五个标签页
        # 使用自定义样式显示性能统计标题
        st.markdown('<div class="debug-header">性能统计</div>', unsafe_allow_html=True)
        # 显示性能统计信息
        display_performance_stats()
        
        # 添加清除性能数据的按钮
        if st.button("清除性能数据"):
            clear_performance_data()
            st.rerun()

def display_debug_panel():
    """
    显示调试面板主函数
    
    功能：
    - 创建包含多个标签页的调试面板
    - 集成执行轨迹、知识图谱、源内容、知识图谱管理和性能监控等功能
    - 实现标签页自动切换逻辑
    - 应用自定义CSS样式
    
    设计思路：
    - 使用标签页组织不同类型的调试信息，提高信息密度
    - 对知识图谱管理使用延迟加载，避免不必要的API请求
    - 使用JavaScript实现自动标签切换，提升用户体验
    - 统一应用自定义样式，确保UI一致性
    """
    st.subheader("🔍 调试信息")
    
    # 创建标签页用于不同类型的调试信息
    tabs = st.tabs(["执行轨迹", "知识图谱", "源内容", "知识图谱管理", "性能监控"])
    
    # 执行轨迹标签 - 显示AI执行过程详情
    display_execution_trace_tab(tabs)
    
    # 知识图谱标签 - 显示相关知识图谱
    display_knowledge_graph_tab(tabs)
    
    # 源内容标签 - 显示AI回答引用的源文本
    display_source_content_tab(tabs)
    
    # 知识图谱管理标签 - 延迟加载，避免不必要的API请求
    if st.session_state.current_tab == "知识图谱管理":
        display_kg_management_tab(tabs)
    else:
        with tabs[3]:
            if st.button("加载知识图谱管理面板", key="load_kg_management"):
                st.session_state.current_tab = "知识图谱管理"
                st.rerun()
            else:
                st.info("点击上方按钮加载知识图谱管理面板")
    
    # 性能监控标签 - 显示性能统计信息
    add_performance_tab(tabs)
    
    # 通过JS脚本直接控制标签切换 - 实现自动标签切换逻辑
    tab_index = 0  # 默认显示执行轨迹标签
    
    if st.session_state.current_tab == "执行轨迹":
        tab_index = 0
    elif st.session_state.current_tab == "知识图谱":
        tab_index = 1
    elif st.session_state.current_tab == "源内容":
        tab_index = 2
    elif st.session_state.current_tab == "知识图谱管理":
        tab_index = 3
    elif st.session_state.current_tab == "性能监控":
        tab_index = 4
    
    # 加载知识图谱管理CSS样式
    kg_management_css = KG_MANAGEMENT_CSS
    st.markdown(kg_management_css, unsafe_allow_html=True)

    # 使用自定义JS自动切换到指定标签页
    tab_js = f"""
    <script>
        // 等待DOM加载完成
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                // 查找所有标签按钮
                const tabs = document.querySelectorAll('[data-baseweb="tab"]');
                if (tabs.length > {tab_index}) {{
                    // 模拟点击目标标签
                    tabs[{tab_index}].click();
                }}
            }}, 100);
        }});
    </script>
    """
    
    # 只有当需要切换标签时才注入JS
    if "current_tab" in st.session_state:
        st.markdown(tab_js, unsafe_allow_html=True)