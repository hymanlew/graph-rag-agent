import streamlit as st
from utils.api import get_knowledge_graph, get_kg_reasoning
from .visualization import visualize_knowledge_graph
import re

"""
知识图谱显示模块

此模块负责在Streamlit前端应用中显示和交互知识图谱相关内容，包括：
1. 知识图谱标签页的内容展示
2. 回答相关图谱与全局知识图谱的切换显示
3. 知识图谱推理问答功能，支持多种推理类型
4. 结果可视化和格式化显示
5. 使用指南和交互帮助

作为Graph RAG系统的前端展示组件，它将后端的知识图谱数据以用户友好的方式呈现，
并提供丰富的交互功能，使用户能够直观地探索知识实体之间的关系。
"""

def display_knowledge_graph_tab(tabs):
    """
    显示知识图谱标签页内容 - 懒加载实现
    
    此函数负责在Streamlit应用中渲染知识图谱标签页，实现了：
    1. 基于Agent类型的条件显示逻辑
    2. 知识图谱显示模式的切换（回答相关/全局）
    3. 知识图谱推理问答功能界面
    
    参数:
        tabs: Streamlit标签页对象，包含知识图谱标签
    """
    with tabs[1]:
        st.markdown('<div class="kg-controls">', unsafe_allow_html=True)

        # 根据当前Agent类型显示不同的提示信息
        if st.session_state.agent_type == "naive_rag_agent":
            # 基础RAG不支持知识图谱功能
            st.info("Naive RAG 是传统的向量搜索方式，没有知识图谱的可视化。")
            return
        elif st.session_state.agent_type == "deep_research_agent":
            # 深度研究Agent有自己的推理可视化方式
            st.info("Deep Research Agent 专注于深度推理过程，没有知识图谱的可视化。请查看执行轨迹标签页了解详细推理过程。")
            return
        elif st.session_state.agent_type == "fusion_agent":
            # 融合Agent支持多种知识图谱技术
            st.info("Fusion Agent 使用多种知识图谱技术进行融合分析。查看图谱可以了解实体间的关联和社区结构。")
        
        # 添加子标签页，分别用于显示知识图谱和执行推理问答
            kg_tabs = st.tabs(["图谱显示", "推理问答"])
        
        with kg_tabs[0]:
            # 提供两种图谱显示模式的切换选项
            kg_display_mode = st.radio(
                "显示模式:",
                ["回答相关图谱", "全局知识图谱"],
                key="kg_display_mode",
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 使用会话状态跟踪图谱加载状态，实现懒加载优化
            # 只有当首次访问或切换显示模式时才重新加载图谱
            should_load_kg = False
            
            # 检查标签页切换和显示模式变化情况
            if "current_tab" in st.session_state and st.session_state.current_tab == "知识图谱":
                if "last_kg_mode" not in st.session_state or st.session_state.last_kg_mode != kg_display_mode:
                    should_load_kg = True
                    st.session_state.last_kg_mode = kg_display_mode
            
            # 显示相应的图谱
            # 根据选择的显示模式加载和显示相应的知识图谱
            if kg_display_mode == "回答相关图谱":
                # 显示与特定回答相关的知识图谱
                if "current_kg_message" in st.session_state and st.session_state.current_kg_message is not None:
                    msg_idx = st.session_state.current_kg_message
                    
                    # 安全检查：确保索引有效且图谱数据存在
                    if (0 <= msg_idx < len(st.session_state.messages) and 
                        "kg_data" in st.session_state.messages[msg_idx] and 
                        st.session_state.messages[msg_idx]["kg_data"] is not None and
                        len(st.session_state.messages[msg_idx]["kg_data"].get("nodes", [])) > 0):
                        
                        # 提取回答预览并显示
                        msg_preview = st.session_state.messages[msg_idx]["content"][:20] + "..."
                        st.success(f"显示与回答「{msg_preview}」相关的知识图谱")
                        
                        # 调用可视化函数展示知识图谱
                        visualize_knowledge_graph(st.session_state.messages[msg_idx]["kg_data"])
                    else:
                        # 如果没有相关数据，尝试加载全局图谱作为备选
                        st.info("未找到与当前回答相关的知识图谱数据")
                        st.warning("尝试加载全局知识图谱...")
                        with st.spinner("加载全局知识图谱..."):
                            kg_data = get_knowledge_graph(limit=100)
                            if kg_data and len(kg_data.get("nodes", [])) > 0:
                                visualize_knowledge_graph(kg_data)
                else:
                    # 提示用户需要发送查询才能获取相关图谱
                    st.info("在调试模式下发送查询获取相关的知识图谱")
            else:
                # 加载并显示全局知识图谱
                with st.spinner("加载全局知识图谱..."):
                    # 调用API获取全局知识图谱数据，限制节点数量为100以优化性能
                    kg_data = get_knowledge_graph(limit=100)
                    if kg_data and len(kg_data.get("nodes", [])) > 0:
                        visualize_knowledge_graph(kg_data)
                    else:
                        st.warning("未能加载全局知识图谱数据")
            
        with kg_tabs[1]:
            # 实现知识图谱推理问答功能界面
            st.markdown("## 知识图谱推理问答")
            st.markdown("探索实体之间的关系和路径，从知识图谱中发现深层次的关联。")
            
            # 提供多种推理类型选择
            reasoning_type = st.selectbox(
                "选择推理类型",
                options=[
                    "shortest_path", 
                    "one_two_hop", 
                    "common_neighbors",
                    "all_paths",
                    "entity_cycles",
                    "entity_influence",
                    "entity_community"
                ],
                format_func=lambda x: {
                    # 将内部类型名称映射为用户友好的中文显示
                    "shortest_path": "最短路径查询",
                    "one_two_hop": "一到两跳关系路径",
                    "common_neighbors": "共同邻居查询",
                    "all_paths": "关系路径查询",
                    "entity_cycles": "实体环路检测",
                    "entity_influence": "影响力分析",
                    "entity_community": "社区检测"
                }.get(x, x),
                key="kg_reasoning_type"
            )
            
            # 根据选择的推理类型显示相应的说明文本
            if reasoning_type == "shortest_path":
                st.info("查询两个实体之间的最短连接路径，了解它们如何关联。")
            elif reasoning_type == "one_two_hop":
                st.info("找出两个实体之间的直接关系或通过一个中间节点的间接关系。")
            elif reasoning_type == "common_neighbors":
                st.info("发现同时与两个实体相关联的其他实体（共同邻居）。")
            elif reasoning_type == "all_paths":
                st.info("探索两个实体之间的所有可能路径，了解它们之间的多种关联方式。")
            elif reasoning_type == "entity_cycles":
                st.info("检测实体的环路，发现循环依赖或递归关系。")
            elif reasoning_type == "entity_influence":
                st.info("分析实体的影响范围，找出它直接和间接关联的所有实体。")
            elif reasoning_type == "entity_community":
                st.info("发现实体所属的社区或集群，分析实体在更大知识网络中的位置。")
                # 社区检测功能特有的算法选择
                algorithm = st.selectbox(
                    "社区检测算法",
                    options=["leiden", "sllpa"],
                    format_func=lambda x: {
                        "leiden": "Leiden算法",
                        "sllpa": "SLLPA算法"
                    }.get(x, x),
                    key="community_algorithm"
                )
                
                # 为每种算法提供详细说明，帮助用户选择合适的算法
                if algorithm == "leiden":
                    st.markdown("""
                    **Leiden算法**是一种优化的社区检测方法，与Louvain算法相似，但能更好地避免出现孤立社区。
                    适合较大规模的图谱，质量更高但计算量也更大。
                    """)
                else:
                    st.markdown("""
                    **SLLPA**（Speaker-Listener Label Propagation Algorithm）是一种标签传播算法，
                    能够快速检测重叠社区，适合中小规模的图谱，速度较快。
                    """)
            
            # 根据推理类型显示不同的输入表单，分为需要两个实体和一个实体的情况
            if reasoning_type in ["shortest_path", "one_two_hop", "common_neighbors", "all_paths"]:
                # 处理需要两个实体作为输入的推理类型
                col1, col2 = st.columns(2)
                
                with col1:
                    entity_a = st.text_input("实体A", key="kg_entity_a", 
                                            help="输入第一个实体的名称")
                
                with col2:
                    entity_b = st.text_input("实体B", key="kg_entity_b", 
                                            help="输入第二个实体的名称")
                
                # 路径类查询需要最大深度参数控制搜索范围
                if reasoning_type in ["shortest_path", "all_paths"]:
                    max_depth = st.slider("最大深度/跳数", 1, 5, 3, key="kg_max_depth",
                                        help="限制搜索的最大深度")
                else:
                    max_depth = 1  # 默认值
                
                # 推理执行按钮，触发后端API调用
                if st.button("执行推理", key="kg_reasoning_button", 
                            help="点击执行知识图谱推理"):
                    # 输入验证
                    if not entity_a or not entity_b:
                        st.error("请输入两个实体名称")
                    else:
                        with st.spinner("正在执行知识图谱推理..."):
                            # 显示处理状态信息，提升用户体验
                            process_info = st.empty()
                            process_info.info(f"正在处理: {reasoning_type} 查询 (可能需要几秒钟...)")
                            
                            try:
                                # 调用后端API执行知识图谱推理
                                result = get_kg_reasoning(
                                    reasoning_type=reasoning_type,
                                    entity_a=entity_a,
                                    entity_b=entity_b,
                                    max_depth=max_depth
                                )
                                
                                # 清除处理信息
                                process_info.empty()
                                
                                # 错误处理
                                if "error" in result and result["error"]:
                                    st.error(f"推理失败: {result['error']}")
                                    return
                                
                                # 检查是否有结果
                                if len(result.get("nodes", [])) == 0:
                                    st.warning("未找到相关的推理结果")
                                    return
                                     
                                # 格式化并显示结果信息
                                display_reasoning_result(reasoning_type, result, entity_a, entity_b)
                                
                                # 可视化知识图谱结果
                                visualize_knowledge_graph(result)
                            except Exception as e:
                                # 异常处理，显示详细错误信息
                                process_info.empty()
                                st.error(f"处理请求时出错: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
            else:
                # 处理只需要一个实体作为输入的推理类型
                entity_id = st.text_input("实体名称", key="kg_entity_single", 
                                        help="输入实体的名称")
                
                # 设置搜索深度限制
                max_depth = st.slider("最大深度", 1, 4, 2, key="kg_max_depth_single",
                                    help="限制搜索的最大深度")
                
                # 对于社区检测，获取之前选择的算法
                algorithm = st.session_state.get("community_algorithm", "leiden") if reasoning_type == "entity_community" else None
                
                # 推理执行按钮
                if st.button("执行推理", key="kg_reasoning_button_single", 
                           help="点击执行知识图谱推理"):
                    # 输入验证
                    if not entity_id:
                        st.error("请输入实体名称")
                    else:
                        with st.spinner("正在执行知识图谱推理..."):
                            # 显示处理状态信息
                            process_info = st.empty()
                            process_info.info(f"正在处理: {reasoning_type} 查询 (可能需要几秒钟...)")
                            
                            try:
                                # 调用后端API执行推理
                                result = get_kg_reasoning(
                                    reasoning_type=reasoning_type,
                                    entity_a=entity_id,
                                    max_depth=max_depth,
                                    algorithm=algorithm
                                )
                                
                                # 清除处理信息
                                process_info.empty()
                                
                                # 错误处理
                                if "error" in result and result["error"]:
                                    st.error(f"推理失败: {result['error']}")
                                    return
                                
                                # 检查是否有结果
                                if len(result.get("nodes", [])) == 0:
                                    st.warning("未找到相关的推理结果")
                                    return
                                
                                # 显示格式化的结果信息
                                display_reasoning_result(reasoning_type, result, entity_id)
                                
                                # 可视化知识图谱
                                visualize_knowledge_graph(result)
                            except Exception as e:
                                # 异常处理
                                process_info.empty()
                                st.error(f"处理请求时出错: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())
            
            # 提供详细的使用指南，帮助用户理解和使用知识图谱推理功能
            with st.expander("📖 推理问答使用指南", expanded=False):
                st.markdown("""
                ### 知识图谱推理功能使用指南
                
                本功能允许您探索知识图谱中实体之间的关系和结构。以下是各种推理类型的说明：
                
                #### 1. 最短路径查询
                查找两个实体之间的最短连接路径，帮助您理解它们是如何关联的。
                - **输入**: 实体A和实体B的名称
                - **参数**: 最大跳数（限制搜索深度）
                - **输出**: 最短路径可视化和路径长度
                
                #### 2. 一到两跳关系路径
                查找两个实体之间的直接关系或通过一个中间节点的间接关系。
                - **输入**: 实体A和实体B的名称
                - **输出**: 所有一跳或两跳路径的列表和可视化
                
                #### 3. 共同邻居查询
                发现同时与两个实体相关联的其他实体（共同邻居）。
                - **输入**: 实体A和实体B的名称
                - **输出**: 共同邻居列表和可视化网络
                
                #### 4. 关系路径查询
                探索两个实体之间的所有可能路径，不限于最短路径。
                - **输入**: 实体A和实体B的名称
                - **参数**: 最大深度（限制搜索深度）
                - **输出**: 发现的所有路径和可视化
                
                #### 5. 实体环路检测
                检测一个实体的环路，即从该实体出发，经过一系列关系后再次回到该实体的路径。
                - **输入**: 实体名称
                - **参数**: 最大环路长度
                - **输出**: 环路列表和可视化
                
                #### 6. 影响力分析
                分析一个实体的影响范围，找出它直接和间接关联的所有实体。
                - **输入**: 实体名称
                - **参数**: 最大深度
                - **输出**: 影响统计和可视化网络
                
                #### 7. 社区检测
                发现实体所属的社区或集群，分析实体在更大知识网络中的位置。
                - **输入**: 实体名称
                - **参数**: 最大深度（定义社区范围）和算法选择
                - **输出**: 社区统计和可视化
                - **算法**: 
                  - Leiden算法 - 精准度更高，适合复杂图谱
                  - SLLPA算法 - 速度更快，适合中小型图谱
                
                ### 使用技巧
                
                - 对于大型知识图谱，建议先限制较小的搜索深度，然后根据需要增加
                - 在可视化图谱中，可以双击节点聚焦，右键点击节点查看更多选项
                - 单击空白处可重置图谱视图
                - 使用右上角的控制面板进行图谱导航
                """)
            
            # 提供图谱可视化图例，帮助用户理解节点颜色和交互方式
            with st.expander("🎨 图谱可视化图例", expanded=False):
                st.markdown("""
                ### 图谱节点颜色说明
                
                - **蓝色**: 源实体/查询实体
                - **红色**: 目标实体
                - **绿色**: 中间节点/共同邻居
                - **紫色**: 社区1成员
                - **青色**: 社区2成员
                - **黄色**: 其他社区成员
                
                ### 图谱交互指南
                
                - **双击节点**: 聚焦显示该节点及其直接相连的节点
                - **右键点击节点**: 打开上下文菜单，提供更多操作
                - **单击空白处**: 重置视图，显示所有节点
                - **拖拽节点**: 手动调整布局
                - **滚轮缩放**: 放大或缩小视图
                - **右上角控制面板**: 提供额外功能，如重置和返回上一步
                """)

def display_reasoning_result(reasoning_type, result, entity_a=None, entity_b=None):
    """
    根据推理类型显示不同的结果信息，使用实体名称而不是ID
    
    此函数负责根据不同的推理类型格式化和显示结果信息，将技术性的ID转换为用户友好的格式。
    
    参数:
        reasoning_type: 推理类型，如'shortest_path'、'common_neighbors'等
        result: 推理结果数据，包含nodes、paths等信息
        entity_a: 第一个实体的名称（可选）
        entity_b: 第二个实体的名称（可选）
    """
    # 最短路径查询结果显示
    if reasoning_type == "shortest_path":
        if "path_info" in result:
            # 格式化路径信息，将实体名称用引号包围以提高可读性
            path_info = result["path_info"]
            if entity_a and entity_b:
                path_info = path_info.replace(entity_a, f"'{entity_a}'")
                path_info = path_info.replace(entity_b, f"'{entity_b}'")
            st.success(f"{path_info} (长度: {result['path_length']})")
    
    # 一到两跳关系路径查询结果显示
    elif reasoning_type == "one_two_hop":
        if "paths_info" in result:
            st.success(f"找到 {result['path_count']} 条路径")
            if result["path_count"] > 0:
                # 使用可展开区域显示详细路径
                with st.expander("查看详细路径", expanded=True):
                    for i, path in enumerate(result["paths_info"]):
                        # 格式化路径显示
                        formatted_path = format_path_with_names(path)
                        st.markdown(f"**路径 {i+1}**: {formatted_path}")
    
    # 共同邻居查询结果显示
    elif reasoning_type == "common_neighbors":
        if "common_neighbors" in result:
            st.success(f"找到 {result['neighbor_count']} 个共同邻居")
            if result["neighbor_count"] > 0:
                # 格式化并显示共同邻居列表，长列表会被截断以优化显示效果
                neighbors = [format_entity_name(neighbor) for neighbor in result["common_neighbors"]]
                neighbors_str = ", ".join(neighbors)
                if len(neighbors_str) > 200:  # 如果太长就截断
                    neighbors_str = neighbors_str[:200] + "..."
                st.write(f"共同邻居: {neighbors_str}")
                
                # 当共同邻居数量较多时，提供完整列表的可折叠视图
                if len(result["common_neighbors"]) > 5:
                    with st.expander("查看所有共同邻居", expanded=False):
                        for i, neighbor in enumerate(result["common_neighbors"]):
                            st.markdown(f"- {format_entity_name(neighbor)}")
    
    # 所有路径查询结果显示
    elif reasoning_type == "all_paths":
        if "paths_info" in result:
            st.success(f"找到 {result['path_count']} 条路径")
            if result["path_count"] > 0:
                with st.expander("查看详细路径", expanded=True):
                    for i, path in enumerate(result["paths_info"]):
                        # 格式化并显示每条路径
                        formatted_path = format_path_with_names(path)
                        st.markdown(f"**路径 {i+1}**: {formatted_path}")
    
    # 实体环路检测结果显示
    elif reasoning_type == "entity_cycles":
        if "cycles_info" in result:
            st.success(f"找到 {result['cycle_count']} 个环路")
            if result["cycle_count"] > 0:
                with st.expander("查看环路详情", expanded=True):
                    for i, cycle in enumerate(result["cycles_info"]):
                        # 格式化环路描述，显示环路长度
                        formatted_desc = format_path_with_names(cycle["description"])
                        st.markdown(f"**环路 {i+1}** (长度: {cycle['length']}): {formatted_desc}")
    
    # 影响力分析结果显示
    elif reasoning_type == "entity_influence":
        if "influence_stats" in result:
            stats = result["influence_stats"]
            # 使用Streamlit的metric组件以卡片形式显示关键统计数据
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("直接关联实体", stats["direct_connections"])
            with col2:
                st.metric("总关联实体", stats["total_connections"])
            with col3:
                st.metric("关系类型数", len(stats["connection_types"]))
            
            # 详细显示各种关系类型的分布情况
            if stats["connection_types"]:
                st.subheader("关系类型分布")
                for rel_type in stats["connection_types"]:
                    st.markdown(f"- **{rel_type['type']}**: {rel_type['count']}次")
    
    # 社区检测结果显示
    elif reasoning_type == "entity_community":
        if "communities" in result:
            st.success(f"检测到 {result['community_count']} 个社区")
            
            # 显示查询实体所属的社区
            if "entity_community" in result:
                entity_name = entity_a if entity_a else "当前实体"
                st.info(f"实体'{entity_name}'所属社区: {result['entity_community']}")
            
            # 显示各个社区的详细信息
            if result["communities"]:
                with st.expander("查看社区详情", expanded=True):
                    for comm in result["communities"]:
                        # 标记社区是否包含中心实体
                        contains = "✓" if comm["contains_center"] else "✗" 
                        st.markdown(f"**社区 {comm['id']}** (包含中心实体: {contains})")
                        st.markdown(f"- 成员数量: {comm['size']}")
                        st.markdown(f"- 连接密度: {comm['density']:.2f}")
                        
                        # 格式化并显示样本成员列表
                        if "sample_members" in comm and comm["sample_members"]:
                            sample_members = [format_entity_name(member) for member in comm["sample_members"]]
                            sample_str = ", ".join(sample_members)
                            if len(sample_str) > 100:  # 长列表截断显示
                                sample_str = sample_str[:100] + "..."
                            st.markdown(f"- 样本成员: {sample_str}")
                        
                        st.markdown("---")
                        
        # 显示社区摘要信息（如果有）
        if "community_info" in result and isinstance(result["community_info"], dict):
            info = result["community_info"]
            if "summary" in info and info["summary"]:
                with st.expander("社区摘要", expanded=True):
                    st.markdown(f"""
                    **社区ID**: {info.get('id', 'N/A')}
                    
                    **实体数量**: {info.get('entity_count', 0)}
                    
                    **关系数量**: {info.get('relation_count', 0)}
                    
                    **摘要**:
                    {info.get('summary', '无摘要')}
                    """)

def format_entity_name(entity_id):
    """
    将实体ID格式化为友好的显示名称
    
    此函数处理不同类型的实体ID，将其转换为更易读的格式，
    特别是对非数字类型的实体名称添加引号以提高可读性。
    
    参数:
        entity_id: 实体的ID或名称
        
    返回:
        str: 格式化后的友好显示名称
    """
    # 处理空值情况
    if not entity_id:
        return "未知实体"
    
    # 对于数字类型的实体ID，直接转换为字符串
    if isinstance(entity_id, (int, float)) or (isinstance(entity_id, str) and entity_id.isdigit()):
        return str(entity_id)
    
    # 对于非数字类型的实体名称，用引号包围以提高可读性
    return f"'{entity_id}'"

def format_path_with_names(path):
    """
    将路径中的实体ID格式化为友好的显示名称
    
    此函数使用正则表达式识别路径描述中的实体ID，
    并将其转换为友好的显示格式，同时保留关系名称不变。
    
    参数:
        path: 路径描述字符串
        
    返回:
        str: 格式化后的路径描述
    """
    # 处理空路径情况
    if not path:
        return ""
    
    # 初始化格式化结果
    formatted = path
    
    # 定义正则表达式模式来识别实体ID（支持中英文和数字下划线）
    entity_pattern = r'\b([a-zA-Z0-9_\u4e00-\u9fa5]+)\b'
    
    # 定义替换函数，处理特殊情况（关系名称等）
    def replace_entity(match):
        entity = match.group(1)
        
        # 跳过关系名称（通常在-[之后）
        if "-[" in match.string[max(0, match.start()-2):match.start()]:
            return entity
        
        # 跳过方括号内的关系类型
        if match.start() > 0 and match.string[match.start()-1:match.start()+len(entity)+1] == f"[{entity}]":
            return entity
        
        # 对普通实体应用格式化
        return format_entity_name(entity)
    
    # 应用正则表达式替换
    formatted = re.sub(entity_pattern, replace_entity, formatted)
    
    return formatted

def get_node_color(node_type, is_center=False):
    """
    根据节点类型和是否为中心节点返回对应的颜色
    
    此函数为知识图谱中的不同类型节点分配不同颜色，
    帮助用户在可视化中区分不同角色的节点。
    
    参数:
        node_type: 节点类型字符串
        is_center: 是否为中心节点
        
    返回:
        str: 颜色的十六进制代码
    """
    from frontend_config.settings import NODE_TYPE_COLORS, KG_COLOR_PALETTE
    
    # 中心节点有特殊颜色，优先级最高
    if is_center:
        return NODE_TYPE_COLORS["Center"]
    
    # 检查是否有预定义的节点类型颜色
    if node_type in NODE_TYPE_COLORS:
        return NODE_TYPE_COLORS[node_type]
    
    # 特殊处理社区节点，根据社区ID分配不同颜色
    if isinstance(node_type, str) and "Community" in node_type:
        try:
            # 从节点类型字符串中提取社区ID
            comm_id_str = node_type.replace("Community", "")
            # 处理空字符串情况，避免转换错误
            if not comm_id_str:
                comm_id = 0
            else:
                comm_id = int(comm_id_str)
                
            # 使用取模运算确保颜色索引在有效范围内
            color_index = (comm_id - 1) % len(KG_COLOR_PALETTE) if comm_id > 0 else 0
            return KG_COLOR_PALETTE[color_index]
        except (ValueError, TypeError):
            # 处理转换异常，使用默认灰色
            return "#757575"  # 灰色
    
    # 其他未定义类型的节点使用默认灰色
    return "#757575"  # 灰色