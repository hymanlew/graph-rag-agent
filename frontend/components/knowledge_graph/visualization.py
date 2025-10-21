import tempfile
import os
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from frontend_config.settings import KG_COLOR_PALETTE, NODE_TYPE_COLORS

def visualize_knowledge_graph(kg_data):
    """
    使用pyvis库可视化知识图谱，实现类似Neo4j的交互式图谱展示
    
    该函数接收知识图谱数据，并通过pyvis库创建交互式网络图，主要功能包括：
    1. 动态节点样式和颜色分配，基于节点类型和社区分类
    2. 自定义交互配置，包括物理引擎、节点悬停效果等
    3. 整合自定义CSS样式和JavaScript交互脚本
    4. 生成交互式图例，帮助用户理解节点类型
    5. 临时文件管理和资源清理
    
    参数:
        kg_data (dict): 知识图谱数据，包含"nodes"和"links"两个主要键
                       - nodes: 节点列表，每个节点包含id、label、group等属性
                       - links: 边列表，每个边包含source、target、label、weight等属性
                        
    实现思路:
    1. 数据验证 - 首先检查输入数据的有效性
    2. 用户交互控制 - 提供图谱显示设置面板
    3. 网络图初始化 - 创建pyvis网络图对象并配置交互选项
    4. 颜色分配策略 - 为不同类型的节点分配颜色，包括预定义颜色和动态颜色
    5. 节点和边添加 - 循环处理数据，为每个节点和边添加自定义样式和交互属性
    6. 自定义样式和脚本注入 - 整合CSS样式和JavaScript交互逻辑
    7. 图例生成 - 创建美观、有组织的图例帮助用户理解图谱
    """
    # 数据验证 - 检查输入数据的有效性，确保包含必要的结构
    if not kg_data or "nodes" not in kg_data or "links" not in kg_data:
        st.warning("无法获取知识图谱数据")
        return
    
    # 检查是否有节点数据
    if len(kg_data["nodes"]) == 0:
        st.info("没有找到相关的实体和关系")
        return
    
    # 添加图表设置控制 - 提供用户自定义图谱显示选项和交互帮助信息
    with st.expander("图谱显示设置与交互说明", expanded=False):
        st.markdown("""
        ### 交互说明
        - **双击节点**: 聚焦查看该节点及其直接相连的节点和关系
        - **右键节点**: 打开上下文菜单，提供更多操作
        - **单击空白处**: 重置图谱，显示所有节点
        - **使用控制面板**: 右上角的控制面板提供重置和返回上一步功能
        
        ### 显示设置
        """)
        
        # 为每个checkbox添加唯一的key参数
        # 通过使用随机生成或基于kg_data一部分内容的哈希值创建唯一键
        import hashlib
        
        # 基于kg_data的节点数量和时间戳创建哈希值的一部分
        import time
        timestamp = str(time.time())
        node_count = str(len(kg_data["nodes"]))
        base_key = hashlib.md5((node_count + timestamp).encode()).hexdigest()[:8]
        
        col1, col2 = st.columns(2)
        with col1:
            physics_enabled = st.checkbox("启用物理引擎", 
                                       value=st.session_state.kg_display_settings["physics_enabled"],
                                       key=f"physics_enabled_{base_key}",
                                       help="控制节点是否可以动态移动")
            node_size = st.slider("节点大小", 10, 50, 
                                st.session_state.kg_display_settings["node_size"],
                                key=f"node_size_{base_key}",
                                help="调整节点的大小")
        
        with col2:
            edge_width = st.slider("连接线宽度", 1, 10, 
                                 st.session_state.kg_display_settings["edge_width"],
                                 key=f"edge_width_{base_key}", 
                                 help="调整连接线的宽度")
            spring_length = st.slider("弹簧长度", 50, 300, 
                                    st.session_state.kg_display_settings["spring_length"],
                                    key=f"spring_length_{base_key}", 
                                    help="调整节点之间的距离")
        
        # 更新设置
        st.session_state.kg_display_settings = {
            "physics_enabled": physics_enabled,
            "node_size": node_size,
            "edge_width": edge_width,
            "spring_length": spring_length,
            "gravity": st.session_state.kg_display_settings["gravity"]
        }
    
    # 创建网络图 - 设置为白色背景，黑色文字，并启用有向图模式
    net = Network(height="600px", width="100%", bgcolor="#FFFFFF", font_color="#333333", directed=True)
    
    # 增强配置 - 设置Neo4j风格的交互体验，包括物理引擎参数和用户交互选项
    net.set_options("""
    {
      "physics": {
        "enabled": %s,  # 控制是否启用物理引擎
        "barnesHut": {  # 配置Barnes-Hut算法参数
          "gravitationalConstant": %d,  # 重力常数，影响节点间的吸引力
          "centralGravity": 0.5,        # 中心引力，控制节点向中心聚集的程度
          "springLength": %d,           # 弹簧长度，影响节点之间的距离
          "springConstant": 0.04,       # 弹簧常数，影响弹性效果
          "damping": 0.15,              # 阻尼系数，控制动画衰减
          "avoidOverlap": 0.1           # 避免重叠的程度
        },
        "solver": "barnesHut",  # 使用Barnes-Hut算法，适合大型网络
        "stabilization": {       # 稳定化配置，改善初始布局
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "interaction": {          # 交互选项配置
        "navigationButtons": true,  # 显示导航按钮
        "keyboard": {               # 键盘控制
          "enabled": true,
          "bindToWindow": true
        },
        "hover": true,              # 启用悬停效果
        "multiselect": true,        # 允许多选
        "tooltipDelay": 200         # 提示框延迟时间
      },
      "layout": {                # 布局选项
        "improvedLayout": true,
        "hierarchical": {         # 不使用层级布局
          "enabled": false
        }
      }
    }
    """ % (str(physics_enabled).lower(), st.session_state.kg_display_settings["gravity"], spring_length))
    
    # 提取所有唯一组类型 - 收集图谱中的所有节点类型，用于颜色分配
    group_types = set()
    for node in kg_data["nodes"]:
        group = node.get("group", "Unknown")
        if group:
            group_types.add(group)
    
    # 为每个组分配颜色 - 实现复杂的颜色分配策略，确保视觉一致性
    group_colors = {}
    
    # 首先分配预定义颜色 - 使用配置文件中定义的颜色映射
    for group in group_types:
        if group in NODE_TYPE_COLORS:
            group_colors[group] = NODE_TYPE_COLORS[group]
    
    # 然后为剩余组分配颜色 - 使用配色方案动态分配
    palette_index = 0
    for group in sorted(group_types):
        if group not in group_colors:
            # 特殊处理社区类型 - 为相同社区ID分配一致的颜色
            if isinstance(group, str) and "Community" in group:
                try:
                    # 提取社区ID数字部分
                    comm_id_str = group.replace("Community", "")
                    if not comm_id_str:
                        comm_id = 0
                    else:
                        comm_id = int(comm_id_str)
                    
                    # 确保使用一致的社区颜色映射 - 相同ID的社区使用相同颜色
                    color_index = (comm_id - 1) % len(KG_COLOR_PALETTE) if comm_id > 0 else 0
                    group_colors[group] = KG_COLOR_PALETTE[color_index]
                except (ValueError, TypeError):
                    # 转换失败，使用默认分配
                    group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                    palette_index += 1
            else:
                # 普通类型按序分配颜色
                group_colors[group] = KG_COLOR_PALETTE[palette_index % len(KG_COLOR_PALETTE)]
                palette_index += 1
    
    # 添加节点，使用现代样式并增强交互体验 - 为每个节点设置视觉和交互属性
    for node in kg_data["nodes"]:
        node_id = node["id"]
        label = node.get("label", node_id)  # 使用标签或ID作为显示文本
        group = node.get("group", "Unknown")  # 获取节点类型组
        description = node.get("description", "")  # 获取节点描述
        
        # 根据节点组类型设置颜色 - 使用之前分配的颜色映射
        color = group_colors.get(group, KG_COLOR_PALETTE[0])  # 默认使用第一个颜色
        
        # 构建节点悬停提示文本
        title = f"{label}" + (f": {description}" if description else "")
        
        # 添加带有阴影和边框的节点 - 增强视觉效果和交互反馈
        net.add_node(
            node_id,                # 节点唯一标识符
            label=label,            # 显示标签
            title=title,            # 悬停提示文本
            color={                 # 自定义颜色配置
                "background": color,        # 背景色
                "border": "#ffffff",        # 边框色
                "highlight": {               # 选中时的样式
                    "background": color, 
                    "border": "#000000"
                },
                "hover": {                   # 悬停时的样式
                    "background": color, 
                    "border": "#000000"
                }
            }, 
            size=node_size,                  # 节点大小（用户可配置）
            font={"color": "#ffffff", "size": 14, "face": "Arial"},  # 字体样式
            shadow={"enabled": True, "color": "rgba(0,0,0,0.2)", "size": 3},  # 阴影效果
            borderWidth=2,                   # 边框宽度
            # 添加自定义数据用于JavaScript交互
            group=group,                     # 节点组类型
            description=description          # 节点描述
        )
    
    # 添加边，使用现代样式并增强交互体验 - 为每条关系设置视觉和交互属性
    for link in kg_data["links"]:
        source = link["source"]      # 源节点ID
        target = link["target"]      # 目标节点ID
        label = link.get("label", "")  # 关系类型标签
        weight = link.get("weight", 1)  # 关系权重
        
        # 根据权重设置线的粗细和不透明度 - 权重越大，线条越粗
        width = edge_width * min(1 + (weight * 0.2), 3)  # 限制最大宽度为3倍基础宽度
        
        # 使用弯曲的箭头和平滑的线条 - 提高视觉美观度
        smooth = {"enabled": True, "type": "dynamic", "roundness": 0.5}
        
        title = label  # 边的悬停提示文本
        
        # 添加带有阴影的边 - 增强视觉效果和交互反馈
        net.add_edge(
            source,                     # 源节点
            target,                     # 目标节点
            title=title,                # 悬停提示文本
            label=label,                # 关系类型标签
            width=width,                # 线宽（基于权重）
            smooth=smooth,              # 平滑曲线配置
            color={                     # 颜色配置
                "color": "#999999",     # 常规颜色
                "highlight": "#666666", # 选中时的颜色
                "hover": "#666666"      # 悬停时的颜色
            },
            shadow={"enabled": True, "color": "rgba(0,0,0,0.1)"},  # 阴影效果
            selectionWidth=2,           # 选中时线宽增量
            # 添加自定义数据用于JavaScript交互
            weight=weight,              # 关系权重
            arrowStrikethrough=False    # 箭头不穿透节点
        )
    
    # 使用临时文件保存并显示网络图 - 实现自定义HTML内容注入和资源清理
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        # 保存网络图到临时文件
        net.save_graph(tmp.name)
        
        # 读取生成的HTML内容
        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
            # 添加自定义样式和交互脚本 - 增强默认的pyvis生成的HTML
            # 导入并注入自定义CSS样式
            from .kg_styles import KG_STYLES
            html_content = html_content.replace('</head>', KG_STYLES + '</head>')
            
            # 导入并注入JavaScript交互脚本，实现Neo4j风格的交互功能
            from .interaction import KG_INTERACTION_SCRIPT
            html_content = html_content.replace('</body>', KG_INTERACTION_SCRIPT + '</body>')
            
            # 在Streamlit应用中显示HTML内容
            components.html(html_content, height=600)
        
        # 清理临时文件 - 确保不会留下临时文件
        try:
            os.unlink(tmp.name)
        except:
            pass  # 忽略删除错误，确保应用继续运行
    
    # 显示图例，使用现代样式 - 帮助用户理解节点类型和颜色对应关系
    st.write("### 图例")

    # 按特定优先级顺序显示图例 - 优化图例的组织和可读性
    priority_groups = ["Center", "Source", "Target", "Common"]  # 优先显示的关键节点类型
    community_groups = []   # 社区类型节点
    other_groups = []       # 其他类型节点

    # 对组类型进行分类 - 将所有节点类型分类到不同列表中
    for group in group_colors.keys():
        if group in priority_groups:
            continue  # 优先类型将在后面单独处理
        elif isinstance(group, str) and "Community" in group:
            community_groups.append(group)  # 社区类型
        else:
            other_groups.append(group)      # 其他类型

    # 排序以确保一致的显示顺序 - 确保每次显示时图例顺序相同
    community_groups.sort()
    other_groups.sort()

    # 合并所有组，保持优先顺序 - 确保关键类型首先显示，然后是普通类型，最后是社区类型
    all_groups = []
    # 先添加优先类型（如果存在）
    for group in priority_groups:
        if group in group_colors:
            all_groups.append(group)
    # 然后添加其他类型
    all_groups.extend(other_groups)
    # 最后添加社区类型
    all_groups.extend(community_groups)

    # 创建多列显示，使用更美观的图例样式 - 采用三列布局提高空间利用率
    cols = st.columns(3)
    # 循环显示每个图例项 - 为每个节点类型创建美观的图例条目
    for i, group in enumerate(all_groups):
        if group in group_colors:
            color = group_colors[group]  # 获取该类型的颜色
            col_idx = i % 3  # 计算应该显示在哪一列
            with cols[col_idx]:
                # 为常见类型提供中文显示名称
                group_display_name = group
                if group == "Center":
                    group_display_name = "中心节点"
                elif group == "Source":
                    group_display_name = "源节点"
                elif group == "Target":
                    group_display_name = "目标节点"
                elif group == "Common":
                    group_display_name = "共同邻居"
                    
                # 使用HTML和CSS创建美观的图例项
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:12px">'  # 容器
                    f'<div style="width:20px;height:20px;border-radius:50%;background-color:{color};margin-right:10px;box-shadow:0 2px 4px rgba(0,0,0,0.1);"></div>'  # 颜色圆点
                    f'<span style="font-family:sans-serif;color:#333;">{group_display_name}</span>'  # 文本标签
                    f'</div>',
                    unsafe_allow_html=True  # 允许渲染HTML
                )