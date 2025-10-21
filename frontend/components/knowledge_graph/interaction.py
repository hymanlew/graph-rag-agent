"""
知识图谱交互模块

该模块定义了知识图谱的交互JavaScript脚本，用于实现图谱的用户交互功能，
包括节点操作、视图控制、右键菜单等。脚本通过vis.js库与知识图谱进行交互，
提供了类似Neo4j的用户体验。

主要功能包括：
1. 图谱初始化和数据保存
2. 物理引擎配置，实现节点动画效果
3. 控制面板创建，提供图谱操作按钮
4. 事件处理系统，支持双击、单击、右键等操作
5. 节点聚焦、重置、历史记录等核心交互功能
"""
KG_INTERACTION_SCRIPT = """
<script>
    /**
     * 全局变量定义
     * - originalNodes: 原始节点数据集，用于重置图谱
     * - originalEdges: 原始边数据集，用于重置图谱
     * - isFiltered: 标记图谱是否处于过滤状态
     * - lastSelectedNode: 上一次选中的节点ID
     * - nodeHistory: 节点浏览历史记录
     */
    var originalNodes = [];
    var originalEdges = [];
    var isFiltered = false;
    var lastSelectedNode = null;
    var nodeHistory = [];
    
    /**
     * 页面加载完成后初始化
     * 保存原始图谱数据，供后续重置操作使用
     */
    document.addEventListener('DOMContentLoaded', function() {
        // 延迟保存，确保图谱已完全初始化
        setTimeout(function() {
            try {
                originalNodes = new vis.DataSet(network.body.data.nodes.get());
                originalEdges = new vis.DataSet(network.body.data.edges.get());
                console.log("图谱数据已保存:", originalNodes.length, "节点,", originalEdges.length, "关系");
            } catch(e) {
                console.error("保存图谱数据出错:", e);
            }
        }, 500);
    });
    
    /**
     * 配置物理引擎，实现节点动画效果
     * 使图谱在初始加载后有轻微的弹性动画，提升用户体验
     */
    setTimeout(function() {
        try {
            network.once("stabilizationIterationsDone", function() {
                network.setOptions({ 
                    physics: { 
                        stabilization: false,
                        barnesHut: {
                            gravitationalConstant: -2000,  // 重力常数，影响节点间吸引力
                            springConstant: 0.04,         // 弹簧常数，影响弹性效果
                            damping: 0.2,                // 阻尼系数，控制动画衰减
                        }
                    } 
                });
            });
            network.stabilize(200);  // 执行稳定化，参数为迭代次数
        } catch(e) {
            console.error("设置物理引擎出错:", e);
        }
    }, 1000);
    
    // 延迟创建控制面板，确保DOM已准备就绪
    setTimeout(createControlPanel, 800);
    
    /**
     * 添加基本事件监听器
     * 实现鼠标交互、节点操作等核心功能
     */
    try {
        // 添加鼠标悬停效果 - 改变鼠标指针样式
        network.on("hoverNode", function(params) {
            document.body.style.cursor = 'pointer';
        });
        
        network.on("blurNode", function(params) {
            document.body.style.cursor = 'default';
        });
        
        // 处理节点双击事件 - 实现类似Neo4j的邻居查看功能
        network.on("doubleClick", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                focusOnNode(nodeId);
            }
        });
        
        // 添加单击背景事件 - 恢复完整图谱视图
        network.on("click", function(params) {
            if (params.nodes.length === 0 && params.edges.length === 0) {
                resetGraph();
            }
        });
        
        // 添加右键菜单功能 - 提供节点操作选项
        network.on("oncontext", function(params) {
            params.event.preventDefault();
            var nodeId = network.getNodeAt(params.pointer.DOM);
            
            if (nodeId) {
                showContextMenu(nodeId, params);
            }
        });
    } catch(e) {
        console.error("添加事件处理出错:", e);
    }
    
    /**
     * 创建浮动控制面板
     * 
     * 动态生成包含标题、重置按钮、后退按钮和信息显示区域的控制面板，
     * 为用户提供图谱交互的便捷入口。
     */
    function createControlPanel() {
        try {
            // 创建控制面板容器
            var controlPanel = document.createElement('div');
            controlPanel.id = 'graph-control-panel';
            controlPanel.className = 'graph-control-panel';
            
            // 添加控制面板标题
            var panelTitle = document.createElement('div');
            panelTitle.style.fontWeight = 'bold';
            panelTitle.style.marginBottom = '8px';
            panelTitle.style.borderBottom = '1px solid #eee';
            panelTitle.style.paddingBottom = '5px';
            panelTitle.textContent = '图谱控制';
            controlPanel.appendChild(panelTitle);
            
            // 添加重置按钮 - 用于恢复完整图谱视图
            var resetButton = document.createElement('button');
            resetButton.textContent = '重置图谱';
            resetButton.className = 'graph-control-button';
            resetButton.onclick = resetGraph;
            controlPanel.appendChild(resetButton);
            
            // 添加后退按钮 - 用于撤销操作，返回上一个视图状态
            var backButton = document.createElement('button');
            backButton.textContent = '返回上一步';
            backButton.className = 'graph-control-button';
            backButton.onclick = goBack;
            controlPanel.appendChild(backButton);
            
            // 添加信息显示区域 - 用于展示节点详情和统计信息
            var infoDiv = document.createElement('div');
            infoDiv.id = 'graph-info';
            infoDiv.className = 'graph-info';
            controlPanel.appendChild(infoDiv);
            
            // 将控制面板添加到文档中的网络容器旁边
            var networkContainer = document.querySelector('.vis-network');
            if (networkContainer && networkContainer.parentNode) {
                networkContainer.parentNode.appendChild(controlPanel);
                console.log("控制面板已创建");
            } else {
                console.error("找不到网络容器");
            }
        } catch(e) {
            console.error("创建控制面板出错:", e);
        }
    }
    
    /**
     * 显示右键菜单
     * 
     * @param {string} nodeId - 节点ID
     * @param {Object} params - 事件参数，包含坐标等信息
     * 
     * 在用户右键点击节点时，显示包含聚焦、隐藏和查看信息等操作的上下文菜单。
     */
    function showContextMenu(nodeId, params) {
        try {
            // 获取节点信息
            var nodeInfo = network.body.data.nodes.get(nodeId);
            
            // 创建或获取上下文菜单
            var contextMenu = document.getElementById('node-context-menu');
            if (!contextMenu) {
                contextMenu = document.createElement('div');
                contextMenu.id = 'node-context-menu';
                contextMenu.className = 'node-context-menu';
                document.body.appendChild(contextMenu);
                
                // 点击其他地方关闭菜单
                document.addEventListener('click', function() {
                    if (contextMenu) contextMenu.style.display = 'none';
                });
            }
            
            // 设置菜单位置 - 基于事件坐标定位
            var canvasRect = params.event.srcElement.getBoundingClientRect();
            contextMenu.style.left = (canvasRect.left + params.pointer.DOM.x) + 'px';
            contextMenu.style.top = (canvasRect.top + params.pointer.DOM.y) + 'px';
            
            // 设置菜单内容 - 包含节点标签、操作选项和类型信息
            var label = nodeInfo.label || nodeId;
            var group = nodeInfo.group || "未知类型";
            
            contextMenu.innerHTML = `
                <div class="node-context-menu-header">
                    ${label}
                </div>
                <div class="node-context-menu-item" id="focus-node">
                    🔍 聚焦此节点
                </div>
                <div class="node-context-menu-item" id="hide-node">
                    🚫 隐藏此节点
                </div>
                <div class="node-context-menu-item" id="show-info">
                    ℹ️ 查看详细信息
                </div>
                <div class="node-context-menu-header" style="margin-top:5px;font-size:11px;color:#666;border-bottom:none;">
                    类型: ${group}
                </div>
            `;
            
            // 显示菜单
            contextMenu.style.display = 'block';
            
            // 添加菜单项点击事件处理
            document.getElementById('focus-node').onclick = function(e) {
                e.stopPropagation();
                focusOnNode(nodeId);
                contextMenu.style.display = 'none';
            };
            
            document.getElementById('hide-node').onclick = function(e) {
                e.stopPropagation();
                // 从当前视图中移除节点
                network.body.data.nodes.remove(nodeId);
                contextMenu.style.display = 'none';
            };
            
            document.getElementById('show-info').onclick = function(e) {
                e.stopPropagation();
                showNodeDetails(nodeId);
                contextMenu.style.display = 'none';
            };
        } catch(e) {
            console.error("显示上下文菜单出错:", e);
        }
    }
    
    /**
     * 显示节点详细信息
     * 
     * @param {string} nodeId - 节点ID
     * 
     * 在信息面板中显示节点的详细属性和关联关系统计，
     * 并通过样式变化高亮显示该节点。
     */
    function showNodeDetails(nodeId) {
        try {
            var node = network.body.data.nodes.get(nodeId);
            if (!node) return;
            
            // 格式化节点基本信息
            var details = '';
            details += `<div style="font-weight:bold;margin-bottom:5px;">节点ID: ${node.id}</div>`;
            details += `<div style="margin-bottom:5px;">标签: ${node.label || '无'}</div>`;
            details += `<div style="margin-bottom:5px;">类型: ${node.group || '未知'}</div>`;
            details += `<div>描述: ${node.description || '无描述'}</div>`;
            
            // 查找连接的边和节点，统计关联信息
            var connectedNodes = [];
            var connectedEdges = [];
            
            var edges = network.body.data.edges.get();
            edges.forEach(function(edge) {
                if (edge.from === nodeId || edge.to === nodeId) {
                    connectedEdges.push(edge);
                    var connectedNodeId = edge.from === nodeId ? edge.to : edge.from;
                    if (!connectedNodes.includes(connectedNodeId)) {
                        connectedNodes.push(connectedNodeId);
                    }
                }
            });
            
            // 添加连接信息到详情面板
            details += `<div style="margin-top:10px;"><strong>相连节点:</strong> ${connectedNodes.length}</div>`;
            details += `<div><strong>关系数量:</strong> ${connectedEdges.length}</div>`;
            
            // 创建或更新信息显示区域
            var infoDiv = document.getElementById('graph-info');
            if (infoDiv) {
                infoDiv.innerHTML = details;
            }
            
            // 高亮选中节点，通过增大边框和改变颜色实现
            network.body.data.nodes.update([{
                id: nodeId,
                borderWidth: 3,
                borderColor: '#FF5733',
                size: 35
            }]);
        } catch(e) {
            console.error("显示节点详情出错:", e);
        }
    }
    
    /**
     * 获取相连节点和边
     * 
     * @param {string} nodeId - 节点ID
     * @returns {Object} - 包含相连节点ID和边ID的对象
     * 
     * 查找与指定节点直接相连的所有节点和边，
     * 用于实现节点聚焦时的视图过滤功能。
     */
    function getConnectedNodes(nodeId) {
        try {
            var connectedNodes = [nodeId];
            var connectedEdges = [];
            
            // 获取与当前节点连接的所有边
            var edges = network.body.data.edges.get();
            for (var i = 0; i < edges.length; i++) {
                var edge = edges[i];
                if (edge.from === nodeId || edge.to === nodeId) {
                    connectedEdges.push(edge.id);
                    
                    // 添加边的另一端节点
                    var connectedNodeId = edge.from === nodeId ? edge.to : edge.from;
                    if (!connectedNodes.includes(connectedNodeId)) {
                        connectedNodes.push(connectedNodeId);
                    }
                }
            }
            
            return {
                nodes: connectedNodes,
                edges: connectedEdges
            };
        } catch(e) {
            console.error("获取相连节点出错:", e);
            return { nodes: [nodeId], edges: [] };
        }
    }
    
    /**
     * 聚焦到节点
     * 
     * @param {string} nodeId - 要聚焦的节点ID
     * 
     * 实现核心的节点聚焦功能，过滤显示该节点及其直接关联的节点和边，
     * 同时保存历史状态以便返回操作。
     */
    function focusOnNode(nodeId) {
        try {
            // 保存历史状态，用于返回上一步操作
            if (lastSelectedNode !== nodeId) {
                nodeHistory.push({
                    nodeId: lastSelectedNode,
                    isFiltered: isFiltered
                });
            }
            
            // 更新节点状态标记
            lastSelectedNode = nodeId;
            isFiltered = true;
            
            // 获取节点信息
            var nodeInfo = network.body.data.nodes.get(nodeId);
            var nodeLabel = nodeInfo.label || nodeId;
            
            // 获取与所选节点连接的节点和边
            var connected = getConnectedNodes(nodeId);
            
            // 更新图谱，只显示连接的节点和边，实现视图过滤
            var connectedNodes = network.body.data.nodes.get(connected.nodes);
            var connectedEdges = network.body.data.edges.get(connected.edges);
            
            network.body.data.nodes.clear();
            network.body.data.edges.clear();
            
            network.body.data.nodes.add(connectedNodes);
            network.body.data.edges.add(connectedEdges);
            
            // 更新信息面板，显示当前节点的关联统计
            updateInfoPanel(nodeLabel, connected.nodes.length - 1, connected.edges.length);
            
            // 突出显示选中的节点，通过样式变化实现
            network.body.data.nodes.update([{
                id: nodeId,
                borderWidth: 3,
                borderColor: '#FF5733',
                size: 35
            }]);
            
            // 聚焦并适应视图，使用动画效果平滑过渡
            network.focus(nodeId, {
                scale: 1.2,
                animation: true
            });
            
            console.log("已聚焦到节点:", nodeId);
        } catch(e) {
            console.error("聚焦节点出错:", e);
        }
    }
    
    /**
     * 重置图谱
     * 
     * 恢复完整的图谱视图，清除所有过滤状态和历史记录，
     * 重置信息面板并调整视图以显示所有节点。
     */
    function resetGraph() {
        try {
            // 检查是否需要重置
            if (!isFiltered || !originalNodes || originalNodes.length === 0) return;
            
            // 清空历史和状态标记
            nodeHistory = [];
            lastSelectedNode = null;
            isFiltered = false;
            
            // 恢复原始数据
            network.body.data.nodes.clear();
            network.body.data.edges.clear();
            
            network.body.data.nodes.add(originalNodes.get());
            network.body.data.edges.add(originalEdges.get());
            
            // 重置视图，适应所有节点
            network.fit({
                animation: true
            });
            
            // 清空信息面板
            var infoDiv = document.getElementById('graph-info');
            if (infoDiv) infoDiv.innerHTML = '';
            
            console.log("图谱已重置");
        } catch(e) {
            console.error("重置图谱出错:", e);
        }
    }
    
    /**
     * 返回上一步
     * 
     * 从历史记录中恢复上一个浏览状态，如果没有历史记录则重置图谱。
     * 实现用户操作的撤销功能。
     */
    function goBack() {
        try {
            if (nodeHistory.length === 0) {
                // 如果没有历史记录，则重置图谱
                resetGraph();
                return;
            }
            
            // 获取上一个状态
            var prevState = nodeHistory.pop();
            
            if (prevState.isFiltered && prevState.nodeId !== null) {
                // 如果上一个状态是过滤的，聚焦到该节点
                focusOnNode(prevState.nodeId);
                // 移除刚刚添加的状态
                nodeHistory.pop();
            } else {
                // 如果上一个状态不是过滤的，重置图谱
                resetGraph();
            }
            
            console.log("返回上一步");
        } catch(e) {
            console.error("返回上一步出错:", e);
        }
    }
    
    /**
     * 更新信息面板
     * 
     * @param {string} nodeLabel - 节点标签
     * @param {number} connectedCount - 相连节点数量
     * @param {number} edgesCount - 关系数量
     * 
     * 在信息面板中显示当前聚焦节点的关联统计信息，
     * 并提供使用提示帮助用户理解操作方式。
     */
    function updateInfoPanel(nodeLabel, connectedCount, edgesCount) {
        try {
            var infoDiv = document.getElementById('graph-info');
            if (!infoDiv) return;
            
            infoDiv.innerHTML = `
                <div style="margin-bottom:5px;"><strong>当前节点:</strong> ${nodeLabel}</div>
                <div><strong>相连节点:</strong> ${connectedCount}</div>
                <div><strong>关系数量:</strong> ${edgesCount}</div>
                <div style="margin-top:8px;font-style:italic;font-size:11px;">双击节点查看其连接</div>
                <div style="font-style:italic;font-size:11px;">单击空白处重置图谱</div>
            `;
        } catch(e) {
            console.error("更新信息面板出错:", e);
        }
    }
</script>
"""