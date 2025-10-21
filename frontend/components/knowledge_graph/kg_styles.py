"""
知识图谱样式模块

该模块定义了知识图谱可视化所需的CSS样式，通过定义KG_STYLES常量提供完整的样式表，
用于美化知识图谱界面，提升用户体验。样式设计融合了现代UI设计理念，
采用了类似Neo4j的视觉风格，提供清晰的布局和交互反馈。

主要样式组件包括：
1. 图谱容器样式 - 定义边框、阴影和布局
2. 提示框样式 - 美化鼠标悬停时的信息提示
3. 节点交互效果 - 添加悬停动画增强用户反馈
4. 右键菜单 - 类似Neo4j风格的上下文菜单
5. 控制面板 - 浮动的操作面板样式
6. 按钮样式 - 控制按钮的外观和交互效果
7. 信息面板 - 节点详情和状态信息显示区域
"""
# 知识图谱CSS样式
KG_STYLES = """
<style>
    /**
     * 图谱容器样式
     * 设置边框、圆角和阴影效果，提升整体美观度
     */
    .vis-network {
        border: 1px solid #e8e8e8;      /* 浅灰色边框 */
        border-radius: 8px;             /* 圆角效果 */
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);  /* 轻微阴影 */
        position: relative;             /* 相对定位，作为内部绝对定位元素的参考 */
    }
    
    /**
     * 提示框样式
     * 美化默认提示框，使用 !important 确保样式优先级
     */
    .vis-tooltip {
        background-color: white !important;      /* 白色背景 */
        color: #333 !important;                  /* 深灰色文字 */
        border: 1px solid #e0e0e0 !important;    /* 边框 */
        border-radius: 4px !important;           /* 圆角 */
        padding: 8px !important;                 /* 内边距 */
        font-family: 'Arial', sans-serif !important;  /* 字体 */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;  /* 阴影 */
    }
    
    /**
     * 节点悬停动画效果
     * 当鼠标悬停在节点上时，节点放大并平滑过渡，提升交互体验
     */
    .vis-node:hover {
        transform: scale(1.1);          /* 放大1.1倍 */
        transition: all 0.3s ease;      /* 0.3秒平滑过渡 */
    }
    
    /**
     * 右键菜单样式
     * 实现类似Neo4j的上下文菜单外观
     */
    .node-context-menu {
        position: absolute;             /* 绝对定位，根据鼠标位置动态放置 */
        background: white;              /* 白色背景 */
        border: 1px solid #ccc;         /* 灰色边框 */
        border-radius: 4px;             /* 圆角 */
        padding: 5px 0;                 /* 垂直内边距 */
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);  /* 阴影 */
        z-index: 1000;                  /* 高z-index确保菜单始终可见 */
        min-width: 150px;               /* 最小宽度 */
    }
    
    /* 菜单项样式 */
    .node-context-menu-item {
        padding: 5px 10px;              /* 内边距 */
        cursor: pointer;                /* 鼠标指针变为手型 */
    }
    
    /* 菜单项悬停效果 */
    .node-context-menu-item:hover {
        background-color: #f0f0f0;      /* 浅灰色背景 */
    }
    
    /* 菜单项标题样式 */
    .node-context-menu-header {
        padding: 5px 10px;              /* 内边距 */
        font-weight: bold;              /* 粗体 */
        border-bottom: 1px solid #eee;  /* 底部边框 */
    }
    
    /**
     * 控制面板样式
     * 浮动在图谱右上角，提供控制功能
     */
    .graph-control-panel {
        position: absolute;             /* 绝对定位 */
        top: 10px;                      /* 距离顶部10px */
        right: 10px;                    /* 距离右侧10px */
        z-index: 999;                   /* 高z-index确保在图谱上方 */
        background: white;              /* 白色背景 */
        border: 1px solid #ccc;         /* 边框 */
        border-radius: 6px;             /* 圆角 */
        padding: 10px;                  /* 内边距 */
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);  /* 阴影 */
        min-width: 180px;               /* 最小宽度 */
    }
    
    /**
     * 控制按钮样式
     * 自定义按钮外观，与整体设计风格保持一致
     */
    .graph-control-button {
        display: block;                 /* 块级元素，占满容器宽度 */
        width: 100%;                    /* 宽度100% */
        margin: 5px 0;                  /* 上下外边距 */
        padding: 6px 12px;              /* 内边距 */
        background-color: #f8f9fa;      /* 浅灰色背景 */
        border: 1px solid #ddd;         /* 边框 */
        border-radius: 4px;             /* 圆角 */
        cursor: pointer;                /* 鼠标指针变为手型 */
        text-align: left;               /* 文字左对齐 */
    }
    
    /* 按钮悬停效果 */
    .graph-control-button:hover {
        background-color: #e9ecef;      /* 更深的灰色背景 */
    }
    
    /**
     * 信息面板样式
     * 用于显示节点详情和统计信息
     */
    .graph-info {
        font-size: 12px;                /* 小字体 */
        margin-top: 8px;                /* 顶部外边距 */
        color: #666;                    /* 灰色文字 */
        border-top: 1px solid #eee;     /* 顶部边框 */
        padding-top: 8px;               /* 顶部内边距 */
    }
</style>
"""