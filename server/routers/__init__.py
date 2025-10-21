"""
API路由系统配置模块

该模块负责创建和组织FastAPI的路由系统，将各个功能模块的路由整合到统一的API路由树中。
它是API端点组织的核心文件，定义了整个后端服务的API结构和分类。

主要功能：
- 创建主API路由器实例
- 导入并整合所有功能模块的路由
- 设置路由标签，用于API文档分类展示
- 提供模块化的路由管理架构

设计特点：
- 采用模块化设计，各功能模块路由独立开发和维护
- 使用标签系统，提高API文档的可读性和可用性
- 统一路由注册入口，简化主应用配置
- 支持路由的灵活扩展，便于添加新功能模块
"""
from fastapi import APIRouter
# 导入各个功能模块的路由器
from . import chat, feedback, knowledge_graph, source

# 创建总路由器
# APIRouter实例作为所有子路由的父容器，统一管理所有API端点
api_router = APIRouter()

# 包含各个子路由器并设置对应的标签
# 标签用于在API文档中对端点进行分类展示
api_router.include_router(chat.router, tags=["聊天"])  # 聊天功能路由，处理用户对话请求
api_router.include_router(feedback.router, tags=["反馈"])  # 用户反馈路由，收集用户对系统的评价
api_router.include_router(knowledge_graph.router, tags=["知识图谱"])  # 知识图谱操作路由，处理实体和关系的增删改查
api_router.include_router(source.router, tags=["源内容"])  # 源内容路由，提供内容溯源和查询功能