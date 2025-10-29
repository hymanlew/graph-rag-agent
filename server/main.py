"""
服务器主入口模块

该模块是整个知识图谱问答系统后端服务的启动入口，负责初始化FastAPI应用、配置路由、
管理数据库连接和处理应用生命周期事件。它是整个后端系统的顶层组件，连接各个功能模块，
提供统一的HTTP服务接口。

主要功能：
- 初始化FastAPI应用实例
- 配置API路由系统
- 管理数据库连接生命周期
- 处理应用启动和关闭事件
- 启动ASGI服务器（uvicorn）

架构设计：
- 采用FastAPI框架作为Web服务基础
- 使用模块化路由结构，便于功能扩展
- 实现资源生命周期管理，确保资源正确释放
- 支持多进程工作模式，提高并发处理能力
"""
import uvicorn
from fastapi import FastAPI
from routers import api_router
from server_config.database import get_db_manager
from services.agent_service import agent_manager
from config.settings import fastapi_workers

# 初始化 FastAPI 应用
# 创建FastAPI实例，设置应用标题和描述，提供API文档自动生成功能
app = FastAPI(title="知识图谱问答系统", description="基于知识图谱的智能问答系统后端API")

# 添加路由
# 注册所有子路由到主路由，构建完整的API路由系统
# 子路由包括：聊天、反馈、知识图谱和源内容管理等功能模块
app.include_router(api_router)

# 获取数据库连接
# 初始化数据库管理器，建立与Neo4j数据库的连接
# 数据库连接作为全局资源，供整个应用使用
# 采用单例模式设计，确保数据库连接的统一管理

db_manager = get_db_manager()  # 数据库管理器实例，负责管理数据库连接池和会话
driver = db_manager.driver  # 数据库驱动实例，直接用于执行数据库操作


@app.on_event("shutdown")
def shutdown_event():
    """
    应用关闭时清理资源
    
    该函数在应用程序关闭时自动调用，负责释放所有关键资源，确保系统优雅退出。
    实现了资源生命周期的完整管理，防止资源泄漏和连接未关闭问题。
    
    清理流程：
    1. 关闭所有Agent资源，释放模型和计算资源
    2. 关闭Neo4j数据库连接，释放数据库会话
    3. 打印日志，记录资源关闭状态
    
    业务意义：
    - 确保资源正确释放，避免内存泄漏
    - 关闭数据库连接，防止连接池耗尽
    - 优雅退出，减少系统错误
    - 提供审计日志，便于问题追踪
    """
    # 关闭所有Agent资源
    # 释放所有AI模型和代理实例占用的计算资源和内存
    agent_manager.close_all()
    
    # 关闭Neo4j连接
    # 确保数据库连接正确关闭，避免连接池耗尽和资源泄漏
    if driver:
        driver.close()
        print("已关闭Neo4j连接")


# 启动服务器
# 当作为主程序运行时，启动uvicorn ASGI服务器
# 配置主机地址为0.0.0.0（监听所有网络接口）
# 配置端口为8000（标准Web服务端口）
# 配置工作进程数，从settings导入，支持水平扩展
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=fastapi_workers)