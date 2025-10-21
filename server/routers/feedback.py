"""
反馈API路由模块

该模块定义了处理用户反馈的API端点，是系统收集用户评价、监控服务质量和持续改进的重要组成部分。
用户可以对系统生成的回答提供正面或负面反馈，这些数据可用于模型调优、服务质量监控和用户体验优化。

主要功能：
- 接收用户对系统回答的反馈评价
- 处理反馈数据并存储
- 支持不同代理类型的反馈收集
- 提供性能监控功能

设计特点：
- 简单直观的API设计
- 异步处理提高并发能力
- 集成性能监控
- 与chat_service紧密协作
"""
from fastapi import APIRouter
from models.schemas import FeedbackRequest, FeedbackResponse
from services.chat_service import process_feedback
from utils.performance import measure_performance

# 创建路由器
# 初始化反馈相关的API路由组，用于管理反馈功能的端点
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
@measure_performance("feedback")
async def feedback(request: FeedbackRequest):
    """
    处理用户对回答的反馈
    
    该端点接收用户对系统生成回答的评价反馈，是系统自我改进和质量监控的核心接口。
    它支持收集正面和负面反馈，同时关联相关的对话上下文信息，便于后续分析和改进。
    
    Args:
        request: 反馈请求对象，包含反馈类型、消息ID、查询内容等信息
        
    Returns:
        FeedbackResponse: 反馈响应对象，包含处理状态和执行的操作信息
        
    业务流程：
    1. 接收并验证用户反馈数据
    2. 调用chat_service中的process_feedback处理反馈
    3. 将处理结果封装为FeedbackResponse格式并返回
    4. 通过性能监控装饰器记录API调用性能数据
    
    业务意义：
        - 收集用户对系统的真实评价
        - 为模型调优提供实际用户反馈数据
        - 监控系统回答质量的变化趋势
        - 识别需要改进的特定问题和场景
        - 提供用户参与系统优化的渠道
    """
    # 调用反馈处理服务，传入所有相关参数
    # process_feedback负责处理反馈逻辑，包括存储、分析和可能的模型更新
    result = await process_feedback(
        message_id=request.message_id,  # 消息唯一标识符
        query=request.query,  # 用户原始查询内容
        is_positive=request.is_positive,  # 是否为正面反馈
        thread_id=request.thread_id,  # 对话线程ID
        agent_type=request.agent_type  # 使用的代理类型
    )
    
    # 将处理结果转换为预定义的响应模型
    # FeedbackResponse确保响应数据符合预期的结构和类型
    return FeedbackResponse(**result)