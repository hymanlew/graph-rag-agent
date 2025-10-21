"""
源内容API路由模块

该模块定义了与源内容相关的所有API端点，提供了获取源文件内容和元信息的功能。
它支持单个和批量操作，是系统获取和管理原始数据的重要接口，为知识图谱和聊天功能提供数据支持。

主要功能：
- 获取单个源文件的详细内容
- 获取源文件的元信息（如文件名、路径等）
- 批量获取多个内容块
- 批量获取多个源文件的信息

设计特点：
- 支持单次和批量操作，提高系统效率
- 统一的错误处理和异常管理
- 集成数据库连接管理
- 与知识图谱服务紧密协作
"""
from typing import Dict
from fastapi import APIRouter, HTTPException
from models.schemas import SourceRequest, SourceResponse, SourceInfoResponse, SourceInfoBatchRequest, ContentBatchRequest
from services.kg_service import get_source_content, get_source_file_info
from utils.neo4j_batch import BatchProcessor
from config.neo4jdb import get_db_manager

# 创建路由器
# 初始化源内容相关的API路由组，用于管理所有与源内容相关的端点
router = APIRouter()


@router.post("/source", response_model=SourceResponse)
async def source(request: SourceRequest):
    """
    处理源内容请求
    
    该端点根据源ID获取完整的源文件内容，是系统访问原始数据的主要接口之一。
    它直接调用知识图谱服务获取内容，并将结果封装为标准响应格式。
    
    Args:
        request: 源内容请求对象，包含需要获取内容的源ID
        
    Returns:
        SourceResponse: 包含源内容的响应对象
            - content: 源文件的完整文本内容
            
    业务流程：
    1. 接收并验证包含source_id的请求
    2. 调用kg_service中的get_source_content获取源文件内容
    3. 将内容封装为SourceResponse格式并返回
    
    业务意义：
        - 提供对原始文档内容的访问接口
        - 支持前端显示完整的源文本
        - 为用户提供回答的证据和参考来源
    """
    content = get_source_content(request.source_id)
    return SourceResponse(content=content)

@router.post("/source_info")
async def source_info(request: SourceRequest):
    """
    处理源文件信息请求
    
    该端点根据源ID获取源文件的元信息，如文件名、路径等，不包含实际内容。
    它用于展示文档的基本信息，帮助用户快速识别和定位特定文档。
    
    Args:
        request: 源内容请求对象，包含需要获取信息的源ID
        
    Returns:
        Dict: 包含源文件元信息的响应字典
            - 可能包含文件名、路径、创建时间、大小等信息
            
    业务流程：
    1. 接收并验证包含source_id的请求
    2. 调用kg_service中的get_source_file_info获取源文件信息
    3. 直接返回获取到的信息字典
    
    业务意义：
        - 提供文档的元数据信息
        - 支持文档管理和浏览功能
        - 帮助用户了解文档的基本属性
    """
    info = get_source_file_info(request.source_id)
    return info

@router.post("/content_batch", response_model=Dict)
async def get_content_batch(request: ContentBatchRequest):
    """
    批量获取内容
    
    该端点支持一次性获取多个内容块，通过批量处理提高系统效率，减少网络请求次数。
    它特别适用于需要同时显示多个相关内容片段的场景，如文档比较或综合展示。
    
    Args:
        request: 内容批量请求对象，包含多个内容块ID
            - chunk_ids: 需要获取的内容块ID列表
            
    Returns:
        Dict: 包含批量内容的字典，以内容块ID为键，内容为值
        
    Raises:
        HTTPException: 批量获取内容失败时抛出500错误
        
    业务流程：
    1. 接收包含多个chunk_ids的批量请求
    2. 获取数据库连接驱动
    3. 使用BatchProcessor批量处理获取内容的请求
    4. 返回所有内容块的映射结果
    5. 发生异常时，抛出适当的HTTP错误
    
    业务意义：
        - 提高内容获取效率，减少网络请求
        - 支持批量展示多个相关内容
        - 优化前端加载性能
    """
    try:
        # 获取数据库驱动
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 使用BatchProcessor批量处理
        result = BatchProcessor.get_content_batch(request.chunk_ids, driver)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量获取内容失败: {str(e)}")


@router.post("/source_info_batch", response_model=Dict)
async def get_source_info_batch(request: SourceInfoBatchRequest):
    """
    批量获取源信息
    
    该端点支持一次性获取多个源文件的元信息，适用于需要同时展示多个文档信息的场景。
    通过批量处理，显著提高了系统效率和用户体验。
    
    Args:
        request: 源信息批量请求对象，包含多个源文件ID
            - source_ids: 需要获取信息的源文件ID列表
            
    Returns:
        Dict: 包含多个源文件元信息的字典，以源文件ID为键，信息字典为值
        
    Raises:
        HTTPException: 批量获取源信息失败时抛出500错误
        
    业务流程：
    1. 接收包含多个source_ids的批量请求
    2. 获取数据库连接驱动
    3. 使用BatchProcessor批量处理获取源信息的请求
    4. 返回所有源文件信息的映射结果
    5. 发生异常时，抛出适当的HTTP错误
    
    业务意义：
        - 提高源文件信息获取效率
        - 支持文档列表和批量管理功能
        - 优化前端多文档显示性能
    """
    try:
        # 获取数据库驱动
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        
        # 使用BatchProcessor批量处理
        result = BatchProcessor.get_source_info_batch(request.source_ids, driver)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量获取源信息失败: {str(e)}")