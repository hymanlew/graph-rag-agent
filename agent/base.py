from typing import Annotated, Sequence, TypedDict, List, Dict, Any, AsyncGenerator
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint
import time
import asyncio

# 导入模型获取函数
from model.get_models import get_llm_model, get_stream_llm_model, get_embeddings_model
# 导入缓存管理相关类
from CacheManage.manager import (
    CacheManager, 
    ContextAwareCacheKeyStrategy, 
    HybridCacheBackend
)
from CacheManage.strategies.global_strategy import GlobalCacheKeyStrategy

class BaseAgent(ABC):
    """
    代理系统抽象基类，为所有具体代理实现提供统一接口和核心功能框架。
    
    这个抽象基类定义了代理系统的核心架构，包括工作流管理、多级缓存系统、
    性能监控、日志记录和决策机制。所有具体代理实现都应继承此类，并实现
    所需的抽象方法以提供特定功能。
    
    主要职责：
    - 提供代理系统的基本结构和生命周期管理
    - 实现多级缓存策略，提高响应速度和系统性能
    - 定义代理决策和工具使用的工作流
    - 收集和记录系统性能指标
    - 提供统一的查询接口和错误处理机制
    """
    
    def __init__(self, cache_dir="./cache", memory_only=False):
        """
        初始化代理系统的核心组件，配置模型、缓存和工作流。
        
        参数:
            cache_dir: 缓存目录路径，用于存储磁盘缓存结果
            memory_only: 是否仅使用内存缓存模式，True表示不使用磁盘持久化
        
        初始化过程包括：
        1. 加载语言模型和嵌入模型
        2. 创建记忆管理器，用于保存对话历史
        3. 初始化多级缓存系统
        4. 配置性能监控
        5. 设置代理工作流图
        """
        # 初始化语言模型组件
        # 获取普通LLM模型，用于生成完整响应
        self.llm = get_llm_model()
        # 获取流式LLM模型，用于增量生成响应，支持实时展示
        self.stream_llm = get_stream_llm_model()
        # 获取嵌入模型，用于文本向量化，支持语义相似度计算
        self.embeddings = get_embeddings_model()
        
        # 初始化记忆系统
        # 用于保存和检索对话历史，支持多轮对话
        self.memory = MemorySaver()
        # 初始化执行日志列表，用于记录代理运行的各个步骤详细信息
        self.execution_log = []
    
        # 初始化会话内缓存管理器
        # 使用上下文感知的缓存键策略，提高缓存命中率
        self.cache_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=200,  # 内存缓存最大条目数
                disk_max_size=2000   # 磁盘缓存最大条目数
            ) if not memory_only else None,
            cache_dir=cache_dir,
            memory_only=memory_only
        )
        
        # 初始化全局缓存管理器
        # 使用全局缓存键策略，支持跨会话共享缓存结果
        self.global_cache_manager = CacheManager(
            key_strategy=GlobalCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=f"{cache_dir}/global",
                memory_max_size=500,
                disk_max_size=5000
            ) if not memory_only else None,
            cache_dir=f"{cache_dir}/global",
            memory_only=memory_only
        )
        
        # 初始化性能指标字典，用于收集和分析系统性能数据
        self.performance_metrics = {}
        
        # 初始化代理工具集，由子类实现具体工具配置
        self.tools = self._setup_tools()
        
        # 设置代理工作流图，定义状态转换和处理节点
        self._setup_graph()
    
    @abstractmethod
    def _setup_tools(self) -> List:
        """
        配置代理可用的工具集
        
        抽象方法，由具体的代理实现类负责实现。该方法应返回代理可使用的工具列表，
        这些工具将用于执行特定任务如检索、知识图谱查询、数据分析或其他领域特定操作。
        
        工具是代理系统的核心能力扩展点，每个工具通常实现一个特定的功能，
        如网络搜索、数据库查询、文件操作或API调用等。代理可以根据任务需求
        选择合适的工具来完成工作。
        
        返回:
            List: 代理可以使用的工具对象列表，每个工具应符合LangChain工具规范
        
        示例:
            子类实现时可能返回如下工具列表：
            [SearchTool(), KnowledgeGraphQueryTool(), CalculatorTool()]
        """
        pass
    
    def _setup_graph(self):
        """
        设置代理工作流图，定义状态机和处理节点
        
        该方法创建代理的工作流状态机，使用LangGraph构建有向图结构，
        定义各个处理节点及其转换关系。工作流是代理处理查询的骨架，
        决定了信息如何在不同处理阶段间流转。
        
        工作流主要组成部分：
        1. 状态定义：AgentState类型，确定工作流中传递的数据结构
        2. 核心节点：
           - agent节点：分析用户输入并决定操作策略
           - retrieve节点：执行工具调用获取外部信息
           - generate节点：基于收集的信息生成最终回答
        3. 转换逻辑：定义节点间的跳转条件和触发规则
        4. 工作流编译：生成可执行的状态机实例
        
        子类可通过重写_add_retrieval_edges方法自定义检索到生成的流程，
        以实现特定代理所需的复杂处理逻辑。
        """
        # 定义状态类型 - 工作流中传递的数据结构
        class AgentState(TypedDict):
            # 消息序列，使用add_messages注解确保消息正确追加
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # 创建工作流图 - 使用StateGraph构建状态机
        workflow = StateGraph(AgentState)
        
        # 添加核心处理节点
        workflow.add_node("agent", self._agent_node)  # 代理决策节点
        workflow.add_node("retrieve", ToolNode(self.tools))  # 工具调用/检索节点
        workflow.add_node("generate", self._generate_node)  # 回答生成节点
        
        # 设置工作流边 - 定义节点间的流转逻辑
        workflow.add_edge(START, "agent")  # 从开始到代理节点
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # 根据代理决策条件分支
            {
                "tools": "retrieve",  # 需要工具调用时转到检索节点
                END: END,  # 直接结束时转到结束
            },
        )
        
        # 添加从检索到生成的边 - 这个逻辑由子类实现自定义流程
        self._add_retrieval_edges(workflow)
        
        # 从生成到结束的边
        workflow.add_edge("generate", END)
        
        # 编译图 - 生成可执行的工作流
        self.graph = workflow.compile(checkpointer=self.memory)  # 使用memory作为检查点保存器
    
    async def _stream_process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        异步流式处理查询，提供实时响应生成功能。
        
        此方法实现查询的异步流式处理，支持增量返回结果，提高用户体验。
        处理流程包括获取输入消息、构建状态字典、调用异步生成节点，
        并将生成的回答按自然语言单位分块返回，同时进行错误处理。
        
        参数:
            inputs: 包含输入消息的数据字典，消息列表作为对话上下文
            config: 配置参数字典，包含会话ID等可配置信息
            
        返回:
            AsyncGenerator[str, None]: 流式响应生成器，产生文本块序列
            每个块是回答的一部分，确保流畅的用户阅读体验
            
        处理流程:
        1. 从输入数据中提取消息列表和查询内容
        2. 构建状态字典，包含消息和配置信息
        3. 调用异步生成节点获取完整回答
        4. 按句子或段落分块处理，实现自然的流式体验
        5. 返回分块的文本内容，支持实时展示
        """
        # 获取消息
        messages = inputs.get("messages", [])
        query = messages[-1].content if messages else ""
        
        # 构建状态字典
        state = {
            "messages": messages,
            "configurable": config.get("configurable", {})
        }
        
        # 获取生成结果
        result = await self._generate_node_async(state)
        
        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            content = message.content if hasattr(message, "content") else str(message)
            
            # 按句子或段落分块，更自然
            import re
            chunks = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""
            
            for i in range(0, len(chunks)):
                if i < len(chunks):
                    buffer += chunks[i]
                    
                    # 当缓冲区包含完整句子或达到合理大小时输出
                    if (i % 2 == 1) or len(buffer) >= 40:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)  # 微小延迟确保流畅显示
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
        else:
            yield "无法生成响应。"

    
    @abstractmethod
    def _add_retrieval_edges(self, workflow):
        """添加从检索节点到生成节点的边和条件
        
        子类必须实现此方法，定义检索结果如何传递给生成节点的逻辑
        不同的代理类型可能有不同的检索后处理流程
        
        参数:
            workflow: StateGraph对象，用于添加边和条件
        """
        pass
    
    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """记录节点执行的详细信息，用于调试、监控和分析。
        
        该方法创建结构化的执行日志条目，包含节点名称、时间戳、输入输出数据，
        记录到执行日志列表中。这些日志对于理解代理工作流程、诊断问题和
        优化系统性能至关重要。
        
        参数:
            node_name: 执行节点的名称，标识当前处理环节
            input_data: 节点处理的输入数据，包含上下文和参数
            output_data: 节点生成的输出结果，反映处理效果
            
        日志条目结构:
        - node: 节点名称
        - timestamp: Unix时间戳
        - input: 输入数据
        - output: 输出结果
        
        用途:
        - 问题诊断和执行追踪
        - 工作流优化
        - 性能分析
        - 代理行为审计
        """
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        })
    
    def _log_performance(self, operation, metrics):
        """记录各种操作的性能指标
        
        用于监控系统性能，包括操作耗时、缓存命中率等
        
        参数:
            operation: 操作名称
            metrics: 性能指标字典
        """
        self.performance_metrics[operation] = {
            "timestamp": time.time(),
            **metrics  # 合并传入的指标
        }
        
        # 输出关键性能指标到控制台
        if "duration" in metrics:
            print(f"性能指标 - {operation}: {metrics['duration']:.4f}s")
    
    def _agent_node(self, state):
        """代理决策节点逻辑
        
        分析用户输入，提取关键词，并决定是否需要调用工具
        这是工作流中的第一个处理节点，作为代理系统的智能中枢，负责分析用户意图
        并确定最合适的处理路径。该节点实现了输入增强和工具决策两项核心功能。
        
        参数:
            state: 当前工作流状态，包含消息列表
            
        返回:
            Dict: 更新后的状态，包含代理的决策结果和生成的消息
            
        处理流程:
        1. 从状态中提取消息列表
        2. 分析最新的用户消息，提取关键词
        3. 增强用户消息，添加关键词元数据
        4. 使用绑定工具的LLM模型分析消息
        5. 记录执行日志
        6. 返回更新后的状态
        """
        messages = state["messages"]
        
        # 提取关键词优化查询 - 对最新的用户消息进行关键词分析
        if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            keywords = self._extract_keywords(query)
            
            # 记录关键词提取结果
            self._log_execution("extract_keywords", query, keywords)
            
            # 增强消息，添加关键词信息作为元数据
            if keywords:
                enhanced_message = HumanMessage(
                    content=query,
                    additional_kwargs={"keywords": keywords}
                )
                # 替换原始消息为增强后的消息
                messages = messages[:-1] + [enhanced_message]
        
        # 将工具绑定到LLM，让模型决定是否使用工具
        model = self.llm.bind_tools(self.tools)
        # 调用模型进行决策
        response = model.invoke(messages)
        
        # 记录代理节点的执行情况
        self._log_execution("agent", messages, response)
        # 返回更新后的状态
        return {"messages": [response]}
    
    @abstractmethod
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """从用户查询中提取关键词
        
        子类必须实现此方法，定义如何从查询中提取有意义的关键词
        关键词通常分为低级关键词和高级关键词等不同类型。关键词提取是代理系统
        理解用户意图的基础，直接影响检索质量和最终回答的相关性。
        
        参数:
            query: 用户查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含不同类型的关键词列表
            - 低级关键词(low_level): 具体的实体、概念和术语
            - 高级关键词(high_level): 抽象概念、主题和意图指示词
            
        实现要点:
            - 识别查询中的核心概念和实体
            - 区分具体术语和抽象概念
            - 考虑领域特定的专业术语
            - 处理同义词和相关概念
        """
        pass
    
    @abstractmethod
    def _generate_node(self, state):
        """生成回答的节点逻辑
        
        子类必须实现此方法，定义如何基于检索结果生成最终回答
        这是工作流的最后一个主要处理节点，负责综合所有收集到的信息，
        生成准确、连贯且全面的最终回答。
        
        参数:
            state: 当前工作流状态，包含检索结果和对话历史
            
        返回:
            Dict: 包含生成的回答消息的状态
            
        主要职责:
        1. 分析工作流状态中的信息，特别是检索到的工具执行结果
        2. 将检索结果与用户原始查询相结合
        3. 生成逻辑连贯、信息准确的回答
        4. 确保回答格式规范，便于用户理解
        5. 记录回答生成过程的日志信息
        
        实现考虑因素:
            - 回答的相关性和准确性
            - 信息的组织和表达清晰度
            - 针对不同类型查询的回答策略
            - 引用来源和提供解释的能力
        """
        pass

    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本
        
        此方法实现异步流式回答生成，提供实时展示生成进度的功能。
        它通过将回答分割成小块并逐个返回，使前端可以实现打字机效果，
        提高用户体验和交互性。这是默认实现，子类应根据需要提供更高效的原生实现。
        
        参数:
            state: 当前工作流状态，包含检索结果和对话历史，是生成回答的上下文来源
            
        返回:
            AsyncGenerator[str, None]: 流式响应生成器，产生文本块序列，
            每个块是回答的一部分，共同构成完整回答
            
        工作原理:
        1. 首先调用同步的_generate_node方法获取完整回答
        2. 从结果中提取消息内容
        3. 将内容按字符分块，每个块4个字符
        4. 逐个yield文本块，并添加微小延迟以创造流畅的流式体验
        
        注意：
        这是一个简单实现，在实际应用中，子类应考虑使用原生流式API，
        避免先生成完整文本再分块，以提高性能和响应速度。
        """
        # 默认实现 - 调用同步的_generate_node方法获取完整结果
        # 实际使用中应由子类提供更高效的实现，避免先生成完整文本再分块
        result = self._generate_node(state)
        
        # 检查结果中是否包含消息
        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            # 获取消息内容
            content = message.content if hasattr(message, "content") else str(message)
            
            # 模拟流式输出 - 将文本按固定大小分块
            chunk_size = 4  # 每个分块的字符数
            for i in range(0, len(content), chunk_size):
                yield content[i:i+chunk_size]  # 产生文本块
                await asyncio.sleep(0.01)  # 微小延迟确保流式效果
    
    async def _generate_node_async(self, state):
        """生成回答节点逻辑的异步版本
        
        此方法提供异步执行_generate_node的能力，确保在异步环境中
        不会阻塞事件循环，从而提高系统的并发处理能力。这是一种桥接
        同步和异步代码的常见模式，允许使用同步实现的生成逻辑在异步环境中工作。
        
        参数:
            state: 当前工作流状态，包含检索结果和对话历史，作为生成回答的上下文
            
        返回:
            Dict: 包含生成的回答消息的状态字典，与_generate_node返回格式一致
            
        工作原理:
        1. 定义内部同步函数sync_generate，封装_generate_node的调用
        2. 使用asyncio的run_in_executor在线程池中执行同步代码
        3. 异步等待执行结果并返回
        
        优化建议:
        - 子类可以提供更高效的原生异步实现，直接使用异步API
        - 在高并发场景中，考虑配置专门的线程池以优化性能
        - 对于长时间运行的生成任务，可以考虑添加超时处理
        
        这种桥接模式特别适合在逐步迁移到全异步架构的过程中使用，
        允许现有同步代码与新的异步代码共存。
        """
        # 定义同步生成函数，用于在线程池中执行
        def sync_generate():
            return self._generate_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        # 这是一种桥接同步和异步代码的常见模式
        return await asyncio.get_event_loop().run_in_executor(None, sync_generate)
    
    def check_fast_cache(self, query: str, thread_id: str = "default") -> str:
        """专用的快速缓存检查方法，用于高性能路径
        
        此方法实现了一个高效的缓存检查机制，专门设计用于快速响应频繁出现的查询。
        它采用上下文感知的缓存键策略，通过结合查询内容和提取的关键词来提高缓存命中率，
        同时避免了完整的缓存验证过程，牺牲一定的准确性换取显著的性能提升。
        
        参数:
            query: 查询字符串，用户输入的原始问题或指令
            thread_id: 会话ID，默认为"default"，用于区分不同会话的缓存
            
        返回:
            str: 如果缓存命中，返回缓存的回答文本；否则返回None
            
        工作流程:
        1. 记录开始时间，用于性能监控
        2. 从查询中提取关键词，包括低级关键词和高级关键词
        3. 构建缓存参数，包括会话ID和关键词信息
        4. 调用缓存管理器的get_fast方法获取结果
        5. 计算操作耗时并记录性能指标
        6. 返回缓存结果（如果命中）或None（如果未命中）
        
        性能特点:
        - 采用优化的缓存查找路径，减少延迟
        - 包含详细的性能监控，便于系统调优
        - 与上下文感知的缓存策略配合，提高缓存命中率
        - 适用于对响应时间要求较高的场景
        
        使用场景:
        - 高频重复查询的快速响应
        - 需要低延迟的交互式应用
        - 已知高质量回答的缓存条目快速检索
        """
        start_time = time.time()
        
        # 提取关键词，确保在缓存键中使用
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),  # 获取低级关键词
            "high_level_keywords": keywords.get("high_level", [])  # 获取高级关键词
        }
        
        # 使用缓存管理器的快速获取方法，传递相关参数
        result = self.cache_manager.get_fast(query, **cache_params)
        
        # 记录性能指标
        duration = time.time() - start_time
        self._log_performance("fast_cache_check", {
            "duration": duration,  # 操作耗时
            "hit": result is not None  # 是否命中缓存
        })
        
        return result
    
    def _check_all_caches(self, query: str, thread_id: str = "default"):
        """整合的缓存检查方法
        
        按优先级检查多种缓存：全局缓存 > 快速路径缓存 > 常规缓存
        实现了多级缓存策略，以提高响应速度和减少重复计算
        
        参数:
            query: 查询字符串
            thread_id: 会话ID，默认为"default"
            
        返回:
            str: 如果任何缓存命中，返回缓存的回答；否则返回None
        """
        cache_check_start = time.time()
        
        # 1. 首先尝试全局缓存（跨会话缓存）- 优先级最高
        # 全局缓存包含可在多个会话间共享的查询结果
        global_result = self.global_cache_manager.get(query)
        if global_result:
            print(f"全局缓存命中: {query[:30]}...")
            
            # 记录缓存检查性能
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "global"
            })
            
            return global_result
        
        # 2. 尝试快速路径 - 跳过验证的高质量缓存
        # 这些缓存通常是经过验证的高质量回答
        fast_result = self.check_fast_cache(query, thread_id)
        if fast_result:
            print(f"快速路径缓存命中: {query[:30]}...")
            
            # 将命中的内容同步到全局缓存，实现缓存预热
            self.global_cache_manager.set(query, fast_result)
            
            # 记录性能
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "fast"
            })
            
            return fast_result
        
        # 3. 尝试常规缓存路径，但优化验证
        # 跳过完整验证以提高性能
        cached_response = self.cache_manager.get(query, skip_validation=True, thread_id=thread_id)
        if cached_response:
            print(f"常规缓存命中，跳过验证: {query[:30]}...")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(query, cached_response)
            
            # 记录性能
            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "standard"
            })
            
            return cached_response
        
        # 没有命中任何缓存
        cache_time = time.time() - cache_check_start
        self._log_performance("cache_check", {
            "duration": cache_time,
            "type": "miss"
        })
        
        return None
    
    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> Dict:
        """执行查询并获取带执行轨迹的回答
        
        这是一个详细版的查询方法，返回查询的回答和完整的执行日志
        适用于调试和监控场景，可以看到每个节点的执行情况
        
        参数:
            query: 用户问题字符串
            thread_id: 会话ID，默认为"default"
            recursion_limit: 递归限制，防止无限递归
            
        返回:
            Dict: 包含回答和执行日志的字典
        """
        overall_start = time.time()
        self.execution_log = []  # 重置执行日志
        
        # 确保查询字符串是干净的，去除首尾空白
        safe_query = query.strip()
        
        # 首先尝试全局缓存（跨会话缓存）
        global_cache_start = time.time()
        global_result = self.global_cache_manager.get(safe_query)
        global_cache_time = time.time() - global_cache_start
        
        if global_result:
            print(f"全局缓存命中: {safe_query[:30]}... ({global_cache_time:.4f}s)")
            
            # 直接返回全局缓存结果
            return {
                "answer": global_result,
                "execution_log": [{"node": "global_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "全局缓存命中"}]
            }
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(safe_query, thread_id)
        fast_cache_time = time.time() - fast_cache_start
        
        if fast_result:
            print(f"快速路径缓存命中: {safe_query[:30]}... ({fast_cache_time:.4f}s)")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, fast_result)
            
            return {
                "answer": fast_result,
                "execution_log": [{"node": "fast_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "高质量缓存命中"}]
            }
        
        # 尝试常规缓存路径
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            print(f"完整问答缓存命中: {safe_query[:30]}... ({cache_time:.4f}s)")
            
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, cached_response)
            
            return {
                "answer": cached_response,
                "execution_log": [{"node": "cache_hit", "timestamp": time.time(), "input": safe_query, "output": "常规缓存命中"}]
            }
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        # 准备配置参数
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        # 准备输入消息
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 执行完整的处理流程 - 流式执行工作流
            for output in self.graph.stream(inputs, config=config):
                # 打印每个节点的输出（详细调试信息）
                pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
                pprint.pprint("---")
                pprint.pprint(output, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
                
            # 从记忆中获取最终的对话历史
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            # 提取最后一条消息作为回答
            answer = chat_history[-1].content
            
            # 缓存处理结果 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:  # 只缓存有效长度的回答
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            # 记录性能指标
            process_time = time.time() - process_start
            print(f"完整处理耗时: {process_time:.4f}s")
            
            overall_time = time.time() - overall_start
            self._log_performance("ask_with_trace", {
                "total_duration": overall_time,
                "cache_check": cache_time,
                "processing": process_time
            })
            
            # 返回回答和完整的执行日志
            return {
                "answer": answer,
                "execution_log": self.execution_log
            }
        except Exception as e:
            # 处理异常情况
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            # 返回错误信息和执行日志
            return {
                "answer": f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}",
                "execution_log": self.execution_log + [{"node": "error", "timestamp": time.time(), "input": query, "output": str(e)}]
            }
        
    def ask(self, query: str, thread_id: str = "default", recursion_limit: int = 5):
        """向Agent提问的主要接口方法
        
        这是代理系统的核心方法，提供用户查询的标准入口
        实现了多级缓存检查和执行流程管理
        
        参数:
            query: 用户问题字符串
            thread_id: 会话ID，默认为"default"
            recursion_limit: 递归限制，防止无限递归
            
        返回:
            str: 查询的回答文本
        """
        overall_start = time.time()
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        # 检查所有缓存，按优先级顺序
        cached_result = self._check_all_caches(safe_query, thread_id)
        if cached_result:
            # 如果缓存命中，直接返回缓存结果
            return cached_result
        
        # 未命中缓存，执行标准流程
        process_start = time.time()
        
        # 准备配置参数
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }
        
        # 准备输入消息
        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # 执行工作流 - 不需要打印中间输出
            for output in self.graph.stream(inputs, config=config):
                pass
                    
            # 从记忆中获取对话历史
            chat_history = self.memory.get(config)["channel_values"]["messages"]
            # 提取最后一条消息作为回答
            answer = chat_history[-1].content
            
            # 缓存处理结果 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:  # 只缓存有效长度的回答
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            # 记录性能指标
            process_time = time.time() - process_start
            overall_time = time.time() - overall_start
            
            self._log_performance("ask", {
                "total_duration": overall_time,
                "cache_check": 0,  # 由_check_all_caches记录
                "processing": process_time
            })
            
            return answer
        except Exception as e:
            # 处理异常
            error_time = time.time() - process_start
            print(f"处理查询时出错: {e} ({error_time:.4f}s)")
            return f"抱歉，处理您的问题时遇到了错误。请稍后再试或换一种提问方式。错误详情: {str(e)}"
    
    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: int = 5) -> AsyncGenerator[str, None]:
        """向Agent提问，返回流式响应
        
        提供异步流式回答接口，适用于需要实时展示回答进度的场景
        实现了流式缓存命中检查和生成
        
        参数:
            query: 用户问题
            thread_id: 会话ID
            recursion_limit: 递归限制
                
        返回:
            AsyncGenerator[str, None]: 流式响应生成器，产生文本块
        """
        overall_start = time.time()
        
        # 确保查询字符串是干净的
        safe_query = query.strip()
        
        # 首先尝试全局缓存（跨会话缓存）
        global_result = self.global_cache_manager.get(safe_query)
        if global_result:
            # 对于缓存响应，按自然语言单位分块返回
            # 使用正则表达式按句子边界分割文本
            import re
            chunks = re.split(r'([.!?。！？]\s*)', global_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 首先尝试快速路径 - 跳过验证的高质量缓存
        fast_result = self.check_fast_cache(safe_query, thread_id)
        if fast_result:
            # 对于缓存响应，按自然语言单位分块返回
            import re
            chunks = re.split(r'([.!?。！？]\s*)', fast_result)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
                
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, fast_result)
            return
        
        # 尝试常规缓存路径
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start
        
        if cached_response:
            # 同样按自然语言单位分块
            import re
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
                
            # 将命中的内容同步到全局缓存
            self.global_cache_manager.set(safe_query, cached_response)
            return
        
        # 未命中缓存，执行标准流程
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit,
                "stream_mode": True  # 指示流式输出模式
            }
        }
        
        inputs = {"messages": [HumanMessage(content=query)]}
        answer = ""
        
        try:
            # 执行流式处理
            async for chunk in self._stream_process(inputs, config):
                yield chunk  # 产生流式文本块
                answer += chunk  # 累积完整回答用于缓存
            
            # 缓存完整回答 - 同时更新会话缓存和全局缓存
            if answer and len(answer) > 10:  # 只缓存有效长度的回答
                # 更新会话缓存
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(safe_query, answer)
            
            # 记录性能指标
            process_time = time.time() - overall_start
            self._log_performance("ask_stream", {
                "total_duration": process_time,
                "processing": process_time
            })
            
        except Exception as e:
            # 处理异常
            error_time = time.time() - overall_start
            error_msg = f"处理查询时出错: {str(e)} ({error_time:.4f}s)"
            print(error_msg)
            yield error_msg
    
    def mark_answer_quality(self, query: str, is_positive: bool, thread_id: str = "default"):
        """标记回答质量，用于缓存质量控制
        
        此方法允许用户对回答进行反馈，系统根据反馈调整缓存策略
        正面反馈会增强缓存条目，负面反馈会降低其优先级或从缓存中移除
        
        参数:
            query: 查询字符串
            is_positive: 是否为正面反馈
            thread_id: 会话ID，默认为"default"
        """
        start_time = time.time()
        
        # 提取关键词用于缓存键构建
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }
        
        # 调用缓存管理器的质量标记方法
        marked = self.cache_manager.mark_quality(query.strip(), is_positive, **cache_params)
        
        # 记录性能指标
        mark_time = time.time() - start_time
        self._log_performance("mark_quality", {
            "duration": mark_time,
            "is_positive": is_positive
        })
    
    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        """清除特定查询的缓存（会话缓存和全局缓存）
        
        提供了完整的缓存清理功能，处理查询的各种变体形式
        支持清理会话缓存和全局缓存中的相关条目
        
        参数:
            query: 查询字符串
            thread_id: 会话ID，默认为"default"
            
        返回:
            bool: 是否成功删除缓存条目
        """
        # 清除会话缓存
        success = False
        
        try:
            # 尝试移除可能存在的前缀，获取干净的查询文本
            clean_query = query.strip()
            if ":" in clean_query:
                parts = clean_query.split(":", 1)
                if len(parts) > 1:
                    clean_query = parts[1].strip()
            
            # 清除原始查询的会话缓存
            session_cache_deleted = self.cache_manager.delete(query.strip(), thread_id=thread_id)
            success = session_cache_deleted
            
            # 清除没有前缀的查询缓存
            if clean_query != query.strip():
                self.cache_manager.delete(clean_query, thread_id=thread_id)
            
            # 清除带前缀的查询缓存变体
            prefixes = ["generate:", "deep:", "query:"]
            for prefix in prefixes:
                self.cache_manager.delete(f"{prefix}{clean_query}", thread_id=thread_id)
            
            # 清除全局缓存 - 使用所有可能的变体
            if hasattr(self, 'global_cache_manager'):
                # 删除原始查询
                global_cache_deleted = self.global_cache_manager.delete(query.strip())
                success = success or global_cache_deleted
                
                # 删除清理后的查询
                if clean_query != query.strip():
                    self.global_cache_manager.delete(clean_query)
                
                # 删除带前缀的查询变体
                for prefix in prefixes:
                    self.global_cache_manager.delete(f"{prefix}{clean_query}")
            
            # 强制刷新缓存写入，确保所有更改立即生效
            if hasattr(self.cache_manager.storage, '_flush_write_queue'):
                self.cache_manager.storage._flush_write_queue()
            
            if hasattr(self, 'global_cache_manager') and hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
                self.global_cache_manager.storage._flush_write_queue()
                
            # 记录日志
            print(f"已清除查询缓存: {query.strip()}")
            
            return success
        except Exception as e:
            # 处理异常情况
            print(f"清除缓存时出错: {e}")
            return False
    
    def _validate_answer(self, query: str, answer: str, thread_id: str = "default") -> bool:
        """验证答案质量
        
        提供内置的答案质量验证机制，确保缓存的回答符合质量标准
        检查内容长度、错误模式和关键词相关性
        
        参数:
            query: 查询字符串
            answer: 回答文本
            thread_id: 会话ID，默认为"default"
            
        返回:
            bool: 回答是否通过质量验证
        """
        # 定义验证函数，使用缓存管理器的验证方法
        def validator(query, answer):
            # 基本检查 - 长度验证
            if len(answer) < 20:
                return False
                
            # 检查是否包含错误消息
            error_patterns = [
                "抱歉，处理您的问题时遇到了错误",
                "技术原因:",
                "无法获取",
                "无法回答这个问题"
            ]
            
            for pattern in error_patterns:
                if pattern in answer:
                    return False
                    
            # 相关性检查 - 检查问题关键词是否在答案中出现
            keywords = self._extract_keywords(query)
            if keywords:
                low_level_keywords = keywords.get("low_level", [])
                if low_level_keywords:
                    # 至少有一个低级关键词应该在答案中出现
                    keyword_found = any(keyword.lower() in answer.lower() for keyword in low_level_keywords)
                    if not keyword_found:
                        return False
            
            # 通过所有检查
            return True
        
        # 调用缓存管理器进行验证
        return self.cache_manager.validate_answer(query, answer, validator, thread_id=thread_id)
    
    def close(self):
        """关闭资源
        
        确保所有延迟写入的缓存项都被保存到磁盘
        在代理实例不再使用时调用此方法以避免数据丢失
        """
        # 确保所有延迟写入的缓存项都被保存
        if hasattr(self.cache_manager.storage, '_flush_write_queue'):
            self.cache_manager.storage._flush_write_queue()
            
        # 同样确保全局缓存的写入被保存
        if hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
            self.global_cache_manager.storage._flush_write_queue()