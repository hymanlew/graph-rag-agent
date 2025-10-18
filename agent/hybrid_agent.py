from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import tools_condition
import asyncio
import re

from config.prompt import LC_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.hybrid_tool import HybridSearchTool

from agent.base import BaseAgent


class HybridAgent(BaseAgent):
    """混合检索Agent实现
    
    该类继承自BaseAgent，实现了基于混合检索策略的智能问答Agent。
    混合检索通过结合多种搜索方法（如关键词搜索与语义搜索），优化查询速度与相关性，
    提供更全面、准确的信息检索结果。
    """
    
    def __init__(self):
        """初始化混合检索Agent
        
        设置混合搜索工具、缓存目录，并调用父类初始化方法。
        使用上下文感知的缓存策略，提高重复查询的响应速度。
        """
        # 初始化混合搜索工具，整合多种检索方法
        self.search_tool = HybridSearchTool()
        
        # 设置缓存目录，存储会话和全局缓存
        self.cache_dir = "./cache/hybrid_agent"
        
        # 调用父类构造函数 - 使用默认的ContextAwareCacheKeyStrategy
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置混合Agent使用的工具列表
        
        配置Agent可以使用的搜索工具，包括本地搜索和全局搜索两种策略，
        以实现多层次、全方位的信息检索。
        
        返回:
            List: 包含可用工具的列表
        """
        return [
            self.search_tool.get_tool(),        # 获取本地搜索工具
            self.search_tool.get_global_tool(), # 获取全局搜索工具
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加工作流中从检索到生成的边
        
        配置工作流的执行路径，定义从检索节点到生成节点的直接跳转关系，
        简化混合检索Agent的工作流程。
        
        参数:
            workflow: LangGraph工作流对象，用于构建Agent执行路径
        """
        # 简单的从检索直接到生成的工作流路径
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词
        
        从用户查询中提取不同层级的关键词，包括低级细节实体和高级主题概念。
        利用缓存机制提高关键词提取的效率，减少重复计算。
        
        参数:
            query: 用户输入的查询字符串
            
        返回:
            Dict[str, List[str]]: 包含低级和高级关键词的字典
        """
        # 检查缓存，避免重复提取
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        try:
            # 使用增强型搜索工具的关键词提取功能
            keywords = self.search_tool.extract_keywords(query)
            
            # 确保返回有效的关键词格式
            # 统一转换为标准格式，包含低级细节关键词和高级概念关键词
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []  # 存储实体、具体细节等低级关键词
            if "high_level" not in keywords:
                keywords["high_level"] = []  # 存储主题、概念等高级关键词
            
            # 缓存结果，提高后续查询效率
            self.cache_manager.set(f"keywords:{query}", keywords)
            
            return keywords
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 出错时返回默认空关键词，确保程序不会崩溃
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑
        
        处理检索到的信息，生成符合要求的回答内容。
        实现了两级缓存策略（全局缓存和会话缓存），提高响应速度和资源利用效率。
        使用RAG链结合检索内容和大语言模型生成回答，并确保回答格式规范。
        
        参数:
            state: 当前工作流状态，包含消息历史
            
        返回:
            Dict: 包含生成回答消息的字典
        """
        messages = state["messages"]
        
        # 安全地获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        # 安全地获取文档内容
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 首先尝试全局缓存 - 优先使用全局缓存，提高整体系统效率
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 获取当前会话ID，用于上下文感知缓存
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存 - 提供个性化的上下文相关缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存，提高整体系统效率
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 构建提示模板，指导LLM如何生成高质量回答
        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            以下是检索到的相关信息，按重要性排序：
            
            {context}
            
            用户的问题是：
            {question}
            
            请以清晰、全面的方式回答问题，确保：
            1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
            2. 使用三级标题(###)组织内容，增强可读性
            3. 结尾处用"#### 引用数据"标记引用来源
            """),
        ])

        # 创建RAG链，连接提示模板、语言模型和输出解析器
        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            # 调用RAG链生成回答
            response = rag_chain.invoke({
                "context": docs,     # 检索到的文档内容
                "question": question,  # 用户问题
                "response_type": response_type  # 响应类型配置
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            if response and len(response) > 10:  # 确保回答不为空且有足够长度
                # 更新会话缓存，提供个性化回答
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 更新全局缓存，提高整体系统效率
                self.global_cache_manager.set(question, response)
            
            # 记录执行信息，用于监控和分析
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            # 错误处理，捕获并记录生成过程中可能出现的任何异常
            error_msg = f"生成回答时出错: {str(e)}"
            # 记录错误信息，便于后续分析和调试
            self._log_execution("generate_error", 
                            {"question": question, "docs_length": len(docs)}, 
                            error_msg)
            # 返回友好的错误消息给用户，避免暴露系统内部错误细节
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
    
    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本
        
        异步实现的回答生成节点，提供流式输出功能，增强用户体验。
        支持缓存检查、分块流式输出，使回答能够逐步呈现给用户，减少等待时间。
        
        参数:
            state: 当前工作流状态，包含消息历史
            
        生成:
            str: 回答内容的片段，逐步输出给用户
        """
        messages = state["messages"]
        
        # 安全地获取问题内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        # 安全地获取文档内容
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 获取当前会话ID，用于上下文感知缓存
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 检查缓存，快速返回已有结果
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            # 按句子分块输出，提供流式体验
            # 按句子分割文本，便于流式输出
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""  # 缓冲区，用于累积输出内容
            
            # 遍历所有文本块
            for i in range(0, len(chunks)):
                buffer += chunks[i]  # 将当前块添加到缓冲区
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                # i % 2 == 1 表示当前块是标点符号和空格，意味着一个完整句子结束
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer  # 返回当前缓冲区内容
                    buffer = ""   # 重置缓冲区
                    await asyncio.sleep(0.01)  # 短暂延迟，确保流畅的流式体验
            
            # 输出任何剩余内容，确保完整输出
            if buffer:
                yield buffer
            return

        # 构建提示模板，指导LLM如何生成高质量的混合检索回答
        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            以下是检索到的相关信息，按重要性排序：
            
            {context}
            
            用户的问题是：
            {question}
            
            请以清晰、全面的方式回答问题，确保：
            1. 回答结合了检索到的低级（实体细节）和高级（主题概念）信息
            2. 使用三级标题(###)组织内容，增强可读性
            3. 结尾处用"#### 引用数据"标记引用来源
            """),
        ])

        # 使用流式模型，创建RAG链，连接提示模板、语言模型和输出解析器
        # 虽然方法名中包含stream，但这里使用同步模型生成完整结果
        # 然后在生成后进行手动分块，模拟流式输出
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,     # 检索到的文档内容
            "question": question,  # 用户问题
            "response_type": response_type  # 响应类型配置
        })

        # 分块输出结果，实现流式响应效果
        # 按句子分割文本，使用正则表达式捕获标点符号和空格
        sentences = re.split(r'([.!?。！？]\s*)', response)
        buffer = ""  # 缓冲区，用于累积输出内容

        # 遍历所有句子块
        for i in range(len(sentences)):
            buffer += sentences[i]  # 将当前句子块添加到缓冲区
            
            # 当积累了一个完整句子或达到最小长度时输出
            # i % 2 == 1 表示当前块是标点符号和空格，意味着一个完整句子结束
            if i % 2 == 1 or len(buffer) >= 40:
                yield buffer  # 返回当前缓冲区内容
                buffer = ""   # 重置缓冲区
                await asyncio.sleep(0.01)  # 短暂延迟，确保流畅的流式体验

        # 确保所有内容都被输出，即使最后一个块不满足前面的条件
        if buffer:
            yield buffer

    async def _stream_process(self, inputs, config):
        """实现混合检索Agent的流式处理过程
        
        异步实现的完整工作流程，包括缓存检查、问题分析、信息检索和回答生成。
        为用户提供实时进度反馈和流式回答输出，显著改善用户体验。
        特别针对混合检索特性进行了优化，支持不同检索策略的灵活切换。
        
        参数:
            inputs: 包含用户输入消息的数据结构
            config: 包含会话配置的字典，如thread_id
            
        生成:
            str: 处理进度或回答内容的片段
        """
        # 实现与 GraphAgent 类似，但针对 HybridAgent 的特性进行了优化
        # 获取会话ID，用于上下文感知缓存
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        # 安全地获取查询内容，处理各种可能的输入格式
        query = ""
        if "messages" in inputs and inputs["messages"] and len(inputs["messages"]) > 0:
            last_message = inputs["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                query = last_message.content
        
        # 检查查询是否有效
        if not query:
            yield "无法获取查询内容，请重试。"  # 返回错误提示
            return
        
        # 缓存检查与处理同GraphAgent相同，但针对混合检索特点优化
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response:
            # 对于缓存的响应，按自然语言单位分块返回，提供流畅的流式体验
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(chunks)):
                buffer += chunks[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer  # 返回当前缓冲区内容
                    buffer = ""   # 重置缓冲区
                    await asyncio.sleep(0.01)  # 短暂延迟，确保流畅体验
            
            # 输出任何剩余内容，确保完整输出
            if buffer:
                yield buffer
            return
        
        # 工作流处理与GraphAgent相同，但添加进度提示，提升用户体验
        workflow_state = {"messages": [HumanMessage(content=query)]}
        
        # 输出处理开始的提示，告知用户Agent正在工作
        yield "**正在分析问题**...\n\n"
        
        # 执行agent节点，异步方式调用_agent_node方法
        # 分析问题并决定下一步操作（使用工具或直接回答）
        agent_output = await self._agent_node_async(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}
        
        # 检查是否需要使用工具（如搜索工具）
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # 告知用户正在检索相关信息，提供进度反馈
            yield "**正在检索相关信息**...\n\n"
            
            # 执行异步检索节点，获取混合搜索结果
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}
            
            # 告知用户正在生成回答，提供进度反馈
            yield "**正在生成回答**...\n\n"
            
            # 流式生成节点输出，将回答分块返回给用户
            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            # 不需要工具，直接返回Agent的响应
            final_msg = workflow_state["messages"][-1]
            content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            
            # 按自然语言单位分块
            chunks = re.split(r'([.!?。！？]\s*)', content)
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
    
    async def _retrieve_node_async(self, state):
        """异步检索节点实现
        
        处理工具调用，从消息中提取查询参数，并执行混合搜索。
        安全地处理各种可能的消息格式，确保工具调用的正确执行。
        实现了异常捕获和友好的错误处理，提高系统稳定性。
        
        参数:
            state: 当前工作流状态，包含消息历史和工具调用信息
            
        返回:
            Dict: 包含工具执行结果的消息字典
        """
        try:
            # 获取最后一条消息（通常包含工具调用信息）
            last_message = state["messages"][-1]
            
            # 安全获取工具调用信息
            tool_calls = []
            
            # 检查additional_kwargs中的tool_calls - 处理OpenAI兼容格式
            if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs.get('tool_calls', [])
            
            # 检查直接的tool_calls属性 - 处理LangChain格式
            if not tool_calls and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_calls = last_message.tool_calls
                
            # 如果没有找到工具调用，返回错误消息
            if not tool_calls:
                return {
                    "messages": [
                        AIMessage(content="无法获取查询信息，请重试。")
                    ]
                }
            
            # 获取第一个工具调用 - 通常只有一个搜索工具调用
            tool_call = tool_calls[0]
            
            # 初始化默认查询和工具信息
            query = ""
            tool_id = "tool_call_0"
            tool_name = "search_tool"
            
            # 根据工具调用格式提取参数 - 支持多种格式以提高兼容性
            if isinstance(tool_call, dict):
                # 提取工具调用ID
                tool_id = tool_call.get("id", tool_id)
                
                # 提取函数名称和参数 - 处理OpenAI API格式
                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", tool_name)
                    
                    # 提取函数参数
                    args = tool_call["function"].get("arguments", {})
                    if isinstance(args, str):
                        # 尝试解析JSON格式的参数字符串
                        try:
                            import json
                            args_dict = json.loads(args)
                            query = args_dict.get("query", "")
                        except:
                            # 如果解析失败，使用整个字符串作为查询
                            query = args
                    elif isinstance(args, dict):
                        # 直接从参数字典中提取查询
                        query = args.get("query", "")
                # 直接在root级别检查 - 处理其他可能的格式
                elif "name" in tool_call:
                    tool_name = tool_call.get("name", tool_name)
                
                # 检查args字段 - 另一种可能的参数位置
                if not query and "args" in tool_call:
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        query = args.get("query", "")
                    elif isinstance(args, str):
                        query = args
            
            # 最后手段：如果仍然没有找到查询，尝试使用消息内容
            if not query and hasattr(last_message, 'content'):
                query = last_message.content
                
            # 执行混合搜索，获取检索结果
            tool_result = self.search_tool.search(query)
            
            # 返回正确格式的工具消息，将搜索结果封装成工具消息对象
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,      # 搜索结果内容
                        tool_call_id=tool_id,     # 工具调用ID，用于关联调用和结果
                        name=tool_name            # 工具名称，标识使用的工具类型
                    )
                ]
            }
        except Exception as e:
            # 全面的错误处理，捕获检索过程中可能出现的任何异常
            error_msg = f"处理工具调用时出错: {str(e)}"
            print(error_msg)  # 记录错误到控制台，便于调试
            # 返回友好的错误消息，避免系统崩溃
            return {
                "messages": [
                    AIMessage(content=error_msg)  # 将错误信息封装为AI消息返回
                ]
            }
    
    async def _agent_node_async(self, state):
        """Agent节点的异步版本
        
        提供异步接口，通过线程池执行同步的_agent_node方法，避免阻塞事件循环。
        确保在异步环境中能够正确执行同步代码，提高系统并发性能。
        
        参数:
            state: 当前工作流状态，包含消息历史
            
        返回:
            Dict: Agent节点的执行结果
        """
        # 定义内部函数包装同步调用
        def sync_agent():
            return self._agent_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        # 这种方式允许在异步工作流中无缝使用同步代码
        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)
    
    def _get_tool_call_info(self, message):
        """从消息中提取工具调用信息
        
        支持多种格式的工具调用信息提取，包括OpenAI API格式和LangChain格式。
        提供统一的工具调用信息接口，简化工具调用处理逻辑。
        
        参数:
            message: 包含工具调用的消息对象
            
        返回:
            Dict: 标准化的工具调用信息，包括id、name和args
        """
        # 检查additional_kwargs中的tool_calls - OpenAI API格式
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                # 提取OpenAI格式的工具调用信息
                return {
                    "id": tool_call.get("id", "tool_call_0"),  # 工具调用ID
                    "name": tool_call.get("function", {}).get("name", "search_tool"),  # 函数名称
                    "args": tool_call.get("function", {}).get("arguments", {})  # 函数参数
                }
        
        # 检查直接的tool_calls属性 - LangChain格式
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            # 提取LangChain格式的工具调用信息
            return {
                "id": tool_call.get("id", "tool_call_0"),  # 工具调用ID
                "name": tool_call.get("name", "search_tool"),  # 工具名称
                "args": tool_call.get("args", {})  # 工具参数
            }
        
        # 默认返回 - 当无法提取工具调用信息时
        return {
            "id": "tool_call_0",  # 默认工具调用ID
            "name": "search_tool",  # 默认工具名称
            "args": {"query": ""}  # 默认空查询参数
        }
    
    def close(self):
        """关闭资源并释放系统资源
        
        清理Agent使用的所有资源，包括父类资源和搜索工具资源。
        确保在Agent不再使用时正确释放资源，避免资源泄露。
        """
        # 先关闭父类资源，释放缓存管理器等基础资源
        super().close()
        
        # 再关闭搜索工具资源，释放与搜索引擎相关的连接和资源
        if self.search_tool:
            self.search_tool.close()