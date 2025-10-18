from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
import asyncio

import json
import re

from config.prompt import LC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT
from config.settings import response_type
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool

from agent.base import BaseAgent


class GraphAgent(BaseAgent):
    """使用图结构的Agent实现
    
    该Agent扩展了基础Agent架构，实现了基于知识图谱的检索和问答功能。
    主要特点包括本地和全局搜索能力、文档相关性评估、以及两级缓存系统。
    使用LangGraph构建工作流，支持复杂的检索后处理逻辑。
    
    继承自BaseAgent，实现了图RAG的核心功能。
    """
    
    def __init__(self):
        """
        初始化GraphAgent
        
        设置必要的搜索工具和缓存目录，初始化Agent的核心组件。
        加载本地搜索工具用于精确查询和全局搜索工具用于广泛检索。
        """
        # 初始化本地和全局搜索工具
        self.local_tool = LocalSearchTool()  # 用于精确、局部搜索的工具
        self.global_tool = GlobalSearchTool()  # 用于广泛、全局搜索的工具
        
        # 设置缓存目录，存储Agent的缓存数据
        self.cache_dir = "./cache/graph_agent"
        
        # 调用父类构造函数，初始化基础组件
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具
        
        配置Agent可用的搜索工具列表，包括本地精确搜索和全局广泛搜索。
        
        返回:
            List: 配置好的工具对象列表
        """
        return [
            self.local_tool.get_tool(),  # 获取本地搜索工具
            self.global_tool.search,     # 获取全局搜索工具
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边
        
        在LangGraph工作流中设置检索后处理的复杂逻辑，包括:
        1. 添加reduce节点处理复杂文档
        2. 基于文档评分实现条件路由
        3. 配置不同处理路径的流转逻辑
        
        参数:
            workflow: LangGraph工作流对象，用于添加节点和边
        """
        # 添加 reduce 节点，用于处理和聚合大量检索结果
        workflow.add_node("reduce", self._reduce_node)
        
        # 添加条件边，根据文档评分决定使用哪种处理路径
        workflow.add_conditional_edges(
            "retrieve",           # 源节点：检索结果
            self._grade_documents, # 条件函数：评估文档相关性
            {
                "generate": "generate",  # 相关性高：直接生成回答
                "reduce": "reduce"      # 结果复杂：先聚合处理
            }
        )
        
        # 添加从 reduce 到结束的边，处理完直接结束流程
        workflow.add_edge("reduce", END)

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """提取查询关键词
        
        从用户查询中提取两类关键词：低级关键词（具体实体、名称）和高级关键词（主题、概念）。
        使用LLM进行智能提取，并实现缓存机制优化性能。
        
        参数:
            query: 用户输入的查询字符串
            
        返回:
            Dict[str, List[str]]: 包含低级和高级关键词列表的字典
        """
        # 检查查询是否为空或格式不正确
        if not query or not isinstance(query, str):
            return {"low_level": [], "high_level": []}
        
        # 检查缓存，避免重复计算
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords
            
        # 使用LLM提取关键词
        try:
            # 使用简单的prompt模板，避免复杂格式
            prompt = f"""提取以下查询的关键词:
            查询: {query}
            
            请提取两类关键词:
            1. 低级关键词: 具体实体、名称、术语
            2. 高级关键词: 主题、概念、领域
            
            以JSON格式返回。
            """
            
            result = self.llm.invoke(prompt)
            
            # 解析LLM返回的内容
            content = result.content if hasattr(result, 'content') else result
            
            # 尝试提取JSON部分
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    keywords = json.loads(json_str)
                    # 确保结果有正确的格式
                    if not isinstance(keywords, dict):
                        keywords = {}
                    if "low_level" not in keywords:
                        keywords["low_level"] = []
                    if "high_level" not in keywords:
                        keywords["high_level"] = []
                        
                    # 缓存结果，提高后续查询效率
                    self.cache_manager.set(f"keywords:{query}", keywords)
                    return keywords
                except:
                    pass
        except Exception as e:
            print(f"关键词提取失败: {e}")
            
        # 如果提取失败，返回默认空值
        default_keywords = {"low_level": [], "high_level": []}
        return default_keywords

    def _grade_documents(self, state) -> str:
        """评估文档相关性
        
        分析检索到的文档与用户问题的相关性，并决定后续处理路径：
        - 如果是全局检索结果，直接进入reduce路径
        - 对文档内容进行质量评估和关键词匹配分析
        - 根据分析结果返回处理路径："generate"或"reduce"
        
        参数:
            state: 当前工作流状态，包含消息历史和检索结果
            
        返回:
            str: 处理路径标识符，"generate"或"reduce"
        """
        messages = state["messages"]
        retrieve_message = messages[-2]
        
        # 检查是否为全局检索工具调用 - 这类结果通常更复杂，需要特殊处理
        tool_calls = retrieve_message.additional_kwargs.get("tool_calls", [])
        if tool_calls and tool_calls[0].get("function", {}).get("name") == "global_retriever":
            self._log_execution("grade_documents", messages, "reduce")
            return "reduce"

        # 获取问题和文档内容
        try:
            question = messages[-3].content
            docs = messages[-1].content
        except Exception as e:
            # 如果出错，默认为 generate 模式
            print(f"文档评分出错: {e}")
            return "generate"
        
        # 检查文档内容是否足够 - 内容太少时尝试更精确的搜索
        if not docs or len(docs) < 100:
            print("文档内容不足，尝试使用本地搜索")
            # 尝试使用local_tool进行更精确搜索
            try:
                local_result = self.local_tool.search(question)
                if local_result and len(local_result) > 100:
                    # 替换原来的结果
                    messages[-1].content = local_result
            except Exception as e:
                print(f"本地搜索失败: {e}")
        
        # 从问题中提取关键词 - 优先使用已提取的关键词
        keywords = []
        if hasattr(messages[-3], 'additional_kwargs') and messages[-3].additional_kwargs:
            kw_data = messages[-3].additional_kwargs.get("keywords", {})
            if isinstance(kw_data, dict):
                keywords = kw_data.get("low_level", []) + kw_data.get("high_level", [])
        
        # 备用关键词提取逻辑
        if not keywords:
            # 如果没有提取到关键词，使用简单的关键词提取
            keywords = [word for word in question.lower().split() if len(word) > 2]
        
        # 计算关键词匹配率，评估文档相关性
        docs_text = docs.lower() if docs else ""
        matches = sum(1 for keyword in keywords if keyword.lower() in docs_text)
        match_rate = matches / len(keywords) if keywords else 0
        
        # 记录匹配情况，用于调试和性能分析
        self._log_execution("grade_documents", {
            "question": question,
            "keywords": keywords,
            "match_rate": match_rate,
            "docs_length": len(docs_text)
        }, f"匹配率: {match_rate}")
        
        # 总是返回 "generate" 而不是 "rewrite"，避免路由错误
        return "generate"

    def _generate_node(self, state):
        """生成回答节点逻辑
        
        基于检索到的文档内容生成最终回答，实现了两级缓存机制：
        1. 首先检查全局缓存
        2. 然后检查会话缓存
        3. 最后使用RAG链生成新回答并更新两级缓存
        
        参数:
            state: 当前工作流状态，包含消息历史和检索到的文档
            
        返回:
            Dict: 包含生成回答的消息字典
        """
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 首先尝试全局缓存 - 提高相同问题的整体响应速度
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 然后检查会话缓存 - 处理会话特定的上下文
        thread_id = state.get("configurable", {}).get("thread_id", "default")
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存，优化全局性能
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 构建RAG提示模板，包含系统指令和用户问题格式
        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {question}
            
            请严格按照以下格式输出回答：
            1. 使用三级标题(###)标记主题
            2. 主要内容用清晰的段落展示
            3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
            """),
        ])

        # 创建RAG链：提示模板 -> 大语言模型 -> 输出解析器
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,    # 检索到的文档内容
            "question": question, # 用户问题
            "response_type": response_type  # 响应类型配置
        })
        
        # 缓存结果 - 同时更新会话缓存和全局缓存
        if response and len(response) > 10:  # 确保响应有意义
            # 更新会话缓存
            self.cache_manager.set(question, response, thread_id=thread_id)
            # 更新全局缓存
            self.global_cache_manager.set(question, response)
        
        # 记录执行情况，用于监控和调试
        self._log_execution("generate", 
                        {"question": question, "docs_length": len(docs)}, 
                        response)
        
        return {"messages": [AIMessage(content=response)]}

    def _reduce_node(self, state):
        """处理全局搜索的Reduce节点逻辑
        
        专门处理复杂的全局搜索结果，对大量或复杂的文档进行聚合和简化。
        使用特定的reduce提示模板，优化对复杂信息的处理。
        
        参数:
            state: 当前工作流状态，包含消息历史和全局检索结果
            
        返回:
            Dict: 包含聚合处理后回答的消息字典
        """
        messages = state["messages"]
        question = messages[-3].content
        docs = messages[-1].content

        # 检查缓存，避免重复处理相同的复杂查询
        cached_result = self.cache_manager.get(f"reduce:{question}")
        if cached_result:
            self._log_execution("reduce", 
                               {"question": question, "docs_length": len(docs)}, 
                               cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 构建reduce专用提示模板，针对复杂文档聚合设计
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", """
                ---分析报告--- 
                {report_data}

                用户的问题是：
                {question}
                """),
        ])
        
        # 创建reduce链：提示模板 -> 大语言模型 -> 输出解析器
        reduce_chain = reduce_prompt | self.llm | StrOutputParser()
        response = reduce_chain.invoke({
            "report_data": docs,        # 全局检索到的文档
            "question": question,       # 用户问题
            "response_type": response_type,  # 响应类型配置
        })
        
        # 缓存结果，提高后续相同复杂查询的处理效率
        self.cache_manager.set(f"reduce:{question}", response)
        
        # 记录执行情况，用于监控和调试
        self._log_execution("reduce", 
                           {"question": question, "docs_length": len(docs)}, 
                           response)
        
        return {"messages": [AIMessage(content=response)]}
    
    async def _generate_node_stream(self, state):
        """生成回答节点逻辑的流式版本
        
        异步实现的流式响应生成器，用于提供实时反馈。
        支持从缓存快速流式输出，或实时生成并流式返回回答。
        
        参数:
            state: 当前工作流状态，包含消息历史和检索到的文档
            
        生成:
            str: 回答的片段，实现流式响应
        """
        
        messages = state["messages"]
        
        # 安全获取问题和文档内容
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception as e:
            yield f"**获取问题或文档时出错**: {str(e)}"
            return

        # 获取线程ID，用于会话特定的缓存处理
        thread_id = state.get("configurable", {}).get("thread_id", "default")
        
        # 检查缓存，流式输出缓存结果以提高响应速度
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result:
            # 分块输出缓存内容，模拟流式响应
            sentences = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer  # 返回当前缓冲区内容
                    buffer = ""   # 重置缓冲区
                    await asyncio.sleep(0.01)  # 短暂延迟，确保流畅的流式体验
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return

        # 构建提示模板，用于生成回答
        prompt = ChatPromptTemplate.from_messages([
        ("system", LC_SYSTEM_PROMPT),
        ("human", """
            ---分析报告--- 
            请注意，下面提供的分析报告按**重要性降序排列**。
            
            {context}
            
            用户的问题是：
            {question}
            
            请严格按照以下格式输出回答：
            1. 使用三级标题(###)标记主题
            2. 主要内容用清晰的段落展示
            3. 最后必须用"#### 引用数据"标记引用部分，列出用到的数据来源
            """),
        ])

        # 使用流式模型，创建RAG链，使用同步模型生成完整结果（然后分块流式输出）
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,     # 检索到的文档内容
            "question": question,  # 用户问题
            "response_type": response_type  # 响应类型配置
        })

        # 分块输出结果，实现流式响应
        sentences = re.split(r'([.!?。！？]\s*)', response)
        buffer = ""

        for i in range(len(sentences)):
            buffer += sentences[i]
            # 当积累了一个完整句子或达到最小长度时输出
            if i % 2 == 1 or len(buffer) >= 40:
                yield buffer  # 返回当前缓冲区内容
                buffer = ""   # 重置缓冲区
                await asyncio.sleep(0.01)  # 短暂延迟，确保流畅的流式体验

        # 确保所有内容都被输出
        if buffer:
            yield buffer
    
    async def _stream_process(self, inputs, config):
        """实现流式处理过程
        
        异步实现的完整工作流程，包括缓存检查、问题分析、信息检索和回答生成。
        提供实时进度反馈和流式回答输出，优化用户体验。
        
        参数:
            inputs: 包含用户输入消息的数据结构
            config: 包含会话配置的字典，如thread_id
            
        生成:
            str: 处理进度或回答内容的片段
        """
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 首先检查缓存，快速返回已有结果
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response:
            # 分块返回缓存结果，提供流式体验
            sentences = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)  # 短暂延迟，确保流畅体验
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
            return
        
        # 处理新查询的完整工作流
        workflow_state = {"messages": [HumanMessage(content=query)]}
        
        # 执行agent节点 - 分析问题并决定后续操作，提供状态更新
        yield "**正在分析问题**...\n\n"  # 提供用户反馈
        agent_output = self._agent_node(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}
        
        # 检查是否需要使用工具（如搜索工具）
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # 执行检索节点，获取相关信息
            yield "**正在检索相关信息**...\n\n"  # 提供用户反馈
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}
            
            # 确保检索到内容足够丰富
            last_message = workflow_state["messages"][-1]
            content = last_message.content if hasattr(last_message, 'content') else ""
            
            # 如果检索结果不足，尝试使用本地搜索进行补充
            if not content or len(content) < 100:
                try:
                    yield "**检索内容不足，正在尝试更深入的搜索**...\n\n"  # 用户反馈
                    local_result = self.local_tool.search(query)
                    if local_result and len(local_result) > 100:
                        # 使用本地搜索结果替换
                        workflow_state["messages"][-1] = ToolMessage(
                            content=local_result,
                            tool_call_id="local_search",
                            name="local_search_tool"
                        )
                        yield "**找到更多相关信息，继续生成回答**...\n\n"  # 用户反馈
                except Exception as e:
                    yield f"**尝试深入搜索时出错**: {str(e)}"
            
            # 流式生成最终回答
            yield "**正在生成回答**...\n\n"  # 用户反馈
            async for token in self._generate_node_stream(workflow_state):
                yield token  # 流式输出回答内容
        else:
            # 不需要工具，直接返回Agent的响应（简单问题的快速回答路径）
            final_msg = workflow_state["messages"][-1]
            content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            
            # 分块返回，提供流式体验
            sentences = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""
            
            for i in range(0, len(sentences)):
                buffer += sentences[i]
                
                # 当缓冲区包含完整句子或达到合理大小时输出
                if (i % 2 == 1) or len(buffer) >= 40:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)  # 短暂延迟，确保流畅体验
            
            # 输出任何剩余内容
            if buffer:
                yield buffer
    
    # 异步检索节点辅助方法
    async def _retrieve_node_async(self, state):
        """检索节点的异步版本，用于流式处理
        
        异步实现的检索节点，负责从消息中提取工具调用信息，执行搜索工具，
        并返回检索结果。支持不同格式的工具调用信息提取。
        
        参数:
            state: 当前工作流状态，包含消息历史
            
        返回:
            Dict: 包含工具执行结果的消息字典
        """
        try:
            # 获取工具调用信息
            last_message = state["messages"][-1]
            
            # 安全获取工具调用信息 - 兼容不同的消息格式
            tool_calls = []
            
            # 检查additional_kwargs中的tool_calls（常见格式）
            if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs.get('tool_calls', [])
            
            # 检查直接的tool_calls属性（备选格式）
            if not tool_calls and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_calls = last_message.tool_calls
                
            # 如果没有找到工具调用，返回错误消息
            if not tool_calls:
                print("无法获取工具调用信息")
                return {
                    "messages": [
                        AIMessage(content="无法获取查询信息，请重试。")
                    ]
                }
            
            # 获取第一个工具调用（通常只有一个）
            tool_call = tool_calls[0]
            
            # 初始化查询参数和工具信息
            query = ""
            tool_id = "tool_call_0"  # 默认工具ID
            tool_name = "search_tool"  # 默认工具名称
            
            # 根据工具调用格式提取参数 - 支持多种格式以提高兼容性
            if isinstance(tool_call, dict):
                # 提取ID
                tool_id = tool_call.get("id", tool_id)
                
                # 提取函数名称
                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", tool_name)
                    
                    # 提取参数 - 处理不同格式的参数
                    args = tool_call["function"].get("arguments", {})
                    if isinstance(args, str):
                        # 尝试解析JSON格式的参数字符串
                        try:
                            import json
                            args_dict = json.loads(args)
                            query = args_dict.get("query", "")
                        except:
                            query = args  # 如果解析失败，使用整个字符串作为查询
                    elif isinstance(args, dict):
                        query = args.get("query", "")
                # 处理直接在root级别定义的工具名称
                elif "name" in tool_call:
                    tool_name = tool_call.get("name", tool_name)
                
                # 检查args字段（备选参数位置）
                if not query and "args" in tool_call:
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        query = args.get("query", "")
                    elif isinstance(args, str):
                        query = args
            
            # 如果仍然没有查询，尝试使用最简单的提取方法（直接使用消息内容）
            if not query and hasattr(last_message, 'content'):
                query = last_message.content
                
            # 执行搜索 - 根据工具类型选择不同的搜索策略
            if tool_name == "global_retriever":
                # 使用全局搜索 - 范围更广但可能返回更多结果
                tool_result = self.global_tool.search(query)
            else:
                # 使用本地搜索 - 更精确但范围较小
                tool_result = self.local_tool.search(query)
            
            # 检查搜索结果质量，确保内容足够丰富
            if not tool_result or (isinstance(tool_result, str) and len(tool_result.strip()) < 50):
                print("搜索结果内容不足，使用备用方法")
                # 尝试使用另一种搜索方法进行补充
                backup_result = self.local_tool.search(query)
                if backup_result and len(backup_result.strip()) > 50:
                    tool_result = backup_result
            
            # 返回正确格式的工具消息，包含搜索结果
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,  # 搜索工具执行结果
                        tool_call_id=tool_id,  # 工具调用标识符
                        name=tool_name  # 工具名称
                    )
                ]
            }
        except Exception as e:
            # 全局错误处理，确保工作流不会中断
            import traceback
            error_msg = f"处理工具调用时出错: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())  # 打印详细错误堆栈，便于调试
            # 返回用户友好的错误消息
            return {
                "messages": [
                    AIMessage(content=f"搜索过程中出现错误: {str(e)}")
                ]
            }
    
    async def _agent_node_async(self, state):
        """Agent 节点的异步版本
        
        将同步的Agent节点逻辑封装为异步函数，使其可以在异步工作流中使用。
        通过线程池执行同步代码，避免阻塞事件循环，保持异步性能。
        
        参数:
            state: 当前工作流状态，包含消息历史
            
        返回:
            Dict: 包含Agent处理结果的消息字典
        """
        # 定义内部同步函数，调用原始的_agent_node方法
        def sync_agent():
            return self._agent_node(state)
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        # 这种方式允许同步代码在异步上下文中执行，保持高性能
        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)
        