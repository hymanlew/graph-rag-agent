from typing import List, Dict
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import asyncio

from config.prompt import NAIVE_PROMPT
from config.settings import response_type
from search.tool.naive_search_tool import NaiveSearchTool
from agent.base import BaseAgent


class NaiveRagAgent(BaseAgent):
    """使用简单向量检索的Naive RAG Agent实现"""
    
    def __init__(self):
        # 初始化Naive搜索工具
        self.search_tool = NaiveSearchTool()
        
        # 设置缓存目录
        self.cache_dir = "./cache/naive_agent"
        
        # 调用父类构造函数
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """设置工具
        
        初始化Agent使用的工具列表，这里仅包含NaiveSearchTool。
        
        返回:
            List: 包含Agent可使用工具的列表
        """
        return [
            self.search_tool.get_tool(),  # 添加简单向量搜索工具
        ]
    
    def _add_retrieval_edges(self, workflow):
        """添加从检索到生成的边
        
        在LangGraph工作流中建立检索节点与生成节点之间的连接。
        Naive RAG采用简单的线性流程：检索结果直接用于生成回答。
        
        参数:
            workflow: LangGraph工作流实例
        """
        # 简单的从检索直接到生成，无需复杂路由
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        提取查询关键词 - 简化版本，不做实际的关键词提取
        
        Naive RAG不需要复杂的关键词提取，因为它使用向量相似度搜索
        而非基于关键词的搜索。保留此方法是为了与基类接口保持一致。
        
        参数:
            query: 查询字符串
            
        返回:
            Dict[str, List[str]]: 关键词字典，包含低级和高级关键词（空列表）
        """
        # Naive实现不需要关键词提取，返回空列表
        return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """生成回答节点逻辑
        
        核心生成方法，处理检索结果并生成最终回答。
        包含两级缓存机制（全局缓存和会话缓存）以提高性能。
        
        参数:
            state: 当前工作流状态，包含消息历史、检索结果等
            
        返回:
            Dict: 包含生成的AI消息的字典
        """
        messages = state["messages"]  # 从状态中获取消息历史
        
        # 安全地获取问题和检索结果
        try:
            question = messages[-3].content if len(messages) >= 3 else "未找到问题"
        except Exception:
            question = "无法获取问题"
            
        try:
            docs = messages[-1].content if messages[-1] else "未找到相关信息"
        except Exception:
            docs = "无法获取检索结果"

        # 首先尝试全局缓存
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "全局缓存命中")
            return {"messages": [AIMessage(content=global_result)]}

        # 获取当前会话ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")
            
        # 然后检查会话缓存
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            "会话缓存命中")
            # 将命中内容同步到全局缓存
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
        ("system", NAIVE_PROMPT),
        ("human", """
            ---检索结果--- 
            {context}
            
            问题：
            {question}
            """),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()  # 创建RAG链
        try:
            # 调用LLM生成回答
            response = rag_chain.invoke({
                "context": docs,  # 检索到的上下文文档
                "question": question,  # 用户问题
                "response_type": response_type  # 响应类型配置
            })
            
            # 缓存结果 - 同时更新会话缓存和全局缓存
            if response and len(response) > 10:  # 确保有实际生成内容
                # 更新会话缓存
                self.cache_manager.set(question, response, thread_id=thread_id)
                # 更新全局缓存
                self.global_cache_manager.set(question, response)
            
            self._log_execution("generate", 
                            {"question": question, "docs_length": len(docs)}, 
                            response)
            
            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            # 错误处理
            error_msg = f"生成回答时出错: {str(e)}"
            self._log_execution("generate_error", 
                            {"question": question, "docs_length": len(docs)}, 
                            error_msg)
            return {"messages": [AIMessage(content=f"抱歉，我无法回答这个问题。技术原因: {str(e)}")]}
    
    async def _stream_process(self, inputs, config):
        """实现流式处理过程
        
        提供异步流式输出接口，用于将检索和生成结果逐步返回给用户。
        增强用户体验，使用户能够看到实时处理进度。
        
        参数:
            inputs: 包含查询消息的输入字典
            config: 配置信息，包含会话ID等
            
        生成:
            str: 逐步生成的处理消息和结果文本
        """
        # 获取会话信息
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = inputs["messages"][-1].content
        
        # 开始处理提示
        yield "**开始处理查询**...\n\n"
        
        try:
            # 执行Naive搜索
            search_result = self.search_tool.search(query)
            
            # 分块返回结果
            if search_result:
                yield "**已找到相关信息，正在生成回答**...\n\n"
                
                # 分块返回 - 将文本按句子分割
                sentences = re.split(r'([.!?。！？]\s*)', search_result)
                buffer = ""
                
                for i in range(0, len(sentences)):
                    buffer += sentences[i]
                    
                    # 当缓冲区包含完整句子或达到合理大小时输出
                    if (i % 2 == 1) or len(buffer) >= 40:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)  # 短暂暂停以模拟流式效果
                
                # 输出任何剩余内容
                if buffer:
                    yield buffer
            else:
                # 当没有找到相关信息时的提示
                yield "未找到与您问题相关的信息。请尝试更换关键词或提供更多细节。"
        
        except Exception as e:
            # 处理流式处理过程中的错误
            yield f"**处理查询时出错**: {str(e)}"
    
    def close(self):
        """关闭资源
        
        释放Agent使用的所有资源，包括父类资源和搜索工具资源。
        确保在Agent不再使用时正确清理，避免资源泄露。
        """
        # 先关闭父类资源，如缓存管理器
        super().close()
        
        # 再关闭搜索工具资源
        if self.search_tool:
            self.search_tool.close()