from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import time
import json
import re

# 导入模型和工具
from model.get_models import get_llm_model, get_stream_llm_model, get_embeddings_model
from search.tool.deeper_research_tool import DeeperResearchTool
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from search.tool.reasoning.chain_of_exploration import ChainOfExplorationSearcher
from search.tool.reasoning.thinking import ThinkingEngine


class GraphRAGAgentCoordinator:
    """
    多Agent协作系统协调器
    
    这是一个复杂的协调系统，负责管理和组织多个专用Agent协同工作，以解决用户的复杂查询。
    实现了Fusion GraphRAG的多Agent协同架构，通过智能任务分解、分布式检索和结果合成，
    提供比单一Agent更全面、深入和准确的查询回答能力。
    
    主要特点：
    - 基于查询复杂度的动态任务规划
    - 多策略检索（本地、全局、深度探索、链式探索）
    - 智能思考引擎支持复杂问题分析
    - 结果合成与多源信息整合
    - 完整的执行轨迹记录和性能监控
    - 同步和异步流式响应支持
    
    整体架构遵循以下流程：
    1. 接收用户查询
    2. 分析查询并生成检索计划
    3. 根据计划协调多个Agent执行检索任务
    4. 整合检索结果
    5. 合成最终答案
    6. 返回结果（同步或流式）
    """
    
    def __init__(self, llm=None):
        """
        初始化Agent协调器
        
        创建并配置多Agent协调系统，初始化各个组件并设置性能监控。
        
        参数:
            llm: 可选的语言模型实例，如果不提供则使用默认模型
                提供自定义模型可用于测试、性能调优或使用特定能力的模型
        
        注意:
            初始化过程中会创建各种专用Agent组件，每个组件负责特定的检索和处理任务。
            所有组件共享相同的基础模型配置，但各司其职以实现复杂问题的高效求解。
        """
        # 初始化语言模型 - 分别用于标准任务和流式响应
        self.llm = llm or get_llm_model()  # 主要语言模型，用于复杂推理和生成
        self.stream_llm = get_stream_llm_model()  # 流式语言模型，用于实时响应
        self.embeddings = get_embeddings_model()  # 嵌入模型，用于向量表示和相似度计算
        
        # 创建各种专用Agent/工具 - 每个组件负责不同的检索和处理任务
        self.retrieval_planner = self._create_retrieval_planner()  # 检索计划生成器 - 分析查询并规划检索策略
        self.local_searcher = self._create_local_searcher()        # 本地搜索工具 - 查询特定上下文的相关信息
        self.global_searcher = self._create_global_searcher()      # 全局搜索工具 - 获取更广泛的知识范围
        self.explorer = self._create_explorer()                    # 深度探索工具 - 针对复杂主题进行深入研究
        self.chain_explorer = self._create_chain_explorer()        # 链式探索工具 - 基于知识图谱进行关系路径探索
        self.synthesizer = self._create_synthesizer()              # 结果合成器 - 整合所有检索结果生成最终答案
        self.thinking_engine = self._create_thinking_engine()      # 思考引擎 - 为复杂问题提供逻辑推理支持
        
        # 执行记录和性能指标 - 用于监控和评估系统表现
        self.execution_trace = []           # 执行轨迹记录 - 记录每个步骤的详细信息
        self.performance_metrics = {}       # 性能指标字典 - 存储时间、资源使用等数据
    
    def _create_retrieval_planner(self):
        """创建检索计划生成器Agent
        
        返回:
            RetrievalPlannerAgent: 负责分析查询并生成检索计划的Agent实例
        """
        class RetrievalPlannerAgent:
            def __init__(self, llm):
                self.llm = llm
                self.name = "retrieval_planner"
                self.description = "负责分析查询并生成最佳检索计划的Agent"
            
            def plan(self, query: str) -> Dict[str, Any]:
                """分析查询并生成检索计划
                
                参数:
                    query: 用户查询字符串
                    
                返回:
                    Dict: 包含检索计划的字典，定义了任务类型、优先级和执行顺序
                """
                prompt = f"""
                分析以下查询，创建一个全面的检索计划以获取所需信息。
                
                查询: "{query}"
                
                请考虑:
                1. 查询的复杂度和所需的检索深度
                2. 可能涉及的知识领域和关键实体
                3. 是否需要全局概览或具体细节
                4. 需要进行的探索步骤
                5. 是否需要追踪实体间的关系路径
                6. 查询是否涉及时间信息
                
                以JSON格式返回检索计划，包括:
                - complexity_assessment: 查询复杂度(0-1)
                - knowledge_areas: 涉及的知识领域
                - key_entities: 关键实体
                - requires_global_view: 是否需要全局概览
                - requires_path_tracking: 是否需要实体关系路径追踪
                - has_temporal_aspects: 是否包含时间相关内容
                - tasks: 检索任务列表，每个任务包含:
                  * type: 任务类型(local_search/global_search/exploration/chain_exploration)
                  * query: 具体的查询内容
                  * priority: 优先级(1-5)
                  * entities: 相关实体(用于chain_exploration类型)
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # 提取JSON
                    import re
                    import json
                    json_match = re.search(r'({.*})', content, re.DOTALL)
                    if json_match:
                        plan = json.loads(json_match.group(1))
                        return plan
                    else:
                        # 如果无法解析，返回基础计划
                        return {
                            "complexity_assessment": 0.5,
                            "requires_global_view": False,
                            "requires_path_tracking": False,
                            "has_temporal_aspects": False,
                            "tasks": [
                                {"type": "local_search", "query": query, "priority": 3}
                            ]
                        }
                except Exception as e:
                    print(f"计划生成失败: {str(e)}")
                    # 返回默认计划
                    return {
                        "complexity_assessment": 0.5,
                        "requires_global_view": False,
                        "requires_path_tracking": False,
                        "has_temporal_aspects": False,
                        "tasks": [
                            {"type": "local_search", "query": query, "priority": 3}
                        ]
                    }
        
        return RetrievalPlannerAgent(self.llm)
    
    def _create_local_searcher(self):
        """创建本地搜索Agent
        
        返回:
            LocalSearchTool: 本地搜索工具实例
        """
        return LocalSearchTool()
    
    def _create_global_searcher(self):
        """创建全局搜索Agent
        
        返回:
            GlobalSearchTool: 全局搜索工具实例
        """
        return GlobalSearchTool()
    
    def _create_explorer(self):
        """创建深度探索Agent
        
        返回:
            DeeperResearchTool: 深度研究工具实例
        """
        return DeeperResearchTool()
    
    def _create_chain_explorer(self):
        """创建Chain of Exploration Agent
        
        返回:
            ChainOfExplorationSearcher: 基于知识图谱的链式探索工具实例
        """
        # 获取图数据库连接
        from config.neo4jdb import get_db_manager
        db_manager = get_db_manager()
        graph = db_manager.get_graph()
        
        # 创建Chain of Exploration搜索器
        return ChainOfExplorationSearcher(graph, self.llm, self.embeddings)
    
    def _create_thinking_engine(self):
        """创建思考引擎
        
        返回:
            ThinkingEngine: 思考引擎实例，用于复杂问题的分析和推理
        """
        return ThinkingEngine(self.llm)
    
    def _create_synthesizer(self):
        """创建结果合成Agent
        
        返回:
            SynthesizerAgent: 负责整合检索结果并生成最终答案的Agent实例
        """
        class SynthesizerAgent:
            def __init__(self, llm):
                self.llm = llm
                self.name = "synthesizer"
                self.description = "负责整合所有检索结果并生成最终答案的Agent"
            
            def synthesize(self, query: str, results: Dict[str, List], plan: Dict[str, Any],
                          thinking_process: str = None) -> str:
                """整合检索结果并生成最终答案
                
                参数:
                    query: 用户原始查询
                    results: 包含各种检索结果的字典
                    plan: 检索计划
                    thinking_process: 可选的思考过程文本
                    
                返回:
                    str: 综合所有信息生成的最终答案
                """
                # 构建提示
                prompt = f"""
                基于以下检索结果，回答用户的问题。
                
                用户问题: "{query}"
                
                ## 检索计划
                {json.dumps(plan, ensure_ascii=False, indent=2)}
                """
                
                # 添加思考过程
                if thinking_process:
                    prompt += f"""
                    ## 思考过程
                    {thinking_process}
                    """
                
                # 添加检索结果
                prompt += f"""
                ## 本地检索结果
                {self._format_results(results.get('local', []))}
                
                ## 全局检索结果
                {self._format_results(results.get('global', []))}
                
                ## 探索结果
                {self._format_results(results.get('exploration', []))}
                
                ## Chain of Exploration结果
                {self._format_coe_results(results.get('chain_exploration', []))}
                
                请提供一个全面、准确的回答。确保:
                1. 综合所有相关信息
                2. 解决问题的核心
                3. 说明清晰，逻辑严密
                4. 适当引用信息来源
                5. 结构清晰，使用段落和标题组织内容
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    return f"合成回答时出错: {str(e)}"
            
            def _format_results(self, results: List) -> str:
                """格式化标准检索结果列表
                
                参数:
                    results: 检索结果列表
                    
                返回:
                    str: 格式化后的结果文本
                """
                if not results:
                    return "无相关结果"
                
                formatted = []
                for i, result in enumerate(results):
                    formatted.append(f"结果 {i+1}:\n{result}\n")
                return "\n".join(formatted)
                
            def _format_coe_results(self, results: List) -> str:
                """格式化Chain of Exploration结果
                
                参数:
                    results: Chain of Exploration结果列表
                    
                返回:
                    str: 格式化后的链式探索结果文本
                """
                if not results:
                    return "无Chain of Exploration结果"
                    
                formatted = []
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        # 提取路径信息
                        path_info = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:  # 只显示前5步
                            path_info += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        
                        # 提取内容
                        content_info = "发现内容:\n"
                        for j, content in enumerate(result.get('content', [])[:3]):  # 只显示前3个内容
                            text = content.get('text', '')[:200]  # 限制长度
                            content_info += f"  内容{j+1}: {text}...\n"
                            
                        formatted.append(f"探索结果 {i+1}:\n{path_info}\n{content_info}\n")
                    else:
                        formatted.append(f"探索结果 {i+1}:\n{str(result)[:500]}...\n")
                
                return "\n".join(formatted)
        
        return SynthesizerAgent(self.llm)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理查询，协调多个Agent完成任务
        
        这是协调器的核心同步处理方法，负责接收用户查询并协调多个专用Agent完成
        从任务规划到结果合成的完整流程。采用多阶段处理策略，根据查询复杂度动态
        调整处理策略。
        
        参数:
            query: 用户查询字符串 - 用户提出的问题或请求
            
        返回:
            Dict: 包含以下内容的综合结果字典:
                - answer: 最终生成的答案
                - plan: 执行的检索计划
                - results: 各Agent的检索结果
                - thinking: 思考引擎的思考过程（如果有）
                - execution_trace: 详细的执行轨迹
                - metrics: 性能指标（如总耗时）
        """
        start_time = time.time()  # 记录开始时间，用于性能监控
        self.execution_trace = []  # 重置执行轨迹
        
        # 1. 生成检索计划 - 分析查询并规划最佳检索策略
        self._log_step("generating_plan", "生成检索计划")
        retrieval_plan = self.retrieval_planner.plan(query)  # 调用计划生成器分析查询
        self._log_step("plan_generated", "检索计划已生成", retrieval_plan)  # 记录生成的计划
        
        # 初始化思考引擎 - 为复杂问题提供逻辑推理支持
        self._log_step("thinking_init", "初始化思考引擎")
        self.thinking_engine.initialize_with_query(query)
        
        # 1.5 如果问题复杂度高，生成初步思考 - 针对复杂问题提供额外分析
        complexity = retrieval_plan.get("complexity_assessment", 0)  # 获取复杂度评估
        if complexity > 0.7:  # 复杂度阈值，可根据需要调整
            self._log_step("initial_thinking", "生成初步思考")
            initial_thinking = self.thinking_engine.generate_initial_thinking()
            self._log_step("initial_thinking_complete", "完成初步思考", {"thinking": initial_thinking})
        else:
            initial_thinking = None
        
        # 2. 根据计划执行搜索任务 - 按照优先级执行各类检索任务
        local_results = []         # 本地搜索结果
        global_results = []        # 全局搜索结果
        exploration_results = []   # 深度探索结果
        chain_exploration_results = []  # 链式探索结果
        
        # 按优先级排序任务 - 确保高优先级任务先执行
        tasks = sorted(retrieval_plan.get("tasks", []), 
                     key=lambda x: x.get("priority", 3), 
                     reverse=True)  # 优先级1-5，5最高
        
        # 遍历并执行每个检索任务
        for task in tasks:
            task_type = task.get("type", "")  # 任务类型
            task_query = task.get("query", query)  # 任务的具体查询
            
            # 添加任务进度到思考引擎
            self.thinking_engine.add_reasoning_step(f"执行任务: {task_type} - {task_query}")
            
            # 根据任务类型执行相应的检索操作
            if task_type == "local_search":
                self._log_step("local_search", f"执行本地搜索: {task_query}")
                try:
                    result = self.local_searcher.search(task_query)
                    if result:
                        local_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"本地搜索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("local_search_completed", "本地搜索完成")
                except Exception as e:
                    self._log_step("local_search_error", f"本地搜索出错: {str(e)}")
                    
            elif task_type == "global_search":
                self._log_step("global_search", f"执行全局搜索: {task_query}")
                try:
                    result = self.global_searcher.search(task_query)
                    if result:
                        global_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"全局搜索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("global_search_completed", "全局搜索完成")
                except Exception as e:
                    self._log_step("global_search_error", f"全局搜索出错: {str(e)}")
                    
            elif task_type == "exploration":
                self._log_step("exploration", f"执行深度探索: {task_query}")
                try:
                    result = self.explorer.search(task_query)
                    if result:
                        exploration_results.append(result)
                        # 添加结果到思考引擎
                        self.thinking_engine.add_reasoning_step(f"深度探索结果摘要:\n{self._summarize_result(result)}")
                    self._log_step("exploration_completed", "深度探索完成")
                except Exception as e:
                    self._log_step("exploration_error", f"探索出错: {str(e)}")
                    
            elif task_type == "chain_exploration":
                self._log_step("chain_exploration", f"执行Chain of Exploration: {task_query}")
                try:
                    # 获取相关实体
                    entities = task.get("entities", [])
                    if not entities:
                        # 如果任务没有指定实体，尝试从其他结果中提取
                        entities = self._extract_entities_from_results(
                            local_results, global_results, exploration_results
                        )
                    
                    # 至少需要一个起始实体
                    if not entities:
                        self._log_step("chain_exploration_warning", "未找到起始实体，跳过Chain of Exploration")
                        continue
                        
                    # 执行Chain of Exploration - 基于实体关系进行深度图谱探索
                    result = self.chain_explorer.explore(
                        task_query, 
                        entities[:3],  # 使用前3个实体作为起点
                        max_steps=3    # 限制探索深度，避免过度探索
                    )
                    
                    if result:
                        chain_exploration_results.append(result)
                        # 添加结果到思考引擎
                        path_summary = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:
                            path_summary += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        self.thinking_engine.add_reasoning_step(f"Chain of Exploration结果:\n{path_summary}")
                    self._log_step("chain_exploration_completed", "Chain of Exploration完成")
                except Exception as e:
                    self._log_step("chain_exploration_error", f"Chain of Exploration出错: {str(e)}")
        
        # 3. 如果问题复杂度高，生成最终思考 - 整合所有发现形成最终见解
        if complexity > 0.7:
            self._log_step("final_thinking", "生成最终思考")
            # 告诉思考引擎基于搜索结果更新想法
            self.thinking_engine.add_reasoning_step("基于所有搜索结果，更新我的思考")
            updated_thinking = self.thinking_engine.update_thinking_based_on_verification([])
            self._log_step("final_thinking_complete", "完成最终思考", {"thinking": updated_thinking})
            
            # 获取完整思考过程
            thinking_process = self.thinking_engine.get_full_thinking()
        else:
            thinking_process = None
        
        # 4. 整合所有结果 - 准备合成最终答案
        all_results = {
            "local": local_results,
            "global": global_results,
            "exploration": exploration_results,
            "chain_exploration": chain_exploration_results
        }
        
        # 5. 合成最终答案 - 使用合成器整合所有检索结果
        self._log_step("synthesizing", "合成最终答案")
        final_answer = self.synthesizer.synthesize(query, all_results, retrieval_plan, thinking_process)
        self._log_step("synthesis_completed", "答案合成完成")
        
        # 记录总耗时 - 更新性能指标
        total_time = time.time() - start_time
        self.performance_metrics["total_time"] = total_time
        
        # 返回包含所有信息的综合结果
        return {
            "answer": final_answer,
            "plan": retrieval_plan,
            "results": all_results,
            "thinking": thinking_process,
            "execution_trace": self.execution_trace,
            "metrics": self.performance_metrics
        }
    
    def _summarize_result(self, result):
        """
        生成检索结果的摘要
        
        为不同类型的检索结果生成简洁明了的摘要，用于在思考引擎推理过程中提供关键信息概览，
        同时避免过长的文本导致推理上下文过载。
        
        参数:
            result: 需要摘要的检索结果 - 可以是字符串、字典、列表或其他类型
            
        返回:
            str: 结果摘要文本 - 限制长度的关键信息摘要，便于后续处理和分析
        """
        # 空结果处理
        if not result:
            return "无结果"
            
        # 根据不同类型的结果进行专门处理
        if isinstance(result, str):
            # 字符串类型结果 - 限制长度，保留开头部分
            if len(result) > 500:
                return result[:500] + "..."  # 添加省略号表示被截断
            return result
        elif isinstance(result, dict):
            # 字典类型结果 - 序列化为JSON字符串后限制长度
            return json.dumps(result, ensure_ascii=False)[:500] + "..."
        elif isinstance(result, list):
            # 列表类型结果 - 转换为字符串后限制长度
            return str(result)[:500] + "..."
        else:
            # 其他类型结果 - 通用处理，转换为字符串后限制长度
            return str(result)[:500] + "..."
    
    def _extract_entities_from_results(self, local_results, global_results, exploration_results):
        """
        从现有检索结果中提取实体
        
        使用正则表达式等简单启发式方法从文本中提取潜在实体，为链式探索等需要实体作为起点的操作提供支持。
        通过多种模式匹配和文本分析策略，从不同类型的检索结果中高效识别和提取有价值的实体信息。
        
        参数:
            local_results: 本地搜索结果 - 包含与查询直接相关的本地上下文信息
            global_results: 全局搜索结果 - 提供更广泛的知识和实体视角
            exploration_results: 探索结果 - 可能包含深度分析的实体信息
            
        返回:
            List[str]: 提取的实体列表 - 去重后的实体集合，用于后续检索和探索操作
        """
        entities = set()  # 使用集合去重，避免重复实体
        
        # 提取实体的简单启发式方法 - 定义多种模式以捕获不同形式的实体
        entity_patterns = [
            r'实体\s*[:：]\s*([^,，\n]+)',  # 显式标注的实体
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',  # 英文专有名词（人名、地名等）
            r'【([^】]+)】',  # 中文方括号内容
            r'"([^"]+)"'  # 引号内容
        ]
        
        # 处理所有结果 - 聚合不同来源的文本信息
        all_text = []
        all_text.extend(local_results)
        all_text.extend(global_results)
        
        # 特别处理探索结果，仅添加字符串类型的结果
        for result in exploration_results:
            if isinstance(result, str):
                all_text.append(result)
                
        # 从文本中提取实体 - 应用模式匹配提取潜在实体
        for text in all_text:
            if not isinstance(text, str):
                continue
                
            for pattern in entity_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # 清理并添加实体 - 确保实体长度合理
                    entity = match.strip()
                    if len(entity) > 1 and len(entity) < 30:  # 合理长度限制（避免太短或太长的无效实体）
                        entities.add(entity)
        
        return list(entities)  # 转换为列表返回
    
    async def process_query_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        流式处理查询，返回处理过程和结果
        
        这是协调器的异步流式处理方法，提供类似process_query的功能，但以流式方式返回
        中间结果和进度信息，增强用户体验。适用于需要实时反馈的场景，如Web界面交互。
        
        参数:
            query: 用户查询字符串 - 用户提出的问题或请求
            
        返回:
            AsyncGenerator[str, None]: 异步生成器，流式返回处理状态和最终答案
        """
        # 1. 生成检索计划 - 分析查询并生成检索策略
        yield "**正在分析问题和制定检索计划**...\n\n"  # 流式反馈进度信息
        retrieval_plan = self.retrieval_planner.plan(query)
        
        # 提取和显示计划摘要 - 向用户展示分析结果
        complexity = retrieval_plan.get("complexity_assessment", 0.5)  # 复杂度评估
        requires_global = retrieval_plan.get("requires_global_view", False)  # 是否需要全局视图
        requires_path = retrieval_plan.get("requires_path_tracking", False)  # 是否需要关系追踪
        has_temporal = retrieval_plan.get("has_temporal_aspects", False)  # 是否包含时间因素
        knowledge_areas = retrieval_plan.get("knowledge_areas", [])  # 涉及的知识领域
        
        # 构建并输出计划摘要
        plan_summary = f"**检索计划制定完成**\n"
        plan_summary += f"- 复杂度评估: {complexity:.2f}\n"
        plan_summary += f"- 需要全局视图: {'是' if requires_global else '否'}\n"
        plan_summary += f"- 需要关系路径追踪: {'是' if requires_path else '否'}\n"
        plan_summary += f"- 包含时间相关内容: {'是' if has_temporal else '否'}\n"
        
        if knowledge_areas:
            plan_summary += f"- 涉及知识领域: {', '.join(knowledge_areas[:3])}\n"
        
        yield plan_summary + "\n"
        
        # 初始化思考引擎 - 为复杂问题提供逻辑支持
        self.thinking_engine.initialize_with_query(query)
        
        # 如果复杂度高，添加初步思考 - 针对复杂问题提供额外分析
        if complexity > 0.7:
            yield "**正在进行初步思考分析**...\n\n"
            initial_thinking = self.thinking_engine.generate_initial_thinking()
            
            # 返回思考摘要 - 为了避免输出过长，只显示前几行
            thinking_lines = initial_thinking.split('\n')
            if len(thinking_lines) > 5:
                thinking_summary = '\n'.join(thinking_lines[:5]) + "...\n"
            else:
                thinking_summary = initial_thinking + "\n"
                
            yield thinking_summary + "\n"
        
        # 2. 根据计划执行搜索任务 - 按优先级执行各类检索任务
        tasks = sorted(retrieval_plan.get("tasks", []), 
                     key=lambda x: x.get("priority", 3), 
                     reverse=True)  # 按优先级排序
        
        # 初始化结果存储
        all_results = {"local": [], "global": [], "exploration": [], "chain_exploration": []}
        
        # 遍历并执行每个检索任务
        for i, task in enumerate(tasks):
            task_type = task.get("type", "")  # 任务类型
            task_query = task.get("query", query)  # 任务的具体查询
            
            # 向用户反馈当前正在执行的任务
            task_msg = f"**执行任务 {i+1}/{len(tasks)}**: {task_type} - {task_query}\n"
            yield task_msg
            
            # 添加任务到思考引擎
            self.thinking_engine.add_reasoning_step(f"执行任务: {task_type} - {task_query}")
            
            # 异步执行各类检索任务
            try:
                if task_type == "local_search":
                    # 异步执行本地搜索
                    result = await self._async_local_search(task_query)
                    if result:
                        all_results["local"].append(result)
                        self.thinking_engine.add_reasoning_step(f"本地搜索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 本地搜索完成\n\n"
                        
                elif task_type == "global_search":
                    # 异步执行全局搜索
                    result = await self._async_global_search(task_query)
                    if result:
                        all_results["global"].append(result)
                        self.thinking_engine.add_reasoning_step(f"全局搜索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 全局搜索完成\n\n"
                        
                elif task_type == "exploration":
                    # 异步执行深度探索
                    yield "**开始深度探索**...\n"
                    result = await self._async_exploration(task_query)
                    if result:
                        all_results["exploration"].append(result)
                        self.thinking_engine.add_reasoning_step(f"深度探索结果摘要:\n{self._summarize_result(result)}")
                        yield "✓ 深度探索完成\n\n"
                        
                elif task_type == "chain_exploration":
                    # 异步执行链式探索
                    yield "**开始Chain of Exploration**...\n"
                    # 获取相关实体
                    entities = task.get("entities", [])
                    if not entities:
                        # 如果任务没有指定实体，尝试从其他结果中提取
                        entities = self._extract_entities_from_results(
                            all_results["local"], 
                            all_results["global"], 
                            all_results["exploration"]
                        )
                        
                        if entities:
                            yield f"- 从已有结果中提取实体: {', '.join(entities[:3])}" + ("..." if len(entities) > 3 else "") + "\n"
                    
                    # 至少需要一个起始实体
                    if not entities:
                        yield "⚠️ 未找到起始实体，跳过Chain of Exploration\n\n"
                        continue
                        
                    # 执行链式探索
                    result = await self._async_chain_exploration(task_query, entities[:3])
                    if result:
                        all_results["chain_exploration"].append(result)
                        
                        # 添加结果到思考引擎
                        path_summary = "探索路径:\n"
                        for step in result.get('exploration_path', [])[:5]:
                            path_summary += f"- 步骤{step.get('step')}: {step.get('node_id')} ({step.get('reasoning', '无理由')})\n"
                        self.thinking_engine.add_reasoning_step(f"Chain of Exploration结果:\n{path_summary}")
                        
                        # 显示探索路径摘要给用户
                        if "exploration_path" in result:
                            yield "- 探索路径:\n"
                            for step in result["exploration_path"][:5]:
                                yield f"  • 步骤{step.get('step')}: {step.get('node_id')}\n"
                            
                        if "content" in result:
                            yield f"- 找到 {len(result['content'])} 条相关内容\n"
                            
                        yield "✓ Chain of Exploration完成\n\n"
            
            except Exception as e:
                # 错误处理 - 向用户反馈任务执行失败
                yield f"❌ {task_type}任务执行失败: {str(e)}\n\n"
        
        # 如果复杂度高，生成最终思考
        if complexity > 0.7:
            yield "**正在基于所有搜索结果进行最终思考**...\n\n"
            self.thinking_engine.add_reasoning_step("基于所有搜索结果，更新我的思考")
            updated_thinking = self.thinking_engine.update_thinking_based_on_verification([])
            
            # 返回思考摘要
            thinking_lines = updated_thinking.split('\n')
            if len(thinking_lines) > 5:
                thinking_summary = '\n'.join(thinking_lines[:5]) + "...\n"
            else:
                thinking_summary = updated_thinking + "\n"
                
            yield thinking_summary + "\n"
        
        # 3. 合成最终答案
        yield "**正在整合所有检索结果，生成最终答案**...\n\n"
        
        # 获取思考过程
        thinking_process = self.thinking_engine.get_full_thinking() if complexity > 0.7 else None
        
        # 异步合成最终答案
        final_answer = await self._async_synthesize(query, all_results, retrieval_plan, thinking_process)
        
        # 清理答案 - 删除"引用数据"部分保留干净的回答
        clean_answer = final_answer
        ref_index = final_answer.find("#### 引用数据")
        if ref_index > 0:
            clean_answer = final_answer[:ref_index].strip()
        
        # 输出最终答案
        yield f"\n\n{clean_answer}"
    
    async def _async_local_search(self, query):
        """异步执行本地搜索
        
        将同步的本地搜索操作转换为异步执行，以支持流式处理。
        使用线程池执行器运行阻塞操作，避免阻塞事件循环。
        
        参数:
            query: 搜索查询
            
        返回:
            搜索结果
        """
        def sync_search():
            return self.local_searcher.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_search)
    
    async def _async_global_search(self, query):
        """异步执行全局搜索
        
        将同步的全局搜索操作转换为异步执行，以支持流式处理。
        
        参数:
            query: 搜索查询
            
        返回:
            搜索结果
        """
        def sync_search():
            return self.global_searcher.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_search)
    
    async def _async_exploration(self, query):
        """异步执行深度探索
        
        将同步的深度探索操作转换为异步执行，以支持流式处理。
        
        参数:
            query: 探索查询
            
        返回:
            探索结果
        """
        def sync_explore():
            return self.explorer.search(query)
        return await asyncio.get_event_loop().run_in_executor(None, sync_explore)
    
    async def _async_chain_exploration(self, query, entities):
        """异步执行Chain of Exploration
        
        将同步的链式探索操作转换为异步执行，以支持流式处理。
        
        参数:
            query: 探索查询
            entities: 起始实体列表
            
        返回:
            链式探索结果
        """
        def sync_explore():
            return self.chain_explorer.explore(query, entities, max_steps=3)
        return await asyncio.get_event_loop().run_in_executor(None, sync_explore)
    
    async def _async_synthesize(self, query, results, plan, thinking_process=None):
        """异步合成最终答案
        
        将同步的答案合成操作转换为异步执行，以支持流式处理。
        
        参数:
            query: 用户查询
            results: 检索结果
            plan: 检索计划
            thinking_process: 思考过程
            
        返回:
            str: 合成的最终答案
        """
        def sync_synthesize():
            return self.synthesizer.synthesize(query, results, plan, thinking_process)
        return await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)
    
    def _log_step(self, step_type: str, description: str, data: Any = None):
        """
        记录执行步骤
        
        为系统执行过程中的每个重要步骤创建详细日志，包含时间戳和可选的附加数据。
        这些日志对于调试、监控和评估系统性能至关重要。
        
        参数:
            step_type: 步骤类型标识符 - 用于分类和过滤日志
            description: 步骤描述 - 提供步骤的人类可读说明
            data: 可选的附加数据 - 包含与步骤相关的额外信息
        """
        self.execution_trace.append({
            "type": step_type,
            "description": description,
            "timestamp": time.time(),
            "data": data
        })