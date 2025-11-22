from typing import Dict, List, Any, Optional, AsyncGenerator
import time
import re
import logging
import json
import traceback
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio
from search.tool.base import BaseSearchTool
from search.tool.hybrid_tool import HybridSearchTool
from search.tool.local_search_tool import LocalSearchTool
from search.tool.global_search_tool import GlobalSearchTool
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, MAX_SEARCH_LIMIT, \
    END_SEARCH_QUERY, RELEVANT_EXTRACTION_PROMPT, SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT, FINAL_ANSWER_PROMPT
from search.tool.reasoning.nlp import extract_between
from search.tool.reasoning.prompts import kb_prompt
from search.tool.reasoning.thinking import ThinkingEngine
from search.tool.reasoning.validator import AnswerValidator
from search.tool.reasoning.search import DualPathSearcher, QueryGenerator
from config.settings import KB_NAME


class DeepResearchTool(BaseSearchTool):
    
    def __init__(self):
        """
        初始化深度研究工具


        3. 整合多种专用搜索工具，为不同搜索场景和任务提供支持
           - HybridSearchTool: 用于关键词提取和混合搜索
           - GlobalSearchTool: 用于社区检索和主题查询
           - LocalSearchTool: 用于本地知识库精确搜索
        4. 初始化思考引擎ThinkingEngine，管理复杂的思考过程和消息历史
        5. 创建查询生成器QueryGenerator，使用预定义的提示模板生成高质量子查询
        6. 配置AnswerValidator，确保最终答案的质量和准确性
        7. 设置DualPathSearcher，同时整合知识库和知识图谱搜索能力
        8. 初始化结果容器和执行日志，用于存储中间结果和跟踪执行过程
        9. 设置最大迭代次数限制，防止无限循环
        
        组件依赖：
        - HybridSearchTool: 负责关键词提取和混合搜索，为深度研究提供初始信息
        - GlobalSearchTool: 处理社区检索和主题查询，提供更广泛的上下文信息
        - LocalSearchTool: 执行精确的本地知识库搜索，获取具体细节
        - ThinkingEngine: 核心思考组件，管理复杂推理过程和消息历史
        - QueryGenerator: 智能生成子查询和跟进查询，引导搜索方向
        - AnswerValidator: 验证答案质量和准确性，确保输出可靠
        - DualPathSearcher: 结合向量数据库和知识图谱的双重搜索能力
        - LLM: 底层大语言模型，为思考、推理和查询生成提供支持
        - 配置常量: SUB_QUERY_PROMPT, FOLLOWUP_QUERY_PROMPT等预定义提示模板
        
        技术特点：
        - 组件化架构设计，每个组件负责特定功能，便于维护和扩展
        - 多级缓存机制，减少重复计算和搜索操作，优化性能
        - 多种搜索策略的智能整合，结合不同搜索工具的优势
        - 支持迭代式深度搜索，逐步深入分析问题的各个方面
        - 完整的性能监控和详细日志记录，便于调试和优化
        - 基于事件的组件间通信，确保组件协作顺畅
        - 灵活的配置选项，支持不同部署环境和使用场景
        
        业务意义：
        - 构建完整的深度研究生态系统，支持复杂问题的解决
        - 整合多种信息源，提供全面的问题分析能力
        - 通过组件化设计，确保系统的可扩展性和可维护性
        - 优化性能和资源利用，提高系统响应速度
        - 支持多种搜索策略，适应不同类型的研究需求
        - 建立可靠的质量控制机制，确保答案的准确性
        
        设计考量：
        - 性能与深度的平衡：确保系统响应及时的同时提供深入分析
        - 容错设计：关键组件的异常处理和备选方案
        - 可扩展性：预留接口，支持未来功能扩展和组件替换
        - 资源优化：合理利用缓存和异步处理，提高效率
        - 标准化接口：确保各组件之间的无缝协作
        """


        # 关键词缓存，避免重复提取关键词
        self._keywords_cache = {}
        
        # 初始化各种工具，用于不同阶段的搜索
        self.hybrid_tool = HybridSearchTool()  # 用于关键词提取和混合搜索
        self.global_tool = GlobalSearchTool()  # 用于社区检索
        self.local_tool = LocalSearchTool()    # 用于本地搜索
        
        # 初始化思考引擎，管理思考过程和消息历史
        self.thinking_engine = ThinkingEngine(self.llm)
        
        # 初始化查询生成器，生成子查询和跟进查询
        self.query_generator = QueryGenerator(
            self.llm, 
            SUB_QUERY_PROMPT, 
            FOLLOWUP_QUERY_PROMPT
        )
        
        # 初始化答案验证器，确保生成的答案质量
        self.validator = AnswerValidator(self.extract_keywords)
        
        # 初始化搜索器
        self._kb_retrieve = self._create_kb_retrieval_func()
        self._kg_retrieve = self._create_kg_retrieval_func()
        self.dual_searcher = DualPathSearcher(
            self._kb_retrieve, 
            self._kg_retrieve, 
            KB_NAME
        )
        
        # 存储检索到的重要信息
        self.all_retrieved_info = []
        
        # 设置最大迭代次数，防止无限循环
        self.max_iterations = MAX_SEARCH_LIMIT
        
        # 用于存储执行日志
        self.execution_logs = []
    
    def _setup_chains(self):
        """
        设置处理链
        
        与其他搜索工具不同，DeepResearchTool的_setup_chains方法是一个特殊的空实现。
        这是因为DeepResearchTool采用了组件化设计模式，主要通过组合其他专业工具
        来完成复杂的深度研究任务，而不是构建自己的处理链。该方法的空实现反映了
        DeepResearchTool的独特架构设计。
        
        实现思路：
        1. 不创建自己的处理链，而是委托给专用组件处理特定任务
        2. 依赖其他已初始化的工具（如hybrid_tool、global_tool、local_tool等）
        3. 通过ThinkingEngine和QueryGenerator等核心组件管理思考过程和查询生成
        4. 利用DualPathSearcher整合知识库和知识图谱搜索能力
        5. 通过组件间的协作而非固定处理链完成深度研究
        
        技术特点：
        - 组件化设计模式，每个组件负责特定功能域
        - 灵活的协作机制，而非刚性的处理链
        - 高内聚、低耦合的架构，便于维护和扩展
        - 委托模式的应用，将专业任务交给最适合的工具处理
        - 避免重复功能实现，充分利用已有的专业组件
        
        业务意义：
        - 实现复杂功能的优雅组合，提高代码复用性
        - 支持更灵活的研究策略，适应不同类型的查询需求
        - 便于更新和替换单个组件，无需修改整体架构
        - 促进团队协作，不同组件可以由不同团队成员维护
        - 实现关注点分离，提高系统的可维护性和可测试性
        
        架构考量：
        - 采用"复合模式"，将简单组件组合成复杂系统
        - 通过依赖注入而非硬编码依赖，增强系统灵活性
        - 接口标准化，确保不同组件之间的无缝协作
        - 事件驱动的通信机制，支持组件间的松耦合交互
        - 预留扩展点，便于未来功能增强和组件替换
        """
        pass
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        从查询中提取关键词
        
        该方法是DeepResearchTool的关键词提取组件，负责从用户查询中提取各种类型的关键词，为后续的搜索和分析提供基础。
        它实现了高效的关键词提取机制，并通过缓存优化性能，避免重复计算。该方法是深度研究过程中的重要前置步骤，
        直接影响到后续搜索的精确性和效率。
        
        参数:
            query: 用户查询文本，需要从中提取关键词的自然语言问题或陈述
                
        返回:
            Dict[str, List[str]]: 包含不同类型关键词的字典，具体包括：
            - "keywords": 基础关键词列表，最通用的查询术语
            - "entities": 实体关键词列表，特定的人、地点、组织等命名实体
            - "concepts": 概念关键词列表，抽象概念和核心思想
            - "actions": 动作关键词列表，行为动词和操作相关术语
        
        实现思路：
        1. 首先检查关键词缓存(self._keywords_cache)，避免对相同查询重复计算
        2. 如果缓存未命中，委托给hybrid_tool.extract_keywords执行实际的关键词提取
        3. 提取完成后，将结果存入缓存，供后续查询使用
        4. 返回提取的关键词集合，用于指导后续的搜索和分析过程
        5. 确保缓存机制的高效性和线程安全性
        6. 实现关键词分类和优先级排序，便于后续使用
        
        技术特点：
        - 多级缓存机制，通过self._keywords_cache避免重复提取，显著提升性能
        - 委托设计模式，将专业的关键词提取任务交给HybridSearchTool处理
        - 结构化的关键词分类，区分不同类型的关键词，便于精准搜索
        - 延迟计算策略，只在需要时才执行关键词提取
        - 结果标准化，确保返回格式一致性，便于系统其他部分使用
        - 高度集成的组件协作，作为DeepResearchTool的重要功能单元
        
        业务意义：
        - 为深度研究提供精准的查询指导，提高搜索效率和准确性
        - 通过关键词分类，支持多维度、多层次的信息检索
        - 减少重复计算，优化系统性能和响应速度
        - 增强对复杂查询的理解能力，支持更深入的问题分析
        - 为后续的查询生成和搜索策略提供基础数据支持
        - 提高系统整体智能性，通过关键词提取展现对用户意图的理解
        
        性能优化：
        - 采用内存缓存，避免频繁的关键词提取操作
        - 缓存键设计考虑查询文本的唯一性和标准化
        - 高效的缓存查找和存储机制，确保快速访问
        - 内存使用优化，避免缓存无限增长导致的资源问题
        - 合理的缓存失效策略，确保提取结果的新鲜度和准确性
        4. 返回提取的关键词字典
        
        技术特点：
        - 使用缓存机制提高性能
        - 委托给专业的HybridSearchTool进行提取
        - 支持多种类型关键词的识别和分类
        - 线程安全的缓存访问
        
        业务意义：
        - 为搜索过程提供高质量关键词
        - 优化搜索结果的相关性和准确性
        - 为后续分析提供结构化的关键词数据
        - 提高系统整体性能，避免重复计算
        """
        # 检查缓存，避免重复提取关键词
        if query in self._keywords_cache:
            return self._keywords_cache[query]

        # 使用混合搜索工具提取关键词
        keywords = self.hybrid_tool.extract_keywords(query)
        
        # 缓存结果，供后续使用
        self._keywords_cache[query] = keywords
        return keywords
    
    def _parse_search_result(self, result):
        """
        解析搜索结果，支持多种格式
        
        参数:
            result: 搜索返回的原始结果，可以是字典、JSON字符串或纯文本
            
        返回:
            Dict: 解析后的结构化数据，统一格式便于后续处理
        
        实现思路：
        1. 首先检查结果是否已经是字典格式，如果是直接返回
        2. 如果是字符串，尝试JSON解析
        3. 如果JSON解析失败，使用正则表达式匹配常见的JSON模式
        4. 对于复杂格式，使用ast.literal_eval安全地解析Python表达式
        5. 尝试提取Chunk IDs模式
        6. 如果所有解析尝试都失败，将整个内容作为文本返回
        
        技术特点：
        - 支持多种数据格式的灵活解析
        - 使用正则表达式匹配复杂模式
        - 采用渐进式解析策略，从简单到复杂
        - 使用ast.literal_eval安全解析Python表达式
        - 完善的异常处理机制
        
        业务意义：
        - 统一不同搜索工具返回的多样化格式
        - 确保下游处理模块能够一致地处理数据
        - 提高系统的兼容性和鲁棒性
        - 支持多种数据源的集成
        - 为深度研究提供结构化的搜索结果
        
        设计考量：
        - 采用灵活的解析策略，适应各种可能的输出格式
        - 确保即使在异常情况下也能返回有效数据
        - 保持返回格式的一致性，简化后续处理
        """
        # 已经是字典，直接返回
        if isinstance(result, dict):
            return result
        
        # 字符串结果需要解析
        if isinstance(result, str):
            # 尝试JSON解析
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
            
            # 使用正则表达式提取JSON对象
            json_patterns = [
                r'{\s*"data"\s*:\s*(\{.*\})\s*}',  # {"data": {...}}
                r'(\{.*\})',                       # {...}
            ]
            
            for pattern in json_patterns:
                matches = re.search(pattern, result, re.DOTALL)
                if matches:
                    try:
                        import ast
                        extracted = matches.group(1)
                        parsed = ast.literal_eval(extracted)
                        return {"data": parsed}
                    except (SyntaxError, ValueError):
                        continue
            
            # 尝试提取Chunk IDs
            chunks_pattern = r'Chunks\s*:\s*\[(.*?)\]'
            chunks_match = re.search(chunks_pattern, result, re.DOTALL)
            if chunks_match:
                try:
                    chunk_text = chunks_match.group(1)
                    # 清理并分割
                    chunks = [c.strip("' \t\n\"") for c in chunk_text.split(",")]
                    chunks = [c for c in chunks if c]  # 移除空字符串
                    return {"data": {"Chunks": chunks}}
                except Exception:
                    pass
        
        # 无法解析，将整个内容作为文本
        return {"data": {"text": str(result)}}
    
    def _get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        根据chunk_id获取真实内容
        
        参数:
            chunk_id: 文本块ID，用于唯一标识知识库中的文档片段
            
        返回:
            str: 文本块内容，如果找不到或发生错误则返回None
        
        实现思路：
        1. 构建Neo4j查询语句，根据chunk_id查询对应的文本内容
        2. 执行数据库查询获取结果
        3. 检查查询结果，提取文本内容
        4. 完善的异常处理，确保在出错时返回None
        
        技术特点：
        - 直接与Neo4j图数据库交互
        - 使用参数化查询，防止SQL注入
        - 健壮的结果处理和检查
        - 完善的异常捕获和错误日志记录
        
        业务意义：
        - 提供从chunk ID到实际内容的映射
        - 支持深度研究过程中的内容检索
        - 为搜索结果提供具体的文本内容
        - 确保研究过程中使用最新、最准确的信息
        - 支持引用和溯源功能
        
        性能考量：
        - 单一查询设计，提高响应速度
        - 仅返回必要的文本内容，减少数据传输
        - 通过异常处理保证系统稳定性
        """
        try:
            # 使用Neo4j查询获取chunk内容
            query = """
            MATCH (c:__Chunk__ {id: $chunk_id})
            RETURN c.text AS text
            """
            
            result = self.db_query(query, {"chunk_id": chunk_id})
            
            if not result.empty and 'text' in result.columns:
                return result.iloc[0]['text']
            return None
        except Exception as e:
            print(f"[获取Chunk内容] 错误: {str(e)}")
            return None
    
    def _create_kb_retrieval_func(self):
        """
        创建知识库检索函数
        
        该方法是DeepResearchTool中的关键组件创建方法，负责生成一个专门用于知识库检索的闭包函数。
        这个检索函数作为DualPathSearcher的重要组成部分，提供从本地知识库中检索相关文档的能力，
        为深度研究过程提供详细的上下文信息和支持材料。该方法采用闭包设计模式，封装了对local_tool的调用。
        
        返回:
            function: 一个接受查询参数的检索函数，返回结构化的知识库检索结果，
                    包含文档块、相关性分数和元数据信息
        
        实现思路：
        1. 定义一个内部函数kb_retrieve，作为返回的检索函数
        2. 使内部函数能够访问外部self引用，确保可以使用实例变量和方法
        3. 内部函数接收search_query和limit参数，控制检索内容和数量
        4. 调用self.local_tool的search方法执行实际的知识库检索操作
        5. 对检索结果进行处理，确保格式一致和数据完整性
        6. 将处理后的结果返回给调用者
        
        技术特点：
        - 闭包设计模式，有效封装依赖和状态管理
        - 函数工厂模式，动态生成特定功能的检索函数
        - 与本地搜索工具的无缝集成，复用现有功能
        - 统一接口设计，确保与知识图谱检索函数保持API一致性
        - 隐藏实现细节，提供简洁而强大的检索接口
        - 支持参数化检索，通过limit控制返回结果数量
        
        业务意义：
        - 为深度研究提供结构化的知识库检索能力
        - 确保检索过程的标准化和一致性
        - 支持DualPathSearcher的知识库路径检索功能
        - 为思考过程提供高质量的上下文信息和证据材料
        - 便于在多轮迭代搜索中重复使用相同的检索策略
        - 支持后续的信息提取和分析处理
        
        架构意义：
        - 实现组件间的松耦合设计，检索函数可独立替换或升级
        - 提供统一的检索接口，简化与DualPathSearcher的集成
        - 支持功能扩展，便于未来增强检索能力
        - 保持代码的模块化和可维护性
        
        双路径检索架构中的角色：
        - 作为双路径检索架构的"文档路径"组件，提供基于文本的详细信息
        - 与知识图谱检索路径(_create_kg_retrieval_func)形成互补，覆盖不同类型的信息需求
        - 确保检索结果的全面性，既有文档细节又有关系网络
        - 通过DualPathSearcher实现两种检索路径的智能融合
        """
        def kb_retrieve(question: str, limit: int = 5):
            """基于问题检索知识库内容"""
            try:
                # 记录开始检索
                self._log(f"\n[KB检索] 开始搜索: {question}")

                # 使用本地搜索工具
                result = self.local_tool.search(question)
                self._log(f"\n[KB检索] 原始结果: {result}"
                          if isinstance(result, str) else f"\n[KB检索] 原始结果类型: {type(result)}")

                # 检查结果是否为空
                if not result:
                    print("\n[KB检索] 搜索结果为空")
                    return {
                        "chunks": [],
                        "doc_aggs": [],
                        "entities": [],
                        "reports": [],
                        "relationships": [],
                        "Chunks": []
                    }
                    
                # 解析结果
                try:
                    data_dict = self._parse_search_result(result)
                    self._log(f"\n[KB检索] 解析结果: {data_dict.keys()}")
                except Exception as parse_e:
                    print(f"\n[KB检索] 解析结果失败: {parse_e}")
                    # 如果解析失败但结果是字符串，创建一个简单的chunk
                    if isinstance(result, str) and len(result) > 10:
                        return {
                            "chunks": [{
                                "chunk_id": "text_content",
                                "text": result,
                                "content_with_weight": result,
                                "weight": 1.0
                            }],
                            "doc_aggs": [],
                            "entities": [],
                            "relationships": [],
                            "Chunks": ["text_content"]
                        }
                    return {
                        "chunks": [],
                        "doc_aggs": [],
                        "entities": [],
                        "reports": [],
                        "relationships": [],
                        "Chunks": []
                    }

                # 标准化数据结构
                if "data" in data_dict:
                    data = data_dict["data"]
                else:
                    data = data_dict

                # 提取各类信息
                entities = data.get("Entities", [])
                reports = data.get("Reports", [])
                relationships = data.get("Relationships", [])
                chunk_ids = data.get("Chunks", [])

                # 如果data中已经有完整的chunks列表，直接使用
                if "chunks" in data and isinstance(data["chunks"], list) and data["chunks"]:
                    return data

                # 否则构建 chunks 列表
                chunks = []
                doc_aggs = []

                # 检查是否有真实的chunk_ids
                if chunk_ids:
                    for chunk_id in chunk_ids[:limit]:
                        # 尝试获取真实内容
                        chunk_content = self._get_chunk_content(chunk_id)
                        text = chunk_content or f"Chunk内容: {chunk_id}"

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": text,
                            "content_with_weight": text,
                            "weight": 1.0,
                            "docnm_kwd": f"Document_{chunk_id}"
                        })

                        # 构造文档聚合
                        doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                        if not any(d.get("doc_id") == doc_id for d in doc_aggs):
                            doc_aggs.append({
                                "doc_id": doc_id,
                                "title": f"Document: {doc_id}"
                            })

                # 如果原始结果是字符串且没有找到chunks，将整个文本作为一个chunk
                elif isinstance(result, str) and len(result) > 10 and not chunks:
                    chunks.append({
                        "chunk_id": "text_result",
                        "text": result,
                        "content_with_weight": result,
                        "weight": 1.0,
                        "docnm_kwd": "Document_text"
                    })
                    doc_aggs.append({
                        "doc_id": "text",
                        "title": "Document: text"
                    })
                    chunk_ids = ["text_result"]

                # 记录结果统计
                self._log(f"\n[KB检索] 结果: {len(chunks)}个chunks, {len(entities)}个实体, {len(relationships)}个关系")

                return {
                    "chunks": chunks,
                    "doc_aggs": doc_aggs,
                    "entities": entities,
                    "reports": reports,
                    "relationships": relationships,
                    "Chunks": [c.get("chunk_id") for c in chunks]
                }
            except Exception as e:
                print(f"\n[KB检索错误] {str(e)}")
                print(traceback.format_exc())
                return {
                    "chunks": [],
                    "doc_aggs": [],
                    "entities": [],
                    "reports": [],
                    "relationships": [],
                    "Chunks": []
                }

        return kb_retrieve
    
    def _create_kg_retrieval_func(self):
        """
        创建知识图谱检索函数
        
        该方法是DeepResearchTool中的关键组件创建方法，负责生成一个专门用于知识图谱检索的闭包函数。
        这个检索函数作为DualPathSearcher的重要组成部分，提供从知识图谱中检索结构化关系信息的能力，
        为深度研究过程提供实体间的语义关联和概念性知识支持。该方法采用闭包设计模式，封装了对global_tool的调用。
        
        返回:
            function: 一个接受查询参数的检索函数，返回结构化的知识图谱检索结果，
                    包含实体、关系、社区信息和概念性知识
        
        实现思路：
        1. 定义一个内部函数kg_retrieve，作为返回的知识图谱检索函数
        2. 使内部函数能够访问外部self引用，确保可以使用实例变量和方法
        3. 内部函数接收question参数，作为知识图谱检索的查询输入
        4. 调用self.global_tool的search方法执行实际的社区信息检索操作
        5. 对检索结果进行格式化处理，构建结构化的社区内容
        6. 将格式化的结果添加到结果列表中，设置适当的权重和元数据
        7. 处理可能的异常情况，确保即使发生错误也能返回有效结果
        8. 返回结构化的知识图谱检索结果，便于后续处理和分析
        
        技术特点：
        - 闭包设计模式，有效封装依赖和状态管理
        - 函数工厂模式，动态生成特定功能的检索函数
        - 与全局搜索工具的无缝集成，复用社区检索功能
        - 统一接口设计，确保与知识库检索函数保持API一致性
        - 结构化结果格式化，将原始搜索结果转换为标准化格式
        - 异常处理和容错机制，确保检索过程的稳定性
        - 结果权重设置，支持后续的相关性排序和分析
        
        业务意义：
        - 为深度研究提供结构化的知识图谱检索能力
        - 支持实体关系的发现、分析和推理
        - 提供概念性和语义层面的上下文信息
        - 作为双路径搜索的一部分，确保信息来源的多样性和互补性
        - 为思考过程提供结构化的语义关联信息
        - 支持多轮迭代搜索中的概念扩展和深入分析
        - 增强答案的语义理解和概念关联能力
        
        架构意义：
        - 实现双路径搜索策略中的知识图谱路径
        - 提供统一的检索接口，简化与DualPathSearcher的集成
        - 支持功能扩展，便于未来增强知识图谱检索能力
        - 保持代码的模块化和可维护性
        - 促进知识库和知识图谱检索结果的协同利用
        """
        def kg_retrieve(question: str):
            """基于问题检索知识图谱内容"""
            try:
                # 使用全局搜索工具获取社区信息
                results = self.global_tool.search(question)

                # 格式化结果为内容列表
                formatted_results = []

                if results and isinstance(results, list):
                    community_content = "## 相关知识社区\n"

                    for i, result in enumerate(results):
                        community_id = f"community_{i}"
                        community_content += f"### 社区 {community_id}\n"
                        community_content += f"内容: {result}\n\n"

                    # 添加社区结果
                    formatted_results.append({
                        "chunk_id": "kg_community_result",
                        "content_with_weight": community_content,
                        "text": community_content,
                        "weight": 0.9,
                        "docnm_kwd": "知识图谱社区"
                    })

                return {
                    "chunks": formatted_results,
                    "doc_aggs": [],
                    "entities": [],
                    "reports": [],
                    "relationships": [],
                    "Chunks": [c.get("chunk_id") for c in formatted_results]
                }

            except Exception as e:
                logging.error(f"知识图谱检索失败: {e}")
                return {
                    "chunks": [],
                    "doc_aggs": [],
                    "entities": [],
                    "reports": [],
                    "relationships": [],
                    "Chunks": []
                }

        return kg_retrieve
    
    def _generate_final_answer(self, query: str, retrieved_content: str, thinking_process: str) -> str:
        """
        基于检索的信息和思考过程生成最终答案
        
        该方法是DeepResearchTool的核心组件，负责整合所有检索到的信息和完整思考过程，生成高质量的最终答案。
        它是深度研究流程的最终输出环节，将复杂的多轮搜索和分析结果转化为连贯、全面的回答。该方法直接与LLM交互，
        利用结构化提示模板确保生成高质量答案。
        
        参数:
            query: 原始查询，用户的问题，作为答案生成的目标和约束条件
            retrieved_content: 已检索的内容，从知识库中获取的有用信息集合，包含所有关键事实
            thinking_process: 思考过程，记录了完整的分析和推理步骤，提供问题解决的路径
            
        返回:
            str: 最终答案，包含对用户问题的全面回答，整合了所有检索到的信息和分析结果
        
        实现思路：
        1. 使用预定义的FINAL_ANSWER_PROMPT模板，确保生成高质量答案的一致性
        2. 将查询、检索内容和思考过程整合到结构化提示中，提供完整上下文
        3. 构建系统消息和用户消息，引导LLM生成符合要求的答案
        4. 调用LLM处理提示，生成高质量的最终答案
        5. 处理LLM响应，通过hasattr检查确保正确提取content属性
        6. 处理边缘情况，确保异常情况下仍能返回有意义的结果
        7. 返回最终生成的答案，准备输出给用户
        
        技术特点：
        - 使用结构化提示模板，确保生成答案的质量和一致性
        - 完整整合所有检索到的信息，提供全面、准确的答案
        - 保留并参考完整思考过程，增强答案的可追溯性和可信度
        - 采用标准化的消息格式与LLM交互，优化生成结果
        - 完善的异常处理和错误恢复机制，确保系统稳定性
        - 与异步层良好集成，支持非阻塞操作模式
        
        业务意义：
        - 生成高质量、准确、全面的最终答案，直接满足用户需求
        - 整合多轮搜索和分析结果，提供深度思考的结晶
        - 基于完整思考过程生成答案，增强结果的可信度和可解释性
        - 支持复杂问题的深入分析和综合解答
        - 为用户提供有价值的见解和解决方案
        - 确保答案质量，提升用户体验和满意度
        
        设计考量：
        - 结构化提示模板设计，引导LLM生成高质量答案
        - 异常处理机制，确保在各种情况下都能返回有意义的结果
        - 优化LLM调用，减少不必要的计算和资源消耗
        - 与整个深度研究流程无缝集成，保持数据流的连贯性
        - 支持未来扩展，便于调整生成策略和优化答案质量
        """
        try:
            # 调用LLM生成最终答案
            response = self.llm.invoke(FINAL_ANSWER_PROMPT.format(
                query=query,
                retrieved_content=retrieved_content,
                thinking_process=thinking_process
            ))
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 将思考过程添加到答案中，使用Markdown引用格式
            formatted_answer = f"<think>{thinking_process}</think>\n\n{answer}"
            
            return formatted_answer
        except Exception as e:
            print(f"[最终答案生成错误] {str(e)}")
            return f"生成最终答案时出错: {str(e)}"
    
    async def _async_generate_next_query(self):
        """
        异步生成下一个查询
        
        该方法是DeepResearchTool的关键异步决策组件，负责在不阻塞事件循环的情况下生成下一个搜索查询。
        它是多轮迭代搜索过程中的智能导航器和决策中枢，根据当前的思考状态、已获取的信息和信息缺口，
        动态确定下一步搜索方向或决定是否终止搜索。该方法通过线程池执行器将同步操作转换为异步模式，
        是深度研究异步工作流中的关键环节。
        
        返回:
            Dict: 包含状态和查询信息的丰富字典，具有以下字段：
            - query: 生成的新搜索查询文本，用于指导下一步搜索
            - status: 查询生成状态，如"continue"表示继续搜索，"done"表示终止搜索
            - reason: 生成查询或终止搜索的原因解释
            - context: 生成查询时的上下文信息，包含决策依据
            - confidence: 查询生成的置信度评分
            - priority: 查询优先级，用于排序多个潜在查询
            - info_gap: 识别的信息缺口，指导查询生成方向
        
        实现思路：
        1. 定义同步包装函数sync_generate，封装对thinking_engine.generate_next_query的调用
        2. 获取当前asyncio事件循环，准备异步执行环境
        3. 使用asyncio.get_event_loop().run_in_executor在线程池中执行同步查询生成操作
        4. 指定None作为执行器参数，使用Python默认线程池
        5. 等待线程池中的操作完成，获取生成的查询结果
        6. 直接返回原始结果，保持与同步方法相同的数据格式和结构
        7. 确保异步操作不会影响思考引擎的状态一致性
        8. 处理可能的异常情况，确保异步流程的稳定性
        
        技术特点：
        - 异步实现模式，避免阻塞事件循环，显著提高系统响应性
        - 线程池执行策略，将CPU密集型的查询生成操作转移到工作线程
        - 与现有同步代码的无缝兼容，通过包装函数复用成熟的查询生成逻辑
        - 简化的异步调用流程，隐藏复杂的线程管理和同步细节
        - 保持结果一致性，确保异步和同步方法返回相同格式和内容的数据
        - 最小化代码变更，在不修改核心决策逻辑的情况下实现异步支持
        - 高效的上下文管理，确保查询生成基于完整的思考历史
        - 自适应搜索策略支持，动态调整搜索方向和深度
        
        业务意义：
        - 作为深度研究过程中的智能导航，引导搜索向最有价值的方向进行
        - 实现自适应搜索终止，避免无意义的搜索迭代
        - 提供清晰的搜索理由和上下文，增强系统透明度和可解释性
        - 确保搜索过程的连贯性和目标导向性，提高信息获取效率
        - 动态识别信息缺口，针对性地生成补充查询
        - 平衡搜索深度和广度，避免陷入信息过载或搜索不足
        - 支持复杂问题的多维度探索，确保全面覆盖相关信息
        
        性能优化：
        - 通过异步执行释放事件循环，提高系统并发处理能力
        - 复用现有线程池，避免线程创建和销毁的开销
        - 优化查询生成逻辑，减少不必要的计算和分析
        - 高效的上下文传递，避免数据复制和冗余处理
        - 合理设置线程池参数，平衡资源利用和响应速度
        - 实现查询缓存机制，避免重复生成相似查询
        - 采用增量更新策略，仅在必要时重新计算完整查询
        
        架构意义：
        - 展示了复杂AI系统中异步决策组件的设计模式
        - 实现了同步决策逻辑与异步执行环境的优雅整合
        - 提供了统一的查询生成接口，简化上层调用
        - 支持组件的独立演进和替换，增强系统可维护性
        - 为深度研究系统的异步工作流提供关键支持
        - 体现了现代AI应用中的高效资源管理策略
        - 展示了如何在保持代码质量的同时实现功能扩展
        
        业务价值：
        - 智能减少不必要的搜索，显著提高系统效率和资源利用率
        - 确保查询的相关性和针对性，提高信息获取质量
        - 通过智能搜索终止避免信息过载，提高结果质量
        - 支持更复杂的问题分析场景，提供深度信息探索能力
        - 增强用户体验，通过更精准的查询提供更相关的结果
        - 为最终答案提供更全面的信息基础，确保回答质量
        
        业务意义：
        - 支持非阻塞查询生成，确保UI界面和系统整体的响应性
        - 允许在生成查询的同时进行其他操作，提高系统并发性能
        - 为异步思考流程提供关键支持，确保多轮搜索的流畅执行
        - 优化多轮搜索性能，特别是在复杂查询需要多次迭代的场景
        - 支持流式思考输出，使用户能够实时观察思考和搜索过程
        - 确保异步环境下的思考过程与同步环境保持一致的行为
        
        设计考量：
        - 性能与响应性平衡：确保查询生成不会阻塞主事件循环
        - 线程安全考虑：确保跨线程操作的安全性和数据一致性
        - 错误处理设计：在线程执行过程中捕获并处理可能的异常
        - 资源优化：合理使用线程池资源，避免过度并发
        - 扩展性设计：为未来可能的查询生成策略优化预留空间
        """
        def sync_generate():
            return self.thinking_engine.generate_next_query()
            
        # 在线程池中运行同步代码，避免阻塞事件循环
        return await asyncio.get_event_loop().run_in_executor(None, sync_generate)

    async def _async_search(self, query: str):
        """
        异步执行搜索，避免阻塞事件循环
        
        参数:
            query: 搜索查询文本，用于指导双路径搜索器检索相关信息
            
        返回:
            Dict: 包含搜索结果的字典，具有chunks、entities、relationships等字段
        """
        try:
            def search_wrapper():
                try:
                    # 执行实际搜索
                    return self.dual_searcher.search(query)
                except Exception as inner_e:
                    # 内层异常处理，捕获搜索器内部错误
                    self._log(f"[深度研究] 搜索器内部异常: {str(inner_e)}\n{traceback.format_exc()}")
                    raise
            
            # 在线程池中运行同步代码，避免阻塞事件循环
            result = await asyncio.get_event_loop().run_in_executor(None, search_wrapper)
            # 确保返回值是字典类型
            return result if isinstance(result, dict) else {"chunks": [], "entities": [], "relationships": []}
        except Exception as e:
            # 外层异常处理，捕获线程池或异步执行错误
            error_msg = f"[深度研究] 搜索异常: {str(e)}"
            self._log(f"{error_msg}\n{traceback.format_exc()}")
            # 返回空结果，允许上层逻辑继续执行
            return {"chunks": [], "entities": [], "relationships": []}

    async def _async_extract_info(self, search_query, prev_reasoning, kb_prompt_result):
        """
        异步提取搜索结果中的有用信息
        
        参数:
            search_query: 当前搜索查询，用于指导信息提取方向
            prev_reasoning: 之前的思考内容，提供上下文信息
            kb_prompt_result: 知识库搜索结果，包含原始文档内容和相关信息
            
        返回:
            str: 包含分析结果和有用信息的结构化文本
        """
        try:
            # 参数验证
            if not search_query or not isinstance(search_query, str):
                error_msg = "无效的搜索查询参数"
                self._log(f"[深度研究] 参数错误: {error_msg}")
                return f"**Final Information**\n参数错误: {error_msg}\n"
            
            # 格式化提示词
            try:
                extract_prompt = RELEVANT_EXTRACTION_PROMPT.format(
                    prev_reasoning=prev_reasoning or "",
                    search_query=search_query,
                    document=kb_prompt_result or ""
                )
            except KeyError as ke:
                format_error = f"提示词格式化错误: 缺少必要的占位符 {ke}"
                self._log(f"[深度研究] {format_error}")
                return f"**Final Information**\n{format_error}\n"
            
            def llm_invoke():
                try:
                    # 构建消息列表
                    messages = [
                        {"role": "system", "content": extract_prompt},
                        {"role": "user", "content": f'基于当前的搜索查询"{search_query}"和前面的推理步骤，分析每个知识来源并找出有用信息。'}
                    ]
                    
                    # 执行LLM调用
                    response = self.llm.invoke(messages)
                    
                    # 处理不同格式的响应
                    if hasattr(response, 'content'):
                        return response.content
                    elif isinstance(response, str):
                        return response
                    else:
                        return str(response)
                except Exception as inner_e:
                    # 内层异常处理，捕获LLM调用错误
                    inner_error = f"LLM调用失败: {str(inner_e)}"
                    self._log(f"[深度研究] {inner_error}\n{traceback.format_exc()}")
                    raise
            
            # 在线程池中运行同步LLM调用，避免阻塞事件循环
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, llm_invoke)
                # 确保返回的是字符串类型
                return str(result) if result else "**Final Information**\n无法提取有用信息。\n"
            except asyncio.TimeoutError:
                timeout_msg = "信息提取操作超时"
                self._log(f"[深度研究] {timeout_msg}")
                return f"**Final Information**\n{timeout_msg}\n"
                
        except Exception as e:
            # 外层异常处理，捕获所有其他错误
            error_msg = f"信息提取异常: {str(e)}"
            self._log(f"[深度研究] {error_msg}\n{traceback.format_exc()}")
            # 返回标准格式的错误响应，确保后续处理能正常进行
            return f"**Final Information**\n提取信息时发生错误，请参考原始搜索结果。\n{error_msg}\n"

    async def _async_generate_final_answer(self, query, retrieved_content, thinking):
        """
        异步生成最终答案
        
        该方法是DeepResearchTool的最终答案生成异步组件，负责在不阻塞主事件循环的情况下生成高质量的最终答案。
        它是深度研究过程的结论生成阶段，通过整合所有检索到的信息、完整思考过程和原始查询，生成全面、准确、有深度的
        最终回答。该方法通过线程池执行器将CPU密集型的LLM生成操作转换为异步模式，是整个异步深度研究工作流的完美收官。
        
        参数:
            query: 原始用户问题，作为答案生成的目标和参考，指导回答的相关性和针对性
            retrieved_content: 检索到的有用信息集合，包含多轮搜索过程中提取的关键事实、数据和知识点
            thinking: 完整思考过程文本，记录了整个分析、分解、搜索和推理过程，提供上下文支持
            
        返回:
            str: 生成的高质量最终答案，具有以下特点：
                - 直接回答用户原始问题，确保相关性和针对性
                - 整合多轮搜索和分析的所有关键信息
                - 保持逻辑连贯性和论证充分性
                - 结构清晰，层次分明，易于理解
                - 包含适当的引用和证据支持
                - 语言流畅自然，符合专业表达习惯
        
        实现思路：
        1. 定义同步生成答案的包装函数generate_wrapper，封装对底层_generate_final_answer方法的调用
        2. 获取当前asyncio事件循环，准备异步执行环境
        3. 使用asyncio.get_event_loop().run_in_executor在线程池中执行同步答案生成操作
        4. 指定None作为执行器参数，使用Python默认线程池
        5. 等待线程池中的LLM调用和答案生成完成
        6. 获取并直接返回生成的最终答案结果
        7. 确保异步操作不会影响系统的整体稳定性和响应性
        8. 实现异常处理机制，确保在生成过程中出现问题时能够优雅处理
        
        技术特点：
        - 异步实现设计，与Python的asyncio框架无缝集成，支持异步工作流
        - 线程池执行模式，将CPU密集型的LLM调用放入工作线程处理
        - 函数包装器设计，将同步答案生成函数优雅转换为异步函数
        - 与现有同步代码的完全兼容，复用成熟的_generate_final_answer核心逻辑
        - 简化的异步调用接口，隐藏复杂的线程管理和异步处理细节
        - 保持与同步方法相同的输入输出接口，确保API一致性
        - 完整的异常传递机制，确保问题能够被上层正确捕获和处理
        - 高度可复用的组件设计，适用于各种异步生成场景
        
        业务意义：
        - 作为深度研究过程的结论生成阶段，提供完整的问题解决闭环
        - 整合多轮复杂搜索和深度分析的结果，生成高质量综合性答案
        - 确保在处理复杂问题时不会阻塞系统其他功能，提供良好用户体验
        - 支持高并发场景，多个深度研究任务可以并行执行而不互相影响
        - 为异步流式交互提供必要的异步答案生成支持
        - 确保最终答案的质量和准确性，反映整个思考和搜索过程的价值
        - 支持复杂问题的全面解答，提供深入的分析和见解
        
        架构意义：
        - 展示了复杂AI系统中同步和异步代码的无缝集成模式
        - 实现了高性能异步答案生成的最佳实践
        - 提供了统一的异步接口，简化上层调用和集成
        - 支持组件化设计，将复杂功能拆分为可管理的模块
        - 为整个深度研究系统的异步化提供了关键支持
        - 体现了现代AI应用中的高效资源管理策略
        - 支持系统的水平扩展，适应更高并发需求
        
        性能优化：
        - 通过异步执行显著提高系统并发处理能力和资源利用率
        - 线程池复用，避免频繁创建和销毁线程的开销
        - 非阻塞执行模式，确保事件循环的流畅运行
        - 充分利用多核CPU资源执行并行计算
        - 优化资源分配策略，确保系统整体性能最优
        - 减少主线程负担，提高用户界面响应速度
        - 实现高效的内存使用，避免不必要的数据复制
        
        业务价值：
        - 提供高质量、全面的最终答案，满足用户深度信息需求
        - 增强系统的并发处理能力，支持更多用户同时使用
        - 确保复杂查询的响应性和用户体验，即使在处理大量信息时
        - 通过异步处理提高系统整体吞吐量和效率
        - 支持实时交互场景，提供流畅的用户体验
        - 为深度研究功能的实用性和易用性提供关键支持
        """
        try:
            def generate_wrapper():
                try:
                    return self._generate_final_answer(query, retrieved_content, thinking)
                except Exception as e:
                    # 内层异常处理：捕获_generate_final_answer中的异常
                    error_msg = f"[深度研究] 答案生成内部异常: {str(e)}"
                    self._log(f"{error_msg}\n{traceback.format_exc()}")
                    return f"**答案生成内部错误**\n在生成最终答案的核心过程中遇到问题。\n错误详情: {str(e)}\n建议尝试简化您的查询或稍后再试。"
            
            # 使用线程池执行同步的答案生成操作，避免阻塞事件循环
            return await asyncio.get_event_loop().run_in_executor(None, generate_wrapper)
        except Exception as e:
            # 外层异常处理：捕获异步执行或事件循环相关的异常
            error_msg = f"[深度研究] 最终答案异步生成异常: {str(e)}"
            self._log(f"{error_msg}\n{traceback.format_exc()}")
            # 返回标准格式的错误响应
            return f"**答案生成失败**\n在处理答案生成过程中遇到系统级问题。\n错误信息: {str(e)}\n请尝试重新查询或联系系统管理员。"
        
    def _log(self, message):
        """记录执行日志"""
        self.execution_logs.append(message)
        # print(message)  # 同时打印到控制台
    
    async def thinking_stream(self, query: str) -> AsyncGenerator[str, None]:
        """
        执行深度研究推理过程，流式返回思考内容
        
        该方法是DeepResearchTool的流式思考控制流程组件，负责以异步流式方式执行完整的多轮迭代思考和搜索过程。
        它是深度研究能力的交互式和实时反馈体现，通过异步生成器实时返回每一步思考内容，极大增强了用户体验和系统透明度。
        该方法本质上是同步thinking方法的异步流式版本，采用相似的处理逻辑，但提供了实时反馈能力。
        
        参数:
            query: 用户问题，作为深度研究和推理过程的起点和目标，要求以自然语言形式表达完整问题
                    
        返回:
            AsyncGenerator[str, None]: 流式生成的思考和答案内容，以异步生成器形式提供实时反馈，
                                   每个yield返回的是思考过程的一个片段或状态更新
        
        实现思路：
        1. 初始化执行环境，包括清空执行日志、重置关键词缓存和准备结果容器
        2. 设置思考引擎，通过ThinkingEngine.initialize_with_query初始化思考状态和上下文
        3. 使用异步方式生成初始子查询，将复杂问题分解为可管理的子问题集合
        4. 构建初始思考内容，提供问题分析框架和研究方向，并通过yield实时返回
        5. 执行多轮异步迭代搜索和分析循环（最多MAX_ROUNDS轮次）：
           - 调用_async_generate_next_query生成下一个搜索查询，基于当前思考状态和信息缺口
           - 使用_async_search执行异步双路径搜索，同时查询知识库和知识图谱
           - 调用_async_extract_info异步提取有用信息，过滤无关内容，标记关键知识点
           - 更新思考过程，整合新获取的信息，完善问题理解
           - 在每一步关键操作后使用yield返回思考进展，提供实时反馈
           - 根据搜索结果质量和思考完整性，决定是否继续搜索
        6. 当达到终止条件时（满足信息充分性或达到最大轮次），异步生成最终答案
        7. 调用_async_generate_final_answer整合所有检索到的信息和思考过程
        8. 最后yield返回最终生成的完整答案给用户
        9. 实现全面的异常处理机制，确保在出现问题时能够优雅处理并提供有意义的错误信息
        
        技术特点：
        - 基于异步生成器的流式输出设计，与Python的asyncio框架无缝集成
        - 完全异步化的操作流程，使用await处理所有潜在的阻塞操作
        - 实时反馈机制，通过yield语句在思考过程的关键点返回更新
        - 异步函数组件化设计，复用多个专门的异步辅助方法（如_async_search、_async_extract_info等）
        - 迭代式自适应搜索策略，动态调整搜索深度和方向
        - 双路径检索机制集成，同时利用知识库和知识图谱获取互补信息
        - 完整的思考过程实时展示，增强系统透明度和可解释性
        - 全面的异常处理和容错机制，确保流式输出的稳定性和可靠性
        
        业务意义：
        - 提供交互式思考体验，实时展示问题分析和解决过程，增强用户参与感
        - 显著提升系统透明度和可信度，用户可直接观察AI的思考路径和决策依据
        - 支持复杂问题的多轮迭代搜索和推理，确保答案的全面性、准确性和深度
        - 通过实时反馈机制减少用户等待焦虑，显著改善用户体验
        - 符合人类认知习惯，以循序渐进的方式展示复杂问题解决过程
        - 支持在思考过程中进行交互和干预，提高系统的实用性
        - 为最终答案提供完整的思考路径和证据支持，增强答案的可解释性
        
        用户体验价值：
        - 减少用户等待焦虑，实时展示系统正在积极工作，提高感知响应性
        - 提供思考过程的可视化和透明化，增强用户对系统能力的信任
        - 使复杂问题的解决过程变得直观可理解，降低用户认知负担
        - 允许用户在思考过程中评估回答质量和完整性，决定是否继续等待
        - 提供丰富的中间状态信息，使用户了解当前处理阶段和进展
        - 增强交互体验，使AI助手感觉更像是一个正在思考的人类专家
        - 支持认知对齐，用户可以理解AI的思考方式
        - 允许用户在必要时提前终止复杂的查询过程
        
        架构意义：
        - 展示了异步流式处理在复杂AI系统中的应用模式和最佳实践
        - 实现了同步和异步流式接口的完整支持，满足不同应用场景需求
        - 采用组件化设计，将复杂流程拆分为可复用的异步功能单元
        - 提供了与现代前端框架和交互式应用的完美集成能力
        - 支持响应式应用设计，使系统能够实时响应用户交互和查询变化
        - 为未来的功能扩展和性能优化提供了灵活的架构基础
        
        实现挑战与解决方案：
        - 异步管理多个并发搜索操作，通过asyncio的事件循环和任务调度机制避免阻塞
        - 平衡搜索深度和响应速度，采用自适应搜索策略确保良好的用户体验
        - 确保流式输出的连贯性和逻辑性，通过结构化思考内容设计实现
        - 处理网络延迟和资源限制，实现健壮的错误处理和重试机制
        - 优化内存使用，避免多轮搜索过程中资源过度消耗
        - 提供足够详细但不过于频繁的更新，平衡信息丰富度和性能开销
        """
        # 清空执行日志，准备新的思考过程
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")

        self._keywords_cache = {}

        # 初始化结果容器，用于存储检索到的信息，确保包含所有必要字段
        chunk_info = {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
        self.all_retrieved_info = []
        
        # 初始化思考引擎，设置初始查询
        try:
            self.thinking_engine.initialize_with_query(query)
        except Exception as e:
            init_error_msg = f"\n**思考引擎初始化失败**: {str(e)}\n"
            self._log(f"引擎初始化错误: {init_error_msg}\n{traceback.format_exc()}")
            yield init_error_msg
            return

        # 使用QueryGenerator生成初始子查询，将复杂问题分解
        yield "\n**正在分析您的问题，生成研究方向**...\n"
        
        # 异步生成子查询，避免阻塞事件循环
        try:
            def generate_sub_queries():
                return self.query_generator.generate_sub_queries(query)
            
            initial_sub_queries = await asyncio.get_event_loop().run_in_executor(None, generate_sub_queries)
            self._log(f"\n[深度研究] 生成了{len(initial_sub_queries)}个初始子查询")
        except Exception as e:
            query_error_msg = f"\n**子查询生成失败**: {str(e)}\n"
            self._log(f"子查询生成错误: {query_error_msg}\n{traceback.format_exc()}")
            yield query_error_msg
            return
        
        think = ""
        
        # 将初始思考添加到思考过程，提供问题分析和研究方向
        initial_thinking = f"我需要回答问题：{query}\n\n"
        initial_thinking += "为了全面解答这个问题，我将从以下几个方面进行研究：\n"
        for i, sub_q in enumerate(initial_sub_queries, 1):
            initial_thinking += f"{i}. {sub_q}\n"
        initial_thinking += "\n让我逐步进行搜索和分析。"
        
        self.thinking_engine.add_reasoning_step(initial_thinking)
        think += initial_thinking
        
        # 分组返回初始思考内容
        yield initial_thinking
        
        # 迭代思考过程，执行多轮搜索和分析
        for iteration in range(self.max_iterations):
            self._log(f"\n[深度研究] 开始第{iteration + 1}轮迭代")
            
            # 检查是否达到最大迭代次数，防止无限循环
            if iteration >= self.max_iterations:
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\n搜索次数已达上限。将基于已有信息生成答案。\n{END_SEARCH_RESULT}\n"
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(summary_think)
                think += self.thinking_engine.remove_result_tags(summary_think)
                yield "\n**搜索次数已达上限，将基于已有信息生成答案**\n"
                break
                
            # 优化的迭代控制：如果已收集到足够信息或没有新的搜索方向，提前结束
            if self.all_retrieved_info and iteration > 0:
                # 检查是否有新的查询方向
                followup_queries = self.query_generator.generate_followup_queries(
                    query, self.all_retrieved_info
                )
                
                # 如果没有新的查询方向，且已有足够信息，结束搜索
                if not followup_queries and len(self.all_retrieved_info) >= 3:
                    enough_info_msg = "\n**已收集到充分信息且无新的搜索方向，开始生成最终答案**\n"
                    self._log(enough_info_msg)
                    yield enough_info_msg
                    break

            # 更新消息历史，请求继续推理
            self.thinking_engine.update_continue_message()
            
            # 确定当前迭代要处理的查询
            queries_to_process = []
            
            # 确保chunk_info在每次迭代开始时正确初始化
            if 'chunk_info' not in locals():
                chunk_info = {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
            
            if iteration == 0 and initial_sub_queries:
                # 第一轮迭代使用预先生成的子查询
                queries_to_process = initial_sub_queries[:2]  # 限制首轮使用的子查询数量
                query_think = "开始根据分解的子问题进行搜索"
                yield "\n**开始按照研究方向进行搜索**...\n"
            else:
                # 非首轮，使用思考引擎生成下一步查询
                result = self.thinking_engine.generate_next_query()
                
                # 处理生成结果，根据不同状态采取不同策略
                if result["status"] == "empty":
                    self._log("\n[深度研究] 生成的思考内容为空")
                    # 尝试使用QueryGenerator的多假设生成功能寻找新方向
                    hypotheses = QueryGenerator.generate_multiple_hypotheses(query, self.llm)
                    if hypotheses:
                        self._log(f"\n[深度研究] 生成了{len(hypotheses)}个新假设，尝试从新角度探索")
                        queries_to_process = hypotheses
                        query_think = "尝试从新的角度思考这个问题:\n" + "\n".join([f"- {h}" for h in hypotheses])
                        self.thinking_engine.add_reasoning_step(query_think)
                        think += query_think
                        yield "\n**尝试从新的角度探索问题**:\n" + query_think
                    else:
                        continue
                elif result["status"] == "error":
                    self._log(f"\n[深度研究] 生成查询出错: {result.get('error', '未知错误')}")
                    break
                elif result["status"] == "answer_ready":
                    self._log("\n[深度研究] AI认为已有足够信息生成答案")
                    yield "\n**已收集到足够的信息，准备生成最终答案**\n"
                    break
                else:
                    # 获取生成的思考内容
                    query_think = result["content"]
                    think += self.thinking_engine.remove_query_tags(query_think)
                    
                    # 获取搜索查询
                    queries_to_process = result["queries"]
            
            # 如果当前迭代没有查询，且我们已经检索到一些信息，尝试生成跟进查询
            if not queries_to_process and self.all_retrieved_info:
                followup_queries = self.query_generator.generate_followup_queries(
                    query, self.all_retrieved_info
                )
                
                if followup_queries:
                    self._log(f"\n[深度研究] 生成了{len(followup_queries)}个跟进查询")
                    queries_to_process = followup_queries
                    followup_think = "\n考虑到已发现的信息，我需要进一步探索以下问题:\n"
                    for i, fq in enumerate(followup_queries, 1):
                        followup_think += f"{i}. {fq}\n"
                    self.thinking_engine.add_reasoning_step(followup_think)
                    think += followup_think
                    yield followup_think
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries_to_process:
                if not self.all_retrieved_info and iteration == 0:
                    # 如果第一轮没检索到任何信息，强制使用原始查询
                    queries_to_process = [query]
                    self._log("\n[深度研究] 没有检索到信息，使用原始查询")
                else:
                    # 已有信息，结束迭代
                    end_msg = "\n\n**没有发现新的查询角度，基于已有信息生成回答**...\n\n"
                    self._log(end_msg)
                    yield end_msg
                    break
            
            # 处理每个搜索查询
            for search_query in queries_to_process:
                try:
                    # 搜索开始通知
                    search_start_msg = f"\n**正在搜索: {search_query}**\n"
                    self._log(search_start_msg)
                    yield search_start_msg
                        
                    # 检查是否已执行过相同查询
                    if self.thinking_engine.has_executed_query(search_query):
                        # 使用统一的格式化消息处理重复查询
                        summary_think = f"\n{BEGIN_SEARCH_RESULT}\n已搜索过该查询。请参考前面的结果。\n{END_SEARCH_RESULT}\n"
                        dupe_msg = f"\n**已搜索过类似查询，跳过重复执行**\n"
                        
                        # 添加到推理历史并更新思考过程
                        self.thinking_engine.add_reasoning_step(summary_think)
                        self.thinking_engine.add_human_message(summary_think)
                        think += self.thinking_engine.remove_result_tags(summary_think)
                        
                        # 记录日志并返回用户友好消息
                        self._log(dupe_msg)
                        yield dupe_msg
                        continue
                        
                    # 记录已执行查询并更新消息历史
                    self.thinking_engine.add_executed_query(search_query)
                    self.thinking_engine.add_ai_message(f"{search_query}")
                    think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                        
                    # 让事件循环有机会执行其他任务
                    await asyncio.sleep(0.01)
                        
                    # 执行实际搜索（使用异步搜索避免阻塞）
                    yield "\n**正在查询知识库**...\n"
                    try:
                        kbinfos = await self._async_search(search_query)
                        if kbinfos is None:
                            kbinfos = {}
                    except Exception as e:
                        search_error_msg = f"\n**搜索操作失败: {str(e)}**\n"
                        self._log(f"搜索错误: {search_error_msg}\n{traceback.format_exc()}")
                        yield search_error_msg
                        continue

                    # 检查搜索结果是否为空，更全面地验证结果
                    has_results = False
                    if kbinfos:
                        # 检查所有可能包含有效信息的字段
                        for key in ["chunks", "entities", "relationships", "doc_aggs"]:
                            if key in kbinfos and kbinfos[key]:
                                # 确保结果列表不为空且包含有效内容
                                if isinstance(kbinfos[key], list) and len(kbinfos[key]) > 0:
                                    # 进一步验证chunks是否包含有意义的文本
                                    if key == "chunks":
                                        for chunk in kbinfos[key]:
                                            # 检查chunk是否为字典且包含有意义的内容
                                            if isinstance(chunk, dict):
                                                for content_key in ["content", "text", "page_content"]:
                                                    if content_key in chunk and chunk[content_key] and len(chunk[content_key].strip()) > 10:
                                                        has_results = True
                                                        break
                                            # 如果chunk是字符串且长度足够
                                            elif isinstance(chunk, str) and len(chunk.strip()) > 10:
                                                has_results = True
                                            if has_results:
                                                break
                                    else:
                                        # 对于entities、relationships等，只要有非空列表就算有结果
                                        has_results = True
                                    if has_results:
                                        break
                         
                    if not has_results:
                        # 无搜索结果处理
                        no_result_msg = f"\n**没有找到与{search_query}相关的有效信息，尝试其他角度**...\n"
                        self._log(no_result_msg)
                        yield no_result_msg
                         
                        # 添加到推理历史
                        formatted_msg = f"\n{BEGIN_SEARCH_RESULT}\n没有找到与'{search_query}'相关的有效信息。系统将尝试使用不同的关键词或角度进行搜索。\n{END_SEARCH_RESULT}\n"
                        self.thinking_engine.add_reasoning_step(formatted_msg)
                        self.thinking_engine.add_human_message(formatted_msg)
                        think += self.thinking_engine.remove_result_tags(formatted_msg)
                        continue
                        
                    # 正常处理有结果的情况
                    truncated_prev_reasoning = self.thinking_engine.prepare_truncated_reasoning()
                        
                    # 合并块信息，确保chunk_info已初始化
                    if 'chunk_info' not in locals() or chunk_info is None:
                        chunk_info = {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
                    
                    # 正确合并搜索结果
                    try:
                        # 如果dual_searcher有_merge_results方法，使用它合并结果
                        if hasattr(self.dual_searcher, '_merge_results'):
                            chunk_info = self.dual_searcher._merge_results(chunk_info, kbinfos)
                        else:
                            # 否则手动合并
                            for key in ["chunks", "entities", "relationships", "doc_aggs"]:
                                if key in kbinfos and kbinfos[key]:
                                    if key not in chunk_info:
                                        chunk_info[key] = []
                                    # 避免重复添加相同内容
                                    for item in kbinfos[key]:
                                        if item not in chunk_info[key]:
                                            chunk_info[key].append(item)
                    except Exception as merge_e:
                        merge_error_msg = f"\n**结果合并失败: {str(merge_e)}**\n"
                        self._log(f"结果合并错误: {merge_error_msg}\n{traceback.format_exc()}")
                        # 使用原始kbinfos作为备选
                        chunk_info = kbinfos
                        
                    # 构建提取相关信息的提示
                    kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                    
                    # 告知用户正在分析结果
                    yield "\n**正在分析搜索结果**...\n"
                    # 使用异步LLM提取有用信息
                    try:
                        summary_think = await self._async_extract_info(search_query, truncated_prev_reasoning, kb_prompt_result)
                        
                        # 改进的有用信息判断逻辑
                        has_useful_info = False
                        if summary_think:
                            # 检查是否包含最终信息标记
                            if "**Final Information**" in summary_think:
                                # 提取最终信息部分
                                final_info_part = summary_think.split("**Final Information**", 1)[1].strip() if len(summary_think.split("**Final Information**")) > 1 else ""
                                # 只有当信息不为空且不包含明显的无信息标记时才视为有用
                                has_useful_info = (final_info_part and 
                                                  "No helpful information found" not in final_info_part.lower() and
                                                  "无法提取有用信息" not in final_info_part and
                                                  len(final_info_part.strip()) > 20)  # 确保有足够长度的内容
                        
                        # 额外的安全检查：如果summary_think格式正确但内容过少，也视为无用信息
                        if has_useful_info and summary_think:
                            info_content = summary_think.split("**Final Information**")[-1].strip()
                            if len(info_content) < 20:
                                has_useful_info = False
                                self._log(f"[深度研究] 信息过于简短，被标记为无用: {info_content}")
                        
                    except Exception as e:
                        extract_error_msg = f"\n**信息提取失败: {str(e)}**\n"
                        self._log(f"信息提取错误: {extract_error_msg}\n{traceback.format_exc()}")
                        yield extract_error_msg
                        has_useful_info = False
                        continue
                except Exception as e:
                    # 异常处理，确保单个查询失败不会影响整个搜索过程
                    error_msg = f"\n**处理查询 '{search_query}' 时发生错误: {str(e)}**\n"
                    self._log(error_msg)
                    yield error_msg
                    continue
                    
                if has_useful_info:
                    useful_info = summary_think.split("**Final Information**")[1].strip()
                    self.all_retrieved_info.append(useful_info)
                    info_msg = f"发现有用信息: {useful_info[:100]}..."
                    self._log(info_msg)
                    yield "\n**找到相关信息！**\n"
                else:
                    no_useful_msg = "**\n未从搜索结果中发现特别有价值的信息**\n"
                    self._log(no_useful_msg)
                    yield no_useful_msg
                    
                # 更新推理历史
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(f"\n{BEGIN_SEARCH_RESULT}{summary_think}{END_SEARCH_RESULT}\n")
                
                # 获取去除标签后的思考内容（避免重复调用）
                processed_think = self.thinking_engine.remove_result_tags(summary_think)
                think += processed_think

                # 分组返回处理后的思考内容，提供更好的流式体验
                result_parts = re.split(r'(\n\n)', processed_think)
                result_buffer = ""
                
                for part in result_parts:
                    result_buffer += part
                    # 当积累了足够内容或遇到段落分隔时yield
                    if len(result_buffer) >= 80 or "\n\n" in result_buffer:
                        yield result_buffer
                        result_buffer = ""
                        # 短暂暂停让事件循环有机会处理其他任务
                        await asyncio.sleep(0.01)
                        
                # 处理剩余内容
                if result_buffer:
                    yield result_buffer
            
            # 在每轮迭代结束后，评估是否需要继续搜索
            if iteration > 0 and self.all_retrieved_info:
                # 异步判断是否需要继续生成查询
                def check_gap_needed():
                    return len(self.query_generator.generate_followup_queries(query, self.all_retrieved_info)) > 0
                
                gap_needed = await asyncio.get_event_loop().run_in_executor(None, check_gap_needed)
                if not gap_needed:
                    reflection_msg = "\n**已收集到足够的信息，可以开始整合分析了**\n"
                    self._log(reflection_msg)
                    yield reflection_msg
                    self.thinking_engine.add_reasoning_step("\n已收集到足够的信息，可以开始整合分析了。")
                    think += "\n已收集到足够的信息，可以开始整合分析了。"
                    break
        
        # 生成最终答案
        try:
            # 确保至少执行了一次搜索
            if not self.thinking_engine.executed_search_queries:
                # 无搜索执行时的处理
                no_search_msg = f"\n**无法找到与{query}相关的信息，尝试给出基础回答**...\n"
                yield no_search_msg
                
                # 返回结构化结果，确保所有字段都有安全的默认值
                safe_chunk_info = locals().get('chunk_info') or {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
                result = {
                    "thinking_process": think or "",
                    "answer": f"抱歉，我无法回答关于'{query}'的问题，因为没有找到相关信息。",
                    "reference": safe_chunk_info,
                    "retrieved_info": [],
                    "execution_logs": getattr(self, 'execution_logs', []),
                }
                
                # 向用户发送最终答案
                yield {"answer": result["answer"], "thinking": think}
                return result
            
            # 生成最终答案
            yield "\n**正在根据所有收集的信息生成最终答案**...\n"
            
            # 使用检索到的信息生成答案
            try:
                retrieved_content = "\n\n".join(getattr(self, 'all_retrieved_info', []))
                final_answer = await self._async_generate_final_answer(query, retrieved_content, think)
            except Exception as e:
                answer_gen_error = f"\n**最终答案生成失败: {str(e)}**\n"
                self._log(f"答案生成错误: {answer_gen_error}\n{traceback.format_exc()}")
                yield answer_gen_error
                final_answer = f"生成答案时发生错误: {str(e)}，请稍后重试。"

            # 构建结构化结果，确保所有字段都有安全的默认值
            safe_chunk_info = locals().get('chunk_info') or {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
            result = {
                "thinking_process": think or "",
                "answer": final_answer or "无法生成答案，请稍后重试。",
                "reference": safe_chunk_info,
                "retrieved_info": getattr(self, 'all_retrieved_info', []),
                "execution_logs": getattr(self, 'execution_logs', []),
            }
            
            # 向用户发送最终答案（一次性发送，因为前端会替换整个响应）
            yield {"answer": final_answer, "thinking": think}
            
            return result
        except Exception as e:
            # 最终答案生成异常处理
            error_msg = f"\n**生成最终答案时发生错误: {str(e)}**\n"
            self._log(f"最终答案生成错误: {error_msg}\n{traceback.format_exc()}")
            yield error_msg
            
            # 返回错误结果，确保所有字段都有安全的默认值
            safe_chunk_info = locals().get('chunk_info') or {"chunks": [], "entities": [], "relationships": [], "doc_aggs": []}
            return {
                "thinking_process": think or "",
                "answer": f"生成答案时发生错误: {str(e)}，请稍后重试。",
                "reference": safe_chunk_info,
                "retrieved_info": getattr(self, 'all_retrieved_info', []),
                "execution_logs": getattr(self, 'execution_logs', []),
            }
    
    async def search(self, query_input: Any) -> str:
        """
        执行深度研究搜索
        
        该方法是DeepResearchTool的主要同步执行入口，负责协调整个深度研究流程，从输入处理、缓存检查、思考过程执行到最终答案生成和验证。
        它实现了一个异步接口，但内部实际上是通过调用同步思考方法来完成深度分析的，同时提供了全面的错误处理和性能监控。
        
        参数:
            query_input: 搜索查询或包含查询的字典，支持多种输入格式以增加灵活性
                    
        返回:
            str: 搜索结果，包含最终答案和可能的引用信息，格式化为便于使用的文本
        
        实现思路：
        1. 记录开始时间，用于性能监控和执行时间分析
        2. 解析输入查询，支持字符串和字典两种格式，增强灵活性
        3. 构建缓存键，考虑查询内容，确保高效缓存查找
        4. 检查缓存，避免对相同查询的重复计算，显著提高性能
        5. 执行思考过程，调用self.thinking方法获取深入分析结果
        6. 提取分析结果中的答案和参考信息
        7. 格式化参考资料，从chunk_info中提取相关文档ID，构建引用列表
        8. 添加引用信息到答案中，增强答案的可信度和可追溯性
        9. 调用AnswerValidator验证答案质量，确保结果的可靠性和准确性
       10. 只缓存通过验证的高质量答案，避免缓存低质量结果
       11. 记录总执行时间，更新性能指标，便于后续优化
       12. 实现全面的异常处理，捕获并记录所有可能的错误
       13. 在出错时返回友好的错误信息，同时打印详细错误日志用于调试
        
        技术特点：
        - 灵活的输入格式处理，支持字符串和结构化字典输入
        - 智能缓存机制，基于查询内容的高效缓存键生成
        - 集成答案质量验证流程，确保输出结果的可靠性
        - 全面的异常处理和错误恢复机制，提高系统稳定性
        - 详细的性能监控和日志记录，便于性能优化和问题诊断
        - 引用信息生成和添加，增强答案的可信度和可追溯性
        - 异步接口设计，支持非阻塞操作模式
        
        业务意义：
        - 提供深度研究能力的标准化入口，简化调用流程
        - 确保输出高质量、可靠的答案，提升用户体验
        - 通过缓存机制优化查询性能，显著提高系统响应速度
        - 为复杂问题提供全面的分析和解决方案，满足专业需求
        - 支持引用信息的添加，增强答案的可信度和学术价值
        - 实现完整的性能监控，便于系统优化和资源管理
        - 提供统一的错误处理，确保系统稳定性和用户体验
        
        处理流程：
        - 输入处理 → 缓存检查 → 思考过程 → 结果提取 → 引用添加 → 质量验证 → 缓存结果 → 返回答案
        
        注意事项：
        - 尽管方法声明为async，但实际思考过程是同步执行的
        - 答案验证失败的结果不会被缓存，避免影响后续查询质量
        - 引用信息限制最多显示5个文档ID，避免结果过长
        - 异常处理机制确保即使在错误情况下也能返回有意义的响应
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入，支持字符串和字典两种格式
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存，避免重复计算
        cache_key = f"deep:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self._log(f"\n[深度搜索] 缓存命中，返回缓存结果")
            return cached_result
        
        try:
            # 执行思考过程，获取深入分析结果
            self._log(f"\n[深度搜索] 开始执行思考过程")
            result = self.thinking(query)
            answer = result["answer"]
            chunk_info = result.get("reference", {})
            
            # 格式化参考资料，提取相关文档ID
            references = []
            if "doc_aggs" in chunk_info:
                for doc in chunk_info["doc_aggs"]:
                    doc_id = doc.get("doc_id", "")
                    if doc_id and doc_id not in references:
                        references.append(doc_id)
            
            # 添加引用信息，增强答案可信度
            if references and "{'data': {'Chunks':" not in answer:
                ref_str = ", ".join([f"'{ref}'" for ref in references[:5]])
                answer += f"\n\n{'data': {'Chunks':[{ref_str}] }}"
            
            # 验证答案质量，确保结果可靠性
            validation_results = self.validator.validate(query, answer)
            if validation_results["passed"]:
                self._log(f"\n[深度搜索] 答案验证通过，缓存结果")
                self.cache_manager.set(cache_key, answer)
            else:
                self._log(f"\n[深度搜索] 答案验证失败，不缓存")
            
            # 记录总时间，更新性能指标
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
    
    def get_thinking_stream_tool(self) -> BaseTool:
        """获取流式思考过程工具"""
        class DeepStreamThinkingTool(BaseTool):
            name : str = "deep_thinking_stream"
            description : str = "流式深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> AsyncGenerator:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 返回流式生成器
                return self.thinking_stream(tk_query)
            
            async def _arun(self_tool, query: Any) -> AsyncGenerator:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 返回流式生成器
                return await self.thinking_stream(tk_query)
        
        return DeepStreamThinkingTool()
    
    async def _fix_answer(self, query, answer):
        """
        尝试修复低质量答案
        
        参数:
            query: 原始问题
            answer: 可能存在问题的答案
            
        返回:
            str: 修复后的高质量答案
            
        实现思路：
        1. 构建修复提示，包含原始问题和有问题的答案
        2. 定义LLM调用包装函数
        3. 使用线程池执行LLM调用，避免阻塞事件循环
        4. 返回修复后的答案
        
        技术特点：
        - 异步实现
        - 避免事件循环阻塞
        - 结构化提示设计
        - 针对常见答案质量问题的修复策略
        
        业务意义：
        - 提高最终答案的质量
        - 修正可能的错误和不完整信息
        - 确保答案的清晰度和准确性
        - 优化用户体验
        """
        fix_prompt = f"""
        原问题是: {query}
        
        生成的答案可能存在问题: {answer}
        
        请提供一个修正后、质量更高的答案，更好地回应用户的问题。
        确保答案:
        1. 直接回答问题核心
        2. 删除不必要的重复内容
        3. 去除表示不确定的语言
        4. 结构清晰，重点突出
        """
        
        def llm_fix():
            response = self.llm.invoke(fix_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        # 在线程池中运行同步LLM调用  
        return await asyncio.get_event_loop().run_in_executor(None, llm_fix)
    
    def close(self):
        """关闭资源
        
        实现思路：
        1. 调用父类的close方法，释放基础资源
        2. 关闭各个复用的工具资源
        3. 清理引用，避免内存泄漏
        
        设计特点：
        - 层次化资源管理
        - 完善的清理流程
        - 防止资源泄漏
        - 确保系统稳定性
        
        业务意义：
        - 正确释放系统资源
        - 避免内存泄漏
        - 确保长时间运行时的稳定性
        - 支持优雅关闭
        """
        # 调用父类方法
        super().close()
        
        # 关闭复用的工具资源
        if hasattr(self, 'hybrid_tool'):
            self.hybrid_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'local_tool'):
            self.local_tool.close()

    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 深度研究搜索工具实例
        
        实现思路：
        1. 定义内部类DeepResearchRetrievalTool，继承自BaseTool
        2. 设置工具名称和描述，说明其功能和适用场景
        3. 实现同步运行方法_run，调用self.search方法执行搜索
        4. 实现异步运行方法_arun，但当前不支持异步执行
        5. 创建并返回工具实例
        
        设计特点：
        - 使用内部类定义工具，封装实现细节
        - 遵循BaseTool接口规范，便于集成
        - 同步和异步方法都有定义，保证接口一致性
        - 清晰的工具描述，帮助用户理解工具功能
        
        业务意义：
        - 提供标准接口，便于集成到各种系统中
        - 实现搜索功能的封装，简化调用流程
        - 为复杂问题提供专业的搜索解决方案
        - 支持统一的工具管理和调度
        """
        class DeepResearchRetrievalTool(BaseTool):
            name : str = "deep_research"
            description : str = "深度研究工具：通过多轮推理和搜索解决复杂问题，尤其适用于需要深入分析的查询。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeepResearchRetrievalTool()
    
    def get_thinking_tool(self) -> BaseTool:
        """
        获取思考过程可见的工具版本
        
        返回:
            BaseTool: 深度思考工具实例，展示完整思考过程
        
        实现思路：
        1. 定义内部类DeepThinkingTool，继承自BaseTool
        2. 设置工具名称和描述，突出其思考过程可见的特点
        3. 实现同步运行方法_run，调用self.thinking方法执行深度思考
        4. 实现异步运行方法_arun，但当前不支持异步执行
        5. 创建并返回工具实例
        
        设计特点：
        - 使用内部类定义工具，封装实现细节
        - 支持返回完整的思考过程和答案
        - 提供结构化的结果，包含思考过程、答案和引用信息
        - 保持接口一致性，与其他工具兼容
        
        业务意义：
        - 提供透明的思考过程，增强用户信任
        - 适用于需要理解分析过程的教育场景
        - 支持复杂问题的深入分析和解释
        - 便于调试和优化思考过程
        """
        class DeepThinkingTool(BaseTool):
            name : str = "deep_thinking"
            description : str = "深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> Dict:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 执行思考过程
                return self.thinking(tk_query)
            
            def _arun(self_tool, query: Any) -> Dict:
                raise NotImplementedError("异步执行未实现")
        
        return DeepThinkingTool()

    def thinking(self, query: str):
        """
        执行深度研究推理过程
        
        该方法是DeepResearchTool的核心控制流程组件，负责协调整个深度研究过程的执行。它实现了复杂问题的多轮
        分析、分解、搜索、推理和综合，是深度研究能力的集中体现。该方法通过模拟人类专家的思考过程，将复杂问题
        分解为可管理的子问题，通过迭代搜索和推理逐步构建对问题的全面理解，并最终生成高质量答案。
        
        参数:
            query: 用户问题，作为深度研究和推理过程的起点和目标，要求以自然语言形式表达完整问题
                    
        返回:
            Dict: 包含完整思考过程和最终答案的结构化字典，具有以下主要字段：
            - thinking_process: 完整思考过程文本，记录了整个分析、分解、搜索和推理路径
            - answer: 最终生成的答案，基于所有检索信息和完整思考过程
            - reference: 参考资料信息，包含所有检索到的文档块、来源和引用
            - retrieved_info: 检索到的重要信息列表，从搜索结果中提取的关键事实和数据点
            - execution_logs: 执行日志，记录整个思考过程的关键步骤、时间点和系统状态
            - steps: 思考过程的结构化步骤列表，便于可视化和分析
            - query_history: 查询历史，记录所有生成和执行的搜索查询
        
        实现思路：
        1. 初始化执行环境，包括清空执行日志、重置关键词缓存和准备结果容器
        2. 设置思考引擎，通过ThinkingEngine.initialize_with_query初始化思考状态和上下文
        3. 使用QueryGenerator分解复杂问题，生成初始子查询集合
        4. 构建初始思考内容，提供问题分析框架和研究方向
        5. 执行多轮迭代搜索和分析循环（最多MAX_ROUNDS轮次）：
           - 调用ThinkingEngine生成下一个搜索查询，基于当前思考状态和信息缺口
           - 使用DualPathSearcher执行双路径搜索，同时查询知识库和知识图谱
           - 调用LLM提取有用信息，过滤无关内容，标记关键知识点
           - 更新思考过程，整合新获取的信息，完善问题理解
           - 根据搜索结果质量和思考完整性，决定是否继续搜索
        6. 当达到终止条件时（满足信息充分性或达到最大轮次），生成最终答案
        7. 整合所有检索到的信息和完整思考过程，调用_generate_final_answer生成高质量答案
        8. 构建并返回完整结果字典，包含所有必要的元数据、参考和过程信息
        
        技术特点：
        - 结构化思考过程设计，通过ThinkingEngine实现复杂推理链的构建和管理
        - 迭代式自适应搜索策略，根据当前信息状态动态调整搜索方向和深度
        - 双路径检索机制集成，同时利用知识库和知识图谱获取互补的信息视角
        - 智能查询生成系统，自动构建高质量搜索查询以填补信息缺口
        - 多轮搜索结果的智能整合与去重，确保信息的连贯性和一致性
        - 完整的思考过程追踪和记录，支持结果的可解释性和透明度
        - 丰富的边界情况处理和异常恢复机制，确保系统稳定性
        - 与多个组件的紧密协作，包括ThinkingEngine、QueryGenerator和DualSearcher
        - 自适应终止条件判断，平衡搜索深度和计算效率
        
        业务意义：
        - 提供深度分析和复杂推理能力，解决传统RAG难以直接回答的复杂问题
        - 为用户提供透明、可解释的思考过程，增强答案可信度和决策支持能力
        - 支持多轮迭代搜索和推理，确保答案的全面性、准确性和深度
        - 适应各种复杂查询场景，包括多跳推理、概念扩展和综合分析任务
        - 提供完整的参考信息和来源追踪，支持结果验证和知识扩展应用
        - 通过结构化思考和迭代分析提高问题解决效率和答案质量
        - 符合人类专家解决复杂问题的认知模式，提供更自然、可理解的结果
        - 支持知识发现和概念关联，在回答问题的同时提供额外价值洞察
        
        架构意义：
        - 作为DeepResearchTool的核心控制中心，协调各组件的协同工作
        - 实现了复杂问题解决的端到端流程，从问题输入到最终答案输出
        - 展示了基于大语言模型的高级推理系统的设计模式和实现方法
        - 体现了多组件协作和模块化设计的优势，便于维护和扩展
        - 支持思考过程的完整记录和复用，为后续优化提供数据基础
        - 提供了标准化的接口和输出格式，便于与其他系统集成
        - 实现了同步思考模式，为异步的thinking_stream方法提供基础实现
        
        处理流程：
        - 问题接收 → 初始化环境 → 问题分解 → 初始子查询生成 → 构建思考框架 → 多轮迭代搜索 → 信息提取与整合
        → 继续搜索决策 → 生成最终答案 → 构建结果字典 → 返回完整结果
        
        核心价值：
        - 模拟人类专家解决复杂问题的思维方式和工作流程
        - 整合多种信息源和检索方式，提供全面、多角度的分析视角
        - 确保答案的准确性、全面性和深度，超越简单检索的局限
        - 增强AI系统的透明度、可解释性和可信度
        - 通过结构化思考和迭代分析提高复杂问题解决的效率和质量
        - 为用户提供不仅是答案，还有完整的思考过程和参考资料
        """
        # 清空执行日志，准备新的思考过程
        self.execution_logs = []
        self._log(f"\n[深度研究] 开始处理查询: {query}")
    
        # 重置关键词缓存，确保分析的新鲜性
        self._keywords_cache = {}
        
        # 初始化结果容器，存储检索到的信息
        chunk_info = {"chunks": [], "doc_aggs": []}
        self.all_retrieved_info = []
        
        # 初始化思考引擎，设置初始查询
        self.thinking_engine.initialize_with_query(query)
    
        # 使用QueryGenerator生成初始子查询，分解复杂问题
        initial_sub_queries = self.query_generator.generate_sub_queries(query)
        self._log(f"\n[深度研究] 生成了{len(initial_sub_queries)}个初始子查询")
        
        think = ""
        
        # 将初始思考添加到思考过程，提供问题分析和研究方向
        initial_thinking = f"我需要回答问题：{query}\n\n"
        initial_thinking += "为了全面解答这个问题，我将从以下几个方面进行研究：\n"
        for i, sub_q in enumerate(initial_sub_queries, 1):
            initial_thinking += f"{i}. {sub_q}\n"
        initial_thinking += "\n让我逐步进行搜索和分析。"
        
        self.thinking_engine.add_reasoning_step(initial_thinking)
        think += initial_thinking
        
        # 分组返回初始思考内容
        yield initial_thinking
        
        # 迭代思考过程，执行多轮搜索和分析
        for iteration in range(self.max_iterations):
            self._log(f"\n[深度研究] 开始第{iteration + 1}轮迭代")
            
            # 检查是否达到最大迭代次数，防止无限循环
            if iteration >= self.max_iterations - 1:
                summary_think = f"\n{BEGIN_SEARCH_RESULT}\n搜索次数已达上限。不允许继续搜索。\n{END_SEARCH_RESULT}\n"
                self.thinking_engine.add_reasoning_step(summary_think)
                self.thinking_engine.add_human_message(summary_think)
                think += self.thinking_engine.remove_result_tags(summary_think)
                yield "\n**搜索次数已达上限，将基于已有信息生成答案**\n"
                break

            # 更新消息历史，请求继续推理
            self.thinking_engine.update_continue_message()
            
            # 确定当前迭代要处理的查询
            queries_to_process = []
            
            if iteration == 0 and initial_sub_queries:
                # 第一轮迭代使用预先生成的子查询
                queries_to_process = initial_sub_queries[:2]  # 限制首轮使用的子查询数量
                query_think = "开始根据分解的子问题进行搜索"
                yield "\n**开始按照研究方向进行搜索**...\n"
            else:
                # 非首轮，使用思考引擎生成下一步查询
                result = self.thinking_engine.generate_next_query()
                
                # 处理生成结果，根据不同状态采取不同策略
                if result["status"] == "empty":
                    self._log("\n[深度研究] 生成的思考内容为空")
                    # 尝试使用QueryGenerator的多假设生成功能寻找新方向
                    hypotheses = QueryGenerator.generate_multiple_hypotheses(query, self.llm)
                    if hypotheses:
                        self._log(f"\n[深度研究] 生成了{len(hypotheses)}个新假设，尝试从新角度探索")
                        queries_to_process = hypotheses
                        query_think = "尝试从新的角度思考这个问题:\n" + "\n".join([f"- {h}" for h in hypotheses])
                        self.thinking_engine.add_reasoning_step(query_think)
                        think += query_think
                        yield "\n**尝试从新的角度探索问题**:\n" + query_think
                    else:
                        continue
                elif result["status"] == "error":
                    self._log(f"\n[深度研究] 生成查询出错: {result.get('error', '未知错误')}")
                    break
                elif result["status"] == "answer_ready":
                    self._log("\n[深度研究] AI认为已有足够信息生成答案")
                    yield "\n**已收集到足够的信息，准备生成最终答案**\n"
                    break
                else:
                    # 获取生成的思考内容
                    query_think = result["content"]
                    think += self.thinking_engine.remove_query_tags(query_think)
                    
                    # 获取搜索查询
                    queries_to_process = result["queries"]
            
            # 如果当前迭代没有查询，且我们已经检索到一些信息，尝试生成跟进查询
            if not queries_to_process and self.all_retrieved_info:
                followup_queries = self.query_generator.generate_followup_queries(
                    query, self.all_retrieved_info
                )
                
                if followup_queries:
                    self._log(f"\n[深度研究] 生成了{len(followup_queries)}个跟进查询")
                    queries_to_process = followup_queries
                    followup_think = "\n考虑到已发现的信息，我需要进一步探索以下问题:\n"
                    for i, fq in enumerate(followup_queries, 1):
                        followup_think += f"{i}. {fq}\n"
                    self.thinking_engine.add_reasoning_step(followup_think)
                    yield followup_think
            
            # 如果没有生成搜索查询但不是第一轮，考虑结束
            if not queries_to_process:
                if not self.all_retrieved_info and iteration == 0:
                    # 如果第一轮没检索到任何信息，强制使用原始查询
                    queries_to_process = [query]
                    self._log("\n[深度研究] 没有检索到信息，使用原始查询")
                else:
                    # 已有信息，结束迭代
                    end_msg = "\n\n**没有发现新的查询角度，基于已有信息生成回答**...\n\n"
                    self._log(end_msg)
                    yield end_msg
                    break
            
            # 处理每个搜索查询
            for search_query in queries_to_process:
                search_start_msg = f"\n**正在搜索: {search_query}**\n"
                self._log(search_start_msg)
                yield search_start_msg
                    
                # 检查是否已执行过相同查询
                if self.thinking_engine.has_executed_query(search_query):
                    dupe_msg = f"\n**已搜索过类似查询，跳过重复执行**\n"
                    self._log(dupe_msg)
                    yield dupe_msg
                    continue
                    
                # 记录已执行查询
                self.thinking_engine.add_executed_query(search_query)
                    
                # 将搜索查询添加到消息历史
                self.thinking_engine.add_ai_message(f"{search_query}")
                think += f"\n\n> {iteration + 1}. {search_query}\n\n"
                    
                # 让事件循环有机会执行其他任务
                await asyncio.sleep(0)
                    
                # 执行实际搜索
                yield "\n**正在查询知识库**...\n"
                try:
                    kbinfos = await self._async_search(search_query)
                    
                    # 检查搜索结果是否为空
                    has_results = kbinfos and (
                        kbinfos.get("chunks", []) or 
                        kbinfos.get("entities", []) or 
                        kbinfos.get("relationships", [])
                    )
                    
                    if not has_results:
                        no_result_msg = f"\n**没有找到与{search_query}相关的信息，尝试其他角度**...\n"
                        self._log(no_result_msg)
                        yield no_result_msg
                        self.thinking_engine.add_reasoning_step(f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n")
                        self.thinking_engine.add_human_message(f"\n没有找到与'{search_query}'相关的信息。请尝试使用不同的关键词进行搜索。\n")
                        think += no_result_msg
                        continue
                    
                    # 正常处理有结果的情况
                    truncated_prev_reasoning = self.thinking_engine.prepare_truncated_reasoning()
                    
                    # 合并块信息
                    chunk_info = self.dual_searcher._merge_results(chunk_info, kbinfos)
                except Exception as e:
                    error_msg = f"\n**搜索过程中发生错误**: {str(e)}\n"
                    self._log(f"搜索失败: {error_msg}\n{traceback.format_exc()}")
                    yield error_msg
                    self.thinking_engine.add_reasoning_step(f"\n搜索'{search_query}'时发生错误: {str(e)}\n")
                    self.thinking_engine.add_human_message(f"\n搜索'{search_query}'时发生错误: {str(e)}\n")
                    think += error_msg
                    continue
                    
                # 构建提取相关信息的提示
                kb_prompt_result = "\n".join(kb_prompt(kbinfos, 4096))
                    
                # 告知用户正在分析结果
                yield "\n**正在分析搜索结果**...\n"
                
                # 使用异步LLM提取有用信息
                try:
                    summary_think = await self._async_extract_info(search_query, truncated_prev_reasoning, kb_prompt_result)
                    
                    # 保存重要信息
                    has_useful_info = summary_think and (
                        "**Final Information**" in summary_think and 
                        "No helpful information found" not in summary_think
                    )
                    
                    if has_useful_info:
                        useful_info = summary_think.split("**Final Information**")[1].strip()
                        self.all_retrieved_info.append(useful_info)
                        info_msg = f"发现有用信息: {useful_info[:100]}..."
                        self._log(info_msg)
                        yield "\n**找到相关信息！**\n"
                    else:
                        no_useful_msg = "**\n未从搜索结果中发现特别有价值的信息**\n"
                        self._log(no_useful_msg)
                        yield no_useful_msg
                except Exception as e:
                    extract_error_msg = f"\n**信息提取过程中发生错误**: {str(e)}\n"
                    self._log(f"信息提取失败: {extract_error_msg}\n{traceback.format_exc()}")
                    yield extract_error_msg
                    self.thinking_engine.add_reasoning_step(f"\n提取信息时发生错误: {str(e)}\n")
                    self.thinking_engine.add_human_message(f"\n提取信息时发生错误: {str(e)}\n")
                    think += extract_error_msg
                    
                # 更新推理历史
                try:
                    self.thinking_engine.add_reasoning_step(summary_think)
                    self.thinking_engine.add_human_message(summary_think)
                    think += self.thinking_engine.remove_result_tags(summary_think)
                    
                    # 分组返回处理后的思考内容
                    try:
                        result_parts = re.split(r'(\n\n)', self.thinking_engine.remove_result_tags(summary_think))
                        result_buffer = ""
                        
                        for part in result_parts:
                            result_buffer += part
                            if len(result_buffer) >= 80 or "\n\n" in result_buffer:
                                yield result_buffer
                                result_buffer = ""
                                await asyncio.sleep(0.01)
                        
                        if result_buffer:
                            yield result_buffer
                    except Exception as e:
                        format_error_msg = f"\n**思考内容格式化失败**: {str(e)}\n"
                        self._log(f"格式化错误: {format_error_msg}\n{traceback.format_exc()}")
                        yield format_error_msg
                except Exception as e:
                    history_error_msg = f"\n**推理历史更新失败**: {str(e)}\n"
                    self._log(f"历史更新错误: {history_error_msg}\n{traceback.format_exc()}")
                    yield history_error_msg
            
            # 在每轮迭代结束后，评估是否需要继续搜索
            if iteration > 0 and self.all_retrieved_info:
                # 异步判断是否需要继续生成查询
                def check_gap_needed():
                    return len(self.query_generator.generate_followup_queries(query, self.all_retrieved_info)) > 0
                
                gap_needed = await asyncio.get_event_loop().run_in_executor(None, check_gap_needed)
                
                if not gap_needed:
                    reflection_msg = "\n**已收集到足够的信息，可以开始整合分析了**\n"
                    self._log(reflection_msg)
                    yield reflection_msg
                    self.thinking_engine.add_reasoning_step("\n已收集到足够的信息，可以开始整合分析了。")
                    think += "\n已收集到足够的信息，可以开始整合分析了。"
                    break
        
        # 确保至少执行了一次搜索
        if not self.thinking_engine.executed_search_queries:
            no_search_msg = f"\n**无法找到与{query}相关的信息，尝试给出基础回答**...\n"
            yield no_search_msg
            return
        
        # 生成最终答案
        yield "\n**正在根据所有收集的信息生成最终答案**...\n"
        
        # 使用检索到的信息生成答案
        retrieved_content = "\n\n".join(self.all_retrieved_info)
        final_answer = await self._async_generate_final_answer(query, retrieved_content, think)
        
        # 向用户发送最终答案（一次性发送，因为前端会替换整个响应）
        yield {"answer": final_answer, "thinking": think}
    
    async def search(self, query_input: Any) -> str:
        """
        执行深度研究搜索
        
        该方法是DeepResearchTool的另一个实现版本，提供异步接口执行深度研究搜索。
        它负责协调整个深度研究流程，包括输入处理、缓存检查、思考过程执行、结果生成和质量验证。
        该方法在处理复杂查询时提供了更灵活的异步执行方式，同时确保结果的质量和可靠性。
        
        参数:
            query_input: 搜索查询或包含查询的字典，支持多种输入格式以增加灵活性和适应性
                    
        返回:
            str: 搜索结果，包含高质量的最终答案和必要的引用信息，格式化为结构化文本
        
        实现思路：
        1. 记录开始时间，用于性能监控和执行时间分析
        2. 解析输入查询，支持字符串和字典两种格式，增强使用灵活性
        3. 构建唯一缓存键，确保高效的缓存查找和命中
        4. 检查缓存，避免对相同查询的重复计算，显著提高响应速度
        5. 执行完整的思考过程，通过self.thinking方法进行深度分析和推理
        6. 从思考结果中提取答案内容和参考信息
        7. 格式化参考资料列表，从chunk_info中提取相关文档ID和元数据
        8. 将引用信息添加到答案中，增强结果的可信度和可追溯性
        9. 调用AnswerValidator验证答案质量，确保输出结果的可靠性
        10. 只缓存通过验证的高质量答案，维护系统输出质量
        11. 记录总执行时间，更新性能监控指标
        12. 实现全面的异常处理，确保系统稳定性
        13. 在出错时返回友好的错误信息，同时提供详细日志用于调试
        
        技术特点：
        - 异步接口设计，支持非阻塞操作模式，优化并发性能
        - 灵活的输入格式处理，适应不同的调用场景
        - 智能缓存机制，基于查询内容生成高效缓存键
        - 集成严格的答案质量验证流程，确保输出可靠性
        - 全面的异常检测和错误恢复机制，提高系统稳定性
        - 详细的性能监控和日志记录，便于系统优化
        - 引用信息生成和管理，增强答案可追溯性
        - 结果格式化和标准化，确保输出一致性
        
        业务意义：
        - 提供高级搜索能力的统一接口，简化复杂查询的处理
        - 确保输出高质量、可靠的答案，满足专业用户需求
        - 通过缓存机制显著提高系统响应速度，优化用户体验
        - 为复杂问题提供深度分析和全面解决方案
        - 支持引用信息添加，增强答案的学术价值和可信度
        - 实现完整的质量控制流程，保证系统输出质量
        - 提供统一的错误处理机制，确保系统稳定性
        - 支持各种应用场景，从简单查询到复杂研究任务
        
        处理流程：
        - 输入解析 → 缓存检查 → 深度思考 → 结果提取 → 引用生成 → 质量验证 → 缓存结果 → 返回答案
        
        注意事项：
        - 方法提供异步接口，适合集成到异步应用框架中
        - 答案验证机制确保只有高质量结果被缓存和返回
        - 引用信息会被格式化为结构化形式，便于用户理解
        - 完善的异常处理确保系统在各种情况下都能稳定运行
        - 性能监控数据可用于后续系统调优和资源分配
        
        业务意义：
        - 提供深度研究能力的主要入口
        - 确保高质量、可靠的答案输出
        - 优化查询性能，提高系统响应速度
        - 为复杂问题提供全面解决方案
        - 支持引用信息的添加，增强答案可信度
        """
        overall_start = time.time()
        
        # 记录开始搜索
        self._log(f"\n[深度搜索] 开始处理查询...")
        
        # 解析输入，支持字符串和字典两种格式
        if isinstance(query_input, dict) and "query" in query_input:
            query = query_input["query"]
        else:
            query = str(query_input)
        
        self._log(f"\n[深度搜索] 解析后的查询: {query}")
        
        # 检查缓存，避免重复计算
        cache_key = f"deep:{query}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self._log(f"\n[深度搜索] 缓存命中，返回缓存结果")
            return cached_result
        
        try:
            # 执行思考过程，获取深入分析结果
            self._log(f"\n[深度搜索] 开始执行思考过程")
            result = self.thinking(query)
            answer = result["answer"]
            chunk_info = result.get("reference", {})
            
            # 格式化参考资料，提取相关文档ID
            references = []
            if "doc_aggs" in chunk_info:
                for doc in chunk_info["doc_aggs"]:
                    doc_id = doc.get("doc_id", "")
                    if doc_id and doc_id not in references:
                        references.append(doc_id)
            
            # 添加引用信息，增强答案可信度
            if references and "{'data': {'Chunks':" not in answer:
                ref_str = ", ".join([f"'{ref}'" for ref in references[:5]])
                answer += f"\n\n{'data': {'Chunks':[{ref_str}] }} }"
            
            # 验证答案质量，确保结果可靠性
            validation_results = self.validator.validate(query, answer)
            if validation_results["passed"]:
                self._log(f"\n[深度搜索] 答案验证通过，缓存结果")
                self.cache_manager.set(cache_key, answer)
            else:
                self._log(f"\n[深度搜索] 答案验证失败，不缓存")
            
            # 记录总时间，更新性能指标
            total_time = time.time() - overall_start
            self._log(f"\n[深度搜索] 完成，耗时 {total_time:.2f}秒")
            self.performance_metrics["total_time"] = total_time
            
            return answer
                
        except Exception as e:
            error_msg = f"深度研究过程中出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"抱歉，处理您的问题时遇到了错误: {str(e)}"
    
    def get_thinking_stream_tool(self) -> BaseTool:
        """获取流式思考过程工具"""
        class DeepStreamThinkingTool(BaseTool):
            name : str = "deep_thinking_stream"
            description : str = "流式深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> AsyncGenerator:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 返回流式生成器
                return self.thinking_stream(tk_query)
            
            async def _arun(self_tool, query: Any) -> AsyncGenerator:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 返回流式生成器
                return await self.thinking_stream(tk_query)
        
        return DeepStreamThinkingTool()
    
    async def _fix_answer(self, query, answer):
        """
        尝试修复低质量答案
        
        参数:
            query: 原始问题
            answer: 可能存在问题的答案
            
        返回:
            str: 修复后的高质量答案
            
        实现思路：
        1. 构建修复提示，包含原始问题和有问题的答案
        2. 定义LLM调用包装函数
        3. 使用线程池执行LLM调用，避免阻塞事件循环
        4. 返回修复后的答案
        
        技术特点：
        - 异步实现
        - 避免事件循环阻塞
        - 结构化提示设计
        - 针对常见答案质量问题的修复策略
        
        业务意义：
        - 提高最终答案的质量
        - 修正可能的错误和不完整信息
        - 确保答案的清晰度和准确性
        - 优化用户体验
        """
        fix_prompt = f"""
        原问题是: {query}
        
        生成的答案可能存在问题: {answer}
        
        请提供一个修正后、质量更高的答案，更好地回应用户的问题。
        确保答案:
        1. 直接回答问题核心
        2. 删除不必要的重复内容
        3. 去除表示不确定的语言
        4. 结构清晰，重点突出
        """
        
        def llm_fix():
            response = self.llm.invoke(fix_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        
        # 在线程池中运行同步LLM调用  
        return await asyncio.get_event_loop().run_in_executor(None, llm_fix)
    
    def close(self):
        """关闭资源
        
        实现思路：
        1. 调用父类的close方法，释放基础资源
        2. 关闭各个复用的工具资源
        3. 清理引用，避免内存泄漏
        
        设计特点：
        - 层次化资源管理
        - 完善的清理流程
        - 防止资源泄漏
        - 确保系统稳定性
        
        业务意义：
        - 正确释放系统资源
        - 避免内存泄漏
        - 确保长时间运行时的稳定性
        - 支持优雅关闭
        """
        # 调用父类方法
        super().close()
        
        # 关闭复用的工具资源
        if hasattr(self, 'hybrid_tool'):
            self.hybrid_tool.close()
        if hasattr(self, 'global_tool'):
            self.global_tool.close()
        if hasattr(self, 'local_tool'):
            self.local_tool.close()

    def get_tool(self) -> BaseTool:
        """
        获取搜索工具
        
        返回:
            BaseTool: 深度研究搜索工具实例
        
        实现思路：
        1. 定义内部类DeepResearchRetrievalTool，继承自BaseTool
        2. 设置工具名称和描述，说明其功能和适用场景
        3. 实现同步运行方法_run，调用self.search方法执行搜索
        4. 实现异步运行方法_arun，但当前不支持异步执行
        5. 创建并返回工具实例
        
        设计特点：
        - 使用内部类定义工具，封装实现细节
        - 遵循BaseTool接口规范，便于集成
        - 同步和异步方法都有定义，保证接口一致性
        - 清晰的工具描述，帮助用户理解工具功能
        
        业务意义：
        - 提供标准接口，便于集成到各种系统中
        - 实现搜索功能的封装，简化调用流程
        - 为复杂问题提供专业的搜索解决方案
        - 支持统一的工具管理和调度
        """
        class DeepResearchRetrievalTool(BaseTool):
            name : str = "deep_research"
            description : str = "深度研究工具：通过多轮推理和搜索解决复杂问题，尤其适用于需要深入分析的查询。"
            
            def _run(self_tool, query: Any) -> str:
                return self.search(query)
            
            def _arun(self_tool, query: Any) -> str:
                raise NotImplementedError("异步执行未实现")
        
        return DeepResearchRetrievalTool()
    
    def get_thinking_tool(self) -> BaseTool:
        """
        获取思考过程可见的工具版本
        
        返回:
            BaseTool: 深度思考工具实例，展示完整思考过程
        
        实现思路：
        1. 定义内部类DeepThinkingTool，继承自BaseTool
        2. 设置工具名称和描述，突出其思考过程可见的特点
        3. 实现同步运行方法_run，调用self.thinking方法执行深度思考
        4. 实现异步运行方法_arun，但当前不支持异步执行
        5. 创建并返回工具实例
        
        设计特点：
        - 使用内部类定义工具，封装实现细节
        - 支持返回完整的思考过程和答案
        - 提供结构化的结果，包含思考过程、答案和引用信息
        - 保持接口一致性，与其他工具兼容
        
        业务意义：
        - 提供透明的思考过程，增强用户信任
        - 适用于需要理解分析过程的教育场景
        - 支持复杂问题的深入分析和解释
        - 便于调试和优化思考过程
        """
        class DeepThinkingTool(BaseTool):
            name : str = "deep_thinking"
            description : str = "深度思考工具：显示完整思考过程的深度研究，适用于需要查看推理步骤的情况。"
            
            def _run(self_tool, query: Any) -> Dict:
                # 解析输入
                if isinstance(query, dict) and "query" in query:
                    tk_query = query["query"]
                else:
                    tk_query = str(query)
                
                # 执行思考过程
                return self.thinking(tk_query)
            
            def _arun(self_tool, query: Any) -> Dict:
                raise NotImplementedError("异步执行未实现")
        
        return DeepThinkingTool()