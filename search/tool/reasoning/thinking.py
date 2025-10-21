import re
import json
import time
from typing import List, Dict, Any
import logging
import traceback
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from search.tool.reasoning.nlp import extract_between
from config.reasoning_prompts import BEGIN_SEARCH_QUERY, BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, REASON_PROMPT, END_SEARCH_QUERY


class ThinkingEngine:
    """
    思考引擎类：负责管理多轮迭代的思考过程
    提供思考历史管理和转换功能
    """
    
    def __init__(self, llm):
        """
        初始化思考引擎
        
        该方法是Graph-RAG系统中思考引擎的初始化入口，负责创建和配置思考引擎的核心组件。
        思考引擎是整个系统的推理核心，管理多轮迭代的思考过程、假设生成、验证和分支推理。
        它维护了完整的推理状态，包括思考历史、假设列表、验证链和推理树结构。
        
        参数:
            llm: 大语言模型实例，用于生成思考内容、假设分析和验证逻辑
        
        实现思路：
        1. 保存传入的大语言模型实例，作为思考生成和分析的核心引擎
        2. 初始化各种状态变量，包括推理步骤列表、消息历史、已执行查询记录等
        3. 创建数据结构用于存储假设和验证链，支持假设驱动的推理过程
        4. 初始化推理树结构，支持多分支推理和反事实分析
        5. 设置默认的主推理分支
        
        技术特点：
        - 状态管理：维护完整的思考过程状态，支持多轮推理
        - 数据结构：使用列表、字典等数据结构高效组织推理内容
        - 可扩展性：支持分支推理和多路径探索
        - 模块化设计：与语言模型解耦，便于替换不同的LLM后端
        - 状态隔离：不同思考过程的数据独立存储
        
        业务意义：
        - 提供系统的核心推理能力，支持深度分析和复杂问题解决
        - 管理多轮迭代的思考过程，实现逐步深入的分析
        - 支持假设生成和验证，实现基于证据的推理
        - 提供分支推理能力，支持多角度思考和反事实分析
        - 为最终答案生成提供结构化的推理依据
        """
        self.llm = llm
        self.all_reasoning_steps = []
        self.msg_history = []
        self.executed_search_queries = []
        self.hypotheses = []       # 存储假设
        self.verification_chain = [] # 验证步骤
        self.reasoning_tree = {}   # 推理树结构
        self.current_branch = "main" # 当前推理分支
    
    def initialize_with_query(self, query: str):
        """
        使用初始查询初始化思考历史
        
        该方法是思考引擎的入口点，用于为新的查询创建一个全新的思考环境。它重置所有推理状态，
        初始化必要的数据结构，并将用户查询添加到消息历史中，为后续的思考过程做好准备。
        这是Graph-RAG系统中开始推理前的关键准备步骤。
        
        参数:
            query: 用户的原始问题或查询字符串，作为推理过程的起点
        
        实现思路：
        1. 清空所有历史状态，包括推理步骤、假设、验证链等
        2. 初始化消息历史，将用户查询格式化后添加为第一条消息
        3. 重置已执行查询记录，避免重复查询
        4. 初始化空的假设列表和验证链，为后续的假设驱动推理做准备
        5. 创建新的推理树结构，初始化主分支
        6. 设置当前工作分支为主分支
        
        技术特点：
        - 完全重置：确保新查询不会受之前推理过程的影响
        - 结构化初始化：为不同类型的数据维护独立的数据结构
        - 统一格式：使用标准化格式存储用户查询
        - 清晰的状态管理：维护推理树和当前分支信息
        - 简洁高效：代码简洁但功能完备，确保系统响应速度
        
        业务意义：
        - 为每个新查询创建干净的推理环境，确保推理的独立性
        - 准备好进行多轮思考和迭代推理的数据结构
        - 支持基于查询的假设生成和验证过程
        - 为后续的思考过程提供统一的起点
        - 确保系统在处理多个查询时不会发生状态混淆
        """
        self.all_reasoning_steps = []
        self.msg_history = [{"role": "user", "content": f'问题:"{query}"\n'}]
        self.executed_search_queries = []
        self.hypotheses = []
        self.verification_chain = []
        self.reasoning_tree = {"main": []} # 初始化主分支
        self.current_branch = "main"
    
    def generate_initial_thinking(self):
        """
        生成初步思考过程
        
        该方法负责基于用户问题生成初步的思考分析，是Graph-RAG系统中推理过程的起点。
        它使用LLM对问题进行深入分析，识别问题核心、所需信息和可能的思考方向，为后续的
        假设生成和验证奠定基础。这一步骤确保系统能够有条理地开始解决复杂问题。
        
        返回:
            str: LLM生成的初步思考内容，包含问题分析和可能的思考方向
        
        实现思路：
        1. 构建系统提示，引导LLM从问题核心、所需信息和可能思考方向三个方面进行分析
        2. 将系统提示与用户原始问题组合成提示序列
        3. 调用LLM生成初步思考内容
        4. 处理LLM响应，提取思考内容
        5. 将思考内容添加到推理步骤记录中
        6. 返回思考内容供后续处理使用
        
        技术特点：
        - 结构化提示：使用清晰的提示结构引导LLM进行深入思考
        - 通用性设计：适用于各种类型的复杂问题
        - 响应处理：优雅处理不同格式的LLM响应
        - 状态更新：自动更新推理步骤记录
        - 简洁高效：代码简洁但功能完备
        
        业务意义：
        - 为整个推理过程提供初始框架和方向
        - 帮助系统深入理解用户问题的本质
        - 识别解决问题所需的关键信息
        - 探索多种可能的思考路径
        - 为后续的假设生成和验证提供基础
        """
        prompt = """
        请对问题进行深入思考，考虑以下方面:
        1. 问题的核心是什么？
        2. 需要哪些信息来回答这个问题？
        3. 有哪些可能的思考方向？
        
        请提供你的推理过程，不需要立即给出答案。
        """
        
        response = self.llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": self.msg_history[0]["content"]}
        ])
        
        content = response.content if hasattr(response, 'content') else str(response)
        self.add_reasoning_step(content)
        
        return content
    
    def generate_hypotheses(self, initial_thinking):
        """
        生成多个可能的假设
        
        该方法负责基于初步思考分析生成3-5个合理的假设，是Graph-RAG系统中假设驱动推理
        的关键组件。它引导LLM从不同角度思考问题，生成可验证的假设，为后续的验证和分析
        提供基础。这种多假设方法有助于系统避免确认偏误，全面探索问题空间。
        
        参数:
            initial_thinking: 初步思考内容，包含问题分析和可能的思考方向
            
        返回:
            List[Dict]: 假设列表，每个假设包含假设内容(hypothesis)和理由(reasoning)
        
        实现思路：
        1. 构建提示，要求LLM基于初步思考生成3-5个合理假设
        2. 指定假设的三个关键特性：解释已有观察、可进一步验证、相互有所不同
        3. 要求LLM以JSON格式返回假设列表
        4. 调用LLM生成假设
        5. 处理响应，尝试提取JSON格式的假设列表
        6. 如果JSON解析失败，则使用正则表达式作为备用方法提取假设
        7. 将提取的假设添加到推理步骤记录中
        
        技术特点：
        - 结构化假设生成：明确定义假设的数量和质量要求
        - 双重解析策略：优先使用JSON解析，失败时使用正则表达式
        - 错误处理：优雅处理各种格式问题，确保系统稳定性
        - 状态管理：自动更新假设列表和推理步骤记录
        - 可验证性：确保生成的假设可以通过后续步骤验证
        
        业务意义：
        - 实现多角度思考，避免单一思路的局限性
        - 提供可验证的推理起点，增强推理的科学性
        - 帮助系统探索问题的多种可能性
        - 为后续的假设验证和分析提供明确目标
        - 增强最终答案的全面性和可靠性
        """
        prompt = f"""
        基于初步思考：
        {initial_thinking}
        
        生成3-5个合理的假设，每个假设应该：
        1. 解释已有的观察结果
        2. 可以被进一步验证
        3. 相互之间有所不同
        
        以JSON格式返回：
        [
            {{"hypothesis": "...", "reasoning": "..."}},
            ...
        ]
        """
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析假设
        try:            
            # 寻找JSON部分
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                hypotheses = json.loads(json_match.group(0))
                self.hypotheses = hypotheses
                
                # 添加假设到推理步骤
                hypothesis_step = "生成的假设：\n"
                for i, hyp in enumerate(hypotheses):
                    hypothesis_step += f"假设 {i+1}: {hyp['hypothesis']}\n"
                    hypothesis_step += f"理由: {hyp['reasoning']}\n\n"
                
                self.add_reasoning_step(hypothesis_step)
                return hypotheses
            else:
                # 使用正则表达式提取假设
                return self._extract_hypotheses_fallback(content)
        except Exception as e:
            print(f"解析假设失败: {e}")
            return self._extract_hypotheses_fallback(content)
    
    def _extract_hypotheses_fallback(self, content):
        """
        当JSON解析失败时，使用正则表达式提取假设
        
        参数:
            content: 包含假设的文本
            
        返回:
            List[Dict]: 假设列表
        """        
        hypotheses = []
        
        # 查找假设模式
        hypothesis_pattern = re.compile(r'假设\s*\d+[:：]?\s*(.*?)(?=假设\s*\d+[:：]?|$)', re.DOTALL)
        matches = hypothesis_pattern.findall(content)
        
        for i, match in enumerate(matches):
            # 尝试分离假设和理由
            parts = re.split(r'理由[:：]', match, 1)
            
            if len(parts) == 2:
                hypothesis, reasoning = parts
            else:
                hypothesis = parts[0]
                reasoning = ""
                
            hypotheses.append({
                "hypothesis": hypothesis.strip(),
                "reasoning": reasoning.strip()
            })
        
        # 如果没有找到假设，创建一个默认假设
        if not hypotheses:
            hypotheses = [{
                "hypothesis": "问题可能需要更多背景信息",
                "reasoning": "初步思考中没有明确的答案方向"
            }]
            
        # 添加假设到思考引擎状态
        self.hypotheses = hypotheses
        
        # 添加假设到推理步骤
        hypothesis_step = "生成的假设：\n"
        for i, hyp in enumerate(hypotheses):
            hypothesis_step += f"假设 {i+1}: {hyp['hypothesis']}\n"
            hypothesis_step += f"理由: {hyp['reasoning']}\n\n"
        
        self.add_reasoning_step(hypothesis_step)
        
        return hypotheses
    
    def verify_hypothesis(self, hypothesis):
        """
        验证假设
        
        该方法负责对单个假设进行深入的验证分析，是Graph-RAG系统中假设驱动推理的核心环节。
        它使用LLM对假设进行评估，检查其是否符合已知信息、是否有证据支持、是否存在逻辑漏洞，
        以及是否需要更多信息进行验证。这种系统性的验证确保了系统的推理过程基于坚实的证据。
        
        参数:
            hypothesis: 要验证的假设，包含hypothesis和reasoning的字典
            
        返回:
            Dict: 验证结果字典，包含原始假设、验证分析和验证状态
        
        实现思路：
        1. 构建提示，要求LLM从四个关键方面评估假设：与已知信息的一致性、支持/反对的证据、逻辑完整性、所需额外信息
        2. 调用LLM生成详细的验证分析
        3. 创建验证结果字典，包含原始假设、验证分析文本
        4. 调用_assess_verification_status方法评估验证状态
        5. 将验证结果添加到验证链中
        6. 将验证过程记录到推理步骤中
        7. 返回完整的验证结果
        
        技术特点：
        - 多维度评估：从多个角度全面评估假设
        - 状态分类：将验证结果分为支持、反对或不确定三种状态
        - 状态跟踪：维护完整的验证链，记录所有假设的验证状态
        - 过程记录：详细记录验证过程，确保推理可追溯性
        - 结构化输出：提供标准化的验证结果格式
        
        业务意义：
        - 对每个假设进行系统性评估，确保推理的严谨性
        - 识别支持或反对假设的关键证据
        - 发现假设中的逻辑漏洞或不足
        - 确定哪些假设更可能是正确的
        - 为最终结论提供可靠的推理基础
        """
        prompt = f"""
        请验证以下假设：
        
        假设: {hypothesis['hypothesis']}
        理由: {hypothesis['reasoning']}
        
        请考虑:
        1. 这个假设是否符合已知信息?
        2. 有哪些证据支持或反对这个假设?
        3. 这个假设是否有逻辑漏洞?
        4. 是否需要更多信息来验证这个假设?
        
        请提供详细的验证分析。
        """
        
        response = self.llm.invoke(prompt)
        verification = response.content if hasattr(response, 'content') else str(response)
        
        # 创建验证结果
        verification_result = {
            "hypothesis": hypothesis['hypothesis'],
            "verification": verification,
            "status": self._assess_verification_status(verification)
        }
        
        # 添加到验证链
        self.verification_chain.append(verification_result)
        
        # 添加到推理步骤
        self.add_reasoning_step(f"验证假设: {hypothesis['hypothesis']}\n\n{verification}")
        
        return verification_result
    
    def _assess_verification_status(self, verification):
        """
        评估验证状态
        
        参数:
            verification: 验证文本
            
        返回:
            str: 验证状态 (supported/rejected/uncertain)
        """
        # 分析验证文本，确定假设状态
        prompt = f"""
        根据以下验证分析，该假设的状态是什么?
        
        验证分析:
        {verification}
        
        请从以下三个选项中选择一个:
        - supported: 假设被证据支持
        - rejected: 假设被证据反驳
        - uncertain: 证据不足，无法确定
        
        只返回一个状态单词，不要包含解释。
        """
        
        try:
            response = self.llm.invoke(prompt)
            status = response.content if hasattr(response, 'content') else str(response)
            
            # 清理并标准化状态
            status = status.strip().lower()
            
            if "support" in status:
                return "supported"
            elif "reject" in status:
                return "rejected"
            else:
                return "uncertain"
        except:
            # 默认不确定
            return "uncertain"
    
    def think_deeper(self, query, context=None):
        """
        启动深度思考模式
        
        该方法是思考引擎的核心功能，负责执行完整的深度思考流程，包括初始化思考、生成
        初步分析、提出假设、验证假设、更新思考和整合所有思考过程。这是Graph-RAG系统中
        进行复杂问题解决和深度分析的主要入口点，实现了假设驱动的结构化推理过程。
        
        参数:
            query: 用户问题，需要进行深度思考的原始查询
            context: 上下文信息，可选参数，提供额外的背景信息用于思考
            
        返回:
            str: 深度思考结果，包含完整的思考过程和结论
        
        实现思路：
        1. 使用用户问题初始化思考历史，准备全新的推理环境
        2. 如果提供了上下文信息，则将其添加到推理步骤中
        3. 生成初步思考，分析问题核心和可能的思考方向
        4. 基于初步思考生成多个合理假设
        5. 对每个假设进行系统性验证，评估其可靠性
        6. 基于验证结果更新思考，整合被支持的假设
        7. 将所有思考过程（初步分析、假设生成、验证、更新思考）整合为完整报告
        8. 返回整合后的完整思考过程
        
        技术特点：
        - 全流程管理：协调整个思考过程的各个环节
        - 模块化设计：将复杂思考分解为可管理的子步骤
        - 假设驱动：基于多假设方法进行推理，避免单一思路局限
        - 状态维护：在整个思考过程中维护完整的推理状态
        - 结构化输出：生成格式化的完整思考报告
        
        业务意义：
        - 提供系统化的深度分析能力，支持复杂问题解决
        - 实现假设驱动的科学推理方法
        - 提供可解释的思考过程，增强系统透明度
        - 整合多源信息进行综合分析
        - 为最终答案提供全面、严谨的推理基础
        """
        # 初始化思考历史
        self.initialize_with_query(query)
        
        # 添加上下文信息（如果有）
        if context:
            self.add_reasoning_step(f"考虑以下背景信息：\n{context}")
        
        # 生成初步思考
        initial_thinking = self.generate_initial_thinking()
        
        # 提出假设
        hypotheses = self.generate_hypotheses(initial_thinking)
        
        # 对每个假设进行验证
        verifications = []
        for hypothesis in hypotheses:
            verification = self.verify_hypothesis(hypothesis)
            verifications.append(verification)
            
        # 基于验证结果更新思考
        updated_thinking = self.update_thinking_based_on_verification(verifications)
        
        # 整合所有思考过程
        final_thinking = self.integrate_thinking_process(
            initial_thinking,
            hypotheses,
            verifications,
            updated_thinking
        )
        
        return final_thinking

    def update_thinking_based_on_verification(self, verifications):
        """
        基于验证结果更新思考
        
        该方法负责根据假设验证的结果，更新整体的思考过程。它汇总所有验证结果，
        区分被支持、被拒绝和不确定的假设，并使用LLM生成一个更新后的、连贯的思考过程，
        这是Graph-RAG系统中实现迭代式推理的关键步骤。
        
        参数:
            verifications: 验证结果列表，每个元素包含假设、验证状态和验证内容
            
        返回:
            str: 更新后的思考过程，综合考虑所有验证结果
        
        实现思路：
        1. 创建验证结果汇总字符串，用于整合所有验证信息
        2. 初始化三个列表，分别用于存储被支持、被拒绝和不确定的假设
        3. 遍历所有验证结果，根据状态将假设分类到不同列表中
        4. 在汇总字符串中添加各类假设的统计信息和内容
        5. 将汇总信息添加到推理步骤记录中
        6. 构建提示，要求LLM基于验证结果生成更新后的思考
        7. 调用LLM生成更新后的思考内容
        8. 将更新后的思考添加到推理步骤记录中
        9. 返回更新后的思考内容
        
        技术特点：
        - 分类汇总：系统地分类和整合验证结果
        - 结构化提示：使用清晰的提示结构引导LLM生成连贯的思考更新
        - 状态管理：自动更新推理步骤记录
        - 灵活适应：能够处理多种验证结果组合
        - 结果可视化：生成易于理解的验证结果统计
        
        业务意义：
        - 实现假设驱动的迭代推理过程
        - 基于证据更新系统思考，增强推理可靠性
        - 提供验证结果的清晰汇总和分析
        - 支持系统化的假设筛选和整合
        - 为最终答案生成提供基于验证的思考基础
        """
        # 汇总验证结果
        verification_summary = "验证结果汇总:\n"
        
        supported = []
        rejected = []
        uncertain = []
        
        for v in verifications:
            if v["status"] == "supported":
                supported.append(v["hypothesis"])
            elif v["status"] == "rejected":
                rejected.append(v["hypothesis"])
            else:
                uncertain.append(v["hypothesis"])
                
        verification_summary += f"- 被支持的假设: {len(supported)}\n"
        if supported:
            verification_summary += "  " + "\n  ".join(supported) + "\n"
            
        verification_summary += f"- 被拒绝的假设: {len(rejected)}\n"
        if rejected:
            verification_summary += "  " + "\n  ".join(rejected) + "\n"
            
        verification_summary += f"- 不确定的假设: {len(uncertain)}\n"
        if uncertain:
            verification_summary += "  " + "\n  ".join(uncertain) + "\n"
        
        # 添加汇总到推理步骤
        self.add_reasoning_step(verification_summary)
        
        # 基于验证结果更新思考
        prompt = f"""
        基于以下验证结果汇总，请更新你的思考过程:
        
        {verification_summary}
        
        请综合考虑所有被支持的假设，并解释为什么拒绝其他假设。
        提供一个更新后的、连贯的思考过程。
        """
        
        response = self.llm.invoke(prompt)
        updated_thinking = response.content if hasattr(response, 'content') else str(response)
        
        # 添加到推理步骤
        self.add_reasoning_step(f"更新后的思考:\n\n{updated_thinking}")
        
        return updated_thinking
    
    def integrate_thinking_process(self, initial_thinking, hypotheses, verifications, updated_thinking):
        """
        整合所有思考过程
        
        该方法负责将思考过程的所有阶段整合为一个连贯、结构化的报告，包括初步分析、
        假设生成、假设验证和最终思考。它将复杂的多步骤推理过程以清晰易读的方式呈现，
        是Graph-RAG系统提供推理透明度和可解释性的重要组件。
        
        参数:
            initial_thinking: 初步思考内容，包含问题分析和可能的思考方向
            hypotheses: 假设列表，每个假设包含假设内容和理由
            verifications: 验证结果列表，包含假设验证的状态和内容
            updated_thinking: 更新后的思考过程，基于验证结果生成
            
        返回:
            str: 整合后的完整思考过程报告，格式化为结构化文档
        
        实现思路：
        1. 创建思考过程报告的基本结构，设置报告标题
        2. 添加初步分析部分，包含原始问题分析内容
        3. 添加假设生成部分，为每个假设编号并显示其内容和理由
        4. 添加假设验证部分，为每个验证结果添加状态标记（支持、拒绝、不确定）
        5. 添加最终思考部分，包含基于验证结果更新后的思考
        6. 返回完整的结构化报告
        
        技术特点：
        - 结构化报告：生成格式良好、层次清晰的思考过程报告
        - 状态可视化：使用符号标记直观显示假设验证状态
        - 层次化组织：将思考过程分为多个逻辑部分
        - 完整性保证：包含思考过程的所有关键阶段
        - 可读性优化：使用适当的标题、缩进和格式提升可读性
        
        业务意义：
        - 提供完整、可追踪的推理过程，增强系统透明度
        - 帮助用户理解系统的思考路径和决策依据
        - 为最终答案提供详细的推理基础和背景
        - 支持对系统思考过程的审核和评估
        - 展示假设驱动推理的完整方法论
        """
        # 构建完整的思考过程
        integrated_thinking = "# 思考过程\n\n"
        integrated_thinking += "## 初步分析\n\n"
        integrated_thinking += initial_thinking + "\n\n"
        
        integrated_thinking += "## 假设生成\n\n"
        for i, hyp in enumerate(hypotheses):
            integrated_thinking += f"### 假设 {i+1}: {hyp['hypothesis']}\n"
            integrated_thinking += f"{hyp['reasoning']}\n\n"
        
        integrated_thinking += "## 假设验证\n\n"
        for i, ver in enumerate(verifications):
            status_map = {
                "supported": "✅ 支持",
                "rejected": "❌ 拒绝", 
                "uncertain": "❓ 不确定"
            }
            status = status_map.get(ver["status"], "未知")
            
            integrated_thinking += f"### 验证 {i+1}: {ver['hypothesis']} [{status}]\n"
            integrated_thinking += f"{ver['verification']}\n\n"
        
        integrated_thinking += "## 最终思考\n\n"
        integrated_thinking += updated_thinking
        
        return integrated_thinking
    
    def add_reasoning_step(self, content: str):
        """
        添加推理步骤
        
        该方法负责将一个新的推理步骤添加到思考引擎的状态中，是Graph-RAG系统中
        维护推理过程连续性和完整性的基础组件。它不仅更新全局推理步骤列表，
        还将步骤添加到当前推理分支的推理树中，并记录时间戳，支持完整的推理过程跟踪。
        
        参数:
            content: 步骤内容，包含该推理步骤的详细描述、分析或结论
        
        实现思路：
        1. 将传入的推理步骤内容添加到全局推理步骤列表中
        2. 检查当前推理分支是否存在于推理树中，如果不存在则创建
        3. 为当前推理分支添加新的推理步骤，包含内容和时间戳
        4. 时间戳用于追踪推理步骤的执行顺序和时间线
        
        技术特点：
        - 双重记录：同时更新全局步骤列表和分支推理树
        - 时间追踪：自动记录每个推理步骤的执行时间
        - 状态同步：确保全局状态和分支状态的一致性
        - 简洁高效：实现简单但功能关键的基础操作
        - 分支支持：支持多分支推理的状态管理
        
        业务意义：
        - 维护完整的推理历史，支持推理过程的回顾和分析
        - 为多分支推理提供状态管理基础
        - 支持推理步骤的时间线追踪
        - 确保推理过程的连续性和完整性
        - 为最终的思考整合提供数据来源
        """
        self.all_reasoning_steps.append(content)
        
        # 更新推理树
        if self.current_branch not in self.reasoning_tree:
            self.reasoning_tree[self.current_branch] = []
            
        self.reasoning_tree[self.current_branch].append({
            "content": content,
            "timestamp": time.time()
        })
    
    def branch_reasoning(self, branch_name: str, base_branch: str = "main"):
        """
        创建推理分支
        
        该方法负责在思考引擎中创建新的推理分支，基于指定的基础分支进行复制。
        这是Graph-RAG系统支持多路径探索和反事实推理的关键组件，允许系统
        从不同角度探索问题解决方案，同时保留原始推理路径。
        
        参数:
            branch_name: 分支名称，用于标识新创建的推理分支
            base_branch: 基础分支名称，新分支将复制该分支的所有推理步骤，默认为"main"
        
        实现思路：
        1. 检查指定的基础分支是否存在于推理树中
        2. 如果基础分支不存在，则默认使用"main"分支
        3. 在推理树中创建新的空分支
        4. 遍历基础分支中的所有推理步骤
        5. 为每个步骤创建一个深拷贝，以避免引用问题
        6. 将复制的步骤添加到新创建的分支中
        7. 不改变当前工作分支，需要单独切换分支
        
        技术特点：
        - 分支支持：实现类似版本控制系统的分支功能
        - 数据隔离：使用深拷贝确保分支间数据互不影响
        - 容错处理：当指定的基础分支不存在时使用默认分支
        - 状态保存：保留分支创建时的完整推理状态
        - 轻量级实现：使用简单的数据结构实现复杂的分支功能
        
        业务意义：
        - 支持多角度思考和反事实推理
        - 允许在不影响主推理路径的情况下探索新方向
        - 实现假设的并行验证和比较
        - 为复杂问题提供更全面的解决方案空间探索
        - 增强系统的创造性思维和探索能力
        """
        # 确保基础分支存在
        if base_branch not in self.reasoning_tree:
            base_branch = "main"
            
        # 创建新分支
        self.reasoning_tree[branch_name] = []
        
        # 复制基础分支内容
        for step in self.reasoning_tree[base_branch]:
            self.reasoning_tree[branch_name].append(step.copy())
            
        # 切换到新分支
        self.current_branch = branch_name
        
        # 添加分支创建记录
        self.add_reasoning_step(f"创建推理分支: {branch_name}，基于: {base_branch}")
    
    def switch_branch(self, branch_name: str):
        """
        切换推理分支
        
        该方法负责在思考引擎中切换当前工作的推理分支，是Graph-RAG系统支持多路径探索的
        重要组件。它允许系统在不同的推理路径之间灵活切换，便于比较不同思考方向的结果，
        实现多角度问题分析。
        
        参数:
            branch_name: 要切换到的分支名称
            
        返回:
            bool: 切换是否成功，如果指定的分支不存在则返回False，否则返回True
        
        实现思路：
        1. 首先检查指定的分支名称是否存在于推理树中
        2. 如果分支不存在，则返回False表示切换失败
        3. 如果分支存在，则将当前工作分支设置为指定的分支
        4. 返回True表示切换成功
        5. 注意：此方法不会自动记录切换操作到推理步骤中
        
        技术特点：
        - 简单高效：实现简洁明了，执行效率高
        - 安全检查：在切换前验证分支存在性
        - 状态更新：更新系统的当前工作状态
        - 返回状态：通过布尔值明确指示操作结果
        - 无副作用：仅更新状态，不修改分支内容
        
        业务意义：
        - 支持多分支推理的灵活切换
        - 便于比较不同推理路径的结果
        - 允许在暂停一个思考方向后继续其他方向的探索
        - 实现更复杂的多路径问题解决策略
        - 增强系统的推理灵活性和探索能力
        """
        # 确保分支存在
        if branch_name not in self.reasoning_tree:
            return False
            
        # 切换分支
        self.current_branch = branch_name
        return True
    
    def merge_branches(self, source_branch: str, target_branch: str = "main"):
        """
        合并推理分支
        
        该方法负责将一个源推理分支的内容合并到目标分支中，是Graph-RAG系统中
        支持多路径探索和信息整合的关键组件。它能够智能地识别源分支中独有的推理步骤，
        避免重复添加，并在合并后自动记录合并操作和切换到目标分支。
        
        参数:
            source_branch: 源分支名称，包含要合并的推理步骤
            target_branch: 目标分支名称，接收合并内容，默认为"main"
            
        返回:
            bool: 合并是否成功，如果任一指定分支不存在则返回False，否则返回True
        
        实现思路：
        1. 检查源分支和目标分支是否都存在于推理树中
        2. 如果任一分支不存在，则返回False表示合并失败
        3. 获取源分支和目标分支中的所有推理步骤
        4. 通过内容比较识别源分支中独有的推理步骤
        5. 将源分支中独有的步骤复制并添加到目标分支
        6. 创建一个合并记录步骤并添加到目标分支中
        7. 自动切换当前工作分支到目标分支
        8. 返回True表示合并成功
        
        技术特点：
        - 智能去重：避免在目标分支中添加重复的推理步骤
        - 内容比较：基于步骤内容而非引用进行比较
        - 数据隔离：使用复制而非引用，避免合并后的数据污染
        - 自动切换：合并后自动切换到目标分支，便于后续操作
        - 操作记录：记录分支合并操作，保持推理过程的可追踪性
        
        业务意义：
        - 支持多路径推理结果的整合
        - 允许系统综合不同思考方向的有效结论
        - 实现多角度分析的信息融合
        - 保留完整的推理过程记录和历史
        - 增强系统的综合分析和决策能力
        """
        # 确保分支存在
        if source_branch not in self.reasoning_tree or target_branch not in self.reasoning_tree:
            return False
            
        # 获取源分支独有的步骤
        source_steps = self.reasoning_tree[source_branch]
        target_steps = self.reasoning_tree[target_branch]
        
        # 找出源分支中独有的步骤
        source_unique_steps = []
        target_step_contents = [step["content"] for step in target_steps]
        
        for step in source_steps:
            if step["content"] not in target_step_contents:
                source_unique_steps.append(step)
        
        # 将源分支独有步骤添加到目标分支
        for step in source_unique_steps:
            self.reasoning_tree[target_branch].append(step.copy())
            
        # 添加合并记录
        merged_step = {
            "content": f"合并分支: {source_branch} → {target_branch}",
            "timestamp": time.time()
        }
        self.reasoning_tree[target_branch].append(merged_step)
        
        # 切换到目标分支
        self.current_branch = target_branch
        
        return True
    
    def counter_factual_analysis(self, hypothesis: str):
        """
        执行反事实分析
        
        该方法负责执行反事实分析，即基于一个可能与已知事实不符的假设进行推理，
        是Graph-RAG系统中支持创新思维和多角度思考的重要组件。它通过创建新的推理分支，
        允许系统探索假设性场景的逻辑结果，而不影响主推理路径。
        
        参数:
            hypothesis: 反事实假设内容，即假设为真的条件语句，即使它可能与已知事实不符
            
        返回:
            str: 反事实分析结果，包含基于假设推理出的结论和分析
        
        实现思路：
        1. 创建一个唯一的反事实分支名称，使用时间戳确保唯一性
        2. 调用branch_reasoning方法创建新分支，基于当前分支的推理历史
        3. 在新分支中添加反事实假设作为推理步骤
        4. 构建提示，要求LLM基于反事实假设重新思考问题
        5. 调用LLM生成反事实分析内容
        6. 从LLM响应中提取分析结果
        7. 将分析结果添加到反事实分支的推理步骤中
        8. 返回反事实分析结果
        9. 注意：系统将保持在新创建的反事实分支中，需要手动切换回原分支
        
        技术特点：
        - 分支隔离：使用分支机制隔离反事实推理，避免影响主推理路径
        - 唯一性保证：使用时间戳确保分支名称的唯一性
        - 结构化提示：明确引导LLM进行反事实思考
        - 结果记录：自动记录分析结果到推理历史中
        - 状态维护：确保推理状态的一致性和可追踪性
        
        业务意义：
        - 支持创新思维和假设性场景探索
        - 提供多角度分析，增强问题理解的全面性
        - 帮助发现常规推理可能忽略的关联和结论
        - 支持决策过程中的敏感性分析和风险评估
        - 增强系统的创造性解决问题能力
        """
        # 创建反事实分支
        branch_name = f"counter_factual_{int(time.time())}"
        self.branch_reasoning(branch_name)
        
        # 添加反事实假设
        self.add_reasoning_step(f"反事实假设: {hypothesis}")
        
        # 基于反事实假设进行推理
        prompt = f"""
        假设以下情况为真:
        {hypothesis}
        
        基于这个假设，请重新思考问题。即使这个假设与事实不符，也请认真推理。
        分析如果这个假设为真，会导致什么结论?
        """
        
        response = self.llm.invoke(prompt)
        counter_analysis = response.content if hasattr(response, 'content') else str(response)
        
        # 添加分析结果
        self.add_reasoning_step(f"反事实分析结果:\n\n{counter_analysis}")
        
        # 对比原始推理和反事实推理
        prompt = f"""
        请比较原始推理和反事实假设下的推理:
        
        原始推理基于已知事实。
        反事实推理基于假设: {hypothesis}
        
        这种对比揭示了什么关键见解?
        这是否帮助我们更好地理解问题?
        """
        
        response = self.llm.invoke(prompt)
        comparison = response.content if hasattr(response, 'content') else str(response)
        
        # 添加比较结果
        self.add_reasoning_step(f"原始推理与反事实推理对比:\n\n{comparison}")
        
        # 回到主分支
        self.switch_branch("main")
        
        # 添加反事实分析的总结
        self.add_reasoning_step(f"反事实分析总结: 如果 {hypothesis}，那么 {self._extract_conclusion(counter_analysis)}")
        
        return comparison
    
    def _extract_conclusion(self, analysis):
        """
        从分析中提取结论
        
        该方法负责从文本分析中智能提取核心结论，是Graph-RAG系统中信息提炼和总结的
        重要组件。它采用多策略方法，优先查找带有明确结论标记的文本，其次考虑文本的最后段落，
        并对提取的结论进行长度限制，确保结果简洁明了。
        
        参数:
            analysis: 分析文本，可能包含推理过程、论证和结论
            
        返回:
            str: 提取的结论，经过长度优化的关键结论摘要
        
        实现思路：
        1. 定义一组常见的结论标记词，如"结论"、"总结"等
        2. 遍历这些标记词，在分析文本中查找它们的位置
        3. 如果找到标记词，提取该标记后的内容作为结论
        4. 对提取的结论进行处理，只保留第一行并限制长度
        5. 如果没有找到标记词，尝试提取分析文本的最后一个段落
        6. 同样对最后段落进行长度限制处理
        7. 如果分析文本为空，则返回默认的提示文本
        8. 返回最终提取的结论
        
        技术特点：
        - 多策略提取：采用多种方法确保能提取到有意义的结论
        - 长度控制：自动限制结论长度，确保简洁明了
        - 结构化查找：使用关键词标记识别结论位置
        - 容错处理：处理各种可能的文本格式和边界情况
        - 轻量级实现：使用简单的字符串操作实现高效的结论提取
        
        业务意义：
        - 从复杂分析中提取核心观点，提高信息处理效率
        - 为系统提供简洁的结论摘要，便于后续处理
        - 支持反事实分析的结果总结和对比
        - 帮助生成更清晰、更精炼的推理报告
        - 增强系统对分析结果的理解和利用能力
        """
        # 查找结论标记
        conclusion_markers = ["结论", "总结", "因此", "所以", "综上所述"]
        
        for marker in conclusion_markers:
            marker_index = analysis.find(marker)
            if marker_index != -1:
                # 提取标记后的内容作为结论
                conclusion = analysis[marker_index:]
                # 限制长度
                conclusion = conclusion.split("\n")[0]
                if len(conclusion) > 100:
                    conclusion = conclusion[:100] + "..."
                return conclusion
                
        # 如果没有找到标记，返回最后一段
        paragraphs = analysis.split("\n\n")
        if paragraphs:
            last_paragraph = paragraphs[-1]
            if len(last_paragraph) > 100:
                last_paragraph = last_paragraph[:100] + "..."
            return last_paragraph
            
        # 如果分析内容为空，返回默认文本
        return "无法提取明确结论"
    
    def remove_query_tags(self, text: str) -> str:
        """
        移除文本中的查询标签
        
        该方法负责从文本中移除预定义的查询标签，是Graph-RAG系统中文本清洗和预处理
        的重要组件。它使用正则表达式识别并删除指定的标签对，同时保留标签内的内容，
        确保文本的干净和一致性。
        
        参数:
            text: 包含查询标签的原始文本
            
        返回:
            str: 移除标签后的干净文本，保留标签内的实际内容
        
        实现思路：
        1. 构建正则表达式模式，使用BEGIN_SEARCH_QUERY和END_SEARCH_QUERY作为匹配标记
        2. 使用re.escape函数确保标记字符被正确转义，避免正则表达式特殊字符的干扰
        3. 使用非贪婪匹配模式(.*?)确保只匹配到最近的标签对
        4. 设置re.DOTALL标志，允许点号匹配换行符，确保能匹配跨越多行的标签内容
        5. 使用re.sub函数将匹配到的标签对替换为空字符串
        6. 返回处理后的文本
        
        技术特点：
        - 精确匹配：使用预定义的标签常量确保匹配的准确性
        - 安全转义：对标签字符进行转义，避免正则表达式注入
        - 多行支持：能够处理跨越多行的标签内容
        - 非贪婪匹配：避免过度匹配导致的内容丢失
        - 高效处理：使用正则表达式实现高效的文本替换
        
        业务意义：
        - 清理LLM输出中的格式标签，提高文本可读性
        - 支持系统提取和处理实际的查询内容
        - 确保不同组件间文本传递的一致性和标准化
        - 为后续的文本处理和分析提供干净的输入
        - 支持结构化查询的识别和管理
        """
        pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def remove_result_tags(self, text: str) -> str:
        """
        移除文本中的结果标签
        
        该方法负责从文本中移除预定义的结果标签，是Graph-RAG系统中文本清洗和后处理的
        重要组件。它使用正则表达式识别并删除结果标签对，同时保留标签内的实际搜索结果，
        确保文本的干净和结构化。
        
        参数:
            text: 包含结果标签的原始文本
            
        返回:
            str: 移除标签后的干净文本，保留标签内的搜索结果内容
        
        实现思路：
        1. 构建正则表达式模式，使用BEGIN_SEARCH_RESULT和END_SEARCH_RESULT作为匹配标记
        2. 使用re.escape函数对标记字符进行转义，确保正则表达式的正确性
        3. 采用非贪婪匹配模式(.*?)确保只匹配到最近的标签对
        4. 设置re.DOTALL标志，支持匹配跨越多行的标签内容
        5. 使用re.sub函数将匹配到的标签对替换为空字符串
        6. 返回处理后的干净文本
        
        技术特点：
        - 精确识别：使用预定义的常量确保标签匹配的准确性
        - 安全转义：处理特殊字符，避免正则表达式语法错误
        - 多行兼容：能够处理标签内的多行文本内容
        - 非贪婪策略：避免过度匹配导致的内容损失
        - 高效实现：使用正则表达式进行高效的文本处理
        
        业务意义：
        - 清理文本中的格式标记，提升可读性
        - 保留实际的搜索结果内容供后续使用
        - 确保文本处理流程的标准化和一致性
        - 为结果整合和分析提供干净的输入
        - 支持系统对搜索结果的有效管理和利用
        """
        pattern = re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT)
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    def extract_queries(self, text: str) -> List[str]:
        """
        从文本中提取搜索查询
        
        该方法负责从文本中智能提取结构化的搜索查询，是Graph-RAG系统中连接思考过程
        和搜索执行的重要桥梁。它利用预定义的查询标签来识别和提取实际的查询内容，
        支持系统自动执行多轮搜索和信息收集。
        
        参数:
            text: 可能包含搜索查询标签的原始文本
            
        返回:
            List[str]: 提取的查询列表，包含所有识别到的搜索查询内容
        
        实现思路：
        1. 直接调用从search.tool.reasoning.nlp模块导入的extract_between函数
        2. 使用BEGIN_SEARCH_QUERY和END_SEARCH_QUERY作为提取的开始和结束标记
        3. extract_between函数会自动处理正则表达式匹配、多行内容和转义字符
        4. 函数返回所有匹配到的查询内容的列表
        5. 如果没有找到查询标签，则返回空列表
        
        技术特点：
        - 模块化设计：复用NLP模块中的提取功能，保持代码的一致性
        - 结构化识别：基于预定义标签进行精确的查询识别
        - 多查询支持：能够从同一文本中提取多个搜索查询
        - 简洁高效：通过函数复用实现简洁而高效的查询提取
        - 标准化处理：使用与其他标签处理方法一致的模式
        
        业务意义：
        - 实现思考到行动的自动化转换，将分析转化为具体的搜索查询
        - 支持多轮迭代搜索，增强系统的信息收集能力
        - 允许LLM以结构化方式表达所需的信息需求
        - 为系统自动执行搜索操作提供基础
        - 支持复杂问题解决中的多步骤信息检索
        """
        return extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
    
    def generate_next_query(self) -> Dict[str, Any]:
        """
        生成下一步搜索查询
        
        该方法是思考引擎的核心功能之一，负责基于当前的思考历史和上下文生成下一步的搜索查询。
        它使用LLM分析当前的推理状态，决定是否需要进一步的信息检索，并以结构化的格式输出
        搜索查询和状态信息。这是Graph-RAG系统中连接思考过程和知识检索的关键环节。
        
        返回:
            Dict: 包含查询和状态信息的字典，具有不同的状态类型：
                - has_query: 生成了有效的搜索查询
                - no_query: 没有生成查询，但需要继续思考
                - answer_ready: 已准备好最终答案
                - empty: 生成了空响应
                - error: 生成查询过程中发生错误
        
        实现思路：
        1. 构建系统消息和历史消息的组合，包括REASON_PROMPT作为系统提示
        2. 调用LLM生成基于当前状态的下一步分析和可能的查询
        3. 清理响应内容，移除可能的思考标记（如"</think>"）
        4. 检查响应是否为空，为空则返回empty状态
        5. 更新思考过程，将清理后的思考内容添加到推理步骤
        6. 从LLM响应中提取搜索查询（使用BEGIN_SEARCH_QUERY和END_SEARCH_QUERY标记）
        7. 根据提取结果决定系统状态：
           - 有查询：返回has_query状态和查询列表
           - 无查询但有答案标记：返回answer_ready状态
           - 无查询且无答案标记：返回no_query状态
        8. 全面的异常处理，捕获并记录任何错误
        
        技术特点：
        - 状态驱动：根据不同情况返回不同的状态标识
        - 结构化输出：使用标准化的标记提取查询内容
        - 错误处理：完善的异常捕获和日志记录机制
        - 灵活响应：适应不同的推理阶段和需求
        - 清晰的决策流程：基于内容特征确定系统下一步行动
        
        业务意义：
        - 智能决定何时需要进一步搜索信息
        - 自动生成高质量的搜索查询，提高信息检索效率
        - 识别推理何时可以结束并生成最终答案
        - 为多轮迭代推理提供决策支持
        - 确保推理过程的连贯性和逻辑性
        """
        # 使用LLM进行推理分析，获取下一个搜索查询
        formatted_messages = [SystemMessage(content=REASON_PROMPT)] + self.msg_history
        
        try:
            # 调用LLM生成查询
            msg = self.llm.invoke(formatted_messages)
            query_think = msg.content if hasattr(msg, 'content') else str(msg)
            
            # 清理响应
            query_think = re.sub(r"<think>.*</think>", "", query_think, flags=re.DOTALL)
            if not query_think:
                return {"status": "empty", "content": None, "queries": []}
                
            # 更新思考过程
            clean_think = self.remove_query_tags(query_think)
            self.add_reasoning_step(query_think)
            
            # 从AI响应中提取搜索查询
            queries = self.extract_queries(query_think)
            
            # 如果没有生成搜索查询，检查是否应该结束
            if not queries:
                # 检查是否包含最终答案标记
                if "**回答**" in query_think or "足够的信息" in query_think:
                    return {
                        "status": "answer_ready", 
                        "content": query_think,
                        "queries": []
                    }
                
                # 没有明确结束标志，就继续
                return {
                    "status": "no_query", 
                    "content": query_think,
                    "queries": []
                }
            
            # 有查询，继续搜索
            return {
                "status": "has_query", 
                "content": query_think,
                "queries": queries
            }
            
        except Exception as e:
            error_msg = f"生成查询时出错: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return {"status": "error", "error": error_msg, "queries": []}
    
    def add_ai_message(self, content: str):
        """
        添加AI消息到历史记录
        
        该方法负责将AI生成的消息添加到思考引擎的消息历史中，是Graph-RAG系统中
        维护对话上下文和推理连续性的重要组件。它使用LangChain的AIMessage类封装AI响应，
        确保消息历史的一致性和结构化。
        
        参数:
            content: AI生成的消息内容，可以是推理、分析、查询或答案等
        
        实现思路：
        1. 导入LangChain的AIMessage类（在文件顶部已完成导入）
        2. 创建一个AIMessage实例，将传入的内容作为消息体
        3. 将创建的AIMessage对象添加到msg_history列表中
        4. 不返回任何值，仅更新内部状态
        
        技术特点：
        - 统一接口：使用LangChain标准消息类型确保系统一致性
        - 简单直接：实现简洁但功能关键
        - 类型安全：使用专门的消息类而非原始字符串，便于后续处理
        - 状态管理：维护完整的对话历史状态
        - 集成兼容：与LangChain生态系统无缝集成
        
        业务意义：
        - 记录AI的思考和分析过程，支持推理历史追踪
        - 为多轮对话和推理提供上下文连续性
        - 确保系统能够基于完整的交互历史生成下一步响应
        - 支持复杂推理任务中的状态持久化
        - 为调试和分析系统行为提供完整的操作记录
        """
        self.msg_history.append(AIMessage(content=content))
    
    def add_human_message(self, content: str):
        """
        添加用户消息到历史记录
        
        该方法负责将用户输入的消息添加到思考引擎的消息历史中，是Graph-RAG系统中
        维护用户交互历史和上下文连续性的重要组件。它使用LangChain的HumanMessage类封装用户输入，
        确保消息历史的一致性和结构化，支持系统进行上下文感知的推理和响应生成。
        
        参数:
            content: 用户输入的消息内容，可以是问题、反馈或其他交互内容
        
        实现思路：
        1. 导入LangChain的HumanMessage类（在文件顶部已完成导入）
        2. 创建一个HumanMessage实例，将传入的内容作为消息体
        3. 将创建的HumanMessage对象添加到msg_history列表中
        4. 不返回任何值，仅更新内部状态
        
        技术特点：
        - 标准化接口：使用LangChain标准消息类型确保系统各组件间的一致性
        - 简洁高效：实现简单但功能关键
        - 类型区分：明确区分用户消息和AI消息，便于后续处理
        - 状态维护：保持完整的对话历史记录
        - 框架集成：与LangChain消息处理框架无缝协作
        
        业务意义：
        - 记录用户交互历史，支持上下文感知的推理过程
        - 为多轮对话提供必要的上下文连续性
        - 确保系统能够理解用户的输入意图和需求
        - 支持复杂任务中的交互状态持久化
        - 为用户与系统的有效沟通提供基础
        """
        self.msg_history.append(HumanMessage(content=content))
    
    def update_continue_message(self):
        """
        更新最后的消息，请求继续推理
        
        该方法负责更新对话历史中的最后一条消息，添加继续推理的指令，是Graph-RAG系统中
        支持多轮迭代推理的重要组件。它能够智能识别不同类型的消息格式，并根据最后一条消息
        的类型采取适当的更新策略，确保推理过程的连续性和上下文的正确性。
        
        实现思路：
        1. 检查消息历史是否为空，如果为空则不做处理
        2. 获取消息历史中的最后一条消息
        3. 根据消息类型进行不同处理：
           a. 对于字典类型的消息，检查"role"字段
              - 如果是"assistant"角色，添加新的用户消息请求继续推理
              - 否则，更新现有用户消息，追加继续推理指令
           b. 对于对象类型的消息（如AIMessage、HumanMessage等）
              - 检查role属性，如果是"assistant"，添加新的用户消息
              - 检查content属性，更新用户消息内容，追加继续推理指令
        4. 提供统一的继续推理指令："继续基于新信息进行推理分析。"
        
        技术特点：
        - 多格式兼容：同时支持字典格式和对象格式的消息处理
        - 智能判断：根据消息类型和角色决定适当的更新策略
        - 上下文感知：保持对话历史的连贯性和逻辑顺序
        - 灵活适应：能够处理不同格式的消息输入
        - 简洁高效：实现简单但功能全面
        
        业务意义：
        - 支持系统在获取新信息后继续推理，实现多轮迭代分析
        - 确保推理过程不会因信息更新而中断
        - 维护对话上下文的一致性和连续性
        - 提供统一的继续推理指令格式，规范化系统行为
        - 支持复杂问题解决中的分步骤推理和信息整合
        """
        if len(self.msg_history) > 0:
            # 检查最后一条消息的类型
            last_message = self.msg_history[-1]
            
            if isinstance(last_message, dict) and "role" in last_message:
                # 处理字典类型的消息
                if last_message["role"] == "assistant":
                    self.add_human_message("继续基于新信息进行推理分析。\n")
                else:
                    # 更新最后的用户消息
                    last_content = last_message.get("content", "")
                    self.msg_history[-1] = {"role": "user", "content": last_content + "\n\n继续基于新信息进行推理分析。\n"}
            else:
                # 处理对象类型的消息 (如AIMessage, HumanMessage等)
                if hasattr(last_message, "role") and last_message.role == "assistant":
                    self.add_human_message("继续基于新信息进行推理分析。\n")
                elif hasattr(last_message, "content"):
                    # 更新最后的用户消息
                    last_content = last_message.content
                    self.msg_history[-1] = {"role": "user", "content": last_content + "\n\n继续基于新信息进行推理分析。\n"}
        
    def prepare_truncated_reasoning(self) -> str:
        """
        准备截断的推理历史，保留关键部分以减少token使用
        
        该方法负责智能地截断和精简推理历史，在保持关键信息的同时减少token使用量。
        它是Graph-RAG系统中优化LLM交互效率和控制成本的重要组件，通过保留最重要的
        推理步骤，确保在有限的上下文窗口中提供最有价值的信息。
        
        返回:
            str: 截断的推理历史，包含最关键的推理步骤，按原始顺序排列
        
        实现思路：
        1. 获取完整的推理步骤列表
        2. 如果步骤少于或等于5个，保留全部步骤
        3. 如果步骤超过5个，采用智能截断策略：
           a. 总是保留第一步（初始思考）
           b. 总是保留最后4步（最近的思考）
           c. 保留中间包含搜索查询或搜索结果的重要步骤
        4. 按原始顺序对重要步骤进行排序
        5. 格式化结果，在步骤间隔处添加省略号，保持步骤顺序和相对位置
        6. 返回格式化的截断推理历史
        
        技术特点：
        - 智能保留策略：优先保留最重要的推理步骤
        - 平衡信息量：在减少token使用和保持信息完整性之间取得平衡
        - 位置感知：通过省略号标识被截断的部分，保持步骤间的相对位置关系
        - 自适应处理：根据步骤数量动态调整截断策略
        - 高效实现：使用简单的数据结构和算法实现高效处理
        
        业务意义：
        - 优化token使用，控制API调用成本
        - 确保在有限的上下文窗口中包含最有价值的信息
        - 提高LLM响应速度和质量
        - 支持处理长时间运行的复杂推理过程
        - 增强系统的可扩展性和效率
        """
        all_reasoning_steps = self.all_reasoning_steps
        
        if not all_reasoning_steps:
            return ""
            
        # 如果步骤少于5个，保留全部
        if len(all_reasoning_steps) <= 5:
            steps_text = ""
            for i, step in enumerate(all_reasoning_steps):
                steps_text += f"Step {i + 1}: {step}\n\n"
            return steps_text.strip()
        
        # 否则，保留第一步、最后4步和包含查询/结果的步骤
        important_steps = []
        
        # 总是包含第一步
        important_steps.append((0, all_reasoning_steps[0]))
        
        # 包含最后4步
        for i in range(max(1, len(all_reasoning_steps) - 4), len(all_reasoning_steps)):
            important_steps.append((i, all_reasoning_steps[i]))
        
        # 包含中间包含搜索查询或结果的步骤
        for i in range(1, len(all_reasoning_steps) - 4):
            step = all_reasoning_steps[i]
            if BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                important_steps.append((i, step))
        
        # 按原始顺序排序
        important_steps.sort(key=lambda x: x[0])
        
        # 格式化结果
        truncated = ""
        prev_idx = -1
        
        for idx, step in important_steps:
            # 如果有间隔，添加省略号
            if idx > prev_idx + 1:
                truncated += "...\n\n"
            
            truncated += f"Step {idx + 1}: {step}\n\n"
            prev_idx = idx
        
        return truncated.strip()
    
    def get_full_thinking(self) -> str:
        """
        获取完整的思考过程
        
        该方法负责生成完整的思考过程文本，包括所有推理步骤，但不包含搜索查询和结果标签，
        是Graph-RAG系统中用于记录、分析和可视化完整推理路径的重要组件。它清理所有步骤中的
        格式标签，生成一个纯文本的思考过程表示。
        
        返回:
            str: 格式化的思考过程文本，使用"</think>"标记包围，包含所有清理后的推理步骤
        
        实现思路：
        1. 创建一个以"</think>"开头的思考过程字符串
        2. 遍历所有的推理步骤(all_reasoning_steps)
        3. 对每个步骤进行清理：
           a. 调用remove_query_tags移除查询标签
           b. 调用remove_result_tags移除结果标签
        4. 将清理后的步骤文本添加到思考过程字符串，并在步骤间添加空行分隔
        5. 在末尾添加"</think>"标记
        6. 返回完整的思考过程字符串
        
        技术特点：
        - 完整记录：包含所有的推理步骤，不做任何截断
        - 格式清理：移除所有的格式标签，提供纯文本内容
        - 结构化表示：使用"</think>"标记包围，便于识别和处理
        - 步骤保留：维持原始的推理顺序和逻辑流程
        - 简洁实现：使用简单的字符串拼接和循环处理
        
        业务意义：
        - 提供完整的推理过程记录，支持系统行为分析和审计
        - 生成用户友好的思考过程展示
        - 为调试和优化系统提供详细的过程信息
        - 支持对推理质量和逻辑链的评估
        - 提供可解释的AI决策过程，增强系统透明度
        """
        thinking = "<think>\n"
        
        for step in self.all_reasoning_steps:
            clean_step = self.remove_query_tags(step)
            clean_step = self.remove_result_tags(clean_step)
            thinking += clean_step + "\n\n"
            
        thinking += "</think>"
        return thinking
    
    def has_executed_query(self, query: str) -> bool:
        """
        检查是否已经执行过相同的查询
        
        该方法负责检查指定的查询是否已经在之前的推理过程中被执行过，是Graph-RAG系统中
        避免重复查询和优化搜索效率的重要组件。它通过简单的集合查找操作，快速确定查询的
        执行状态，帮助系统智能地管理搜索资源和避免冗余计算。
        
        参数:
            query: 要检查的查询字符串
            
        返回:
            bool: 如果查询已经执行过返回True，否则返回False
        
        实现思路：
        1. 接收一个查询字符串作为输入
        2. 直接检查该查询是否存在于executed_search_queries列表中
        3. 使用Python的in运算符进行快速查找
        4. 返回布尔值表示查询的执行状态
        
        技术特点：
        - 高效查找：使用简单的集合操作实现快速查询检查
        - 直接准确：直接比对查询文本，确保检查的准确性
        - 轻量级实现：代码简洁，执行效率高
        - 状态维护：帮助维护系统的查询执行状态
        - 集成便捷：易于与其他搜索和推理组件集成
        
        业务意义：
        - 避免重复执行相同查询，提高系统效率
        - 节省API调用和计算资源
        - 减少不必要的等待时间，提升用户体验
        - 支持智能的搜索策略，优先执行未执行过的查询
        - 确保推理过程的高效性和资源的合理利用
        """
        return query in self.executed_search_queries
    
    def add_executed_query(self, query: str):
        """
        添加已执行的查询
        
        该方法负责将已执行完成的查询添加到系统的已执行查询记录中，是Graph-RAG系统中
        跟踪和管理查询执行状态的重要组件。它确保系统能够记录所有已执行的查询，为后续的
        查询去重和搜索策略优化提供基础数据支持。
        
        参数:
            query: 已执行完成的查询字符串，将被添加到执行记录中
        
        实现思路：
        1. 接收一个已执行的查询字符串作为输入
        2. 直接将该查询字符串追加到executed_search_queries列表中
        3. 不返回任何值，仅更新内部状态
        4. 简单直接的实现确保高效的执行性能
        
        技术特点：
        - 简单高效：直接的列表追加操作，执行速度快
        - 状态跟踪：帮助系统维护完整的查询执行历史
        - 数据完整性：确保所有执行过的查询都被准确记录
        - 轻量级实现：代码简洁，没有复杂的逻辑处理
        - 状态持久化：维护系统状态以便后续查询去重
        
        业务意义：
        - 支持查询去重机制，避免重复执行相同查询
        - 维护完整的查询执行记录，便于系统行为分析
        - 为搜索策略优化提供历史数据支持
        - 帮助控制API调用成本，提高资源利用效率
        - 支持系统的记忆功能，避免重复计算和资源浪费
        """
        self.executed_search_queries.append(query)