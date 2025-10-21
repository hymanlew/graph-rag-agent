"""
推理模块初始化

本模块是Graph-RAG系统的核心推理组件，提供了从自然语言处理到深度推理的完整能力栈，
支持多步思考、知识整合、搜索增强和证据追踪等高级功能。

模块组织结构：
1. NLP工具：提供文本提取和处理的基础功能
2. 提示管理：处理和优化LLM提示模板
3. 思考引擎：实现多步推理和深度思考逻辑
4. 答案验证：确保生成内容的质量和相关性
5. 搜索增强：提供高级搜索策略和查询生成
6. 社区增强：利用社区信息提升搜索质量
7. 知识图谱：动态构建和查询知识关系
8. 证据追踪：记录和管理推理过程中的证据链

核心功能组件：
- ThinkingEngine: 多步推理引擎，实现复杂问题的深度思考
- AnswerValidator: 答案质量验证器，确保生成内容满足质量标准
- DualPathSearcher: 双路径搜索器，协调知识库和知识图谱检索
- QueryGenerator: 查询生成器，优化搜索查询以获取更准确结果
- DynamicKnowledgeGraphBuilder: 动态知识图谱构建器
- EvidenceChainTracker: 证据链追踪器，确保推理过程可追溯

使用场景：
- 深度研究任务：解决复杂的多步骤问题
- 知识整合：从多个来源整合信息
- 复杂推理：执行需要多步思考的推理过程
- 答案验证：确保生成内容的质量和准确性
- 知识发现：发现和构建概念之间的关联

技术特点：
- 模块化设计：各组件独立且可组合使用
- 灵活性强：支持多种推理策略和搜索方法
- 可扩展性：易于添加新的推理模块和功能
- 高性能：优化的推理算法和数据结构
- 可追溯性：完整记录推理过程和证据来源
"""

# 导入NLP处理工具
from search.tool.reasoning.nlp import extract_between, extract_from_templates, extract_sentences

# 导入提示和令牌管理工具
from search.tool.reasoning.prompts import kb_prompt, num_tokens_from_string

# 导入核心思考引擎
from search.tool.reasoning.thinking import ThinkingEngine

# 导入答案验证器
from search.tool.reasoning.validator import AnswerValidator

# 导入搜索增强工具
from search.tool.reasoning.search import DualPathSearcher, QueryGenerator

# 导入社区感知搜索增强器
from search.tool.reasoning.community_enhance import CommunityAwareSearchEnhancer

# 导入动态知识图谱构建器
from search.tool.reasoning.kg_builder import DynamicKnowledgeGraphBuilder

# 导入证据链追踪器
from search.tool.reasoning.evidence import EvidenceChainTracker

# 定义模块公开接口，便于外部导入和使用
__all__ = [
    "extract_between",
    "extract_from_templates",
    "extract_sentences",
    "kb_prompt",
    "num_tokens_from_string",
    "ThinkingEngine",
    "AnswerValidator",
    "DualPathSearcher",
    "QueryGenerator",
    "CommunityAwareSearchEnhancer",
    "DynamicKnowledgeGraphBuilder",
    "EvidenceChainTracker",
]