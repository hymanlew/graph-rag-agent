"""
评估数据结构模块

此模块定义了GraphRAG评估系统中使用的核心数据结构，采用Python的dataclasses设计，
提供结构化的评估数据存储、管理和序列化功能。主要包括：
- 基础序列化类：支持对象与字典/JSON的转换
- 答案评估数据结构：管理问题、标准答案和系统回答及评分
- 检索评估数据结构：管理检索相关的实体、关系和引用信息
- 数据序列化和持久化功能：支持数据的保存和加载
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple

# 导入预处理相关的工具函数
from evaluator.preprocessing.text_cleaner import clean_thinking_process, clean_references
from evaluator.preprocessing.reference_extractor import extract_references_from_answer

class JsonSerializable:
    """
    可序列化为JSON的基类
    
    提供对象与字典之间的互相转换功能，为评估数据结构提供序列化支持。
    所有需要序列化的评估数据类都可以继承此基类，实现数据的持久化和跨平台传输。
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转换为字典格式
        
        使用dataclasses的asdict函数将对象的所有属性转换为字典格式，
        确保对象可以被序列化和持久化存储。
        
        Returns:
            Dict[str, Any]: 对象属性的字典表示，键为属性名，值为属性值
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JsonSerializable':
        """
        从字典创建类实例
        
        根据提供的字典数据创建类的新实例，支持从持久化存储中恢复对象状态。
        字典的键需要与类的属性名匹配。
        
        Args:
            data: 包含对象属性的字典，键需要与类属性名对应
            
        Returns:
            JsonSerializable: 创建的类实例
            
        Raises:
            TypeError: 如果字典中缺少必要的属性或者属性类型不匹配
        """
        return cls(**data)

@dataclass
class AnswerEvaluationSample:
    """
    答案评估样本类，用于存储和更新单个答案评估的数据
    
    表示一个评估样本，包含问题、标准答案、系统回答和评分等信息，
    为答案质量评估提供基础数据结构。支持数据更新、评分记录和序列化功能。
    """
    
    # 问题文本
    question: str
    # 标准答案（用于比较）
    golden_answer: str
    # 系统生成的回答
    system_answer: str = ""
    # 各评估指标的得分
    scores: Dict[str, float] = field(default_factory=dict)
    # Agent类型：naive, hybrid, graph, deep
    agent_type: str = ""
    # 检索到的实体列表
    retrieved_entities: List[str] = field(default_factory=list)
    # 检索到的关系列表
    retrieved_relationships: List = field(default_factory=list)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """
        更新系统回答，自动清理引用数据和思考过程
        
        对原始系统回答进行预处理，去除思考过程和引用标记，
        确保评估的是纯文本回答内容。同时更新Agent类型信息。
        
        Args:
            answer: 原始系统回答，可能包含引用标记和思考过程
            agent_type: Agent类型标识符，如'naive'、'hybrid'、'graph'、'deep'等
        """
        # 预处理流程：先清理思考过程，再清理引用数据
        cleaned_answer = clean_thinking_process(answer)
        cleaned_answer = clean_references(cleaned_answer)
        
        self.system_answer = cleaned_answer
        # 如果提供了Agent类型，则更新
        if agent_type:
            self.agent_type = agent_type
            
    def update_evaluation_score(self, metric: str, score: float):
        """
        更新指定评估指标的得分
        
        将评估指标的得分记录到样本中，支持多个不同指标的得分存储。
        分数通常在0-1范围内，表示性能的相对好坏。
        
        Args:
            metric: 评估指标名称，如'em'、'f1'、'response_coherence'等
            score: 评分结果，范围通常为0-1，数值越大表示性能越好
        """
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将样本数据转换为字典格式
        
        将AnswerEvaluationSample对象转换为字典格式，方便序列化和存储。
        使用dataclasses.asdict自动处理所有属性的转换。
        
        Returns:
            Dict[str, Any]: 样本属性的字典表示，包含所有字段及其当前值
        """
        return asdict(self)

@dataclass
class AnswerEvaluationData:
    """
    答案评估数据类，用于管理多个答案评估样本
    
    提供样本集合的管理功能，支持添加样本、获取样本和数据持久化，
    实现了类似列表的访问接口，便于批量处理多个评估样本。
    作为评估数据的容器，提供了统一的数据访问和管理方式。
    """
    
    # 评估样本列表，存储所有答案评估样本
    samples: List[AnswerEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        """
        返回样本数量
        
        实现类似列表的长度查询功能，返回数据集中包含的样本总数。
        
        Returns:
            int: 评估样本的数量
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> AnswerEvaluationSample:
        """
        通过索引访问样本
        
        实现类似列表的索引访问功能，允许通过下标直接访问特定样本。
        
        Args:
            idx: 样本的索引位置，从0开始
            
        Returns:
            AnswerEvaluationSample: 位于指定索引位置的评估样本
            
        Raises:
            IndexError: 如果索引超出范围
        """
        return self.samples[idx]
    
    def append(self, sample: AnswerEvaluationSample):
        """
        添加评估样本到集合中
        
        将新的评估样本添加到数据集的末尾，扩展数据集的容量。
        
        Args:
            sample: 要添加的评估样本
        """
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """
        获取所有问题文本
        
        返回数据集中所有评估样本的问题列表，顺序与样本列表一致。
        提供便捷的数据访问方式，便于批量处理问题数据。
        
        Returns:
            List[str]: 所有评估样本的问题列表
        """
        return [sample.question for sample in self.samples]
    
    @property
    def golden_answers(self) -> List[str]:
        """
        获取所有标准答案
        
        返回数据集中所有评估样本的标准答案列表，顺序与样本列表一致。
        提供便捷的数据访问方式，便于批量处理标准答案数据。
        
        Returns:
            List[str]: 所有评估样本的标准答案列表
        """
        return [sample.golden_answer for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """
        获取所有系统回答
        
        返回数据集中所有评估样本的系统回答列表，顺序与样本列表一致。
        提供便捷的数据访问方式，便于批量处理系统回答数据。
        
        Returns:
            List[str]: 所有评估样本的系统回答列表
        """
        return [sample.system_answer for sample in self.samples]
    
    def save(self, path: str):
        """
        将评估数据保存到JSON文件
        
        将数据集中的所有样本转换为字典格式，然后序列化为JSON文件，
        实现数据的持久化存储。使用UTF-8编码确保中文等非ASCII字符正确保存。
        
        Args:
            path: 保存文件路径
        """
        with open(path, "w", encoding='utf-8') as f:
            # 将所有样本转换为字典并保存
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AnswerEvaluationData':
        """
        从JSON文件加载评估数据
        
        从指定路径的JSON文件中加载评估数据，重建样本集合，
        恢复之前保存的评估数据状态。
        
        Args:
            path: 数据文件路径
            
        Returns:
            AnswerEvaluationData: 加载的评估数据对象，包含所有保存的样本
        """
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        # 创建并添加所有样本
        for sample_data in samples_data:
            sample = AnswerEvaluationSample(**sample_data)
            data.append(sample)
        
        return data

@dataclass
class RetrievalEvaluationSample:
    """
    检索评估样本类，用于存储和管理单个检索评估的数据
    
    提供检索性能评估所需的所有数据字段，包括检索实体、关系、引用信息、
    检索时间和日志等，为各种检索指标的计算提供数据支持。是检索性能评估的核心数据结构。
    """
    
    # 问题文本
    question: str
    # 系统生成的回答，保留原始格式（包括引用标记）
    system_answer: str = ""
    # 检索到的实体ID列表
    retrieved_entities: List[str] = field(default_factory=list)
    # 检索到的关系三元组列表 (头实体, 关系, 尾实体)
    retrieved_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    # 系统回答中引用的实体ID列表
    referenced_entities: List[str] = field(default_factory=list)
    # 系统回答中引用的关系ID列表
    referenced_relationships: List = field(default_factory=list)
    # 各评估指标的得分
    scores: Dict[str, float] = field(default_factory=dict)
    # Agent类型：naive, hybrid, graph, deep
    agent_type: str = ""
    # 检索耗时（秒）
    retrieval_time: float = 0.0
    # 检索过程日志，记录检索过程中的各种信息
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    # 实体详细信息列表，包含实体的完整属性信息
    entity_details: List[Dict[str, str]] = field(default_factory=list)
    # 增强的关系三元组列表，经过数据增强后的关系信息
    enhanced_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """
        更新系统回答并提取引用信息
        
        处理系统回答，针对不同类型的Agent进行适当的清理，并自动提取
        回答中引用的实体和关系信息，为检索评估提供必要数据。
        
        Args:
            answer: 原始系统回答，可能包含引用标记和思考过程
            agent_type: Agent类型标识符，如'naive'、'hybrid'、'graph'、'deep'等
        """
        # 针对深度研究Agent，特殊处理思考过程
        if agent_type == "deep":
            answer = clean_thinking_process(answer)
            
        # 保存原始答案（包含引用标记）
        self.system_answer = answer
        
        if agent_type:
            self.agent_type = agent_type
                
        # 从回答中提取引用的实体和关系信息
        refs = extract_references_from_answer(answer)
        
        # 存储提取的实体和关系引用
        self.referenced_entities = refs.get("entities", [])
        # 关系暂时存储为ID，后续在评估方法中会转换为三元组
        self.referenced_relationships = refs.get("relationships", [])
    
    def update_retrieval_data(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        """
        更新检索到的实体和关系数据
        
        更新样本中存储的检索数据，包括实体列表和关系三元组列表，
        为检索性能评估指标的计算提供基础数据。
        
        Args:
            entities: 检索到的实体ID列表
            relationships: 检索到的关系三元组列表，每个三元组格式为(头实体, 关系, 尾实体)
        """
        self.retrieved_entities = entities
        self.retrieved_relationships = relationships
        
    def update_logs(self, logs: Dict[str, Any]):
        """
        更新检索过程的日志信息
        
        更新检索过程中的详细日志信息，包括执行步骤、中间结果、
        错误信息等，用于调试和分析检索过程中的问题。
        
        Args:
            logs: 包含检索过程详细信息的日志字典
        """
        self.retrieval_logs = logs
    
    def update_evaluation_score(self, metric: str, score: float):
        """
        更新指定评估指标的得分
        
        将检索评估指标的得分记录到样本中，支持多个不同指标的得分存储。
        分数通常在0-1范围内，表示检索性能的相对好坏。
        
        Args:
            metric: 评估指标名称，如'retrieval_precision'、'entity_coverage'等
            score: 评分结果，范围通常为0-1，数值越大表示性能越好
        """
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将样本数据转换为可JSON序列化的字典格式
        
        处理特殊数据类型，确保可以正确序列化，特别是处理：
        - 元组转列表（JSON不支持元组）
        - 消息对象序列化
        - 其他可能的不可序列化对象
        确保所有数据都可以正确转换为JSON格式。
        
        Returns:
            Dict[str, Any]: 可序列化的字典表示，所有值都可以被JSON序列化
        """
        result = asdict(self)
        
        # 处理关系元组（JSON序列化时需要转换为列表）
        result["retrieved_relationships"] = [list(rel) for rel in self.retrieved_relationships]
        # 处理enhanced_relationships字段（如果存在）
        if hasattr(self, 'enhanced_relationships') and self.enhanced_relationships:
            result["enhanced_relationships"] = [list(rel) for rel in self.enhanced_relationships]
        
        # 处理检索日志中可能存在的消息对象（如HumanMessage）
        if "retrieval_logs" in result and isinstance(result["retrieval_logs"], dict):
            logs = result["retrieval_logs"]
            if "execution_log" in logs and isinstance(logs["execution_log"], list):
                for i, log in enumerate(logs["execution_log"]):
                    # 处理输入中可能的HumanMessage对象
                    if "input" in log and hasattr(log["input"], "__class__") and log["input"].__class__.__name__ == "HumanMessage":
                        logs["execution_log"][i]["input"] = str(log["input"])
                    # 处理输出中可能的HumanMessage或AIMessage对象
                    if "output" in log and hasattr(log["output"], "__class__") and log["output"].__class__.__name__ in ["HumanMessage", "AIMessage"]:
                        logs["execution_log"][i]["output"] = str(log["output"])
        
        return result

@dataclass
class RetrievalEvaluationData:
    """
    检索评估数据类，用于管理多个检索评估样本
    
    提供检索评估样本集合的管理功能，支持添加样本、访问样本和数据持久化，
    实现了类似列表的接口，并提供了便捷的数据访问属性。作为检索评估数据的容器，
    提供了统一的数据管理和访问方式。
    """
    
    # 评估样本列表，存储所有检索评估样本
    samples: List[RetrievalEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        """
        返回样本数量
        
        实现类似列表的长度查询功能，返回数据集中包含的样本总数。
        
        Returns:
            int: 检索评估样本的数量
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RetrievalEvaluationSample:
        """
        通过索引访问样本
        
        实现类似列表的索引访问功能，允许通过下标直接访问特定样本。
        
        Args:
            idx: 样本的索引位置，从0开始
            
        Returns:
            RetrievalEvaluationSample: 位于指定索引位置的检索评估样本
            
        Raises:
            IndexError: 如果索引超出范围
        """
        return self.samples[idx]
    
    def append(self, sample: RetrievalEvaluationSample):
        """
        添加评估样本到集合中
        
        将新的检索评估样本添加到数据集的末尾，扩展数据集的容量。
        
        Args:
            sample: 要添加的检索评估样本
        """
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """
        获取所有问题文本
        
        返回数据集中所有检索评估样本的问题列表，顺序与样本列表一致。
        提供便捷的数据访问方式，便于批量处理问题数据。
        
        Returns:
            List[str]: 所有评估样本的问题列表
        """
        return [sample.question for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """
        获取所有系统回答
        
        返回数据集中所有检索评估样本的系统回答列表，顺序与样本列表一致。
        提供便捷的数据访问方式，便于批量处理系统回答数据。
        
        Returns:
            List[str]: 所有评估样本的系统回答列表
        """
        return [sample.system_answer for sample in self.samples]
    
    @property
    def retrieved_entities(self) -> List[List[str]]:
        """
        获取所有检索到的实体列表
        
        返回数据集中所有检索评估样本的检索实体列表，是一个二维列表，
        第一维对应样本，第二维对应每个样本检索到的实体ID。
        
        Returns:
            List[List[str]]: 每个样本检索到的实体ID列表的集合
        """
        return [sample.retrieved_entities for sample in self.samples]
    
    @property
    def referenced_entities(self) -> List[List[str]]:
        """
        获取所有引用的实体列表
        
        返回数据集中所有检索评估样本的引用实体列表，是一个二维列表，
        第一维对应样本，第二维对应每个样本回答中引用的实体ID。
        
        Returns:
            List[List[str]]: 每个样本回答中引用的实体ID列表的集合
        """
        return [sample.referenced_entities for sample in self.samples]
    
    @property
    def retrieved_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """
        获取所有检索到的关系列表
        
        返回数据集中所有检索评估样本的检索关系列表，是一个二维列表，
        第一维对应样本，第二维对应每个样本检索到的关系三元组。
        
        Returns:
            List[List[Tuple[str, str, str]]]: 每个样本检索到的关系三元组列表的集合
        """
        return [sample.retrieved_relationships for sample in self.samples]
    
    @property
    def referenced_relationships(self) -> List[List]:
        """
        获取所有引用的关系列表
        
        返回数据集中所有检索评估样本的引用关系列表，是一个二维列表，
        第一维对应样本，第二维对应每个样本回答中引用的关系ID。
        
        Returns:
            List[List]: 每个样本回答中引用的关系列表的集合
        """
        return [sample.referenced_relationships for sample in self.samples]
    
    def save(self, path: str):
        """
        将评估数据保存到JSON文件
        
        使用自定义JSON编码器处理特殊对象类型，特别是langchain的消息对象，
        确保所有数据都能正确序列化和持久化存储。使用UTF-8编码支持中文等非ASCII字符。
        
        Args:
            path: 保存文件路径
        """
        class CustomEncoder(json.JSONEncoder):
            """
            自定义JSON编码器，处理特殊对象类型
            
            专门处理langchain消息对象等无法直接JSON序列化的特殊数据类型，
            确保复杂对象也能正确转换为字符串格式保存。
            """
            def default(self, obj):
                try:
                    from langchain_core.messages import BaseMessage
                    # 处理langchain消息对象，将其转换为字符串
                    if isinstance(obj, BaseMessage):
                        return str(obj)
                except ImportError:
                    # 如果langchain未安装，跳过处理
                    pass
                # 其他对象使用默认编码器
                return super().default(obj)
        
        with open(path, "w", encoding='utf-8') as f:
            # 转换所有样本并保存
            samples_data = [sample.to_dict() for sample in self.samples]
            json.dump(samples_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalEvaluationData':
        """
        从JSON文件加载评估数据
        
        从指定路径的JSON文件中加载检索评估数据，包括特殊格式转换，
        如将列表转回元组，以正确重建原始数据结构。
        
        Args:
            path: 数据文件路径
            
        Returns:
            RetrievalEvaluationData: 加载的评估数据对象，包含所有保存的检索评估样本
        """
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            # JSON中的关系是列表格式，需要转回元组
            if "retrieved_relationships" in sample_data:
                sample_data["retrieved_relationships"] = [tuple(rel) for rel in sample_data["retrieved_relationships"]]
            # 处理增强关系字段
            if "enhanced_relationships" in sample_data:
                sample_data["enhanced_relationships"] = [tuple(rel) for rel in sample_data["enhanced_relationships"]]
                
            # 创建样本并添加到集合
            sample = RetrievalEvaluationSample(**sample_data)
            data.append(sample)
        
        return data