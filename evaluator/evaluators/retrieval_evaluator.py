import re
import time
from typing import Dict, List, Tuple

from evaluator.core.base_evaluator import BaseEvaluator
from evaluator.core.evaluation_data import RetrievalEvaluationData, RetrievalEvaluationSample
from evaluator.preprocessing.reference_extractor import extract_references_from_answer

"""
图检索评估器模块

此模块实现了GraphRAGRetrievalEvaluator类，专门用于评估GraphRAG系统中
不同Agent的检索性能。该评估器能够分析检索的实体、关系的覆盖率、精确度和相关性，
支持多种评估指标，并提供了Agent之间的性能比较功能。

核心功能：
- 处理不同类型Agent的检索结果
- 增强实体和关系信息
- 从回答中提取引用数据
- 计算多种检索评估指标
- 保存和比较评估结果
"""

class GraphRAGRetrievalEvaluator(BaseEvaluator):
    """
    GraphRAG检索评估器类
    
    继承自BaseEvaluator，专门用于评估GraphRAG系统中不同Agent的检索性能。
    支持评估实体覆盖、关系覆盖率、检索精确度、相关性等多个维度的指标，
    并提供了从Neo4j获取实体和关系信息的功能，以增强评估数据。
    """
    
    def __init__(self, config):
        """
        初始化图检索评估器
        
        Args:
            config: 评估配置对象，包含评估指标、保存路径等设置，以及Neo4j客户端和QA Agent
            
        初始化过程：
        1. 调用父类初始化方法
        2. 获取Neo4j客户端和QA Agent
        3. 初始化实体和关系映射字典
        """
        super().__init__(config)
        # 获取Neo4j客户端，用于查询图数据库中的实体和关系信息
        self.neo4j_client = config.get('neo4j_client', None)
        # 获取QA Agent，用于生成答案和检索相关信息
        self.qa_agent = config.get('qa_agent', None)
        # 实体ID到描述的映射字典
        self.entity_map = {}
        # 关系ID到关系信息的映射字典
        self.relation_map = {}
    
    def evaluate(self, data: RetrievalEvaluationData) -> Dict[str, float]:
        """
        执行图检索性能评估
        
        处理评估数据中的每个样本，增强实体和关系信息，然后计算配置的评估指标。
        支持不同类型Agent的评估，针对不同Agent类型采用相应的数据处理策略。
        
        Args:
            data: RetrievalEvaluationData对象，包含待评估的检索样本集合
            
        Returns:
            Dict[str, float]: 评估结果字典，键为指标名称，值为得分
            
        评估流程:
        1. 建立实体和关系映射
        2. 处理每个样本，根据Agent类型采取不同的处理策略
        3. 增强实体和关系信息
        4. 从回答中提取引用数据
        5. 计算各个评估指标
        6. 更新样本评分并保存结果
        """
        # 记录评估开始信息
        self.log("\n======== 开始评估检索性能 ========")
        
        # 打印样本信息
        self.log(f"样本总数: {len(data.samples)}")

        # 预处理阶段 - 建立实体和关系映射，用于后续数据增强
        self._prepare_entity_relation_maps()
        
        # 处理每个样本的数据，确保引用的实体和关系信息完整
        for i, sample in enumerate(data.samples):
            self.log(f"\n处理样本 {i+1}:")
            
            # 打印基本信息
            self.log(f"  问题: {sample.question[:50]}...")
            self.log(f"  Agent类型: {sample.agent_type}")

            # 增强实体和关系处理，丰富样本数据
            self._enhance_entity_data(sample)
            self._enhance_relation_data(sample)
            
            # 打印处理后的信息
            self.log(f"  处理后的引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  处理后的引用关系数量: {len(sample.referenced_relationships)}")
            
            # 打印回答的一部分以及从回答中提取的引用数据
            answer = sample.system_answer
            self.log(f"  回答前100字符: {answer[:100]}...")
            
            # 显示当前样本的引用实体和关系信息
            self.log(f"  当前引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  当前引用关系数量: {len(sample.referenced_relationships)}")
            
            # 从回答中提取引用数据并打印
            refs = extract_references_from_answer(answer)
            
            self.log(f"  提取的引用数据:")
            self.log(f"    实体: {refs.get('entities', [])[:5]}{'...' if len(refs.get('entities', [])) > 5 else ''}") 
            self.log(f"    关系: {refs.get('relationships', [])[:5]}{'...' if len(refs.get('relationships', [])) > 5 else ''}")
            self.log(f"    文本块: {refs.get('chunks', [])[:3]}{'...' if len(refs.get('chunks', [])) > 3 else ''}")
            
            # 1. 处理naiveAgent - 确保文本块数据正确存储
            if sample.agent_type.lower() == "naive":
                self.log("  处理NaiveAgent的引用数据...")
                
                # NaiveAgent特殊处理：将文本块ID从referenced_relationships移到referenced_entities
                # 因为NaiveAgent主要基于文本块检索而非图结构检索
                if not sample.referenced_entities and isinstance(sample.referenced_relationships, list):
                    for item in sample.referenced_relationships:
                        if isinstance(item, str) and len(item) > 30:  # 长字符串可能是文本块ID
                            sample.referenced_entities.append(item)
                    sample.referenced_relationships = []
                    self.log(f"  将文本块从关系移到实体字段，现在实体数: {len(sample.referenced_entities)}")
                
                # 确保从json数据中提取的文本块ID在referenced_entities中
                for chunk_id in refs.get("chunks", []):
                    if chunk_id not in sample.referenced_entities:
                        sample.referenced_entities.append(chunk_id)
                        self.log(f"  添加文本块ID: {chunk_id[:10]}...")
            
            # 2. 处理其他Agent - 确保实体和关系ID正确存储
            else:
                self.log("  处理非NaiveAgent的引用数据...")
                
                # 图结构Agent处理：更新实体ID
                added_entities = 0
                for entity_id in refs.get("entities", []):
                    if entity_id and entity_id not in sample.referenced_entities:
                        sample.referenced_entities.append(entity_id)
                        added_entities += 1
                
                # 图结构Agent处理：更新关系ID
                added_relationships = 0
                for rel_id in refs.get("relationships", []):
                    if rel_id and rel_id not in sample.referenced_relationships:
                        sample.referenced_relationships.append(rel_id)
                        added_relationships += 1
                        
                self.log(f"  添加了{added_entities}个实体和{added_relationships}个关系")
                
            # 显示最终引用信息，便于评估跟踪
            self.log(f"  最终引用实体数量: {len(sample.referenced_entities)}")
            self.log(f"  最终引用关系数量: {len(sample.referenced_relationships)}")
        
        # 执行评估计算，使用配置的所有评估指标
        result_dict = {}
        
        for metric_name in self.metrics:
            try:
                self.log(f"\n开始计算指标: {metric_name}")
                metric_class_name = self.metric_class[metric_name].__class__.__name__
                self.log(f"\n使用评估类: {metric_class_name}")
                
                # 调用具体指标的calculate_metric方法计算得分
                metric_result, metric_scores = self.metric_class[metric_name].calculate_metric(data)
                result_dict.update(metric_result)
                
                # 更新每个样本的评分，确保样本级别的评分被记录
                for sample, metric_score in zip(data.samples, metric_scores):
                    sample.update_evaluation_score(metric_name, metric_score)
                    
                self.log(f"完成指标 {metric_name} 计算，平均得分: {list(metric_result.values())[0]:.4f}")
            except Exception as e:
                # 异常处理：记录错误信息但继续评估其他指标
                import traceback
                self.log(f'评估 {metric_name} 时出错: {e}')
                self.log(traceback.format_exc())
                continue
        
        # 打印所有评估指标的计算结果
        self.log("\n所有指标计算结果:")
        for metric, score in result_dict.items():
            self.log(f"  {metric}: {score:.4f}")
        
        self.log("======== 检索性能评估结束 ========\n")
        
        # 根据配置保存评估结果到文件
        if self.save_metric_flag:
            self.save_metric_score(result_dict)
        
        # 根据配置保存完整的评估数据
        if self.save_data_flag:
            self.save_data(data)
        
        return result_dict
    
    def _prepare_entity_relation_maps(self):
        """
        准备实体和关系映射，用于快速查找
        
        从Neo4j图数据库中加载实体和关系信息，构建映射字典，
        用于后续增强评估样本中的实体和关系数据。
        
        此方法是评估前的关键预处理步骤，为后续的数据增强提供基础数据支持。
        如果无法连接Neo4j，则保持映射为空字典。
        """
        # 初始化映射字典
        self.entity_map = {}
        self.relation_map = {}
        
        # 检查是否配置了Neo4j客户端
        if not self.neo4j_client:
            return
        
        try:
            # 获取所有实体
            entity_query = """
            MATCH (n)
            RETURN n.id AS id, n.description AS description
            LIMIT 2000
            """
            entity_result = self.neo4j_client.execute_query(entity_query)
            
            if entity_result.records:
                for record in entity_result.records:
                    ent_id = record.get("id")
                    ent_desc = record.get("description", "")
                    if ent_id:
                        self.entity_map[str(ent_id)] = ent_desc
            
            # 获取所有关系
            relation_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, type(r) AS relation, b.id AS target, r.id AS rel_id
            LIMIT 1000
            """
            relation_result = self.neo4j_client.execute_query(relation_query)
            
            if relation_result.records:
                for record in relation_result.records:
                    rel_id = record.get("rel_id")
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    
                    if rel_id and source and relation and target:
                        self.relation_map[str(rel_id)] = {
                            "source": str(source),
                            "relation": relation,
                            "target": str(target)
                        }
        except Exception as e:
            self.log(f"准备实体和关系映射时出错: {e}")

    def _enhance_entity_data(self, sample):
        """
        增强实体数据处理
        
        将实体ID转换为字符串，并尝试从实体映射中获取实体描述，
        丰富样本中的实体信息，便于后续评估。
        
        Args:
            sample: 评估样本，包含需要增强的实体信息
        """
        # 1. 确保实体ID是字符串
        sample.referenced_entities = [str(e) for e in sample.referenced_entities]
        
        # 2. 尝试添加实体描述
        if self.entity_map:
            enhanced_entities = []
            for ent_id in sample.referenced_entities:
                if ent_id in self.entity_map:
                    desc = self.entity_map[ent_id]
                    enhanced_entity = {
                        "id": ent_id,
                        "description": desc
                    }
                    enhanced_entities.append(enhanced_entity)
                else:
                    enhanced_entities.append({
                        "id": ent_id,
                        "description": f"实体 {ent_id}"
                    })
            
            # 将增强的实体信息保存到样本中
            sample.entity_details = enhanced_entities

    def _enhance_relation_data(self, sample):
        """
        增强关系数据处理
        
        处理样本中的关系ID，尝试从关系映射中获取完整的关系信息（源实体、关系类型、目标实体），
        丰富样本中的关系数据。如果无法获取完整信息，创建占位关系。
        
        Args:
            sample: 评估样本，包含需要增强的关系信息
        """
        # 1. 处理字符串ID的关系
        if not isinstance(sample.referenced_relationships, list):
            sample.referenced_relationships = []
            return
        
        string_rel_ids = [r for r in sample.referenced_relationships if isinstance(r, str)]
        
        # 2. 尝试使用关系映射增强关系信息
        enhanced_relations = []
        for rel_id in string_rel_ids:
            if rel_id in self.relation_map:
                rel_data = self.relation_map[rel_id]
                enhanced_relation = (
                    rel_data["source"],
                    rel_data["relation"],
                    rel_data["target"]
                )
                enhanced_relations.append(enhanced_relation)
        
        # 3. 如果成功增强了关系，更新样本
        if enhanced_relations:
            sample.enhanced_relationships = enhanced_relations
        else:
            # 使用更智能的方式创建占位关系
            relation_types = ["MENTIONS", "RELATES_TO", "PART_OF", "CONTAINS"]
            
            for i, rel_id in enumerate(string_rel_ids):
                rel_type = relation_types[i % len(relation_types)]
                source = f"entity_{i}"
                target = f"entity_{i+1}"
                
                enhanced_relations.append((source, rel_type, target))
            
            sample.enhanced_relationships = enhanced_relations
    
    def get_entities_info(self, entity_ids: List[str]) -> List[Tuple[str, str]]:
        """
        获取实体信息（ID和描述）
        
        从Neo4j数据库中查询指定实体ID的详细信息，包括实体描述。
        
        Args:
            entity_ids: 实体ID列表
            
        Returns:
            List[Tuple[str, str]]: 实体信息列表，每个元素是(ID, 描述)的元组
        """
        if not self.neo4j_client or not entity_ids:
            return []
        
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $ids
            RETURN e.id AS id, e.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": entity_ids})
            
            entities_info = []
            if result.records:
                for record in result.records:
                    entity_id = record.get("id", "未知ID")
                    entity_desc = record.get("description", "")
                    # 使用实体ID和描述
                    entities_info.append((str(entity_id), entity_desc or ""))
            
            # 如果没有找到实体，返回原始ID
            if not entities_info:
                entities_info = [(eid, "") for eid in entity_ids]
                
            return entities_info
                
        except Exception as e:
            self.log(f"查询实体信息失败: {e}")
            return [(eid, "") for eid in entity_ids]

    def get_relationships_info(self, relationship_ids: List[str]) -> List[Tuple[str, str, str]]:
        """
        获取关系信息（源实体-关系类型-目标实体）
        
        从Neo4j数据库中查询指定关系ID的详细信息，包括源实体、关系类型和目标实体。
        
        Args:
            relationship_ids: 关系ID列表
            
        Returns:
            List[Tuple[str, str, str]]: 关系信息列表，每个元素是(源实体ID, 关系类型, 目标实体ID)的元组
        """
        if not self.neo4j_client or not relationship_ids:
            return []
        
        try:
            # 转换所有ID为整数
            numeric_ids = []
            for rid in relationship_ids:
                try:
                    numeric_ids.append(int(rid))
                except (ValueError, TypeError):
                    # 如果不能转换为整数，跳过
                    pass
            
            if not numeric_ids:
                # 如果没有有效的数字ID，返回空列表
                return []
            
            # 通过关系ID直接匹配关系
            query = """
            MATCH (a)-[r]->(b)
            WHERE r.id IN $ids
            RETURN a.id AS source, type(r) AS relation, b.id AS target, 
                r.description AS description
            """
            
            result = self.neo4j_client.execute_query(query, {"ids": numeric_ids})
            
            relationships_info = []
            if result.records:
                for record in result.records:
                    source = record.get("source")
                    relation = record.get("relation")
                    target = record.get("target")
                    description = record.get("description", "")
                    
                    # 只有当所有值都存在时才添加关系
                    if source and relation and target:
                        # 使用关系的描述补充关系类型
                        rel_info = relation
                        if description:
                            rel_info = f"{relation}({description})"
                            
                        relationships_info.append((str(source), rel_info, str(target)))
            
            return relationships_info
                
        except Exception as e:
            self.log(f"查询关系信息失败: {e}")
            return []
        
    def evaluate_agent(self, agent_name: str, questions: List[str]) -> Dict[str, float]:
        """
        评估特定Agent的检索性能
        
        使用指定的Agent回答问题，并评估其检索性能。
        支持不同类型的Agent，如naive、hybrid、graph和deep。
        
        Args:
            agent_name: Agent名称 (naive, hybrid, graph, deep)
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 评估结果字典，键为指标名称，值为得分
            
        Raises:
            ValueError: 当未找到指定的Agent时
        """
        # 获取指定名称的Agent
        agent = self.config.get_agent(agent_name)
        if not agent:
            raise ValueError(f"未找到Agent: {agent_name}")
        
        # 创建评估数据集
        eval_data = RetrievalEvaluationData()
        
        # 处理每个问题
        for question in questions:
            # 创建评估样本
            sample = RetrievalEvaluationSample(
                question=question,
                agent_type=agent_name
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 使用Agent回答问题
            answer = agent.ask(question)
            
            # 计算检索时间
            retrieval_time = time.time() - start_time
            
            # 更新样本信息
            sample.update_system_answer(answer, agent_name)
            sample.retrieval_time = retrieval_time
            
            # 使用Neo4j获取相关图数据
            if self.neo4j_client:
                entities, relationships = self._get_relevant_graph_data(question)
                sample.update_retrieval_data(entities, relationships)
            
            # 添加到评估数据
            eval_data.append(sample)
        
        # 执行评估
        return self.evaluate(eval_data)
    
    def compare_agents(self, questions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        比较所有Agent的检索性能
        
        对配置中的多个Agent进行评估，比较它们在相同问题上的检索性能。
        支持的Agent包括naive、hybrid、graph和deep。
        
        Args:
            questions: 用于评估的问题列表
            
        Returns:
            Dict[str, Dict[str, float]]: 每个Agent的评估结果，外层键为Agent名称，内层为指标和得分
        """
        results = {}
        
        # 遍历支持的Agent类型
        for agent_name in ["naive", "hybrid", "graph", "deep"]:
            agent = self.config.get_agent(agent_name)
            if agent:
                self.log(f"评估Agent: {agent_name}")
                # 评估当前Agent
                agent_results = self.evaluate_agent(agent_name, questions)
                results[agent_name] = agent_results
                
                # 打印结果
                self.log(f"{agent_name} 评估结果:")
                for metric, score in agent_results.items():
                    self.log(f"  {metric}: {score:.4f}")
                self.log("")
        
        return results
    
    def _get_relevant_graph_data(self, question: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        从Neo4j获取与问题相关的实体和关系
        
        实现了智能的图数据检索策略，通过关键词提取和多级图数据库查询，
        获取与问题最相关的实体和关系数据，为检索评估提供基础数据支持。
        
        核心策略：
        1. 首先使用jieba提取问题关键词
        2. 通过关键词查询相关实体
        3. 查询实体之间的关系
        4. 如果实体不足，尝试通过文本块查找
        5. 提供多种回退方案确保数据可用性
        
        Args:
            question: 用户问题文本
            
        Returns:
            Tuple[List[str], List[Tuple[str, str, str]]]: 
                - 第一个元素：相关实体ID列表
                - 第二个元素：关系三元组列表，每个三元组格式为(源实体ID, 关系类型, 目标实体ID)
        """
        # 检查Neo4j客户端是否可用
        if not self.neo4j_client:
            return [], []
            
        try:
            # 提取问题关键词 - 支持中文文本分析
            try:
                # 首选：使用jieba进行关键词提取，获取top 5关键词
                import jieba.analyse
                question_words = jieba.analyse.extract_tags(question, topK=5)
            except Exception as e:
                # 回退方案：使用正则表达式进行简单分词
                self.log(f"关键词提取失败: {e}")
                question_words = re.findall(r'\b[\w\u4e00-\u9fa5]{2,}\b', question)
                question_words = [w for w in question_words if len(w) > 1]  # 过滤掉单字符
            
            # 初始化实体和关系集合
            entities = []
            relationships = []
            
            # 第一级查询：通过关键词查询相关实体
            # 匹配实体ID或描述中包含任何关键词的实体
            entity_query = """
            MATCH (e:__Entity__)
            WHERE ANY(word IN $keywords WHERE 
                e.id CONTAINS word OR
                e.description CONTAINS word)
            RETURN e.id AS id
            LIMIT 15
            """
            
            # 执行实体查询
            entity_result = self.neo4j_client.execute_query(entity_query, {"keywords": question_words})
            
            # 提取查询结果中的实体ID
            if entity_result.records:
                for record in entity_result.records:
                    entity_id = record.get("id")
                    if entity_id:
                        entities.append(entity_id)
            
            # 第二级查询：如果找到实体，查询这些实体之间的关系
            if entities:
                # 查询包含至少一个相关实体的所有关系
                rel_query = """
                MATCH (a:__Entity__)-[r]->(b:__Entity__)
                WHERE a.id IN $entity_ids OR b.id IN $entity_ids
                RETURN DISTINCT a.id AS source, type(r) AS relation, b.id AS target
                LIMIT 30
                """
                
                # 执行关系查询
                rel_result = self.neo4j_client.execute_query(rel_query, {"entity_ids": entities})
                
                # 提取查询结果中的关系三元组
                if rel_result.records:
                    for record in rel_result.records:
                        source = record.get("source")
                        relation = record.get("relation")
                        target = record.get("target")
                        if source and relation and target:
                            relationships.append((source, relation, target))
            
            # 第三级查询：如果找到的实体不足3个，尝试通过文本块查找更多实体
            if len(entities) < 3:
                chunk_query = """
                MATCH (c:__Chunk__)
                WHERE ANY(word IN $keywords WHERE c.text CONTAINS word)
                RETURN c.id AS chunk_id
                LIMIT 5
                """
                
                chunk_result = self.neo4j_client.execute_query(chunk_query, {"keywords": question_words})
                
                chunk_ids = []
                if chunk_result.records:
                    for record in chunk_result.records:
                        chunk_id = record.get("chunk_id")
                        if chunk_id:
                            chunk_ids.append(chunk_id)
                
                # 如果找到文本块，获取相关实体
                if chunk_ids:
                    chunk_entity_query = """
                    MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
                    WHERE c.id IN $chunk_ids
                    RETURN DISTINCT e.id AS entity_id
                    """
                    
                    chunk_entity_result = self.neo4j_client.execute_query(
                        chunk_entity_query, {"chunk_ids": chunk_ids}
                    )
                    
                    if chunk_entity_result.records:
                        for record in chunk_entity_result.records:
                            entity_id = record.get("entity_id")
                            if entity_id and entity_id not in entities:
                                entities.append(entity_id)
        except Exception as e:
            self.log(f"获取图数据时出错: {e}")
        
        return entities, relationships
    
    def format_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        将比较结果格式化为Markdown表格
        
        实现了一个灵活的表格生成功能，将不同Agent的评估结果组织成格式化的Markdown表格，
        便于直观比较多个Agent在各个评估指标上的表现差异。
        
        核心步骤：
        1. 收集所有Agent使用的指标
        2. 构建Markdown表格的表头和分隔线
        3. 为每个指标生成一行数据
        4. 计算并添加平均得分行
        
        Args:
            results: 比较结果字典，格式为 {"agent_name": {"metric_name": score, ...}, ...}
            
        Returns:
            str: 格式化的Markdown表格字符串，可以直接显示或保存
        """
        # 收集所有Agent使用的评估指标
        all_metrics = set()
        for agent_results in results.values():
            all_metrics.update(agent_results.keys())
        
        # 构建表头行，包含指标列和各个Agent列
        header = "| 指标 | " + " | ".join(results.keys()) + " |"
        # 构建分隔线行，确保表格格式正确
        separator = "| --- | " + " | ".join(["---" for _ in results]) + " |"
        
        # 构建行
        rows = []
        for metric in sorted(all_metrics):
            row = f"| {metric} |"
            for agent in results:
                score = results[agent].get(metric, "N/A")
                if isinstance(score, float):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                row += f" {score_str} |"
            rows.append(row)
        
        # 拼接表格
        table = "\n".join([header, separator] + rows)
        return table