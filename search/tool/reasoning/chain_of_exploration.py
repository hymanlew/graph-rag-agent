from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import asyncio
import pandas as pd
import re

class ChainOfExplorationSearcher:
    """
    增强版Chain of Exploration检索器
    
    实现多步自主探索图谱的能力，具有适应性搜索宽度、记忆机制和路径优化功能。该类是Graph-RAG系统中
    实现深度知识探索的核心组件，通过模拟人类探索过程，在知识图谱中进行有目的、有策略的多步探索，
    显著提高系统对复杂问题的理解和回答能力。
    
    核心功能：
    - 多步自主探索：从起始实体开始，逐步扩展探索范围
    - 适应性搜索宽度：根据查询复杂度和探索步骤动态调整搜索广度
    - 记忆机制：记录已探索路径，避免重复探索
    - 智能决策：利用LLM决定探索方向
    - 多维度评分：综合多种因素对节点进行相关性评估
    - 异步支持：提供异步探索接口
    
    设计特点：
    - 策略驱动：基于查询动态生成探索策略
    - 自适应参数：根据查询复杂度和当前状态动态调整
    - 容错设计：完善的异常处理和降级策略
    - 性能监控：记录各环节性能指标
    - 结构化输出：返回标准化的探索结果
    """
    
    def __init__(self, graph, llm, embeddings_model):
        """
        初始化Chain of Exploration检索器
        
        该方法负责初始化探索检索器的核心组件，包括图数据库连接、语言模型和向量嵌入模型。
        它还设置了探索过程中需要的状态变量，如已访问节点、探索路径和记忆机制。
        
        参数:
            graph: 图数据库连接，用于查询实体和关系信息
            llm: 语言模型，用于生成策略和决策下一步探索方向
            embeddings_model: 向量嵌入模型，用于计算文本和实体的语义相似度
        
        实现思路：
        1. 保存图数据库连接，用于后续查询
        2. 保存语言模型，用于策略生成和决策
        3. 保存向量嵌入模型，用于相似度计算
        4. 初始化已访问节点集合，用于避免重复探索
        5. 初始化探索路径列表，用于记录探索过程
        6. 初始化探索记忆字典，用于缓存探索决策
        7. 初始化性能指标字典，用于监控性能
        
        技术特点：
        - 依赖注入：通过构造函数注入依赖组件
        - 状态管理：初始化探索过程需要的各种状态变量
        - 组件解耦：各功能组件独立，便于替换和扩展
        - 资源准备：为后续探索操作准备必要的资源
        
        业务意义：
        - 为深度知识探索提供必要的组件和配置
        - 设置探索过程的状态管理机制
        - 为复杂查询的深入分析奠定基础
        - 支持基于图结构的多步推理
        """
        self.graph = graph
        self.llm = llm
        self.embeddings = embeddings_model
        self.visited_nodes = set()
        self.exploration_path = []
        self.exploration_memory = {}  # 存储已探索路径的记忆
        self.performance_metrics = {}
    
    def explore(self, query: str, starting_entities: List[str], max_steps: int = 5, exploration_width: int = 3):
        """
        从起始实体开始探索图谱
        
        该方法是Chain of Exploration的核心实现，负责执行完整的多步探索过程。它从给定的起始实体出发，
        按照预设的最大步数和基础探索宽度，智能地扩展探索范围，收集相关实体、关系和内容信息。
        通过动态调整探索策略和利用LLM进行决策，该方法能够有效地在知识图谱中发现与查询相关的信息。
        
        参数:
            query: 用户查询，原始问题
            starting_entities: 起始实体列表，作为探索的起点
            max_steps: 最大探索步数，控制探索的深度，默认为5
            exploration_width: 基础探索宽度，控制每步探索的广度，默认为3
            
        返回:
            Dict: 探索结果，包含实体、关系、内容、探索路径等信息
        
        实现思路：
        1. 记录开始时间，用于性能监控
        2. 检查起始实体是否为空，如果为空则返回空结果
        3. 重置探索状态（已访问节点、探索路径）
        4. 计算查询的嵌入向量，用于后续相似度计算
        5. 添加起始实体到探索路径
        6. 根据查询内容生成探索策略
        7. 多步探索过程：
           a. 获取当前实体的邻居节点
           b. 动态计算当前步骤的探索宽度
           c. 为邻居节点评分，综合考虑多种因素
           d. 让LLM决定下一步探索的实体
           e. 更新已访问节点集合
           f. 获取新发现实体的详细信息
           g. 收集关系信息
           h. 收集相关内容信息
           i. 尝试获取社区信息
           j. 记录当前步骤的探索路径
           k. 更新当前实体集合
        8. 根据查询相关性对收集的内容进行排序
        9. 添加探索路径、已访问实体和统计信息
        10. 记录总耗时，返回完整的探索结果
        
        技术特点：
        - 多步探索：支持多步骤的深入探索
        - 动态参数：自适应调整探索宽度
        - LLM辅助决策：利用大语言模型进行智能决策
        - 多维度评分：综合多种因素评估节点相关性
        - 全面收集：获取实体、关系、内容和社区等多方面信息
        - 性能监控：记录各环节的耗时和性能指标
        
        业务意义：
        - 实现深度知识探索，超越简单的关键词匹配
        - 模拟人类探索过程，逐步深入相关领域
        - 提高对复杂查询的理解和回答能力
        - 发现实体之间的潜在连接和路径
        - 为用户提供全面、结构化的信息
        - 显著提升Graph-RAG系统的检索质量
        """
        start_time = time.time()
        
        if not starting_entities:
            return {
                "entities": [],
                "relationships": [],
                "content": [],
                "exploration_path": []
            }
            
        # 重置状态
        self.visited_nodes = set(starting_entities)
        self.exploration_path = []
        query_embedding = self.embeddings.embed_query(query)
        
        # 添加起始节点到探索路径
        for entity in starting_entities:
            self.exploration_path.append({
                "step": 0,
                "node_id": entity,
                "action": "start",
                "reasoning": "初始实体"
            })
        
        # 根据查询内容生成探索策略
        exploration_strategy = self._generate_exploration_strategy(query, starting_entities)
        self.performance_metrics["strategy_generation_time"] = time.time() - start_time
        
        current_entities = starting_entities
        results = {
            "entities": [],
            "relationships": [],
            "content": [],
            "communities": []
        }
        
        # 多步探索
        for step in range(max_steps):
            step_start_time = time.time()
            
            if not current_entities:
                break
                
            # 1. 找出邻居节点
            neighbors = self._get_neighbors(current_entities)
            if not neighbors:
                break
                
            # 2. 动态宽度控制
            current_width = self._calculate_adaptive_width(
                step, 
                query, 
                neighbors, 
                base_width=exploration_width
            )
            
            # 3. 评估每个邻居与查询的相关性
            scored_neighbors = self._score_neighbors_enhanced(
                neighbors, 
                query, 
                query_embedding,
                exploration_strategy
            )
            
            # 4. 让LLM决定探索方向
            next_entities, reasoning = self._decide_next_step_with_memory(
                query, 
                current_entities, 
                scored_neighbors, 
                current_width,
                step
            )
            
            # 5. 更新已访问节点
            new_entities = [e for e in next_entities if e not in self.visited_nodes]
            self.visited_nodes.update(new_entities)
            
            # 6. 获取新发现实体的内容
            entity_info = self._get_entity_info(new_entities)
            results["entities"].extend(entity_info)
            
            # 7. 收集关系信息
            rel_info = self._get_relationship_info(new_entities)
            results["relationships"].extend(rel_info)
            
            # 8. 收集内容信息（如chunk）
            content_info = self._get_content_info(new_entities)
            results["content"].extend(content_info)
            
            # 9. 尝试获取所属社区信息
            community_info = self._get_community_info(new_entities)
            if community_info:
                results["communities"].extend(community_info)
            
            # 10. 记录探索路径
            for entity in new_entities:
                self.exploration_path.append({
                    "step": step + 1,
                    "node_id": entity,
                    "action": "explore",
                    "reasoning": reasoning
                })
            
            # 11. 更新当前实体
            current_entities = new_entities
            
            # 记录每步耗时
            self.performance_metrics[f"step_{step+1}_time"] = time.time() - step_start_time
        
        # 根据查询对所有收集的内容进行最终排序
        results["content"] = self._rank_content_by_relevance(query_embedding, results["content"])
        
        # 添加探索路径、数据统计和性能指标
        results["exploration_path"] = self.exploration_path
        results["visited_entities"] = list(self.visited_nodes)
        results["statistics"] = {
            "entity_count": len(results["entities"]),
            "relationship_count": len(results["relationships"]),
            "content_count": len(results["content"]),
            "path_length": len(self.exploration_path)
        }
        
        # 记录总耗时
        self.performance_metrics["total_time"] = time.time() - start_time
        results["performance_metrics"] = self.performance_metrics
        
        return results
    
    def _generate_exploration_strategy(self, query: str, starting_entities: List[str]) -> Dict[str, Any]:
        """
        为查询生成探索策略
        
        该方法负责基于用户查询和起始实体，生成一个结构化的探索策略，指导后续的图谱探索过程。
        通过利用LLM分析查询意图和起始实体，该方法能够识别应该关注的关系类型、实体类型，
        以及应该避免的关系类型，并为不同类型的关系提供重要性权重，从而使探索过程更有针对性。
        
        参数:
            query: 查询字符串，用户的原始问题
            starting_entities: 起始实体列表，探索的起点
            
        返回:
            Dict: 探索策略，包含关注的关系和实体类型、应避免的关系类型、终止条件和关系权重
        
        实现思路：
        1. 构建提示，要求LLM为给定查询和起始实体生成探索策略
        2. 明确要求LLM提供四个关键信息：探索重点、终止条件、重要程度评分和避免的关系
        3. 指定LLM以JSON格式返回结果，便于后续解析
        4. 调用LLM生成策略
        5. 从LLM响应中提取JSON部分
        6. 解析JSON，得到结构化的探索策略
        7. 如果解析失败，返回默认策略
        8. 实现全面的异常处理，确保方法健壮性
        
        技术特点：
        - LLM辅助生成：利用大语言模型生成智能探索策略
        - 结构化输出：要求LLM以JSON格式返回结果
        - 正则表达式提取：使用正则表达式从LLM响应中提取结构化数据
        - 默认策略：当解析失败时提供合理的默认值
        - 异常处理：完善的错误捕获机制
        
        业务意义：
        - 为探索过程提供明确的指导方向
        - 根据查询特点动态调整探索策略
        - 提高探索效率，避免无关信息干扰
        - 增强系统对不同类型查询的适应能力
        - 为后续的节点评分和决策提供依据
        - 使探索过程更有针对性，显著提升结果质量
        """
        prompt = f"""
        为以下查询生成图谱探索策略，从给定的起始实体开始探索:
        
        查询: "{query}"
        起始实体: {starting_entities}
        
        请提供以下信息:
        1. 探索重点: 探索应该关注哪些类型的关系和实体?
        2. 终止条件: 什么情况下应该终止特定方向的探索?
        3. 重要程度评分: 为不同类型的关系提供重要性权重(0-1)
        
        以JSON格式返回结果:
        {{
            "focus_relations": ["关系类型1", "关系类型2", ...],
            "focus_entity_types": ["实体类型1", "实体类型2", ...],
            "avoid_relations": ["应避免的关系类型1", ...],
            "termination_conditions": ["条件1", ...],
            "relation_weights": {{"关系类型1": 0.9, "关系类型2": 0.7, ...}}
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取JSON部分
            import re
            import json
            
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group(0))
                return strategy
            
            # 如果无法解析，返回默认策略
            return {
                "focus_relations": [],
                "focus_entity_types": [],
                "avoid_relations": [],
                "termination_conditions": [],
                "relation_weights": {}
            }
        except Exception as e:
            print(f"生成探索策略失败: {e}")
            # 默认策略
            return {
                "focus_relations": [],
                "focus_entity_types": [],
                "avoid_relations": [],
                "termination_conditions": [],
                "relation_weights": {}
            }
    
    def _calculate_adaptive_width(self, step, query, neighbors, base_width=3):
        """
        根据查询复杂度和当前步骤动态调整探索宽度
        
        该方法负责智能地调整每一步探索的宽度，根据当前探索步骤、查询复杂度和邻居节点数量等因素，
        动态计算最适合当前情况的探索宽度。这种自适应机制能够在探索初期关注广度，随着探索深入
        逐步聚焦，有效地平衡探索的广度和深度。
        
        参数:
            step: 当前步骤，从0开始计数的探索步骤
            query: 查询字符串，用户的原始问题
            neighbors: 邻居节点列表，当前可探索的候选节点
            base_width: 基础宽度，默认的探索宽度，默认为3
            
        返回:
            int: 调整后的宽度，控制当前步骤探索的实体数量
        
        实现思路：
        1. 计算步骤因素：随着探索深入，逐步减小宽度，避免指数爆炸
        2. 计算邻居因素：根据邻居节点数量动态调整，但设置上限
        3. 计算查询复杂度因素：通过_estimate_query_complexity方法评估查询难度
        4. 综合三个因素计算最终宽度：基础宽度 × 步骤因素 × 邻居因素 × 复杂度因素
        5. 将结果转换为整数
        6. 确保宽度在合理范围内（1-5之间）
        
        技术特点：
        - 多因素综合：同时考虑多个影响因素
        - 动态调整：根据当前状态实时计算
        - 范围控制：确保结果在合理范围内
        - 自适应策略：自动平衡广度和深度
        - 简洁高效：计算逻辑简单但有效
        
        业务意义：
        - 优化探索效率，避免资源浪费
        - 适应不同复杂度的查询需求
        - 在探索初期关注广度，后期聚焦深度
        - 动态平衡探索的全面性和针对性
        - 避免信息过载或信息不足
        - 提高探索结果的相关性和准确性
        """
        # 步骤越深，宽度越小，避免指数爆炸
        step_factor = max(0.5, 1.0 - step * 0.2)
        
        # 邻居节点数量因素 - 邻居越多，宽度越大但有上限
        neighbor_factor = min(1.5, len(neighbors) / 10)
        
        # 查询复杂度因素
        complexity_factor = self._estimate_query_complexity(query)
        
        # 计算最终宽度
        adjusted_width = int(base_width * step_factor * neighbor_factor * complexity_factor)
        
        # 确保宽度在合理范围内
        return max(1, min(5, adjusted_width))
    
    def _estimate_query_complexity(self, query):
        """
        估计查询复杂度
        
        该方法负责评估用户查询的复杂度，为动态调整探索宽度提供依据。通过分析查询的长度、问题数量、
        关键词等特征，该方法能够生成一个复杂度评分，帮助系统更好地适应不同类型的查询需求。
        
        参数:
            query: 查询字符串，用户的原始问题
            
        返回:
            float: 复杂度评分，范围在0.5-1.5之间，数值越大表示查询越复杂
        
        实现思路：
        1. 计算长度因素：基于查询长度，最长支持50个字符，超过则达到最大值1.5
        2. 计算问题因素：统计查询中的问号数量，每增加一个问号增加0.1的复杂度
        3. 定义复杂度关键词列表，包含表示复杂问题的关键词
        4. 计算指标因素：统计查询中包含的复杂度关键词数量，每个关键词增加0.1的复杂度
        5. 综合评分：基础值为0.5，结合长度因素(30%)、问题因素(30%)和指标因素(40%)
        6. 确保最终评分不超过最大值1.5
        
        技术特点：
        - 多维度评估：从多个角度评估查询复杂度
        - 关键词匹配：识别表示复杂查询的特定词汇
        - 加权计算：对不同因素赋予不同权重
        - 范围控制：确保评分在合理范围内
        - 简单高效：计算逻辑简单但有效
        
        业务意义：
        - 为动态探索提供复杂度指导
        - 使系统能够适应不同难度的查询
        - 复杂查询分配更多资源，简单查询更高效
        - 提高系统整体的资源利用效率
        - 增强对复杂推理问题的处理能力
        - 为用户提供更准确的响应
        """
        # 基于查询长度、问号数量和关键词数量的简单启发式方法
        length_factor = min(1.5, len(query) / 50)
        question_marks = query.count("?") + query.count("？")
        question_factor = 1.0 + (question_marks * 0.1)
        
        # 识别复杂问题的关键词
        complexity_indicators = [
            "为什么", "如果", "原因", "关系", "比较", "区别",
            "影响", "分析", "评估", "预测"
        ]
        
        # 检查关键词
        indicator_count = sum(1 for indicator in complexity_indicators if indicator.lower() in query.lower())
        indicator_factor = 1.0 + (indicator_count * 0.1)
        
        # 综合评分,基础值0.5,最大1.5
        complexity = 0.5 + (length_factor * 0.3 + question_factor * 0.3 + indicator_factor * 0.4) / 3
        
        return min(1.5, complexity)
    
    def _get_neighbors(self, entities):
        """
        获取实体的邻居节点
        
        Args:
            entities: 实体ID列表
            
        Returns:
            List: 邻居节点列表
        """
        try:
            query = """
            MATCH (e:__Entity__)-[r]-(neighbor:__Entity__)
            WHERE e.id IN $entity_ids AND NOT neighbor.id IN $visited_ids
            RETURN neighbor.id AS id, neighbor.description AS description,
                   type(r) AS relation_type, startNode(r).id AS source,
                   endNode(r).id AS target,
                   CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight
            LIMIT 100
            """
            
            params = {
                "entity_ids": entities, 
                "visited_ids": list(self.visited_nodes)
            }
            
            result = self.graph.query(query, params=params)
            
            # 结果为空的处理
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换为列表格式
            if isinstance(result, pd.DataFrame):
                neighbors_list = result.to_dict('records')
                return neighbors_list
            else:
                return result
                
        except Exception as e:
            print(f"获取邻居节点失败: {e}")
            return []
    
    def _score_neighbors_enhanced(self, neighbors, query, query_embedding, exploration_strategy):
        """
        增强版邻居评分，考虑策略权重、相似度和关系权重
        
        该方法负责为当前实体的邻居节点计算相关性得分，是整个探索过程的关键环节。通过综合考虑
        语义相似度、策略相关性、关系类型权重和图谱结构等多个因素，该方法能够为每个邻居节点生成
        一个最终评分，帮助系统选择最有价值的节点进行下一步探索。
        
        参数:
            neighbors: 邻居节点列表，待评分的候选节点
            query: 查询字符串，用于上下文理解
            query_embedding: 查询嵌入向量，用于计算语义相似度
            exploration_strategy: 探索策略，提供关系权重等信息
            
        返回:
            List: 评分后的邻居节点列表，按最终得分降序排列
        
        实现思路：
        1. 初始化评分列表
        2. 遍历每个邻居节点
        3. 获取邻居节点的描述信息
        4. 计算语义相似度：使用嵌入模型计算描述与查询的相似度
        5. 计算策略得分：根据探索策略中关系类型的重要性评估
        6. 计算关系权重：基于关系类型分配不同权重
        7. 计算图谱权重：考虑节点在图谱中的结构位置
        8. 计算最终得分：综合多个因素的加权乘积
        9. 将评分结果添加到列表
        10. 实现异常处理，确保即使单个节点评分失败也不影响整体
        11. 按最终得分降序排序
        
        技术特点：
        - 多因素评分：综合考虑多个维度的信息
        - 语义理解：使用向量嵌入计算语义相似度
        - 策略感知：根据探索策略调整评分权重
        - 异常处理：完善的错误捕获机制
        - 结果排序：按相关性得分排序便于后续决策
        
        业务意义：
        - 识别最相关的邻居节点
        - 确保探索过程聚焦于有价值的信息
        - 提供客观的量化标准指导决策
        - 平衡多种因素，避免单一维度评价的局限性
        - 提高探索效率，避免无关信息干扰
        - 显著提升探索结果的相关性和质量
        """
        scored_neighbors = []
        relation_weights = exploration_strategy.get("relation_weights", {})
        focus_relations = exploration_strategy.get("focus_relations", [])
        focus_entity_types = exploration_strategy.get("focus_entity_types", [])
        avoid_relations = exploration_strategy.get("avoid_relations", [])
        
        for neighbor in neighbors:
            # 构建描述文本
            description = neighbor.get('description', '')
            relation_type = neighbor.get('relation_type', '')
            neighbor_id = neighbor.get('id', '')
            
            # 初始权重 - 基础值0.5
            base_weight = 0.5
            
            try:
                # 计算语义相似度
                if description:
                    neighbor_embedding = self.embeddings.embed_query(description)
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(neighbor_embedding).reshape(1, -1)
                    )[0][0]
                else:
                    similarity = 0.0
                    
                # 获取关系权重
                relation_weight = relation_weights.get(relation_type, 1.0)
                
                # 计算策略匹配分数
                strategy_score = 1.0
                
                # 增加关注的关系类型的权重
                if relation_type in focus_relations:
                    strategy_score += 0.5
                
                # 检查实体类型
                entity_type = self._get_entity_type(neighbor_id)
                if entity_type in focus_entity_types:
                    strategy_score += 0.3
                
                # 降低需要避免的关系类型的权重
                if relation_type in avoid_relations:
                    strategy_score -= 0.5
                
                # 添加来自图的原始权重
                graph_weight = float(neighbor.get('weight', 1.0))
                
                # 计算最终得分(语义相似度*策略分数*关系权重*图权重)
                final_score = similarity * strategy_score * relation_weight * graph_weight
                
                # 添加到评分列表
                scored_neighbors.append({
                    "id": neighbor_id,
                    "description": description,
                    "relation_type": relation_type,
                    "source": neighbor.get('source', ''),
                    "target": neighbor.get('target', ''),
                    "similarity": similarity,
                    "strategy_score": strategy_score,
                    "relation_weight": relation_weight,
                    "graph_weight": graph_weight,
                    "final_score": final_score
                })
            except Exception as e:
                print(f"计算节点相似度失败: {e}")
                
        # 按最终得分排序
        return sorted(scored_neighbors, key=lambda x: x['final_score'], reverse=True)
    
    def _get_entity_type(self, entity_id):
        """
        获取实体类型
        
        该方法负责获取指定实体ID的类型信息。实体类型对于理解实体的语义类别和属性非常重要，
        有助于后续的策略生成和决策过程。该方法通过查询图数据库获取实体的标签，并过滤掉通用标签。
        
        参数:
            entity_id: 实体ID，需要获取类型的实体标识
            
        返回:
            str: 实体类型，如果无法获取则返回"unknown"
        
        实现思路：
        1. 构建Cypher查询，匹配指定ID的实体节点
        2. 查询返回实体的所有标签
        3. 执行查询，获取结果
        4. 检查结果是否为空
        5. 如果结果是DataFrame格式，提取第一个记录的标签列表
        6. 过滤掉通用的"__Entity__"标签
        7. 如果存在其他标签，返回第一个标签作为实体类型
        8. 如果没有其他标签，返回"unknown"
        9. 处理可能的异常，确保即使查询失败也返回默认值
        
        技术特点：
        - Cypher查询：使用图数据库查询语言
        - 标签过滤：排除通用标签
        - 结果转换：支持不同格式的结果处理
        - 异常处理：完善的错误捕获机制
        - 默认值处理：在无法获取类型时返回默认值
        
        业务意义：
        - 提供实体的语义类别信息
        - 帮助理解实体的属性和行为特征
        - 为策略生成提供类型相关的依据
        - 增强对实体的分类和管理能力
        - 支持更精确的知识检索和推理
        - 提高系统对实体的语义理解能力
        """
        try:
            query = """
            MATCH (e:__Entity__ {id: $entity_id})
            RETURN labels(e) AS types
            """
            
            result = self.graph.query(query, params={"entity_id": entity_id})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return "unknown"
                
            if isinstance(result, pd.DataFrame):
                types = result.iloc[0]['types']
                # 过滤掉 "__Entity__" 标签
                entity_types = [t for t in types if t != "__Entity__"]
                return entity_types[0] if entity_types else "unknown"
            else:
                # 处理其他结果格式
                return "unknown"
        except Exception as e:
            print(f"获取实体类型失败: {e}")
            return "unknown"
    
    def _decide_next_step_with_memory(self, query, current_entities, scored_neighbors, width, current_step):
        """
        让LLM决定下一步探索方向，考虑已探索的记忆
        
        该方法负责利用大语言模型来智能地决定下一步探索哪些实体，是整个探索过程的决策核心。
        它会考虑当前的查询、已探索的实体、评分后的邻居节点、探索宽度和当前步骤等因素，
        同时利用记忆机制避免重复决策，显著提高系统效率。
        
        参数:
            query: 查询字符串，用户的原始问题
            current_entities: 当前实体列表，当前步骤正在探索的实体
            scored_neighbors: 评分后的邻居列表，按相关性排序的候选实体
            width: 探索宽度，限制每步可选择的实体数量
            current_step: 当前步骤，从0开始计数
            
        返回:
            Tuple[List[str], str]: 下一步实体列表和推理过程
        
        实现思路：
        1. 构建记忆键，基于查询和当前实体列表
        2. 检查是否存在相同情况的历史决策记录
        3. 如果存在有效的记忆记录，直接返回记忆中的结果
        4. 如果没有记忆或记忆已过期，构建LLM提示
        5. 在提示中包含查询、当前实体、当前步骤和评分最高的候选实体
        6. 要求LLM按照指定格式返回选择的实体和推理过程
        7. 调用LLM生成决策
        8. 使用正则表达式从LLM响应中提取实体列表和推理过程
        9. 如果解析失败，使用得分最高的几个实体作为备选
        10. 将决策保存到记忆中，以供将来使用
        11. 实现完善的异常处理，确保方法健壮性
        
        技术特点：
        - 记忆机制：缓存决策结果，避免重复计算
        - LLM决策：利用大语言模型进行智能选择
        - 结构化提示：精心设计的提示格式提高决策质量
        - 正则表达式提取：从非结构化响应中提取结构化数据
        - 降级策略：当LLM失败时使用启发式方法
        - 异常处理：全面的错误捕获和恢复机制
        
        业务意义：
        - 智能选择最有价值的探索方向
        - 平衡探索的相关性和多样性
        - 避免重复探索相同的路径
        - 提高探索效率和质量
        - 为用户提供合理的决策理由
        - 增强系统的可解释性
        """
        memory_key = f"{query}_{','.join(sorted(current_entities))}"
        
        # 检查是否有记忆
        if memory_key in self.exploration_memory:
            remembered = self.exploration_memory[memory_key]
            # 检查记忆是否过期(根据步骤差异判断)
            if remembered["step"] == current_step:
                return remembered["entities"], remembered["reasoning"]
        
        # 构建提示
        prompt = f"""
        我正在使用Chain of Exploration方法探索知识图谱，以回答问题: "{query}"
        
        当前探索的实体有:
        {', '.join(current_entities)}
        
        当前是探索的第{current_step+1}步，需要决定下一步探索哪些实体。
        
        下面是一些可能的下一步探索选项(已按综合得分排序):
        """
        
        # 添加前10个最相关的选项(或全部如果少于10个)
        top_options = scored_neighbors[:10] if len(scored_neighbors) > 10 else scored_neighbors
        for i, neighbor in enumerate(top_options):
            prompt += f"{i+1}. {neighbor['id']} (得分: {neighbor['final_score']:.2f})\n"
            prompt += f"   - 描述: {neighbor['description']}\n"
            prompt += f"   - 关系类型: {neighbor['relation_type']} (连接到: {neighbor['source'] if neighbor['target'] in current_entities else neighbor['target']})\n\n"
            
        prompt += f"""
        请选择最多{width}个最有价值的实体继续探索。你的选择应该:
        1. 平衡相关性和覆盖广度
        2. 避免过于相似的实体
        3. 考虑探索多种关系类型的可能性
        4. 优先选择有助于回答问题的实体
        
        要求回复格式:
        ```
        实体: [实体1, 实体2, ...]
        推理: 你的选择理由...
        ```
        """
        
        try:
            # 调用LLM决策
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析结果
            entities_match = re.search(r'实体:\s*\[(.*?)\]', content, re.DOTALL)
            reasoning_match = re.search(r'推理:(.*?)($|```)', content, re.DOTALL)
            
            selected_entities = []
            reasoning = "无具体推理过程"
            
            if entities_match:
                entities_str = entities_match.group(1).strip()
                # 处理实体列表
                if entities_str:
                    # 分割并清理实体
                    entities = [e.strip().strip('"\'') for e in entities_str.split(',')]
                    selected_entities = [e for e in entities if e]
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # 如果没有解析出实体，使用得分最高的几个
            if not selected_entities:
                selected_entities = [n['id'] for n in scored_neighbors[:width]]
                reasoning = "基于相似度得分自动选择"
            
            # 保存到记忆
            self.exploration_memory[memory_key] = {
                "entities": selected_entities,
                "reasoning": reasoning,
                "step": current_step
            }
            
            return selected_entities, reasoning
                
        except Exception as e:
            print(f"LLM决策失败: {e}")
            # 出错时使用简单启发式方法
            fallback_entities = [n['id'] for n in scored_neighbors[:width]]
            fallback_reasoning = "决策过程出错，默认选择得分最高的实体"
            return fallback_entities, fallback_reasoning
    
    def _get_entity_info(self, entities):
        """
        获取实体详细信息
        
        该方法负责根据实体ID列表获取这些实体的详细信息，包括实体的ID、描述和类型标签。
        这些信息对于理解实体的含义和上下文非常重要，是构建完整探索结果的关键组成部分。
        
        参数:
            entities: 实体ID列表，需要获取详细信息的实体集合
            
        返回:
            List: 实体信息列表，每个元素是包含id、description和types的字典
        
        实现思路：
        1. 检查实体列表是否为空，如果为空直接返回空列表
        2. 构建Cypher查询，匹配指定ID的实体节点
        3. 查询返回实体的ID、描述和类型标签
        4. 执行查询，获取结果
        5. 检查结果是否为空
        6. 如果结果是DataFrame格式，将其转换为字典列表
        7. 实现异常处理，确保即使查询失败也不会影响整体功能
        
        技术特点：
        - Cypher查询：使用图数据库查询语言
        - 结果转换：支持不同格式的结果处理
        - 异常处理：完善的错误捕获机制
        - 空值处理：对空输入和空结果的处理
        
        业务意义：
        - 提供实体的详细上下文信息
        - 帮助理解实体的类型和描述
        - 为后续的评分和决策提供基础信息
        - 丰富探索结果的数据维度
        - 增强用户对探索过程的理解
        - 支持更准确的知识检索和推理
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN e.id AS id, e.description AS description,
                   labels(e) AS types
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取实体信息失败: {e}")
            return []
    
    def _get_relationship_info(self, entities):
        """
        获取实体关系信息
        
        该方法负责获取实体之间的关系信息，包括源实体、目标实体、关系类型、描述和权重等。
        这些关系信息对于理解实体之间的连接和相互作用非常重要，是构建完整知识图谱视图的核心部分。
        
        参数:
            entities: 实体ID列表，需要获取关系信息的实体集合
            
        返回:
            List: 关系信息列表，每个元素包含source、target、type、description和weight等字段
        
        实现思路：
        1. 检查实体列表是否为空，如果为空直接返回空列表
        2. 构建Cypher查询，匹配实体之间的关系
        3. 确保查询中的实体与已访问的节点相关联
        4. 查询返回源实体、目标实体、关系类型、描述和权重
        5. 使用CASE语句处理权重可能为NULL的情况，提供默认值1.0
        6. 执行查询，获取结果
        7. 检查结果是否为空
        8. 如果结果是DataFrame格式，将其转换为字典列表
        9. 实现异常处理，确保即使查询失败也不会影响整体功能
        
        技术特点：
        - Cypher高级查询：使用复杂的图查询模式
        - 数据完整性：处理可能缺失的属性
        - 结果转换：支持不同格式的结果处理
        - 异常处理：完善的错误捕获机制
        - 空值处理：对空输入和空结果的处理
        
        业务意义：
        - 揭示实体之间的连接和关系
        - 提供更完整的知识图谱视图
        - 帮助理解实体间的语义联系
        - 为后续的路径分析提供数据基础
        - 增强对复杂知识结构的理解
        - 支持更准确的推理和回答
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1.id IN $entity_ids AND e2.id IN $visited_ids
            RETURN startNode(r).id AS source, endNode(r).id AS target,
                   type(r) AS type, r.description AS description,
                   CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1.0 END AS weight
            """
            
            result = self.graph.query(
                query, 
                params={
                    "entity_ids": entities,
                    "visited_ids": list(self.visited_nodes)
                }
            )
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取关系信息失败: {e}")
            return []
    
    def _get_content_info(self, entities):
        """
        获取与实体相关的内容信息
        
        该方法负责获取与指定实体相关的内容信息，即包含这些实体提及的文档片段。
        这些内容是回答用户查询的重要依据，提供了实体相关的具体文本信息。
        
        参数:
            entities: 实体ID列表，需要获取相关内容的实体集合
            
        返回:
            List: 内容信息列表，每个元素包含id和text字段
        
        实现思路：
        1. 检查实体列表是否为空，如果为空直接返回空列表
        2. 构建Cypher查询，匹配提及指定实体的文档片段
        3. 使用MENTIONS关系连接文档片段和实体
        4. 查询返回文档片段的ID和文本内容
        5. 设置LIMIT 20，限制返回结果数量，避免信息过载
        6. 执行查询，获取结果
        7. 检查结果是否为空
        8. 如果结果是DataFrame格式，将其转换为字典列表
        9. 实现异常处理，确保即使查询失败也不会影响整体功能
        
        技术特点：
        - 关系查询：使用MENTIONS关系查找相关内容
        - 结果限制：通过LIMIT控制返回结果数量
        - 结果转换：支持不同格式的结果处理
        - 异常处理：完善的错误捕获机制
        - 空值处理：对空输入和空结果的处理
        
        业务意义：
        - 获取实体相关的具体文本内容
        - 为回答用户查询提供原始素材
        - 丰富探索结果的信息维度
        - 提供上下文相关的文本信息
        - 支持更准确的知识检索和答案生成
        - 增强系统的回答能力和准确性
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS id, c.text AS text
            LIMIT 20
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取内容信息失败: {e}")
            return []
    
    def _get_community_info(self, entities):
        """
        获取实体所属社区信息
        
        该方法负责获取指定实体所属的社区信息，包括社区ID和摘要。社区信息对于理解实体的上下文环境
        和知识群体关系非常重要，能够帮助系统从更高层次理解实体之间的关联模式。
        
        参数:
            entities: 实体ID列表，需要获取社区信息的实体集合
            
        返回:
            List: 社区信息列表，每个元素包含community_id和summary字段
        
        实现思路：
        1. 检查实体列表是否为空，如果为空直接返回空列表
        2. 构建Cypher查询，匹配属于指定实体的社区
        3. 使用IN_COMMUNITY关系连接实体和社区
        4. 查询返回社区的ID和摘要
        5. 使用DISTINCT确保每个社区只返回一次
        6. 执行查询，获取结果
        7. 检查结果是否为空
        8. 如果结果是DataFrame格式，将其转换为字典列表
        9. 实现异常处理，确保即使查询失败也不会影响整体功能
        
        技术特点：
        - 关系查询：使用IN_COMMUNITY关系查找所属社区
        - 去重处理：通过DISTINCT确保结果唯一
        - 结果转换：支持不同格式的结果处理
        - 异常处理：完善的错误捕获机制
        - 空值处理：对空输入和空结果的处理
        
        业务意义：
        - 获取实体所属的知识社区
        - 理解实体的群体上下文
        - 发现实体间的隐含关系模式
        - 从更高层次理解知识结构
        - 支持基于社区的知识检索和组织
        - 增强系统对知识网络的整体理解
        """
        if not entities:
            return []
            
        try:
            query = """
            MATCH (e:__Entity__)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE e.id IN $entity_ids
            RETURN DISTINCT c.id AS community_id, c.summary AS summary
            """
            
            result = self.graph.query(query, params={"entity_ids": entities})
            
            if not result or (hasattr(result, 'empty') and result.empty):
                return []
                
            # 转换结果
            if isinstance(result, pd.DataFrame):
                return result.to_dict('records')
            return result
            
        except Exception as e:
            print(f"获取社区信息失败: {e}")
            return []
    
    def _rank_content_by_relevance(self, query_embedding, content_list):
        """
        根据与查询的相关性排序内容
        
        该方法负责根据内容与查询的语义相关性对内容列表进行排序，确保最相关的内容排在前面。
        通过计算内容文本与查询的嵌入向量之间的余弦相似度，该方法能够评估内容的相关程度，
        为用户提供最相关、最有价值的信息。
        
        参数:
            query_embedding: 查询嵌入向量，用户查询的语义表示
            content_list: 内容列表，待排序的文档片段
            
        返回:
            List: 排序后的内容列表，按相关性得分降序排列
        
        实现思路：
        1. 检查内容列表是否为空，如果为空直接返回空列表
        2. 初始化评分内容列表
        3. 遍历每个内容项
        4. 提取内容的文本信息
        5. 如果文本为空，跳过该项
        6. 计算文本的嵌入向量
        7. 计算文本嵌入与查询嵌入的余弦相似度
        8. 将相似度作为相关性得分添加到内容项中
        9. 处理可能的异常，确保单个内容评分失败不影响整体
        10. 按相关性得分降序排序内容列表
        
        技术特点：
        - 语义相似度计算：使用向量嵌入和余弦相似度
        - 异常处理：完善的错误捕获机制
        - 空值处理：对空文本和空输入的处理
        - 结果排序：按相关性降序排列
        - 数据增强：为内容添加相关性得分
        
        业务意义：
        - 确保最相关的内容优先展示
        - 提高信息检索的准确性
        - 减少用户查找相关信息的时间
        - 提升系统的整体搜索质量
        - 为回答用户查询提供最相关的素材
        - 增强用户对系统的信任度
        """
        if not content_list:
            return []
            
        scored_content = []
        
        for content in content_list:
            text = content.get("text", "")
            
            if not text:
                continue
                
            try:
                # 计算文本嵌入
                text_embedding = self.embeddings.embed_query(text)
                
                # 计算相似度
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(text_embedding).reshape(1, -1)
                )[0][0]
                
                # 添加相似度分数
                scored_item = content.copy()
                scored_item["relevance_score"] = similarity
                scored_content.append(scored_item)
                
            except Exception as e:
                print(f"计算内容相似度失败: {e}")
                scored_content.append(content)
        
        # 按相关性排序
        return sorted(scored_content, key=lambda x: x.get("relevance_score", 0), reverse=True)

    async def explore_async(self, query: str, starting_entities: List[str], max_steps: int = 5):
        """
        异步执行探索过程
        
        该方法提供了异步接口，允许在不阻塞主线程的情况下执行探索过程，并通过进度更新生成器实时报告探索进度。
        它是explore方法的异步版本，为Web应用或需要响应用户输入的场景提供了更好的用户体验。
        
        参数:
            query: 用户查询，原始问题
            starting_entities: 起始实体列表，作为探索的起点
            max_steps: 最大探索步数，控制探索的深度，默认为5
            
        返回:
            Tuple[Dict, AsyncGenerator]: 探索结果和进度更新生成器
        
        实现思路：
        1. 定义内部进度更新生成器函数，用于实时返回探索状态
        2. 初始化进度更新存储字典
        3. 创建异步任务，执行实际的探索实现
        4. 返回结果和进度生成器
        
        内部进度生成器实现:
        1. 首先发送开始状态更新
        2. 小延迟后，在每一步探索完成后发送状态更新
        3. 最后发送完成状态更新
        
        技术特点：
        - 异步接口：支持异步操作，不阻塞主线程
        - 进度报告：通过异步生成器实时报告进度
        - 任务分离：将探索实现和进度报告分离
        - 兼容性：在保持功能不变的情况下提供异步接口
        - 可扩展性：易于集成到异步应用框架
        
        业务意义：
        - 提高用户体验：实时显示探索进度
        - 支持并发操作：允许系统同时处理多个探索请求
        - 适用于Web应用：与现代Web框架良好集成
        - 增强系统响应性：即使在长时间探索过程中也能响应其他请求
        - 提供进度可视化：便于前端展示探索状态
        - 适应高并发场景：提高系统的整体吞吐量
        """
        async def progress_generator():
            """生成进度更新"""
            yield {"status": "started", "message": "开始探索过程"}
            
            # 等待探索开始
            await asyncio.sleep(0.1)
            
            for step in range(max_steps):
                if step in self.progress_updates:
                    yield self.progress_updates[step]
                    
                await asyncio.sleep(0.5)
                
            # 最终更新
            if "final" in self.progress_updates:
                yield self.progress_updates["final"]
        
        # 初始化进度更新存储
        self.progress_updates = {}
        
        # 创建任务
        exploration_task = asyncio.create_task(self._explore_async_impl(
            query, starting_entities, max_steps
        ))
        
        # 返回结果和进度生成器
        return await exploration_task, progress_generator()
        
    async def _explore_async_impl(self, query, starting_entities, max_steps):
        """异步探索实现"""
        # 包装同步方法
        def sync_explore():
            return self.explore(query, starting_entities, max_steps)
            
        # 更新进度
        self.progress_updates[0] = {
            "status": "exploring", 
            "step": 0,
            "message": f"开始从{len(starting_entities)}个起始实体探索"
        }
        
        # 执行同步探索
        result = await asyncio.get_event_loop().run_in_executor(None, sync_explore)
        
        # 更新最终进度
        self.progress_updates["final"] = {
            "status": "completed",
            "message": f"探索完成，共发现{len(result.get('entities', []))}个实体，{len(result.get('content', []))}条内容"
        }
        
        return result