from typing import List, Dict, Any
import time
import numpy as np
import jieba.analyse
import re
from sklearn.metrics.pairwise import cosine_similarity

class CommunityAwareSearchEnhancer:
    """
    社区感知搜索增强器
    
    整合社区检测结果到搜索流程中，提供更有结构化的知识视角。该类是Graph-RAG系统
    中实现社区感知搜索的核心组件，通过分析知识图谱中的社区结构，为搜索过程提供
    结构化的背景知识和优化的搜索策略。
    
    核心功能：
    - 查找与查询相关的知识社区
    - 提取社区中的关键实体和关系
    - 基于社区知识生成优化的搜索策略
    - 提供时序信息分析和结构化知识整合
    - 缓存搜索结果以提高性能
    
    设计特点：
    - 向量相似度计算：使用语义相似度识别相关社区
    - 多维度评分机制：结合语义相似度、关键词匹配和社区重要性
    - 知识结构化：将非结构化文本组织为实体、关系和摘要
    - 缓存机制：减少重复计算，提高响应速度
    - 异常处理：全面的错误捕获和降级策略
    """
    
    def __init__(self, graph, embeddings_model, llm):
        """
        初始化社区感知搜索增强器
        
        该方法负责设置社区感知搜索增强器的核心组件，包括图数据库连接、向量嵌入模型和语言模型。
        它是社区感知搜索功能的基础配置步骤，为后续的搜索增强操作提供必要的资源。
        
        参数:
            graph: Neo4j图数据库连接，用于查询知识社区、实体和关系信息
            embeddings_model: 向量嵌入模型，用于计算查询与社区之间的语义相似度
            llm: 语言模型，用于生成基于社区知识的搜索策略
        
        实现思路：
        1. 保存图数据库连接，用于后续查询社区信息
        2. 保存向量嵌入模型，用于计算语义相似度
        3. 保存语言模型，用于生成搜索策略
        4. 初始化缓存字典，用于存储和复用搜索结果
        
        技术特点：
        - 依赖注入：通过构造函数注入依赖组件，提高代码的可测试性和灵活性
        - 缓存机制：使用内存缓存减少重复计算
        - 简洁设计：保持初始化逻辑简单明了
        
        业务意义：
        - 为搜索增强提供必要的组件和资源
        - 配置社区感知功能的核心参数
        - 优化性能，减少重复计算
        """
        self.graph = graph
        self.embeddings = embeddings_model
        self.llm = llm
        self.cache = {}  # 缓存社区检索结果
        
    def enhance_search(self, query: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        增强搜索过程
        
        该方法是社区感知搜索增强器的主入口，负责协调整个搜索增强流程，包括查找相关社区、
        提取社区知识和生成优化的搜索策略。它是Graph-RAG系统中实现社区感知搜索的核心
        功能，通过利用知识图谱中的社区结构，显著提高搜索的相关性和结构化程度。
        
        参数:
            query: 用户查询，用户提出的原始问题
            keywords: 提取的关键词字典，包含高级和低级关键词
            
        返回:
            Dict: 增强的搜索上下文，包含社区信息、搜索策略和搜索时间
        
        实现思路：
        1. 检查缓存，避免重复计算
        2. 记录搜索开始时间，用于性能监控
        3. 查找与查询相关的社区
        4. 提取社区中的知识（实体、关系、摘要）
        5. 生成基于社区知识的搜索策略
        6. 构建完整的增强上下文
        7. 缓存结果，提高后续查询性能
        8. 返回增强上下文
        
        技术特点：
        - 缓存优化：使用查询作为键，缓存结果避免重复计算
        - 性能监控：记录搜索时间，便于性能优化
        - 模块化设计：将功能分解为独立方法，提高代码可维护性
        - 异常处理：在各子方法中处理异常，确保整体流程稳定
        - 结构化输出：提供标准化的增强上下文格式
        
        业务意义：
        - 显著提高搜索的相关性和准确性
        - 提供结构化的背景知识，增强推理质量
        - 生成优化的搜索策略，提高后续查询效率
        - 整合社区知识，提供更全面的信息视角
        - 优化系统性能，减少重复计算
        """
        # 缓存检查
        cache_key = f"comm_search:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        search_start = time.time()
        
        # 步骤1: 找到相关社区
        relevant_communities = self.find_relevant_communities(query, keywords)
        
        # 步骤2: 提取社区知识
        if relevant_communities:
            community_knowledge = self.extract_community_knowledge(relevant_communities)
        else:
            community_knowledge = {"entities": [], "relationships": [], "summaries": []}
        
        # 步骤3: 创建增强上下文
        enhanced_context = {
            "community_info": community_knowledge,
            "search_strategy": self.generate_search_strategy(query, community_knowledge),
            "search_time": time.time() - search_start
        }
        
        # 缓存结果
        self.cache[cache_key] = enhanced_context
        return enhanced_context
        
    def find_relevant_communities(self, query: str, 
                                 keywords: Dict[str, List[str]], 
                                 top_k: int = 3) -> List[Dict]:
        """
        查找与查询最相关的社区
        
        该方法负责识别与用户查询最相关的知识社区，是社区感知搜索的核心步骤之一。它通过
        计算查询与社区摘要的语义相似度、关键词匹配度和考虑社区重要性等多维度因素，
        为用户查询找到最相关的知识社区，为后续的知识提取提供基础。
        
        参数:
            query: 用户查询，用户提出的原始问题
            keywords: 查询关键词，包含高级和低级关键词
            top_k: 返回的社区数量，默认为3个最相关社区
            
        返回:
            List[Dict]: 相关社区列表，每个社区包含ID、得分和摘要信息
        
        实现思路：
        1. 计算查询的向量嵌入，用于后续相似度计算
        2. 从图数据库查询社区信息，获取候选社区列表（最多20个）
        3. 对每个社区进行评分计算：
           a. 计算查询与社区摘要的语义相似度
           b. 计算关键词匹配得分（高级关键词权重更高）
           c. 获取并归一化社区重要性评分
           d. 综合上述因素计算最终得分
        4. 对社区按最终得分排序，返回top_k个最相关社区
        5. 实现全面的异常处理，确保方法健壮性
        
        技术特点：
        - 向量语义相似度：使用余弦相似度计算语义相关性
        - 多维度评分：结合语义相似度、关键词匹配和社区重要性
        - 差异化权重：高级关键词权重高于低级关键词
        - 异常处理：对每个社区的处理都有独立的异常捕获
        - Cypher查询：使用Neo4j图数据库查询语言获取社区信息
        
        业务意义：
        - 为用户查询找到最相关的知识社区，提高信息检索效率
        - 多维度评分确保社区选择的全面性和准确性
        - 社区重要性考虑，优先选择更有价值的社区
        - 为后续的知识提取和搜索策略生成提供关键输入
        """
        # 嵌入查询文本
        query_embedding = self.embeddings.embed_query(query)
        
        # 查询社区信息
        community_query = """
        MATCH (c:__Community__)
        WHERE c.summary IS NOT NULL
        RETURN c.id AS community_id, c.summary AS summary, 
               c.community_rank AS rank
        ORDER BY c.community_rank DESC
        LIMIT 20
        """
        
        try:
            communities = self.graph.query(community_query)
            
            # 如果找不到社区，返回空列表
            if not communities:
                return []
                
            # 计算社区与查询的相关性
            scored_communities = []
            for comm in communities:
                # 计算语义相似度
                if not comm.get('summary'):
                    continue
                    
                try:
                    comm_embedding = self.embeddings.embed_query(comm['summary'])
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(comm_embedding).reshape(1, -1)
                    )[0][0]
                    
                    # 关键词匹配得分
                    high_level_kw = keywords.get('high_level', [])
                    low_level_kw = keywords.get('low_level', [])
                    
                    kw_score = sum(1 for kw in high_level_kw 
                                   if kw.lower() in comm['summary'].lower()) * 2.0
                    kw_score += sum(0.5 for kw in low_level_kw 
                                    if kw.lower() in comm['summary'].lower())
                    
                    # 社区重要性（如果有）
                    importance = comm.get('rank', 1) or 1
                    if isinstance(importance, str):
                        try:
                            importance = float(importance)
                        except:
                            importance = 1.0
                    
                    # 归一化重要性
                    importance_norm = min(importance / 10.0, 1.0)
                    
                    # 综合得分 (语义相似度、关键词匹配、社区重要性)
                    final_score = (similarity * 0.6 + 
                                  (min(kw_score, 5) / 5.0) * 0.3 + 
                                  importance_norm * 0.1)
                    
                    scored_communities.append({
                        'community_id': comm['community_id'],
                        'score': final_score,
                        'summary': comm['summary']
                    })
                except Exception as e:
                    print(f"计算社区相似度时出错: {e}")
                    continue
            
            # 排序并返回top_k个社区
            return sorted(scored_communities, 
                        key=lambda x: x['score'], 
                        reverse=True)[:top_k]
        
        except Exception as e:
            print(f"查询社区信息时出错: {e}")
            return []
    
    def extract_community_knowledge(self, communities: List[Dict]) -> Dict:
        """
        从社区中提取有用的知识，增加摘要的权重
        
        该方法负责从已识别的相关社区中提取结构化知识，包括社区中的关键实体、实体间关系和社区摘要。
        它是社区感知搜索中知识提取的核心实现，通过从知识图谱中查询相关信息，为用户提供
        结构化的背景知识，显著增强搜索结果的语义丰富度和结构化程度。
        
        参数:
            communities: 相关社区列表，包含最相关的知识社区信息
            
        返回:
            Dict: 社区知识，包括实体列表、关系列表和带时序信息的摘要列表
        
        实现思路：
        1. 检查社区列表是否为空，如果为空则返回空的知识结构
        2. 提取社区ID和摘要信息
        3. 查询社区中的核心实体，按提及次数排序（PageRank思想）
        4. 如果有实体，进一步查询实体之间的关系，计算路径重要性
        5. 为每个社区摘要提取时序信息
        6. 返回包含实体、关系和增强摘要的完整知识结构
        7. 实现全面的异常处理，确保方法健壮性
        
        技术特点：
        - Cypher查询：使用Neo4j查询语言提取图数据
        - 实体重要性评估：基于提及次数计算实体权重
        - 关系路径分析：计算关系路径的重要性
        - 时序信息提取：识别摘要中的时间信息
        - 结构化输出：提供标准化的知识结构
        - 异常处理：全面的错误捕获和降级策略
        
        业务意义：
        - 提供结构化的背景知识，增强搜索和推理质量
        - 识别社区中的重要实体和关系，突出关键信息
        - 提供时序视角，丰富信息维度
        - 为搜索策略生成提供知识基础
        - 增强系统对复杂查询的理解和响应能力
        """
        if not communities:
            return {"entities": [], "relationships": [], "summaries": []}
            
        community_ids = [c['community_id'] for c in communities]
        summaries = [c['summary'] for c in communities]
        
        try:
            # 获取社区中的核心实体，加入PageRank权重
            entity_query = """
            MATCH (c:__Community__)<-[:IN_COMMUNITY]-(e:__Entity__)
            WHERE c.id IN $community_ids
            WITH e, c
            MATCH (chunk:__Chunk__)-[:MENTIONS]->(e)
            WITH e, c, count(chunk) as mention_count
            RETURN e.id AS entity_id, e.description AS description,
                c.id AS community_id, mention_count
            ORDER BY mention_count DESC
            LIMIT 50
            """
            
            entities = self.graph.query(entity_query, params={"community_ids": community_ids})
            
            # 获取实体间的关系，并获取路径重要性
            if entities:
                entity_ids = [e['entity_id'] for e in entities]
                
                rel_query = """
                MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
                WHERE e1.id IN $entity_ids AND e2.id IN $entity_ids
                WITH e1, e2, r
                OPTIONAL MATCH (chunk:__Chunk__)-[:MENTIONS]->(e1)
                WITH e1, e2, r, count(chunk) as e1_mentions
                OPTIONAL MATCH (chunk:__Chunk__)-[:MENTIONS]->(e2)
                WITH e1, e2, r, e1_mentions, count(chunk) as e2_mentions
                WITH e1, e2, r, e1_mentions + e2_mentions as path_importance
                RETURN e1.id AS source, e2.id AS target,
                    type(r) AS relation_type,
                    r.weight AS weight,
                    path_importance
                ORDER BY path_importance DESC
                LIMIT 100
                """
                
                relationships = self.graph.query(rel_query, params={"entity_ids": entity_ids})
            else:
                relationships = []
                
            # 获取社区摘要的时序信息
            summaries_with_time = []
            for summary in summaries:
                # 提取时序信息或其他增强数据
                time_info = self._extract_temporal_info(summary)
                summaries_with_time.append({
                    "summary": summary,
                    "temporal_info": time_info
                })
                
            # 返回增强的结构化知识
            return {
                "entities": entities,
                "relationships": relationships,
                "summaries": summaries_with_time
            }
                
        except Exception as e:
            print(f"提取社区知识时出错: {e}")
            return {"entities": [], "relationships": [], "summaries": []}

    def _extract_temporal_info(self, text):
        """
        提取文本中的时序信息
        
        该方法负责从文本中提取各种格式的时序信息，为社区知识增加时间维度。通过识别
        文本中的日期、年份等时间表达式，系统能够建立知识的时序框架，提高对时间敏感
        查询的响应质量。
        
        参数:
            text: 要分析的文本，通常是社区摘要或其他内容
            
        返回:
            List[str]: 提取的时序信息列表，包含识别出的所有时间表达式
        
        实现思路：
        1. 定义多种常见的时间格式正则表达式模式
        2. 按优先级顺序应用正则表达式（从详细到简略）
        3. 收集所有匹配的时间表达式
        4. 返回完整的时间表达式列表
        
        技术特点：
        - 正则表达式匹配：使用正则表达式识别不同格式的时间表达式
        - 多层级模式：覆盖从具体日期到年份的多种时间粒度
        - 格式多样性：支持中文日期格式和ISO日期格式
        - 简单高效：轻量级实现，性能开销小
        
        业务意义：
        - 为知识增加时间维度，丰富信息结构
        - 支持时间相关查询的更好处理
        - 提供时序框架，帮助用户理解事件发展
        - 增强搜索结果的上下文感知能力
        - 为后续的时间推理提供基础
        """
        # 简单的正则表达式匹配时序信息
        import re
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{4}年\d{1,2}月',
            r'\d{4}-\d{1,2}',
            r'\d{4}年'
        ]
        
        matches = []
        for pattern in time_patterns:
            matches.extend(re.findall(pattern, text))
        
        return matches
    
    def generate_search_strategy(self, query: str, 
                               community_knowledge: Dict) -> Dict:
        """
        基于社区知识生成搜索策略
        
        该方法负责基于用户查询和提取的社区知识智能生成搜索策略，是社区感知搜索的最终输出步骤。
        通过利用LLM分析社区知识，生成结构化的搜索建议，包括后续查询、关注实体和关键词等，
        显著提高搜索的针对性和效率，是Graph-RAG系统智能搜索优化的重要组成部分。
        
        参数:
            query: 用户查询，用户提出的原始问题
            community_knowledge: 社区知识，包含实体、关系和摘要信息
            
        返回:
            Dict: 搜索策略，包含策略类型、后续查询、关注实体和关键词等信息
        
        实现思路：
        1. 检查社区知识中的实体数量，判断是否有足够信息生成增强策略
        2. 如果实体数量不足，返回基本搜索策略
        3. 构建提示，引导LLM基于社区知识生成搜索策略
        4. 调用LLM生成策略内容
        5. 使用jieba提取关键词
        6. 尝试多种方法从LLM响应中提取结构化信息：
           a. 首先尝试提取引号中的查询
           b. 如果失败，尝试提取符合长度要求的句子
        7. 识别可能的关注实体
        8. 构建完整的搜索策略对象
        9. 实现全面的异常处理，在失败时返回降级策略
        
        技术特点：
        - LLM辅助生成：利用大语言模型生成智能搜索策略
        - 多策略提取：使用不同方法从LLM响应中提取结构化信息
        - 关键词提取：使用jieba进行中文关键词提取
        - 自动降级：当实体不足或处理失败时提供降级策略
        - 结构化输出：提供标准化的搜索策略格式
        - 异常处理：全面的错误捕获和恢复机制
        
        业务意义：
        - 生成针对性的后续查询，引导更深入的信息收集
        - 识别关键实体，提供重点关注方向
        - 提取核心关键词，优化搜索相关性
        - 显著提高搜索效率和信息质量
        - 为用户提供多维度的信息探索路径
        - 增强系统对复杂查询的处理能力
        """
        entities = community_knowledge.get("entities", [])
        relationships = community_knowledge.get("relationships", [])
        
        # 如果没有足够的社区信息，返回基本策略
        if len(entities) < 3:
            return {
                "strategy_type": "basic",
                "follow_up_queries": [],
                "focus_entities": []
            }
        
        # 构建提示
        prompt = f"""
        基于用户查询和社区知识，生成一个最多3个后续搜索查询的列表。
        
        用户查询: {query}
        
        社区中的关键实体:
        {', '.join([e['entity_id'] for e in entities[:10]])}
        
        请考虑这些实体之间的关系，生成更深入的查询以获取全面信息。
        返回JSON格式的后续查询和关注实体。
        """
        
        try:
            # 调用LLM生成搜索策略
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(content, topK=10)
            
            # 从原始内容中提取可能的查询
            query_pattern = r'"([^"]+)"'
            queries = re.findall(query_pattern, content)
            
            # 如果没有找到引号引起的查询，尝试提取句子
            if not queries:
                sentence_pattern = r'[？?!！。；;][^？?!！。；;]{5,50}[？?!！。；;]'
                sentences = re.findall(sentence_pattern, content)
                queries = [s.strip() for s in sentences if len(s.strip()) > 10][:3]
            
            # 提取可能的实体
            entities = []
            for line in content.split('\n'):
                if ':' in line or '：' in line:
                    parts = re.split(r'[：:]', line, 1)
                    if len(parts) == 2 and len(parts[1].strip()) > 0:
                        entities.append(parts[1].strip())
            
            # 构建策略对象
            strategy = {
                "strategy_type": "jieba_extracted",
                "follow_up_queries": queries[:3] if queries else [],
                "focus_entities": entities[:5] if entities else keywords[:5],
                "keywords": keywords
            }
            
            return strategy
            
        except Exception as e:
            print(f"生成搜索策略时出错: {e}")
            # 返回基本策略
            return {
                "strategy_type": "fallback",
                "follow_up_queries": [],
                "focus_entities": [e['entity_id'] for e in entities[:5]]
            }