from typing import List, Dict, Any
import time
import numpy as np
import jieba.analyse
import re
from sklearn.metrics.pairwise import cosine_similarity

class CommunityAwareSearchEnhancer:
    """
    社区感知搜索增强器，通过分析知识图谱中的社区结构，提供结构化的背景知识和优化的搜索策略。
    
    核心功能：
    - 查询相关的知识社区（结合语义相似度、关键词匹配和社区重要性）
    - 提取社区中的关键实体和关系
    - 基于社区知识生成优化的搜索策略
    - 提供时序信息分析和结构化知识整合
    - 缓存搜索结果以提高性能
    """
    
    def __init__(self, graph, embeddings_model, llm):
        """
        初始化社区感知搜索增强器

        参数:
            graph: Neo4j图数据库连接，用于查询知识社区、实体和关系信息
            embeddings_model: 向量嵌入模型，用于计算查询与社区之间的语义相似度
            llm: 语言模型，用于生成基于社区知识的搜索策略
        """
        self.graph = graph
        self.embeddings = embeddings_model
        self.llm = llm
        self.cache = {}  # 缓存社区检索结果
        
    def enhance_search(self, query: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        增强搜索过程

        参数:
            query: 用户查询，用户提出的原始问题
            keywords: 提取的关键词字典，包含高级和低级关键词
            
        返回:
            Dict: 增强的搜索上下文，包含社区信息、搜索策略和搜索时间
        """
        # 缓存检查
        cache_key = f"comm_search:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        search_start = time.time()
        
        # 步骤1: 找到相关社区
        relevant_communities = self.find_relevant_communities(query, keywords)
        
        # 步骤2: 提取社区知识（实体、关系、摘要）
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
        查找与查询最相关的社区，计算查询与社区摘要的语义相似度、关键词匹配度和考虑社区重要性等多维度因素
        
        参数:
            query: 用户查询，用户提出的原始问题
            keywords: 查询关键词，包含高级和低级关键词
            top_k: 返回的社区数量，默认为3个最相关社区
            
        返回:
            List[Dict]: 相关社区列表，每个社区包含ID、得分和摘要信息
        """
        # 获取嵌入查询文本
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
                    
                    # 归一化重要性值
                    importance_norm = min(importance / 10.0, 1.0)
                    
                    # 多维度评分,综合得分 (语义相似度、关键词匹配、社区重要性)
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
        从社区中提取有用的结构化知识（包括社区中的关键实体、实体间关系和社区摘要），增加摘要的权重

        参数:
            communities: 相关社区列表，包含最相关的知识社区信息
            
        返回:
            Dict: 社区知识，包括实体列表、关系列表和带时序信息的摘要列表
        """
        if not communities:
            return {"entities": [], "relationships": [], "summaries": []}

        #  提取社区ID和摘要信息
        community_ids = [c['community_id'] for c in communities]
        summaries = [c['summary'] for c in communities]
        
        try:
            # 获取社区中的核心实体，加入PageRank权重（按提及次数排序（PageRank思想））
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
                
            # 获取社区摘要的时序信息，识别摘要中的时间信息
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
        提取文本中的,各种格式的,时序信息，为社区知识增加时间维度。
        
        参数:
            text: 要分析的文本，通常是社区摘要或其他内容
            
        返回:
            List[str]: 提取的时序信息列表，包含识别出的所有时间表达式
        """
        # 简单的正则表达式，匹配常见的格式时序信息
        import re
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{4}年\d{1,2}月',
            r'\d{4}-\d{1,2}',
            r'\d{4}年'
        ]

        # 按优先级顺序应用正则表达式（从详细到简略）
        # 多层级模式：覆盖从具体日期到年份的多种时间粒度
        matches = []
        for pattern in time_patterns:
            matches.extend(re.findall(pattern, text))
        
        return matches
    
    def generate_search_strategy(self, query: str, 
                               community_knowledge: Dict) -> Dict:
        """
        利用 LLM 基于用户查询和提取的社区知识智能生成搜索策略，包括后续查询、关注的实体和关键词等

        参数:
            query: 用户查询，用户提出的原始问题
            community_knowledge: 社区知识，包含实体、关系和摘要信息
            
        返回:
            Dict: 搜索策略，包含策略类型、后续查询、关注实体和关键词等信息
        """
        entities = community_knowledge.get("entities", [])
        relationships = community_knowledge.get("relationships", [])
        
        # 如果没有足够的社区信息（如果实体数量不足），返回基本策略
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

            # 尝试多种方法从LLM响应中提取结构化信息
            # a.首先尝试提取引号中的查询
            # b.如果失败，尝试提取符合长度要求的句子

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