import networkx as nx
from typing import Dict, List
import re
import time

class DynamicKnowledgeGraphBuilder:
    """
    动态知识图谱构建器，在推理过程中实时构建与问题相关的知识子图，支持因果推理和关系发现。
    显著增强系统的推理能力和信息整合能力。
    
    核心功能：
    - 基于种子实体构建知识子图（内存图存储：使用NetworkX管理内存中的知识图谱）
    - 递归探索和扩展知识图谱（支持多层级的图谱扩展）
    - 构建文档层级结构的图谱（支持查询驱动和文档驱动的图谱构建）
    - 从文本块中提取实体和关系
    - 识别图谱中的核心实体（中心性分析：使用PageRank识别重要实体）
    """
    
    def __init__(self, graph, entity_relation_extractor=None):
        """
        初始化动态知识图谱构建器

        参数:
            graph: 图数据库连接，用于查询实体关系信息
            entity_relation_extractor: 实体关系提取器，用于从文本中提取实体和关系
        """
        self.graph = graph
        # 实体关系提取器（可选）
        self.extractor = entity_relation_extractor
        # 初始化NetworkX有向图，作为内存中的知识图谱
        self.knowledge_graph = nx.DiGraph()
        # 初始化种子实体集合，用于后续跟踪
        self.seed_entities = set()
        
    def build_query_graph(self, 
                        query: str, 
                        entities: List[str], 
                        depth: int = 2) -> nx.DiGraph:
        """
        为查询构建动态知识图谱，负责基于用户查询和初始实体列表构建完整的知识子图。
        
        参数:
            query: 用户查询，原始问题
            entities: 初始实体列表，作为图谱构建的种子
            depth: 图谱探索深度，控制递归探索的层级，默认为2
            
        返回:
            nx.DiGraph: 构建的知识图谱，包含实体、关系及其属性
        """
        # 确保有有效的实体
        if not entities:
            return self.knowledge_graph
            
        # 重置图谱
        self.knowledge_graph = nx.DiGraph()
        self.seed_entities = set(entities)
        
        # 添加所有种子实体到图谱，并标记为种子类型
        start_time = time.time()
        for entity in entities:
            self.knowledge_graph.add_node(
                entity, 
                type="seed_entity",
                properties={"source": "query"}
            )
        
        # 递归探索图谱，基于种子实体扩展图谱
        self._explore_graph(entities, current_depth=0, max_depth=depth)
        
        # 添加图谱构建元数据（构建时间、查询、节点数、边数）
        self.knowledge_graph.graph['build_time'] = time.time() - start_time
        self.knowledge_graph.graph['query'] = query
        self.knowledge_graph.graph['entity_count'] = self.knowledge_graph.number_of_nodes()
        self.knowledge_graph.graph['relation_count'] = self.knowledge_graph.number_of_edges()
        
        print(f"构建查询图谱完成，包含 {self.knowledge_graph.number_of_nodes()} 个实体和 "
              f"{self.knowledge_graph.number_of_edges()} 个关系，耗时 "
              f"{time.time() - start_time:.2f}秒")
              
        return self.knowledge_graph
    
    def _explore_graph(self, entities: List[str], current_depth: int, max_depth: int):
        """
        递归探索和扩展图谱，递归地从当前实体集合出发，查询它们的相邻实体和关系，并将这些新发现的实体和关系添加到图谱中。
        通过这种递归扩展机制，系统能够构建一个完整的、多层次的知识子图。
        
        参数:
            entities: 当前层次的实体列表，要探索的实体集合
            current_depth: 当前探索深度，用于控制递归终止
            max_depth: 最大探索深度，设定递归的上限
        """
        # 检查递归终止条件：当前深度达到最大深度或实体列表为空
        if current_depth >= max_depth or not entities:
            return
            
        # 查询实体的相邻节点和关系
        try:
            query = """
            MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
            WHERE e1.id IN $entity_ids
            RETURN e1.id AS source, 
                   e2.id AS target,
                   type(r) AS relation,
                   e2.description AS target_description
            LIMIT 100
            """

            relationships = self.graph.query(
                query, 
                params={"entity_ids": entities}
            )
            # 如果没有找到关系，返回
            if not relationships:
                return
                
            # 收集新发现的实体，添加关系到图谱
            new_entities = []
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                relation = rel['relation']
                
                # 检查目标实体是否已在图谱中
                if target not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        target,
                        type="entity",
                        properties={"description": rel.get('target_description', '')}
                    )
                    new_entities.append(target)
                    
                # 添加边
                if not self.knowledge_graph.has_edge(source, target):
                    self.knowledge_graph.add_edge(
                        source, 
                        target, 
                        type=relation
                    )
            
            # 使用新发现的实体递归调用自身，继续探索下一层
            # 递归探索新发现的实体
            if new_entities:
                self._explore_graph(
                    new_entities, 
                    current_depth + 1, 
                    max_depth
                )
                
        except Exception as e:
            print(f"探索图谱时出错: {e}")
    
    def build_hierarchical_graph(self, documents):
        """
        构建包含文档层级、章节和特殊元素的图谱（基于文档结构）
        它将文档分解为文档节点、章节节点、段落节点和特殊元素节点，并建立它们之间的层级关系，为文档内容的结构化理解提供基础。
        
        参数:
            documents: 文档列表，每个文档包含id、title、sections等信息
            
        返回:
            nx.DiGraph: 构建的层级知识图谱
        """
        # 清理原图谱
        self.knowledge_graph = nx.DiGraph()
        # 遍历每个文档，添加文档节点
        for doc in documents:
            doc_id = doc.get('id')
            self.knowledge_graph.add_node(
                doc_id,
                type="document",
                properties={"title": doc.get('title', ''), "type": doc.get('type', '')}
            )
            
            # 添加章节节点和关系
            for section in doc.get('sections', []):
                section_id = f"{doc_id}_section_{section.get('id')}"
                self.knowledge_graph.add_node(
                    section_id,
                    type="section",
                    properties={"title": section.get('title', ''), "content": section.get('content', '')}
                )
                # 添加文档到章节的关系
                self.knowledge_graph.add_edge(doc_id, section_id, type="HAS_SECTION")
                
                # 添加段落节点和关系
                for i, paragraph in enumerate(section.get('paragraphs', [])):
                    para_id = f"{section_id}_para_{i}"
                    self.knowledge_graph.add_node(
                        para_id,
                        type="paragraph",
                        properties={"content": paragraph, "index": i}
                    )
                    self.knowledge_graph.add_edge(section_id, para_id, type="HAS_PARAGRAPH")
                    
                # 添加特殊元素（图表、公式等）
                for element in section.get('special_elements', []):
                    element_id = f"{section_id}_{element.get('type')}_{element.get('id')}"
                    self.knowledge_graph.add_node(
                        element_id,
                        type=element.get('type'),  # 如：formula, table, figure
                        properties={"content": element.get('content', ''), "description": element.get('description', '')}
                    )
                    # 添加章节到特殊元素的关系
                    self.knowledge_graph.add_edge(section_id, element_id, type=f"HAS_{element.get('type').upper()}")
        
        print(f"构建层级图谱完成，包含 {self.knowledge_graph.number_of_nodes()} 个节点和 "
              f"{self.knowledge_graph.number_of_edges()} 个关系")
        
        return self.knowledge_graph

    def extract_subgraph_from_chunk(self, chunk_text: str, chunk_id: str) -> bool:
        """
        从文本块中提取知识子图（提取实体和关系信息，并构建相应的知识子图）

        参数:
            chunk_text: 文本块内容，要从中提取实体和关系的文本
            chunk_id: 文本块ID，用于标记知识来源
            
        返回:
            bool: 是否成功提取知识并添加到图谱
        """
        if not self.extractor:
            return False
            
        try:
            # 使用实体关系提取器分析文本，使用正则解析提取结果，最后将提取的实体和关系添加到知识图谱中
            extraction_result = self.extractor._process_single_chunk(chunk_text)
            if not extraction_result:
                return False
                
            # 定义正则表达式模式，解析结果
            entity_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
            relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')
            
            # 提取实体并添加到图谱
            for match in entity_pattern.findall(extraction_result):
                entity_id, entity_type, description = match
                if entity_id not in self.knowledge_graph:
                    self.knowledge_graph.add_node(
                        entity_id,
                        type=entity_type,
                        properties={
                            "description": description,
                            "source": f"chunk:{chunk_id}"
                        }
                    )
            
            # 提取关系
            for match in relationship_pattern.findall(extraction_result):
                source_id, target_id, rel_type, description, weight = match
                
                # 确保节点存在
                for node_id in [source_id, target_id]:
                    if node_id not in self.knowledge_graph:
                        self.knowledge_graph.add_node(
                            node_id,
                            type="unknown",
                            properties={
                                "description": "从关系中提取的实体",
                                "source": f"chunk:{chunk_id}"
                            }
                        )
                
                # 添加关系
                self.knowledge_graph.add_edge(
                    source_id,
                    target_id,
                    type=rel_type,
                    properties={
                        "description": description,
                        "weight": float(weight),
                        "source": f"chunk:{chunk_id}"
                    }
                )
            
            return True
        except Exception as e:
            print(f"从文本块提取子图时出错: {e}")
            return False
    
    def get_central_entities(self, limit: int = 5) -> List[Dict]:
        """
        获取图谱中最重要或最中心的实体，使用图算法分析实体的连接情况和影响力。
        首先尝试使用PageRank算法，如果失败则使用图的度中心性作为备选方案，返回排序后的重要实体列表。
        
        参数:
            limit: 返回实体数量，控制返回的实体个数，默认为5
            
        返回:
            List[Dict]: 重要实体列表，每个实体包含id、中心性指标、类型和属性等信息
        """
        if not self.knowledge_graph.nodes:
            return []
            
        try:
            # 使用PageRank算法找出重要节点
            pagerank = nx.pagerank(self.knowledge_graph)
            
            # 按中心性得分排序，取前limit个实体
            top_entities = sorted(
                pagerank.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, score in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "centrality": score,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result
        except Exception as e:
            print(f"计算中心实体时出错: {e}")
            # 使用度中心性作为备选方案
            in_degree = dict(self.knowledge_graph.in_degree())
            out_degree = dict(self.knowledge_graph.out_degree())
            
            # 合并入度和出度，得到总度数
            total_degree = {
                node: in_degree.get(node, 0) + out_degree.get(node, 0)
                for node in set(in_degree) | set(out_degree)
            }
            
            # 按总度数排序，取前limit个实体
            top_entities = sorted(
                total_degree.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # 格式化结果
            result = []
            for entity_id, degree in top_entities:
                node_data = self.knowledge_graph.nodes[entity_id]
                result.append({
                    "id": entity_id,
                    "degree": degree,
                    "type": node_data.get("type", "unknown"),
                    "properties": node_data.get("properties", {})
                })
                
            return result