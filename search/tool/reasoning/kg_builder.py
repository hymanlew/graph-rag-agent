import networkx as nx
from typing import Dict, List
import re
import time

class DynamicKnowledgeGraphBuilder:
    """
    动态知识图谱构建器
    
    在推理过程中实时构建与问题相关的知识子图，支持因果推理和关系发现。该类是Graph-RAG系统
    中实现动态知识图谱构建的核心组件，通过在推理过程中实时构建和扩展与查询相关的知识
    子图，显著增强系统的推理能力和信息整合能力。
    
    核心功能：
    - 基于种子实体构建知识子图
    - 递归探索和扩展知识图谱
    - 构建文档层级结构的图谱
    - 从文本块中提取实体和关系
    - 识别图谱中的核心实体
    
    设计特点：
    - 内存图存储：使用NetworkX管理内存中的知识图谱
    - 递归探索：支持多层级的图谱扩展
    - 多种构建策略：支持查询驱动和文档驱动的图谱构建
    - 中心性分析：使用PageRank识别重要实体
    - 异常处理：完善的错误捕获和降级策略
    """
    
    def __init__(self, graph, entity_relation_extractor=None):
        """
        初始化动态知识图谱构建器
        
        该方法负责设置动态知识图谱构建器的核心组件，包括图数据库连接和可选的实体关系提取器。
        它为后续的知识图谱构建操作提供必要的资源和初始化配置。
        
        参数:
            graph: 图数据库连接，用于查询实体关系信息
            entity_relation_extractor: 实体关系提取器，用于从文本中提取实体和关系
        
        实现思路：
        1. 保存图数据库连接，用于后续查询
        2. 保存实体关系提取器（可选）
        3. 初始化NetworkX有向图作为内存知识图谱
        4. 初始化种子实体集合，用于后续跟踪
        
        技术特点：
        - 依赖注入：通过构造函数注入依赖组件
        - 内存图存储：使用NetworkX管理知识图谱
        - 组件可选：实体关系提取器是可选的，提高灵活性
        - 简洁设计：保持初始化逻辑简单明了
        
        业务意义：
        - 为动态知识图谱构建提供必要的组件和资源
        - 配置图谱构建的核心参数
        - 支持不同场景下的知识图谱构建需求
        """
        self.graph = graph
        self.extractor = entity_relation_extractor
        self.knowledge_graph = nx.DiGraph()  # 内存中的知识图谱
        self.seed_entities = set()  # 种子实体
        
    def build_query_graph(self, 
                        query: str, 
                        entities: List[str], 
                        depth: int = 2) -> nx.DiGraph:
        """
        为查询构建动态知识图谱
        
        该方法是动态知识图谱构建的核心入口，负责基于用户查询和初始实体列表构建完整的知识子图。
        它通过初始化图谱、添加种子实体、递归探索关系等步骤，构建与用户查询相关的知识网络，
        为后续的推理和分析提供结构化的知识基础。
        
        参数:
            query: 用户查询，原始问题
            entities: 初始实体列表，作为图谱构建的种子
            depth: 图谱探索深度，控制递归探索的层级，默认为2
            
        返回:
            nx.DiGraph: 构建的知识图谱，包含实体、关系及其属性
        
        实现思路：
        1. 检查实体列表是否为空，为空则返回空图谱
        2. 重置内部知识图谱和种子实体集合
        3. 记录构建开始时间，用于性能监控
        4. 添加所有种子实体到图谱，并标记为种子类型
        5. 调用递归探索方法，基于种子实体扩展图谱
        6. 添加图谱构建元数据（构建时间、查询、节点数、边数）
        7. 打印构建结果信息，包括实体数、关系数和耗时
        8. 返回构建完成的知识图谱
        
        技术特点：
        - 动态构建：根据查询和实体动态生成知识图谱
        - 递归扩展：支持多层级的关系探索
        - 元数据记录：存储构建过程的关键信息
        - 性能监控：记录构建耗时，便于性能优化
        - 结构化输出：返回标准的NetworkX有向图对象
        
        业务意义：
        - 为用户查询构建相关的知识网络
        - 提供结构化的知识表示，便于推理
        - 支持因果关系分析和路径发现
        - 提高系统对复杂问题的理解能力
        - 为多步骤推理提供知识基础
        """
        # 确保有有效的实体
        if not entities:
            return self.knowledge_graph
            
        # 重置图谱
        self.knowledge_graph = nx.DiGraph()
        self.seed_entities = set(entities)
        
        start_time = time.time()
        
        # 添加种子实体
        for entity in entities:
            self.knowledge_graph.add_node(
                entity, 
                type="seed_entity",
                properties={"source": "query"}
            )
        
        # 递归探索图谱
        self._explore_graph(entities, current_depth=0, max_depth=depth)
        
        # 添加图谱构建元数据
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
        递归探索和扩展图谱
        
        该方法是动态知识图谱构建的核心扩展机制，负责递归地从当前实体集合出发，
        查询它们的相邻实体和关系，并将这些新发现的实体和关系添加到图谱中。
        通过这种递归扩展机制，系统能够构建一个完整的、多层次的知识子图。
        
        参数:
            entities: 当前层次的实体列表，要探索的实体集合
            current_depth: 当前探索深度，用于控制递归终止
            max_depth: 最大探索深度，设定递归的上限
        
        实现思路：
        1. 检查递归终止条件：当前深度达到最大深度或实体列表为空
        2. 构建Cypher查询，查找实体的相邻节点和关系
        3. 执行图数据库查询，获取关系信息
        4. 检查查询结果是否为空，如果为空则直接返回
        5. 初始化一个列表，用于收集新发现的实体
        6. 遍历每个关系，将目标实体和关系添加到图谱中：
           a. 如果目标实体不在图谱中，添加它并记录为新发现
           b. 添加源实体到目标实体的关系边
        7. 使用新发现的实体递归调用自身，继续探索下一层
        8. 实现全面的异常处理，确保探索过程不会中断
        
        技术特点：
        - 递归算法：使用递归实现多层级的图谱探索
        - Cypher查询：利用Neo4j查询语言高效提取关系
        - 增量构建：逐步扩展图谱，避免重复处理
        - 深度控制：通过深度参数限制探索范围
        - 异常处理：完善的错误捕获机制
        
        业务意义：
        - 实现知识图谱的自动扩展和丰富
        - 发现实体之间的潜在连接和关系路径
        - 支持复杂的关系推理和分析
        - 构建完整的知识网络，提高推理质量
        - 提供可配置的探索深度，平衡详细度和性能
        """
        if current_depth >= max_depth or not entities:
            return
            
        # 查询实体的相邻节点和关系
        try:
            # 构建查询
            query = """
            MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
            WHERE e1.id IN $entity_ids
            RETURN e1.id AS source, 
                   e2.id AS target,
                   type(r) AS relation,
                   e2.description AS target_description
            LIMIT 100
            """
            
            # 执行查询
            relationships = self.graph.query(
                query, 
                params={"entity_ids": entities}
            )
            
            # 如果没有找到关系，返回
            if not relationships:
                return
                
            # 收集新发现的实体
            new_entities = []
            
            # 添加关系到图谱
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
        构建包含文档层级、章节和特殊元素的图谱
        
        该方法负责从文档结构出发构建层级化的知识图谱，捕获文档的组织关系和内容结构。
        它将文档分解为文档节点、章节节点、段落节点和特殊元素节点，并建立它们之间的层级关系，
        为文档内容的结构化理解提供基础。
        
        参数:
            documents: 文档列表，每个文档包含id、title、sections等信息
            
        返回:
            nx.DiGraph: 构建的层级知识图谱
        
        实现思路：
        1. 清理原有的知识图谱，准备构建新的层级图谱
        2. 遍历每个文档，添加文档节点：
           a. 创建文档节点，包含标题和类型属性
        3. 遍历每个文档的章节，添加章节节点和关系：
           a. 创建章节节点，包含标题和内容属性
           b. 添加文档到章节的HAS_SECTION关系
        4. 遍历章节的段落，添加段落节点和关系：
           a. 为每个段落创建节点，包含内容和索引
           b. 添加章节到段落的HAS_PARAGRAPH关系
        5. 遍历章节的特殊元素，添加特殊元素节点和关系：
           a. 为图表、公式等特殊元素创建节点
           b. 添加章节到特殊元素的关系
        6. 打印构建结果信息，包括节点数和关系数
        7. 返回构建完成的层级知识图谱
        
        技术特点：
        - 层级构建：按文档结构创建多层次节点
        - ID生成：为不同层级的节点生成唯一标识符
        - 属性丰富：为每个节点添加详细的属性信息
        - 关系类型化：使用不同类型的关系表示不同的层级连接
        - 结构化表示：将非结构化文档转化为结构化的图表示
        
        业务意义：
        - 提供文档的结构化表示，便于内容分析和导航
        - 支持基于层级关系的文档内容推理
        - 便于识别文档中的重要部分和特殊元素
        - 为文档理解提供丰富的上下文信息
        - 支持基于文档结构的信息检索和问答
        """
        # 清理原图谱
        self.knowledge_graph = nx.DiGraph()
        
        for doc in documents:
            doc_id = doc.get('id')
            # 添加文档节点
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
                
                # 添加段落节点
                for i, paragraph in enumerate(section.get('paragraphs', [])):
                    para_id = f"{section_id}_para_{i}"
                    self.knowledge_graph.add_node(
                        para_id,
                        type="paragraph",
                        properties={"content": paragraph, "index": i}
                    )
                    # 添加章节到段落的关系
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
        从文本块中提取知识子图
        
        该方法负责从单个文本块中提取实体和关系信息，并构建相应的知识子图。它通过调用实体
        关系提取器处理文本，然后使用正则表达式解析提取结果，最后将提取的实体和关系添加
        到知识图谱中。这是动态知识图谱构建中从非结构化文本获取结构化知识的关键方法。
        
        参数:
            chunk_text: 文本块内容，要从中提取实体和关系的文本
            chunk_id: 文本块ID，用于标记知识来源
            
        返回:
            bool: 是否成功提取知识并添加到图谱
        
        实现思路：
        1. 检查实体关系提取器是否存在，如果不存在则返回False
        2. 调用实体关系提取器处理文本块
        3. 检查提取结果是否为空，如果为空则返回False
        4. 定义正则表达式模式，用于解析实体和关系
        5. 提取实体信息并添加到图谱：
           a. 使用正则表达式匹配实体信息
           b. 为每个实体创建节点，设置类型和描述属性
           c. 记录实体来源为当前文本块
        6. 提取关系信息并添加到图谱：
           a. 使用正则表达式匹配关系信息
           b. 确保源实体和目标实体都在图谱中，如果不在则创建
           c. 添加实体之间的关系边，设置类型、描述和权重
           d. 记录关系来源为当前文本块
        7. 返回成功标志
        8. 实现全面的异常处理，确保方法健壮性
        
        技术特点：
        - 正则表达式解析：使用复杂的正则表达式解析结构化输出
        - 来源追踪：记录知识的来源文本块
        - 错误处理：全面的异常捕获和错误处理
        - 动态补全：自动创建关系中提到但尚未在图谱中的实体
        - 权重支持：为关系添加权重属性
        
        业务意义：
        - 实现从非结构化文本到结构化知识的转换
        - 丰富知识图谱的内容和关系
        - 支持基于文本的知识发现和推理
        - 追踪知识来源，提高知识可信度
        - 为后续的知识整合和分析提供基础
        """
        if not self.extractor:
            return False
            
        try:
            # 使用实体关系提取器分析文本
            extraction_result = self.extractor._process_single_chunk(chunk_text)
            
            if not extraction_result:
                return False
                
            # 解析结果
            entity_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
            relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')
            
            # 提取实体
            for match in entity_pattern.findall(extraction_result):
                entity_id, entity_type, description = match
                
                # 添加到图谱
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
        获取图谱中最重要的实体
        
        该方法负责识别知识图谱中最重要或最中心的实体，通过使用图算法分析实体的连接情况和
        影响力。它首先尝试使用PageRank算法，如果失败则使用度中心性作为备选方案，返回排序后的
        重要实体列表。这是Graph-RAG系统中识别核心概念和关键信息的重要方法。
        
        参数:
            limit: 返回实体数量，控制返回的实体个数，默认为5
            
        返回:
            List[Dict]: 重要实体列表，每个实体包含id、中心性指标、类型和属性等信息
        
        实现思路：
        1. 检查图谱是否为空，如果为空则返回空列表
        2. 尝试使用PageRank算法计算节点中心性：
           a. 调用NetworkX的PageRank实现
           b. 按中心性得分排序，取前limit个实体
           c. 格式化结果，包含id、centrality、type和properties
        3. 如果PageRank计算失败，使用度中心性作为备选：
           a. 计算节点的入度和出度
           b. 合并入度和出度得到总度数
           c. 按总度数排序，取前limit个实体
           d. 格式化结果，包含id、degree、type和properties
        4. 实现全面的异常处理，确保方法健壮性
        
        技术特点：
        - 双算法支持：主用PageRank，备选度中心性
        - 容错设计：当主要算法失败时自动降级
        - 结果排序：按中心性得分降序排列
        - 结构化输出：返回标准化的实体信息
        - 异常处理：完善的错误捕获和恢复机制
        
        业务意义：
        - 识别知识图谱中的核心概念和重要实体
        - 为查询理解和答案生成提供关键参考
        - 帮助系统聚焦于最相关的信息
        - 提高推理和回答的准确性
        - 支持基于重要性的信息优先级排序
        """
        if not self.knowledge_graph.nodes:
            return []
            
        try:
            # 使用PageRank算法找出重要节点
            pagerank = nx.pagerank(self.knowledge_graph)
            
            # 排序
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
            
            # 合并入度和出度
            total_degree = {
                node: in_degree.get(node, 0) + out_degree.get(node, 0)
                for node in set(in_degree) | set(out_degree)
            }
            
            # 排序
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