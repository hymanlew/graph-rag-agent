"""
增量图谱构建模块

此模块是Graph-RAG系统中实现增量图谱更新功能的核心组件，基于LightRAG理念设计，
提供高效的知识图谱增量更新机制。相比完全重建图谱，增量更新显著提高了系统处理效率，
特别是在处理大规模文档集时，可以仅对变更的部分进行处理，避免不必要的计算开销。

在实际应用中，文档库经常会有新增、修改和删除等变更操作。为了保持知识图谱的准确性和
时效性，系统需要能够识别这些变更并相应地更新图谱结构。该模块实现了这一功能，
包括文件变更检测、新增文件处理、修改文件嵌入更新、删除文件清理等核心功能。

主要功能：
1. 高效检测文件变更 - 识别新增、修改和删除的文件
2. 新文件处理 - 执行文档分块、实体关系抽取和图谱构建
3. 嵌入更新 - 为修改的文件和受影响的实体更新嵌入向量
4. 删除文件清理 - 移除与删除文件相关的图数据，同时保护手动编辑的实体
5. 图结构合并 - 智能合并新旧图结构，确保数据一致性

此模块的设计理念是保持高效性和灵活性，在保证图谱质量的同时，最小化更新开销，
提高系统的可维护性和响应速度。
"""

import time
from typing import Dict, List, Any
from pathlib import Path
import os
import tempfile
import shutil

from rich.console import Console
from rich.table import Table

# 导入LLM模型相关组件
from model.get_models import get_llm_model
# 导入提示模板配置
from config.prompt import system_template_build_graph, human_template_build_graph
# 导入系统配置参数
from config.settings import entity_types, relationship_types, CHUNK_SIZE, OVERLAP, MAX_WORKERS, BATCH_SIZE
# 导入文档处理组件
from processor.document_processor import DocumentProcessor
# 导入图谱构建相关组件
from graph import EntityRelationExtractor, GraphWriter, GraphStructureBuilder
# 导入数据库管理组件
from config.neo4jdb import get_db_manager
# 导入文件变更管理组件
from build.incremental.file_change_manager import FileChangeManager
# 导入嵌入管理组件
from graph.indexing.embedding_manager import EmbeddingManager

class IncrementalGraphUpdater:
    """
    增量图谱更新器，基于LightRAG理念实现高效的增量更新。
    
    该类是Graph-RAG系统增量更新机制的核心实现，负责在现有知识图谱基础上，
    通过检测文件变更并执行相应处理，实现图谱的高效增量更新。与完全重建图谱不同，
    增量更新只处理变更的部分，显著提高了系统效率，特别适用于大型文档库的维护。
    
    设计思路：
    - 变更感知 - 精确识别文件的新增、修改和删除
    - 最小化更新 - 只处理必要的节点和关系，避免重复计算
    - 数据保护 - 优先保留手动编辑的数据，避免被自动化更新覆盖
    - 完整性保证 - 在增量更新过程中维持图谱结构的一致性
    
    主要功能：
    1. 无缝集成新数据到现有图结构 - 新增文件的完整处理流程
    2. 仅更新变更部分，避免重建整个索引 - 针对修改文件的定向更新
    3. 高效合并新旧图结构 - 基于时间戳和ID的智能合并
    4. 保护现有图谱的完整性 - 删除文件时保护孤立实体和手动编辑
    5. 嵌入向量更新 - 智能更新受影响实体和Chunk的嵌入
    6. 图谱统计与监控 - 提供完整的处理统计和可视化展示
    """
    
    def __init__(self, files_dir: str, registry_path: str = "./file_registry.json"):
        """
        初始化增量图谱更新器
        
        初始化过程包括创建所有必要的处理组件、建立数据库连接、设置处理参数和统计数据结构。
        这些组件协同工作，形成完整的增量更新处理流水线。
        
        Args:
            files_dir: 文件目录 - 包含需要监控和处理的文档的目录路径
            registry_path: 文件注册表路径 - 用于存储文件状态信息的JSON文件路径
        """
        # 创建控制台输出对象，用于格式化显示处理状态和结果
        self.console = Console()
        # 获取图数据库连接
        self.graph = get_db_manager().graph
        
        # 初始化文件变更管理器，负责检测文件的新增、修改和删除
        self.file_manager = FileChangeManager(files_dir, registry_path)
        
        # 初始化嵌入管理器，负责更新实体和Chunk的向量嵌入
        self.embedding_manager = EmbeddingManager(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 保存文件目录路径，用于后续处理
        self.files_dir = files_dir
        
        # 初始化LLM模型，用于实体关系抽取
        self.llm = get_llm_model()
        
        # 初始化文档处理器，负责文档解析和分块
        self.document_processor = DocumentProcessor(files_dir, CHUNK_SIZE, OVERLAP)
        
        # 初始化图结构构建器，负责创建文档节点和文本块关系
        self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)
        
        # 初始化实体关系抽取器，使用LLM从文本中提取实体和关系
        self.entity_extractor = EntityRelationExtractor(
            self.llm,
            system_template_build_graph,
            human_template_build_graph,
            entity_types,
            relationship_types,
            max_workers=MAX_WORKERS
        )
        
        # 初始化图写入器，负责将提取的实体和关系写入图数据库
        self.graph_writer = GraphWriter(self.graph, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 初始化处理统计数据结构，用于记录整个更新过程的关键指标
        self.stats = {
            "start_time": None,  # 更新开始时间
            "end_time": None,    # 更新结束时间
            "total_time": 0,     # 总处理时间
            "files_processed": 0, # 处理的文件总数
            "entities_integrated": 0, # 新集成的实体数
            "relations_integrated": 0, # 新集成的关系数
            "entities_updated": 0, # 更新的实体数
            "chunks_updated": 0   # 更新的文本块数
        }
    
    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变更
        
        调用FileChangeManager组件检测指定目录中的文件变更，包括新增、修改和删除的文件。
        这是增量更新流程的第一步，通过比较当前文件状态与上次记录的状态，识别需要处理的变更。
        
        Returns:
            Dict: 包含文件变更信息的字典，键包括：
                - "added": 新增文件列表
                - "modified": 修改文件列表
                - "deleted": 删除文件列表
        """
        return self.file_manager.detect_changes()
    
    def process_new_files(self, added_files: List[str]) -> Dict[str, Any]:
        """
        对新文件执行完整的处理流程（分块、实体抽取、关系创建）
        
        此方法是处理新增文件的核心，执行完整的知识图谱构建流程，包括文件复制、文档处理、
        文本分块、实体关系抽取和图谱写入等步骤。为了确保处理的隔离性和灵活性，
        该方法使用临时目录进行文件处理，避免干扰原始目录结构。
        
        处理流程：
        1. 验证输入文件的有效性
        2. 创建临时目录并复制文件
        3. 修改文档处理器的工作目录指向临时目录
        4. 处理文档并生成文本块
        5. 为文档创建节点和文本块关系
        6. 从文本块中抽取实体和关系
        7. 将抽取的实体和关系写入图数据库
        8. 恢复文档处理器的原始目录设置
        
        Args:
            added_files: 新增文件路径列表 - 包含需要处理的新文件的绝对或相对路径
            
        Returns:
            Dict: 处理结果统计，包含以下键：
                - "files_processed": 成功处理的文件数量
                - "entities_extracted": 从中抽取的实体数量
                - "relations_created": 创建的关系数量
        """
        if not added_files:
            return {"files_processed": 0, "entities_extracted": 0, "relations_created": 0}
        
        results = {
            "files_processed": 0,
            "entities_extracted": 0,
            "relations_created": 0
        }
        
        self.console.print(f"[bold cyan]正在处理 {len(added_files)} 个新文件...[/bold cyan]")
        
        # 打印文件路径以便调试
        for file_path in added_files:
            self.console.print(f"[blue]处理文件路径: {file_path}[/blue]")
            if not os.path.exists(file_path):
                self.console.print(f"[red]警告: 文件不存在: {file_path}[/red]")
        
        # 使用临时目录
        # 1. 创建临时目录并复制新文件
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 复制文件到临时目录
                copy_success = False
                for file_path in added_files:
                    try:
                        if os.path.exists(file_path):
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(file_path, dest_path)
                            self.console.print(f"[green]已复制 {file_path} 到临时目录[/green]")
                            copy_success = True
                        elif os.path.exists(os.path.join(self.files_dir, file_path)):
                            # 尝试将文件路径视为相对于files_dir的路径
                            full_path = os.path.join(self.files_dir, file_path)
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(full_path, dest_path)
                            self.console.print(f"[green]已复制 {full_path} 到临时目录[/green]")
                            copy_success = True
                        else:
                            # 最后尝试直接使用文件名
                            file_name = os.path.basename(file_path)
                            full_path = os.path.join(self.files_dir, file_name)
                            if os.path.exists(full_path):
                                dest_path = os.path.join(temp_dir, file_name)
                                shutil.copy2(full_path, dest_path)
                                self.console.print(f"[green]已复制 {full_path} 到临时目录[/green]")
                                copy_success = True
                            else:
                                self.console.print(f"[red]复制文件失败，文件不存在: {file_path}[/red]")
                    except Exception as e:
                        self.console.print(f"[red]复制文件 {file_path} 到临时目录时出错: {e}[/red]")
                
                if not copy_success:
                    self.console.print("[red]没有成功复制任何文件到临时目录，无法继续处理[/red]")
                    return results
                
                # 2. 保存原始目录并临时修改
                original_dir = self.document_processor.directory_path
                self.document_processor.directory_path = temp_dir
                self.document_processor.file_reader.directory_path = temp_dir
                
                # 3. 处理临时目录中的文件
                processed_documents = self.document_processor.process_directory()
                
                # 4. 恢复原始目录
                self.document_processor.directory_path = original_dir
                self.document_processor.file_reader.directory_path = original_dir
                
                # 记录处理的文件数
                if processed_documents:
                    results["files_processed"] = len(processed_documents)
                    self.console.print(f"[green]成功处理 {len(processed_documents)} 个文件[/green]")
                    
                    # 5. 构建图谱结构
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            # 创建文档节点
                            self.console.print(f"[blue]为文件 {doc['filename']} 创建文档节点[/blue]")
                            self.struct_builder.create_document(
                                type="local",
                                uri=str(self.files_dir),
                                file_name=doc["filename"],
                                domain="document"
                            )
                            
                            # 创建chunk节点和关系
                            chunks_count = len(doc['chunks']) if doc['chunks'] else 0
                            self.console.print(f"[blue]为文件 {doc['filename']} 创建 {chunks_count} 个文本块节点[/blue]")
                            doc["graph_result"] = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                doc["chunks"]
                            )
                    
                    # 6. 准备实体抽取的数据
                    file_contents_format = []
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            file_contents_format.append([
                                doc["filename"], 
                                doc["content"], 
                                doc["chunks"]
                            ])
                    
                    # 7. 抽取实体和关系
                    if file_contents_format:
                        self.console.print(f"[cyan]开始抽取实体和关系，文件数: {len(file_contents_format)}[/cyan]")
                        
                        total_chunks = sum(len(content[2]) for content in file_contents_format)
                        self.console.print(f"[blue]总计 {total_chunks} 个文本块需要处理[/blue]")
                        
                        processed_chunk_count = 0
                        def progress_callback(i):
                            nonlocal processed_chunk_count
                            processed_chunk_count += 1
                            if processed_chunk_count % 5 == 0 or processed_chunk_count == total_chunks:
                                self.console.print(f"[blue]已处理 {processed_chunk_count}/{total_chunks} 个文本块[/blue]")
                        
                        # 确保禁用缓存以处理新文件
                        original_cache_setting = getattr(self.entity_extractor, 'enable_cache', True)
                        self.entity_extractor.enable_cache = False
                        
                        try:
                            processed_contents = self.entity_extractor.process_chunks(
                                file_contents_format, 
                                progress_callback=progress_callback
                            )
                            
                            # 恢复缓存设置
                            self.entity_extractor.enable_cache = original_cache_setting
                            
                            # 输出处理结果
                            if processed_contents:
                                self.console.print(f"[green]实体抽取完成，已处理 {len(processed_contents)} 个文件[/green]")
                            
                                # 打印调试信息
                                for i, content in enumerate(processed_contents):
                                    if len(content) > 3:
                                        entity_data = content[3]
                                        entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                        relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                        self.console.print(f"[blue]文件 {i+1}: {content[0]}, 抽取了 {entity_count} 个实体和 {relation_count} 个关系[/blue]")
                                    else:
                                        self.console.print(f"[yellow]文件 {i+1}: {content[0]}, 没有返回实体数据[/yellow]")
                            
                                # 8. 处理结果并写入图数据库
                                graph_writer_data = []
                                for doc in processed_documents:
                                    if "chunks" in doc and doc["chunks"] and "graph_result" in doc:
                                        # 查找对应的处理结果
                                        entity_data = None
                                        for content in processed_contents:
                                            if content[0] == doc["filename"] and len(content) > 3:
                                                entity_data = content[3]
                                                break
                                        
                                        if entity_data:
                                            # 估算实体和关系数量
                                            entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                            relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                            self.console.print(f"[green]文件 {doc['filename']} 中识别出 {entity_count} 个实体和 {relation_count} 个关系[/green]")
                                            
                                            # 添加到写入数据
                                            graph_writer_data.append([
                                                doc["filename"],
                                                doc["content"],
                                                doc["chunks"],
                                                doc["graph_result"],
                                                entity_data
                                            ])
                                            
                                            # 更新统计
                                            results["entities_extracted"] += entity_count
                                            results["relations_created"] += relation_count
                                
                                # 9. 写入图数据库
                                if graph_writer_data:
                                    self.console.print(f"[cyan]开始写入 {len(graph_writer_data)} 个文件的图数据...[/cyan]")
                                    self.graph_writer.process_and_write_graph_documents(graph_writer_data)
                                    self.console.print(f"[green]图数据写入完成[/green]")
                                else:
                                    self.console.print("[yellow]没有有效的图数据需要写入[/yellow]")
                            else:
                                self.console.print("[yellow]实体抽取过程没有返回有效结果[/yellow]")
                        
                        except Exception as e:
                            self.console.print(f"[red]实体抽取过程中出错: {e}[/red]")
                            import traceback
                            self.console.print(f"[red]{traceback.format_exc()}[/red]")
                    else:
                        self.console.print("[yellow]没有找到可用于抽取实体的文本块[/yellow]")
                else:
                    self.console.print("[yellow]没有处理到任何文件[/yellow]")
            
            except Exception as e:
                self.console.print(f"[red]处理新文件时发生错误: {e}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")
        
        self.console.print(f"[green]已完成处理 {results['files_processed']} 个新文件[/green]")
        if results["entities_extracted"] > 0 or results["relations_created"] > 0:
            self.console.print(f"[green]抽取了 {results['entities_extracted']} 个实体和 {results['relations_created']} 个关系[/green]")
        
        return results
    
    def integrate_new_entities(self, new_entities: List[Dict[str, Any]]) -> int:
        """
        无缝集成新实体到现有图结构
        
        此方法负责将新识别的实体整合到现有知识图谱中，采用MERGE操作确保数据一致性，
        避免重复创建。对于已存在的实体，会更新其属性并标记需要重新生成嵌入；
        对于新实体，则创建并设置初始属性。
        
        设计亮点：
        - 使用Neo4j的MERGE操作实现智能合并
        - 区分创建和匹配场景，分别设置创建时间和更新时间
        - 标记所有涉及的实体需要重新生成嵌入，确保查询准确性
        - 批量处理提高效率，同时返回精确的集成数量
        
        Args:
            new_entities: 新实体列表 - 包含待集成的实体数据，每个实体需包含id和其他属性
            
        Returns:
            int: 成功集成的实体数量
        """
        if not new_entities:
            return 0
            
        # Neo4j查询语句，使用MERGE操作实现实体的智能合并
        query = """
        UNWIND $entities AS entity
        MERGE (e:`__Entity__` {id: entity.id})
        ON CREATE 
            SET e += entity.properties,
                e.created_at = datetime(),
                e.needs_reembedding = true
        ON MATCH 
            SET e += entity.properties,
                e.last_updated = datetime(),
                e.needs_reembedding = true
        RETURN count(e) AS entity_count
        """
        
        # 准备实体数据，将列表格式转换为Neo4j查询所需的参数格式
        entities_data = []
        for entity in new_entities:
            entity_data = {
                "id": entity.get("id", ""),  # 实体ID作为唯一标识
                "properties": {
                    k: v for k, v in entity.items() if k != "id"  # 提取除ID外的所有属性
                }
            }
            entities_data.append(entity_data)
        
        # 执行Neo4j查询，将实体数据集成到图谱中
        result = self.graph.query(query, params={"entities": entities_data})
        entity_count = result[0]["entity_count"] if result else 0
        
        # 更新统计信息，记录集成的实体数量
        self.stats["entities_integrated"] += entity_count
        
        # 输出集成结果信息
        self.console.print(f"[green]已集成 {entity_count} 个实体[/green]")
        
        return entity_count
    
    def integrate_new_relationships(self, new_relationships: List[Dict[str, Any]]) -> int:
        """
        无缝集成新关系到现有图结构
        
        Args:
            new_relationships: 新关系列表
            
        Returns:
            int: 集成的关系数量
        """
        if not new_relationships:
            return 0
            
        # 合并关系
        query = """
        UNWIND $relationships AS rel
        MATCH (s:`__Entity__` {id: rel.source_id})
        MATCH (t:`__Entity__` {id: rel.target_id})
        CALL apoc.merge.relationship(s, rel.type, 
            {}, 
            rel.properties, 
            t, 
            {
                onMatch: {
                    properties: rel.properties,
                    last_updated: datetime()
                },
                onCreateProperties: {
                    created_at: datetime()
                }
            }
        )
        YIELD rel as created
        RETURN count(created) AS rel_count
        """
        
        try:
            # 执行查询
            result = self.graph.query(query, params={"relationships": new_relationships})
            rel_count = result[0]["rel_count"] if result else 0
            
            # 更新统计
            self.stats["relations_integrated"] += rel_count
            
            self.console.print(f"[green]已集成 {rel_count} 个关系[/green]")
            
            return rel_count
        except Exception as e:
            self.console.print(f"[yellow]集成关系时出错 (可能APOC未安装): {e}[/yellow]")
            
            # 使用备用方法
            integrated = 0
            for rel in new_relationships:
                source_id = rel.get("source_id", "")
                target_id = rel.get("target_id", "")
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                
                if source_id and target_id:
                    try:
                        backup_query = """
                        MATCH (s:`__Entity__` {id: $source_id})
                        MATCH (t:`__Entity__` {id: $target_id})
                        MERGE (s)-[r:`%s`]->(t)
                        ON CREATE SET r += $properties, r.created_at = datetime()
                        ON MATCH SET r += $properties, r.last_updated = datetime()
                        RETURN count(r) AS created
                        """ % rel_type
                        
                        backup_result = self.graph.query(
                            backup_query, 
                            params={
                                "source_id": source_id, 
                                "target_id": target_id,
                                "properties": properties
                            }
                        )
                        
                        if backup_result and backup_result[0]["created"] > 0:
                            integrated += 1
                    except Exception as e2:
                        self.console.print(f"[red]使用备用方法集成关系时出错: {e2}[/red]")
            
            # 更新统计
            self.stats["relations_integrated"] += integrated
            
            self.console.print(f"[green]使用备用方法集成了 {integrated} 个关系[/green]")
            
            return integrated
    
    def merge_graph_structures(self, old_graph: Dict[str, Any], new_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并原有图结构与新图结构
        
        此方法实现了图结构的智能合并，是增量更新的关键组件之一。它能够将新生成的图结构
        与现有图结构合并，同时保持数据一致性和完整性。合并逻辑基于时间戳进行冲突解决，
        确保更新的节点和关系得到优先处理。
        
        合并策略：
        1. 节点合并 - 使用ID作为唯一标识，基于时间戳决定是否更新
        2. 边合并 - 使用(source, type, target)三元组作为唯一键，避免重复关系
        3. 冲突解决 - 优先保留较新的数据，同时保留必要的属性标记
        
        该方法特别适用于增量更新场景，能够有效处理数据变化并保持图谱的一致性。
        
        Args:
            old_graph: 原有图结构 - 包含现有知识图谱的节点和关系信息
            new_graph: 新图结构 - 包含新增或更新的节点和关系信息
        
        Returns:
            Dict: 合并后的图结构，包含"nodes"和"edges"两个键
        """
        # 节点集合并 - 使用字典结构和节点ID作为键，实现高效的查找和合并
        merged_nodes = {**old_graph.get("nodes", {})}
        for node_id, node in new_graph.get("nodes", {}).items():
            if node_id in merged_nodes:
                # 如果节点已存在，根据时间戳决定是否更新
                old_timestamp = merged_nodes[node_id].get("last_updated", 0)
                new_timestamp = node.get("last_updated", time.time())
                
                if new_timestamp > old_timestamp:
                    # 新数据更新，更新节点属性并保留原有必要信息
                    merged_nodes[node_id] = {**merged_nodes[node_id], **node}
                    merged_nodes[node_id]["last_updated"] = new_timestamp
                    merged_nodes[node_id]["needs_reembedding"] = True  # 标记需要重新生成嵌入
            else:
                # 新节点直接添加到合并结果中
                merged_nodes[node_id] = node
                # 确保新节点有标记需要嵌入，以便后续更新
                if "needs_reembedding" not in merged_nodes[node_id]:
                    merged_nodes[node_id]["needs_reembedding"] = True
        
        # 边集合并，使用字典避免重复关系，键为(source, type, target)三元组
        merged_edges = {}
        
        # 先添加旧图的边，确保所有现有关系被保留
        for edge in old_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            
            # 创建唯一键，确保每个关系都有唯一标识
            key = f"{source}_{rel_type}_{target}"
            merged_edges[key] = edge
        
        # 再添加或更新新图的边，根据时间戳决定是否覆盖现有关系
        for edge in new_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")
            
            # 创建唯一键，与上面相同的格式
            key = f"{source}_{rel_type}_{target}"
            
            if key in merged_edges:
                # 如果关系已存在，根据时间戳决定是否更新
                old_timestamp = merged_edges[key].get("last_updated", 0)
                new_timestamp = edge.get("last_updated", time.time())
                
                if new_timestamp > old_timestamp:
                    # 新数据更新，更新关系
                    merged_edges[key] = {**merged_edges[key], **edge}
                    merged_edges[key]["last_updated"] = new_timestamp
            else:
                # 新关系直接添加
                merged_edges[key] = edge
        
        # 返回合并后的图结构，将边从字典转回列表格式
        return {
            "nodes": merged_nodes,
            "edges": list(merged_edges.values())
        }
    
    def update_changed_file_embeddings(self, changed_files: List[str]) -> Dict[str, int]:
        """
        更新变更文件相关的Embedding
        
        此方法是增量更新机制的关键部分，专门处理修改文件的嵌入向量更新。
        当文件内容发生变化时，不仅需要更新该文件对应的文本块嵌入，还需要更新
        这些文本块中提到的所有实体的嵌入，确保查询结果的准确性。
        
        处理流程：
        1. 标记与变更文件相关的文本块需要更新嵌入
        2. 查找并标记这些文本块中提到的所有实体需要更新嵌入
        3. 执行实体和文本块的嵌入更新
        4. 更新统计信息并返回结果
        
        这种方法能够确保知识图谱中嵌入向量的一致性，同时避免不必要的全局重新计算，
        显著提高增量更新的效率。
        
        Args:
            changed_files: 变更的文件列表 - 包含需要更新嵌入的修改文件路径
            
        Returns:
            Dict: 更新统计，包含以下键：
                - "entities": 成功更新的实体数量
                - "chunks": 成功更新的文本块数量
        """
        if not changed_files:
            return {"entities": 0, "chunks": 0}
            
        # 步骤1: 标记变更文件的Chunk需要更新Embedding
        marked_chunks = self.embedding_manager.mark_changed_files_chunks(changed_files)
        
        # 步骤2: 查找这些Chunk关联的实体，标记它们也需要更新Embedding
        # 这是因为实体的语义可能随着上下文变化而变化
        query = """
        MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
        WHERE c.fileName IN $filenames OR c.needs_reembedding = true
        SET e.needs_reembedding = true,
            e.last_updated = datetime()
        RETURN count(DISTINCT e) AS entity_count
        """
        
        # 获取文件名（不包含路径）
        filenames = [Path(file).name for file in changed_files]
        
        # 执行查询，标记相关实体
        result = self.graph.query(query, params={"filenames": filenames})
        marked_entities = result[0]["entity_count"] if result else 0
        
        # 输出标记结果信息
        self.console.print(f"[blue]已标记 {marked_chunks} 个Chunk和 {marked_entities} 个实体需要更新Embedding[/blue]")
        
        # 步骤3: 执行嵌入更新操作
        # 先更新实体嵌入
        updated_entities = self.embedding_manager.update_entity_embeddings()
        # 再更新文本块嵌入
        updated_chunks = self.embedding_manager.update_chunk_embeddings()
        
        # 步骤4: 更新统计信息
        self.stats["entities_updated"] += updated_entities
        self.stats["chunks_updated"] += updated_chunks
        
        # 返回更新统计结果
        return {
            "entities": updated_entities,
            "chunks": updated_chunks
        }
    
    def process_deleted_files(self, deleted_files: List[str]) -> int:
        """
        处理已删除的文件
        
        此方法负责清理与已删除文件相关的知识图谱数据，同时确保不会删除
        被其他文件引用的实体或手动编辑过的实体。这种选择性删除策略能够
        维护知识图谱的完整性，同时避免丢失重要信息。
        
        处理流程：
        1. 查找与删除文件关联的所有文本块节点
        2. 识别仅由这些文本块引用且非手动编辑的实体
        3. 删除文档节点、文本块节点和相关关系
        4. 仅删除那些完全孤立且不受保护的实体
        
        设计亮点：
        - 保护手动编辑的实体不被自动删除
        - 保留被其他文件引用的共享实体
        - 完整清理文档相关的所有节点和关系
        - 提供详细的删除统计信息
        
        Args:
            deleted_files: 删除的文件列表 - 包含已从文件系统中删除的文件路径
            
        Returns:
            int: 成功删除的节点总数（文档节点、文本块节点和实体节点的总和）
        """
        if not deleted_files:
            return 0
            
        self.console.print(f"[cyan]处理 {len(deleted_files)} 个已删除的文件...[/cyan]")
        
        deleted_count = 0
        for file_path in deleted_files:
            file_name = Path(file_path).name
            
            # 查找与文件关联的所有Chunk节点
            chunk_query = """
            MATCH (d:`__Document__` {fileName: $fileName})<-[:PART_OF]-(c:`__Chunk__`)
            RETURN collect(c.id) AS chunk_ids, count(c) AS chunk_count
            """
            
            chunk_result = self.graph.query(chunk_query, params={"fileName": file_name})
            
            if not chunk_result or not chunk_result[0]["chunk_ids"]:
                self.console.print(f"[yellow]文件 {file_name} 没有相关的数据需要删除[/yellow]")
                continue
                
            chunk_ids = chunk_result[0]["chunk_ids"]
            chunk_count = chunk_result[0]["chunk_count"]
            
            # 查找这些Chunk关联的实体，但排除被其他Chunk引用的实体
            entity_query = """
            MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
            WHERE c.id IN $chunk_ids
            WITH e, count(c) AS references
            MATCH (chunk:`__Chunk__`)-[:MENTIONS]->(e)
            WITH e, references, count(chunk) AS total_references
            WHERE references = total_references 
              AND NOT e.manual_edit = true  // 排除手动编辑的实体
              AND NOT e.protected = true    // 排除受保护的实体
            RETURN collect(e.id) AS entity_ids, count(e) AS entity_count
            """
            
            entity_result = self.graph.query(entity_query, params={"chunk_ids": chunk_ids})
            
            entity_ids = []
            entity_count = 0
            if entity_result and entity_result[0]["entity_ids"]:
                entity_ids = entity_result[0]["entity_ids"]
                entity_count = entity_result[0]["entity_count"]
            
            # 删除与文件关联的所有数据
            delete_query = """
            // 删除文档节点和关系
            MATCH (d:`__Document__` {fileName: $fileName})
            OPTIONAL MATCH (d)-[r]-()
            DELETE r
            
            // 删除Chunk节点和关系
            WITH d
            MATCH (c:`__Chunk__`)-[r1:PART_OF]->(d)
            OPTIONAL MATCH (c)-[r2]-()
            WHERE NOT type(r2) = 'PART_OF'
            DELETE r2
            
            // 删除孤立的实体节点
            WITH d, collect(c.id) as chunk_ids
            UNWIND $entity_ids AS entity_id
            MATCH (e:`__Entity__` {id: entity_id})
            WHERE NOT e.manual_edit = true AND NOT e.protected = true // 再次检查保护
            DELETE e
            
            // 删除Chunk节点
            WITH d, chunk_ids
            MATCH (c:`__Chunk__`)
            WHERE c.id IN chunk_ids
            DELETE c
            
            // 最后删除文档节点
            DELETE d
            
            RETURN count(d) AS deleted_docs
            """
            
            delete_result = self.graph.query(delete_query, params={
                "fileName": file_name, 
                "entity_ids": entity_ids
            })
            
            deleted_docs = delete_result[0]["deleted_docs"] if delete_result else 0
            file_deleted_count = chunk_count + entity_count + deleted_docs
            deleted_count += file_deleted_count
            
            self.console.print(f"[blue]已删除文件 {file_name} 相关数据: "
                              f"{chunk_count} 个Chunk节点, "
                              f"{entity_count} 个实体节点, "
                              f"{deleted_docs} 个文档节点[/blue]")
        
        self.console.print(f"[green]已完成删除文件处理，共删除 {deleted_count} 个节点[/green]")
        return deleted_count
    
    def export_graph_structure(self) -> Dict[str, Any]:
        """
        导出当前图谱结构
        
        此方法负责将知识图谱的实体和关系数据导出为结构化字典格式，便于保存、传输或合并操作。
        导出的数据包含所有实体节点及其属性、所有关系及其属性，以及必要的时间戳信息。
        
        导出的结构设计：
        - 节点使用字典格式，以ID为键，包含标签、属性和时间戳
        - 关系使用列表格式，每个关系包含源节点、目标节点、类型、属性和时间戳
        - 保留时间戳信息，用于后续的增量合并和冲突解决
        
        此功能在备份、恢复或迁移图谱数据时特别有用，也可以用于与其他系统进行数据交换。
        
        Returns:
            Dict: 图谱结构数据，包含以下键：
                - "nodes": 实体节点字典，键为节点ID，值为节点详情
                - "edges": 关系列表，每个元素表示一条关系
        """
        # 查询所有实体节点，获取ID、标签、属性和更新时间
        node_query = """
        MATCH (e:`__Entity__`)
        RETURN e.id AS id, 
               labels(e) AS labels,
               properties(e) AS properties,
               e.last_updated AS last_updated
        """
        
        node_result = self.graph.query(node_query)
        
        # 查询所有关系，获取源节点、目标节点、关系类型、属性和更新时间
        edge_query = """
        MATCH (s:`__Entity__`)-[r]->(t:`__Entity__`)
        RETURN s.id AS source,
               t.id AS target,
               type(r) AS type,
               properties(r) AS properties,
               CASE WHEN r.last_updated IS NOT NULL THEN r.last_updated ELSE null END AS last_updated
        """
        
        edge_result = self.graph.query(edge_query)
        
        # 构建图谱结构数据，节点使用字典格式以提高查找效率
        nodes = {}
        for node in node_result:
            node_id = node["id"]
            nodes[node_id] = {
                "id": node_id,
                "labels": node["labels"],  # 节点标签列表
                "properties": node["properties"],  # 节点属性
                "last_updated": node["last_updated"] if node["last_updated"] else time.time()  # 更新时间
            }
        
        # 构建关系列表，每条关系记录必要的连接信息
        edges = []
        for edge in edge_result:
            edges.append({
                "source": edge["source"],  # 源节点ID
                "target": edge["target"],  # 目标节点ID
                "type": edge["type"],  # 关系类型
                "properties": edge["properties"],  # 关系属性
                "last_updated": edge["last_updated"] if edge["last_updated"] else time.time()  # 更新时间
            })
        
        # 返回完整的图谱结构
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def import_graph_structure(self, graph_structure: Dict[str, Any]) -> Dict[str, int]:
        """
        导入图谱结构
        
        Args:
            graph_structure: 图谱结构数据
            
        Returns:
            Dict: 导入统计
        """
        if not graph_structure:
            return {"nodes": 0, "edges": 0}
            
        # 导入节点
        nodes = list(graph_structure.get("nodes", {}).values())
        node_count = 0
        
        if nodes:
            # 准备节点数据
            node_data = []
            for node in nodes:
                node_data.append({
                    "id": node["id"],
                    "labels": node.get("labels", ["__Entity__"]),
                    "properties": node.get("properties", {})
                })
            
            # 执行导入
            node_query = """
            UNWIND $nodes AS node
            CALL apoc.merge.node(
                node.labels,
                {id: node.id},
                node.properties,
                node.properties
            )
            YIELD node as n
            RETURN count(n) AS node_count
            """
            
            try:
                node_result = self.graph.query(node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0
            except Exception as e:
                self.console.print(f"[yellow]导入节点时出错 (可能APOC未安装): {e}[/yellow]")
                
                # 使用备用方法
                simple_node_query = """
                UNWIND $nodes AS node
                MERGE (n:`__Entity__` {id: node.id})
                SET n += node.properties
                RETURN count(n) AS node_count
                """
                
                node_result = self.graph.query(simple_node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0
        
        # 导入边
        edges = graph_structure.get("edges", [])
        edge_count = 0
        
        if edges:
            # 执行导入
            self.integrate_new_relationships(edges)
            edge_count = len(edges)
        
        return {
            "nodes": node_count,
            "edges": edge_count
        }
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 节点统计
        node_query = """
        MATCH (n)
        RETURN 
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """
        
        node_result = self.graph.query(node_query)
        
        # 关系统计
        rel_query = """
        MATCH ()-[r]->()
        RETURN count(r) AS total_relations,
               count(DISTINCT type(r)) AS relation_types
        """
        
        rel_result = self.graph.query(rel_query)
        
        # 嵌入统计
        embedding_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN 
            count(n) AS nodes_with_embedding,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entities_with_embedding,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunks_with_embedding
        """
        
        embedding_result = self.graph.query(embedding_query)
        
        # 合并统计信息
        stats = {
            "total_nodes": node_result[0]["total_nodes"] if node_result else 0,
            "document_count": node_result[0]["doc_count"] if node_result else 0,
            "chunk_count": node_result[0]["chunk_count"] if node_result else 0,
            "entity_count": node_result[0]["entity_count"] if node_result else 0,
            "total_relations": rel_result[0]["total_relations"] if rel_result else 0,
            "relation_types": rel_result[0]["relation_types"] if rel_result else 0,
            "nodes_with_embedding": embedding_result[0]["nodes_with_embedding"] if embedding_result else 0,
            "entities_with_embedding": embedding_result[0]["entities_with_embedding"] if embedding_result else 0,
            "chunks_with_embedding": embedding_result[0]["chunks_with_embedding"] if embedding_result else 0
        }
        
        return stats
    
    def display_graph_statistics(self):
        """显示图谱统计信息"""
        stats = self.get_graph_statistics()
        
        # 创建统计表格
        stats_table = Table(title="图谱统计信息")
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("数量", justify="right")
        
        # 添加节点统计
        stats_table.add_row("总节点数", str(stats["total_nodes"]))
        stats_table.add_row("文档节点数", str(stats["document_count"]))
        stats_table.add_row("文本块节点数", str(stats["chunk_count"]))
        stats_table.add_row("实体节点数", str(stats["entity_count"]))
        
        # 添加关系统计
        stats_table.add_row("总关系数", str(stats["total_relations"]))
        stats_table.add_row("关系类型数", str(stats["relation_types"]))
        
        # 添加嵌入统计
        stats_table.add_row("具有嵌入的节点数", str(stats["nodes_with_embedding"]))
        stats_table.add_row("具有嵌入的实体数", str(stats["entities_with_embedding"]))
        stats_table.add_row("具有嵌入的文本块数", str(stats["chunks_with_embedding"]))
        
        # 显示统计表格
        self.console.print(stats_table)
    
    def process_incremental_update(self) -> Dict[str, Any]:
        """
        执行增量更新流程
        
        这是增量更新的主入口方法，协调整个更新流程，包括文件变更检测、新增文件处理、
        修改文件嵌入更新、删除文件清理等步骤。该方法实现了一个完整的增量更新流水线，
        能够高效地更新知识图谱，同时保持数据的一致性和完整性。
        
        处理流程：
        1. 检测文件变更 - 识别新增、修改和删除的文件
        2. 处理已删除的文件 - 清理相关的图谱数据
        3. 处理新增文件 - 执行完整的文档处理和图谱构建流程
        4. 更新变更文件的嵌入向量 - 为修改的文件和相关实体更新嵌入
        5. 更新文件注册表 - 保存文件状态，用于下次比较
        6. 显示图谱统计信息 - 提供处理结果的可视化展示
        
        该方法设计为事务性流程，即使在处理过程中出现错误，也能记录已完成的操作和处理时间，
        方便问题排查和恢复。同时，它提供了丰富的进度和结果反馈，提高了系统的可观测性。
        
        Returns:
            Dict: 更新结果统计，包含处理的文件数、新增/更新的实体和关系数、处理时间等信息
            
        Raises:
            Exception: 如果处理过程中出现错误，会重新抛出异常
        """
        # 记录开始时间
        start_time = time.time()
        self.stats["start_time"] = start_time
        
        try:
            # 步骤1: 检测文件变更
            self.console.print("[bold cyan]检测文件变更...[/bold cyan]")
            changes = self.detect_changes()
            
            # 分离不同类型的文件变更
            added_files = changes.get("added", [])      # 新增文件列表
            modified_files = changes.get("modified", [])  # 修改文件列表
            deleted_files = changes.get("deleted", [])   # 删除文件列表
            
            # 确定需要更新嵌入的文件
            changed_files = modified_files  # 只有修改的文件需要更新embedding
            # 更新处理的文件总数
            self.stats["files_processed"] = len(added_files) + len(modified_files) + len(deleted_files)
            
            # 如果没有检测到任何变更，提前返回
            if not added_files and not changed_files and not deleted_files:
                self.console.print("[yellow]未检测到文件变更[/yellow]")
                return self.stats
            
            # 步骤2: 处理已删除的文件
            if deleted_files:
                self.console.print("[bold cyan]处理已删除的文件...[/bold cyan]")
                self.process_deleted_files(deleted_files)
            
            # 步骤3: 处理新文件 - 执行完整的处理流程
            if added_files:
                self.console.print("[bold cyan]处理新增文件...[/bold cyan]")
                new_file_results = self.process_new_files(added_files)
                # 更新统计信息，记录新集成的实体和关系数
                self.stats["entities_integrated"] += new_file_results.get("entities_extracted", 0)
                self.stats["relations_integrated"] += new_file_results.get("relations_created", 0)
            
            # 步骤4: 更新变更文件的Embedding
            if changed_files:
                self.console.print("[bold cyan]更新变更文件的Embedding...[/bold cyan]")
                embedding_stats = self.update_changed_file_embeddings(changed_files)
                
                # 显示Embedding更新结果
                self.console.print(f"[green]更新的实体Embedding: {embedding_stats['entities']}[/green]")
                self.console.print(f"[green]更新的Chunk Embedding: {embedding_stats['chunks']}[/green]")
            
            # 步骤5: 更新文件注册表，保存当前文件状态
            self.file_manager.update_registry()
            
            # 步骤6: 显示图谱统计信息
            self.console.print("[bold cyan]图谱统计信息[/bold cyan]")
            self.display_graph_statistics()
            
            # 计算结束时间和总时间
            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time
            
            # 显示处理结果摘要
            self.console.print("\n[bold green]增量更新完成![/bold green]")
            self.console.print(f"[green]总耗时: {self.stats['total_time']:.2f}秒[/green]")
            self.console.print(f"[green]处理的文件数: {self.stats['files_processed']}[/green]")
            if added_files:
                self.console.print(f"[green]新增实体数: {self.stats['entities_integrated']}[/green]")
                self.console.print(f"[green]新增关系数: {self.stats['relations_integrated']}[/green]")
            
            return self.stats
            
        except Exception as e:
            # 错误处理：显示错误信息并记录时间
            self.console.print(f"[red]增量更新过程中出现错误: {e}[/red]")
            
            # 即使出错也要记录结束时间和总时间
            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time
            
            # 重新抛出异常，允许上层调用者处理
            raise