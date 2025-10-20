"""
Graph-RAG增量更新管理器模块

此模块实现了Graph-RAG系统的增量更新功能，作为系统的核心组件之一，负责协调和管理
各个子模块执行增量更新操作。它提供了完整的增量更新流程，包括文件变更检测、图谱更新、
嵌入更新、一致性验证、社区检测等功能，并支持定时调度和手动触发两种运行模式。

主要功能点：
- 集成各增量更新组件，提供统一接口
- 实现完整的增量更新工作流程
- 支持定时检测和手动触发更新
- 管理手动编辑的同步和保护
- 提供运行统计和状态监控
"""

import os
import time
import signal
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# 导入子组件
from incremental_graph_builder import IncrementalGraphUpdater
from graph.graph_consistency_validator import GraphConsistencyValidator
from build.incremental.manual_edit_manager import ManualEditManager
from graph.indexing.embedding_manager import EmbeddingManager
from community import CommunityDetectorFactory, CommunitySummarizerFactory
from config.neo4jdb import get_db_manager
from config.settings import FILES_DIR, community_algorithm, MAX_WORKERS, BATCH_SIZE
from build.incremental.incremental_update_scheduler import IncrementalUpdateScheduler

class IncrementalUpdateManager:
    """
    增量更新管理器，整合所有增量更新功能，支持后台运行和定期检测。
    
    该类是Graph-RAG系统增量更新机制的中枢，负责协调和集成各个子组件，形成完整的
    增量更新流程。它实现了从文件变更检测到知识图谱更新、嵌入更新、一致性验证等一系列
    功能，并提供了定时调度和手动触发两种运行模式，确保系统能够高效地适应文档库的变化。
    
    主要功能：
    1. 检测文件变更并更新图谱 - 通过IncrementalGraphUpdater实现
    2. 更新实体和Chunk的Embedding - 通过EmbeddingManager处理
    3. 验证图谱一致性 - 使用GraphConsistencyValidator确保数据完整性
    4. 处理社区检测和摘要生成 - 通过社区检测算法和摘要工具
    5. 支持手动编辑同步 - 使用ManualEditManager确保用户编辑不被覆盖
    6. 后台运行和定时调度 - 通过IncrementalUpdateScheduler实现自动化
    """
    
    def __init__(self, files_dir: str = FILES_DIR, config=None):
        """
        初始化增量更新管理器
        
        初始化过程中，设置文件监控目录，加载配置参数，并初始化所有必要的子组件。
        这些子组件负责不同的更新任务，如文件变更检测、图更新、嵌入更新等。同时，
        初始化运行状态和性能统计信息。
        
        Args:
            files_dir: 监控的文件目录，默认为配置中的FILES_DIR
            config: 配置参数字典，包含各组件的更新频率等配置
        """
        # 初始化控制台输出对象
        self.console = Console()
        
        # 设置配置参数
        self.files_dir = files_dir  # 监控的文件目录
        self.config = config or {}  # 合并用户配置和默认配置
        
        # 初始化核心子组件
        self.graph = get_db_manager().graph  # 获取图数据库连接
        self.updater = IncrementalGraphUpdater(files_dir)  # 文件变更检测器和图更新器
        self.validator = GraphConsistencyValidator()  # 图一致性验证器
        self.edit_manager = ManualEditManager()  # 手动编辑管理器
        # 嵌入管理器，设置批处理大小和最大工作线程数
        self.embedding_manager = EmbeddingManager(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
        
        # 初始化调度器，用于定期执行更新任务
        self.scheduler = IncrementalUpdateScheduler(self.config)
        
        # 运行状态标志
        self.running = False  # 调度器运行状态
        self.stop_event = None  # 用于停止调度器的事件对象
        
        # 性能和操作统计信息
        self.stats = {
            "updates_performed": 0,  # 执行的更新次数
            "files_processed": 0,    # 处理的文件数量
            "entities_updated": 0,   # 更新的实体数量
            "communities_detected": 0,  # 检测的社区数量
            "errors": 0              # 发生的错误数量
        }
    
    def detect_file_changes(self):
        """
        检测文件变更并更新图谱
        
        该方法是增量更新流程的起点，负责检测文档目录中的文件变化，并相应地更新知识图谱。
        通过调用IncrementalGraphUpdater的方法，它能够识别新增、修改和删除的文件，
        并对知识图谱进行相应的更新。如果检测到文件删除，还会触发图一致性验证，
        确保图结构的完整性。
        
        Returns:
            Dict: 包含文件变更信息的字典，包括新增、修改和删除的文件列表
        """
        self.console.print("[bold cyan]检测文件变更...[/bold cyan]")
        
        try:
            # 使用IncrementalGraphUpdater检测文件变更
            changes = self.updater.detect_changes()
            
            # 统计各类变更数量
            added_count = len(changes.get("added", []))      # 新增文件数
            modified_count = len(changes.get("modified", []))  # 修改文件数
            deleted_count = len(changes.get("deleted", []))   # 删除文件数
            total_changed = added_count + modified_count + deleted_count  # 总变更数
            
            if total_changed > 0:
                # 输出变更统计信息
                self.console.print(f"[green]检测到 {total_changed} 个文件变更：[/green]")
                self.console.print(f"[green]新增: {added_count}, 修改: {modified_count}, 删除: {deleted_count}[/green]")
                
                # 如果有变更，执行增量更新
                self.updater.process_incremental_update()
                
                # 如果有文件删除，执行图一致性检查，确保图谱完整性
                if deleted_count > 0:
                    self.verify_graph_consistency()
                
                # 更新统计信息
                self.stats["updates_performed"] += 1
                self.stats["files_processed"] += total_changed
            else:
                # 没有检测到变更，输出提示信息
                self.console.print("[yellow]未检测到文件变更[/yellow]")
            
            return changes
            
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]检测文件变更时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def update_entity_embeddings(self):
        """
        更新实体Embedding向量
        
        此方法负责更新知识图谱中实体的嵌入向量。在增量更新过程中，当有新实体添加或现有
        实体修改时，需要更新其嵌入向量以确保语义检索的准确性。该方法使用EmbeddingManager
        来高效地批量处理实体嵌入更新。
        
        Returns:
            int: 成功更新的实体数量
        """
        self.console.print("[bold cyan]更新实体Embedding...[/bold cyan]")
        
        try:
            # 获取需要更新嵌入的实体列表
            entities = self.embedding_manager.get_entities_needing_update()
            
            if entities:
                # 输出待更新实体数量
                self.console.print(f"[green]发现 {len(entities)} 个需要更新Embedding的实体[/green]")
                
                # 执行嵌入更新
                updated_count = self.embedding_manager.update_entity_embeddings()
                
                # 更新统计信息
                self.stats["entities_updated"] += updated_count
                
                return updated_count
            else:
                # 没有需要更新的实体
                self.console.print("[yellow]没有需要更新Embedding的实体[/yellow]")
                return 0
                
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]更新实体Embedding时出错: {e}[/red]")
            self.stats["errors"] += 1
            return 0
    
    def update_chunk_embeddings(self):
        """
        更新文本块(Chunk)的Embedding向量
        
        该方法负责更新知识图谱中文本块的嵌入向量。文本块是文档的基本单位，其嵌入向量
        对于语义检索至关重要。当文档更新时，相关文本块的嵌入向量也需要更新，以确保
        检索结果的准确性。
        
        Returns:
            int: 成功更新的文本块数量
        """
        self.console.print("[bold cyan]更新Chunk Embedding...[/bold cyan]")
        
        try:
            # 获取需要更新嵌入的文本块列表
            chunks = self.embedding_manager.get_chunks_needing_update()
            
            if chunks:
                # 输出待更新文本块数量
                self.console.print(f"[green]发现 {len(chunks)} 个需要更新Embedding的Chunk[/green]")
                
                # 执行嵌入更新
                updated_count = self.embedding_manager.update_chunk_embeddings()
                
                return updated_count
            else:
                # 没有需要更新的文本块
                self.console.print("[yellow]没有需要更新Embedding的Chunk[/yellow]")
                return 0
                
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]更新Chunk Embedding时出错: {e}[/red]")
            self.stats["errors"] += 1
            return 0
    
    def verify_graph_consistency(self, repair=True):
        """
        验证图谱一致性并执行修复
        
        此方法负责确保知识图谱的结构完整性和数据一致性。在增量更新过程中，特别是当有文件删除时，
        图中可能会出现孤立节点或不一致的数据关系。通过图一致性验证，可以检测并修复这些问题，
        确保知识图谱的完整性和可靠性。
        
        Args:
            repair: 是否在检测到问题时自动执行修复，默认为True
            
        Returns:
            Dict: 包含验证结果和统计信息的字典
        """
        self.console.print("[bold cyan]验证图谱一致性...[/bold cyan]")
        
        try:
            if repair:
                # 执行验证和自动修复
                result = self.validator.repair_graph()
                
                # 显示修复结果
                repaired_count = result["validation_stats"]["repaired_issues"]
                self.console.print(f"[green]图谱一致性验证完成，修复了 {repaired_count} 个问题[/green]")
            else:
                # 仅执行验证，不进行修复
                result = self.validator.validate_graph()
                
                # 显示验证结果
                issues_count = result["validation_stats"]["total_issues"]
                self.console.print(f"[green]图谱一致性验证完成，发现 {issues_count} 个问题[/green]")
            
            return result
            
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]验证图谱一致性时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def detect_communities(self):
        """
        执行社区检测和摘要生成
        
        该方法负责在知识图谱中检测社区结构并生成社区摘要。社区检测是图分析的重要任务，
        它能够识别知识图谱中紧密相关的实体集合，有助于理解知识的组织方式和主题分布。
        同时，为每个社区生成摘要可以提供社区内容的概览，提升用户体验。
        
        Returns:
            Dict: 包含社区检测结果和摘要信息的字典
        """
        self.console.print("[bold cyan]执行社区检测...[/bold cyan]")
        
        try:
            # 获取数据库连接
            db_manager = get_db_manager()
            graph = db_manager.graph
            
            # 尝试导入GraphDataScience库
            try:
                from graphdatascience import GraphDataScience
                gds = GraphDataScience(
                    os.environ["NEO4J_URI"],
                    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
                )
            except Exception as e:
                self.console.print(f"[yellow]导入GDS库失败，无法执行社区检测: {e}[/yellow]")
                return {"status": "error", "message": str(e)}
            
            # 创建社区检测器，使用配置中指定的算法
            self.console.print(f"[blue]使用 {community_algorithm} 算法执行社区检测[/blue]")
            detector = CommunityDetectorFactory.create(
                algorithm=community_algorithm,
                gds=gds,
                graph=graph
            )
            
            # 执行社区检测
            detection_result = detector.process()
            
            if detection_result.get('status', '') == 'success':
                # 提取检测到的社区数量
                community_count = detection_result.get('details', {}).get('detection', {}).get('communityCount', 0)
                self.console.print(f"[green]社区检测完成，共检测到 {community_count} 个社区[/green]")
                
                # 更新统计信息
                self.stats["communities_detected"] += community_count
                
                # 执行社区摘要生成
                self.console.print("[blue]开始生成社区摘要...[/blue]")
                summarizer = CommunitySummarizerFactory.create_summarizer(
                    community_algorithm,
                    graph
                )
                summaries = summarizer.process_communities()
                
                self.console.print(f"[green]社区摘要生成完成，共生成 {len(summaries) if summaries else 0} 个摘要[/green]")
                
                # 返回检测结果
                return {
                    "status": "success", 
                    "communities": community_count,
                    "summaries": len(summaries) if summaries else 0
                }
            else:
                # 检测失败
                self.console.print(f"[yellow]社区检测失败: {detection_result.get('message', '未知错误')}[/yellow]")
                return detection_result
                
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]执行社区检测时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"status": "error", "message": str(e)}
    
    def sync_manual_edits(self, changed_files=None):
        """
        同步手动编辑
        
        此方法负责处理用户对知识图谱的手动编辑与自动增量更新之间的同步。在文件发生变更并触发
        自动更新时，需要确保用户的手动编辑不会被覆盖，同时新的自动更新也能适当地与手动编辑
        进行合并。这对于维护用户修改的持久性和图谱数据的一致性至关重要。
        
        Args:
            changed_files: 变更的文件列表，如果为None则自动检测所有变更
            
        Returns:
            Dict: 包含同步结果的字典
        """
        self.console.print("[bold cyan]同步手动编辑...[/bold cyan]")
        
        try:
            # 如果没有提供变更文件列表，获取所有变更
            if changed_files is None:
                changes = self.updater.detect_changes()
                changed_files = changes.get("added", []) + changes.get("modified", [])
            
            if changed_files:
                # 执行手动编辑同步处理
                result = self.edit_manager.process(changed_files)
                
                return result
            else:
                # 没有变更文件，跳过同步
                self.console.print("[yellow]没有变更的文件，跳过手动编辑同步[/yellow]")
                return {"status": "skipped"}
                
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]同步手动编辑时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def check_manual_edits(self):
        """
        检查Neo4j中的手动编辑，并确保这些编辑在增量更新中被保留
        
        该方法检测图数据库中用户进行的手动编辑，并设置相应的保护机制，确保在后续增量
        更新过程中这些手动编辑不会被自动更新覆盖。这对于维护用户修改的完整性和持久性
        非常重要，特别是在团队协作环境中。
        
        Returns:
            Dict: 包含手动编辑统计信息的字典，包括手动编辑的实体数、关系数等
        """
        self.console.print("[bold cyan]检查手动编辑...[/bold cyan]")
        
        try:
            # 使用ManualEditManager检测手动编辑
            edit_stats = self.edit_manager.detect_manual_edits()
            
            # 提取手动编辑的实体和关系数量
            manual_entities = edit_stats.get("manual_entities", 0)
            manual_relations = edit_stats.get("manual_relations", 0)
            
            if manual_entities > 0 or manual_relations > 0:
                # 输出手动编辑统计信息
                self.console.print(f"[green]检测到 {manual_entities} 个手动编辑的实体和 {manual_relations} 个手动编辑的关系[/green]")
                
                # 确保增量更新时保留这些手动编辑
                changes = self.updater.detect_changes()
                changed_files = []
                if changes:
                    changed_files = changes.get("added", []) + changes.get("modified", [])
                
                # 应用手动编辑保护机制
                if changed_files:
                    preserved_count = self.edit_manager.preserve_manual_edits(changed_files)
                    self.console.print(f"[green]已保护 {preserved_count} 个手动编辑，确保增量更新不会覆盖它们[/green]")
                
                return {
                    "manual_entities": manual_entities,
                    "manual_relations": manual_relations,
                    "preserved_edits": preserved_count if changed_files else 0
                }
            else:
                # 没有检测到手动编辑
                self.console.print("[blue]没有检测到手动编辑[/blue]")
                return {
                    "manual_entities": 0,
                    "manual_relations": 0
                }
                
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]检查手动编辑时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def run_once(self):
        """
        执行一次完整的增量更新流程
        
        此方法是增量更新管理器的核心方法，它按照预定顺序调用各个组件的方法，
        执行一次完整的增量更新操作。更新流程包括文件变更检测、实体嵌入更新、
        文本块嵌入更新、图谱一致性验证、手动编辑同步和社区检测等步骤。该方法
        适用于需要立即执行更新的场景，如系统启动时的初始化或用户手动触发更新。
        
        Returns:
            Dict: 包含所有更新步骤结果的字典
        """
        # 记录开始时间，用于计算总耗时
        start_time = time.time()
        
        self.console.print("\n[bold cyan]开始执行增量更新流程...[/bold cyan]")
        
        # 初始化结果字典
        results = {}
        
        try:
            # 1. 检测文件变更并更新图谱
            changes = self.detect_file_changes()
            results["file_changes"] = changes
            
            # 2. 更新实体Embedding
            entity_updates = self.update_entity_embeddings()
            results["entity_updates"] = entity_updates
            
            # 3. 更新Chunk Embedding
            chunk_updates = self.update_chunk_embeddings()
            results["chunk_updates"] = chunk_updates
            
            # 4. 验证图谱一致性
            consistency_check = self.verify_graph_consistency()
            results["consistency_check"] = consistency_check
            
            # 5. 同步手动编辑（仅在有新增或修改文件时执行）
            if changes and (changes.get("added") or changes.get("modified")):
                edit_sync = self.sync_manual_edits(
                    changes.get("added", []) + changes.get("modified", [])
                )
                results["edit_sync"] = edit_sync
            
            # 6. 执行社区检测（仅在文件有变更时执行，避免不必要的计算）
            if changes and (changes.get("added") or changes.get("modified") or changes.get("deleted")):
                community_detection = self.detect_communities()
                results["community_detection"] = community_detection
            
            # 计算总耗时
            end_time = time.time()
            total_time = end_time - start_time
            
            self.console.print(f"[bold green]增量更新流程完成，总耗时: {total_time:.2f}秒[/bold green]")
            
            return results
            
        except Exception as e:
            # 异常处理
            self.console.print(f"[red]执行增量更新流程时出错: {e}[/red]")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def start_scheduler(self):
        """
        启动增量更新调度器，开始后台运行增量更新任务
        
        此方法负责初始化并启动增量更新调度器，将各个更新组件注册到调度器中，
        并设置相应的更新频率。调度器启动后会在后台线程中运行，按照配置的频率
        自动执行各个更新任务，无需人工干预。这对于持续运行的服务环境非常有用。
        """
        self.console.print("[bold cyan]启动增量更新调度器...[/bold cyan]")
        
        # 注册各个更新组件到调度器，设置相应的处理方法
        self.scheduler.schedule_component("file_change", self.detect_file_changes)
        self.scheduler.schedule_component("entity_embedding", self.update_entity_embeddings)
        self.scheduler.schedule_component("chunk_embedding", self.update_chunk_embeddings)
        self.scheduler.schedule_component("graph_consistency", self.verify_graph_consistency)
        self.scheduler.schedule_component("community_detection", self.detect_communities)
        self.scheduler.schedule_component("manual_edit_check", self.check_manual_edits)
        
        # 启动调度器，获取停止事件对象
        self.stop_event = self.scheduler.start()
        # 更新运行状态标志
        self.running = True
        
        self.console.print("[green]增量更新调度器已启动，正在后台运行...[/green]")
        
        # 显示调度器当前状态
        self.scheduler.print_status()
    
    def stop_scheduler(self):
        """
        停止增量更新调度器
        
        安全地停止正在运行的增量更新调度器，清理相关资源。该方法会检查调度器是否正在运行，
        如果是，则通知调度器停止，并更新运行状态标志。此方法通常在程序退出前调用，
        确保资源被正确释放。
        """
        # 检查调度器是否正在运行
        if self.running and self.stop_event:
            # 停止调度器
            self.scheduler.stop(self.stop_event)
            # 更新状态标志
            self.running = False
            self.stop_event = None
            
            self.console.print("[yellow]增量更新调度器已停止[/yellow]")
        else:
            # 调度器未运行，输出提示信息
            self.console.print("[yellow]调度器未运行[/yellow]")
    
    def display_stats(self):
        """
        显示增量更新的统计信息
        
        此方法打印增量更新管理器的运行统计信息，包括执行的更新次数、处理的文件数量、
        更新的实体数量、检测的社区数量和发生的错误数量等。这些统计信息有助于监控
        系统的运行状态和性能表现。如果调度器正在运行，还会显示调度器的当前状态。
        """
        self.console.print("\n[bold cyan]增量更新统计信息[/bold cyan]")
        self.console.print(f"[blue]执行的更新次数: {self.stats['updates_performed']}[/blue]")
        self.console.print(f"[blue]处理的文件数: {self.stats['files_processed']}[/blue]")
        self.console.print(f"[blue]更新的实体数: {self.stats['entities_updated']}[/blue]")
        self.console.print(f"[blue]检测的社区数: {self.stats['communities_detected']}[/blue]")
        self.console.print(f"[blue]错误数: {self.stats['errors']}[/blue]")
        
        # 如果调度器正在运行，也显示调度器状态
        if self.running:
            self.scheduler.print_status()
    
    def signal_handler(self, sig, frame):
        """
        信号处理函数，用于处理终止信号
        
        此方法是一个信号处理器，用于捕获和处理操作系统发送的终止信号（如SIGINT和SIGTERM）。
        当收到这些信号时，它会安全地停止调度器，显示统计信息，然后优雅地退出程序。
        这对于确保程序在被终止时能够正确清理资源非常重要。
        
        Args:
            sig: 接收到的信号
            frame: 信号发生时的栈帧
        """
        self.console.print("\n[yellow]正在退出...[/yellow]")
        
        # 如果调度器正在运行，停止它
        if self.running:
            self.stop_scheduler()
            
        # 显示统计信息
        self.display_stats()
        
        self.console.print("[green]增量更新管理器已安全退出[/green]")
        exit(0)

def main():
    """
    主函数，提供命令行接口运行增量更新管理器
    
    此函数是增量更新管理器的入口点，它解析命令行参数，初始化管理器，并根据参数决定
    以何种模式运行：单次执行模式或后台调度模式。同时支持交互模式和守护进程模式，
    提供了灵活的运行选项。此外，还注册了信号处理器，确保程序能够在被终止时正确清理资源。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="增量更新管理器")
    parser.add_argument("--dir", type=str, default=FILES_DIR, help="监控的文件目录")
    parser.add_argument("--once", action="store_true", help="执行一次更新后退出")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行")
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    parser.add_argument("--community-interval", type=int, default=1800, help="社区检测间隔（秒）")
    parser.add_argument("--manual-check-interval", type=int, default=900, help="手动编辑检查间隔（秒）")
    args = parser.parse_args()
    
    # 创建控制台输出对象
    console = Console()
    
    # 显示启动信息
    start_text = Text("启动增量更新管理器", style="bold cyan")
    console.print(Panel(start_text, border_style="cyan"))
    
    # 构建配置参数字典
    config = {
        "file_change_threshold": args.interval,
        "entity_embedding_threshold": args.interval * 2,
        "chunk_embedding_threshold": args.interval * 2,
        "graph_consistency_threshold": args.interval * 6,
        "community_detection_threshold": args.community_interval,
        "manual_edit_check_threshold": args.manual_check_interval
    }
    
    # 初始化增量更新管理器
    manager = IncrementalUpdateManager(args.dir, config)
    
    # 注册信号处理函数，用于优雅退出
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        if args.once:
            # 单次执行模式：执行一次更新后退出
            console.print("[cyan]执行一次更新后退出...[/cyan]")
            manager.run_once()
        else:
            # 调度运行模式：启动调度器
            manager.start_scheduler()
            
            if args.daemon:
                # 守护进程模式：在后台持续运行
                console.print("[cyan]以守护进程模式运行，按Ctrl+C终止...[/cyan]")
                
                # 无限循环，保持进程运行
                while True:
                    time.sleep(60)
            else:
                # 交互模式：提供命令行界面
                console.print("[cyan]增量更新管理器已启动，输入 'exit' 退出[/cyan]")
                
                # 交互式命令循环
                while True:
                    cmd = input(">>> ").strip().lower()
                    
                    if cmd == "exit":
                        # 退出命令
                        manager.stop_scheduler()
                        break
                    elif cmd == "stats":
                        # 显示统计信息
                        manager.display_stats()
                    elif cmd == "run":
                        # 手动触发一次更新
                        manager.run_once()
                    elif cmd == "help":
                        # 显示帮助信息
                        console.print("[blue]命令列表:[/blue]")
                        console.print("[blue]  exit: 退出程序[/blue]")
                        console.print("[blue]  stats: 显示统计信息[/blue]")
                        console.print("[blue]  run: 执行一次更新[/blue]")
                        console.print("[blue]  help: 显示帮助[/blue]")
                    else:
                        # 未知命令
                        console.print("[yellow]未知命令，输入 'help' 获取帮助[/yellow]")
    finally:
        # 确保无论程序如何退出，都会正确停止调度器
        if manager.running:
            manager.stop_scheduler()
            
        # 显示最终统计信息
        manager.display_stats()
        
        # 显示结束信息
        end_text = Text("增量更新管理器已退出", style="bold green")
        console.print(Panel(end_text, border_style="green"))

if __name__ == "__main__":
    main()