"""
索引和社区构建模块

此模块是Graph-RAG系统中知识图谱后期处理的关键组件，负责在基础图谱构建完成后，
执行一系列的高级处理任务，包括实体索引创建、相似实体检测与合并、社区检测和社区摘要生成。

知识图谱的质量和性能不仅取决于初始构建，还需要这些后期优化步骤。特别是在处理大型文档集合时，
可能会出现实体重复、图结构冗余等问题，通过这些优化可以显著提高知识图谱的准确性和查询效率。

模块主要功能：
1. 实体向量索引创建 - 为实体建立向量表示，支持高效的相似性搜索
2. 相似实体检测与合并 - 识别并合并语义相似的实体，减少冗余
3. 社区检测 - 基于图结构发现实体间的自然社区，揭示概念聚类
4. 社区摘要生成 - 为每个检测到的社区生成文本摘要，便于理解社区主题

整个流程设计采用了流水线架构，各阶段独立但有序衔接，形成完整的后处理流水线。
同时，该模块还实现了详细的性能监控和可视化报告功能，便于系统调优和结果评估。
"""

import os
import time
import psutil
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# 导入配置参数
from config.settings import community_algorithm
# 导入图处理相关组件
from graph import EntityIndexManager
from graph import GDSConfig, SimilarEntityDetector
from graph import EntityMerger
# 导入社区检测相关组件
from community import CommunityDetectorFactory
from community import CommunitySummarizerFactory
# 导入Neo4j图数据科学库
from graphdatascience import GraphDataScience

# 导入数据库管理和配置
from config.neo4jdb import get_db_manager
from config.settings import MAX_WORKERS, ENTITY_BATCH_SIZE, GDS_MEMORY_LIMIT

# 静默警告信息
import shutup
shutup.please()

class IndexCommunityBuilder:
    """
    索引和社区构建器，负责在基础图谱构建后的进一步处理。
    
    该类是Graph-RAG系统知识图谱后处理流水线的核心组件，在基础实体-关系图构建完成后，
    执行一系列高级处理任务，以优化图谱质量、减少冗余并提取概念结构。
    
    设计理念：
    - 模块化架构 - 各处理阶段独立封装，便于维护和扩展
    - 性能监控 - 记录每个处理阶段的时间和资源消耗
    - 结果可视化 - 使用Rich库提供直观的处理状态和结果展示
    - 容错处理 - 完善的异常捕获和处理机制
    
    主要功能包括：
    1. 实体索引的创建和管理 - 构建实体向量索引，支持高效相似性搜索
    2. 相似实体的检测和合并 - 识别并合并语义重复的实体，提高图谱精度
    3. 社区检测 - 基于图结构识别实体集群，发现潜在的概念分组
    4. 社区摘要生成 - 为每个检测到的社区生成文本摘要，增强可解释性
    
    这些处理步骤共同构成了一个完整的知识图谱优化流程，显著提升了Graph-RAG系统的
    查询效率和知识准确性。
    """
    
    def __init__(self):
        """
        初始化索引和社区构建器
        
        创建所有必要的组件和数据结构，设置性能监控统计，并初始化与Neo4j数据库的连接。
        初始化过程包括：
        1. 创建控制台输出对象用于可视化展示
        2. 初始化性能统计数据结构
        3. 设置计时器变量
        4. 初始化所有核心处理组件
        """
        # 初始化终端界面，用于生成美观的进度和结果展示
        self.console = Console()
        
        # 阶段性能统计 - 用于记录各处理阶段的时间消耗
        # 确保在_initialize_components之前定义，以便初始化阶段也能被统计
        self.performance_stats = {
            "初始化": 0,
            "索引创建": 0,
            "相似实体检测": 0,
            "实体合并": 0,
            "社区检测": 0,
            "社区摘要": 0
        }
        
        # 添加计时器，用于记录整个处理流程的开始和结束时间
        self.start_time = None
        self.end_time = None
        
        # 初始化所有必要的处理组件
        self._initialize_components()

    def _create_progress(self):
        """
        创建进度显示器
        
        使用Rich库创建一个视觉上友好的进度显示组件，用于在处理过程中提供实时的进度反馈。
        进度显示包括旋转指示器、任务描述、进度条和百分比，提高用户体验并提供可观测性。
        
        Returns:
            Progress: Rich库的Progress对象，用于显示处理进度
        """
        return Progress(
            SpinnerColumn(),  # 旋转指示器，显示活动状态
            TextColumn("[progress.description]{task.description}"),  # 任务描述
            BarColumn(),  # 进度条
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # 百分比显示
            console=self.console  # 使用预定义的控制台对象
        )

    def _initialize_components(self):
        """
        初始化所有必要的组件
        
        创建并配置所有需要的处理组件，建立数据库连接，并初始化结果存储。初始化过程分为三个主要步骤：
        1. 建立Neo4j图数据库连接和GDS（图数据科学）库连接
        2. 创建并配置各处理组件，如实体索引管理器、相似实体检测器和实体合并器
        3. 初始化处理结果存储结构
        
        同时，该方法还负责记录初始化阶段的性能统计数据，并向用户显示关键配置参数。
        """
        init_start = time.time()
        
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=3)
            
            # 步骤1: 初始化图数据库连接
            self.gds = GraphDataScience(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
            )
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)
            
            # 步骤2: 初始化各个处理组件
            # 实体索引管理器，用于创建实体向量索引
            self.index_manager = EntityIndexManager(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            
            # GDS配置对象，用于配置图数据科学操作
            self.gds_config = GDSConfig(
                memory_limit=GDS_MEMORY_LIMIT,
                batch_size=ENTITY_BATCH_SIZE
            )
            
            # 相似实体检测器，用于识别语义相似的实体
            self.entity_detector = SimilarEntityDetector(self.gds_config)
            
            # 实体合并器，用于合并检测到的相似实体
            self.entity_merger = EntityMerger(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            
            # 输出使用的关键配置参数，帮助用户了解系统设置
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {ENTITY_BATCH_SIZE}[/blue]")
            self.console.print(f"[blue]GDS内存限制: {GDS_MEMORY_LIMIT}GB[/blue]")
            
            progress.advance(task)
            
            # 步骤3: 初始化结果存储数据结构
            self.processing_results = {
                'index_process': {}
            }
            progress.advance(task)
        
        # 记录初始化阶段的时间消耗
        self.performance_stats["初始化"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格"""
        table = Table(title=title, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")
        
        for key, value in data.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        
    def _format_time(self, seconds: float) -> str:
        """格式化时间为小时:分钟:秒.毫秒"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_index_and_communities(self):
        """
        构建索引和社区
        
        执行完整的索引和社区构建流程，包括五个主要阶段：
        1. 创建实体向量索引 - 为知识图谱中的实体建立向量表示，支持相似性搜索
        2. 检测相似实体 - 使用图算法识别语义相似的实体组
        3. 合并相似实体 - 合并检测到的相似实体，减少冗余
        4. 执行社区检测 - 发现实体间的自然分组和社区结构
        5. 生成社区摘要 - 为每个检测到的社区创建文本摘要
        
        每个阶段完成后，都会记录性能统计数据并向用户显示进度和结果。
        整个流程采用流水线设计，确保各阶段有序执行，同时保持数据一致性。
        
        Returns:
            bool: 处理是否成功完成
        """
        self._display_stage_header("构建索引和社区")
        
        try:
            # 1. 创建实体索引
            index_start = time.time()
            self.console.print("[cyan]正在创建实体索引...[/cyan]")
            
            vector_store = self.index_manager.create_entity_index()
            if not vector_store:
                self.console.print("[yellow]警告: 实体索引创建可能不完整[/yellow]")
            
            self.performance_stats["索引创建"] = time.time() - index_start
            
            # 显示嵌入计算性能
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["索引创建"]
            
            self.console.print(f"[blue]索引创建完成，总耗时: {index_total:.2f}秒[/blue]")
            self.console.print(f"[blue]其中: 嵌入计算: {embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%), "
                              f"数据库操作: {db_time:.2f}秒 ({db_time/index_total*100:.1f}%)[/blue]")
            
            # 2. 检测和合并相似实体
            similar_start = time.time()
            self.console.print("[cyan]正在检测相似实体...[/cyan]")
            
            duplicates = self.entity_detector.process_entities()
            
            self.performance_stats["相似实体检测"] = time.time() - similar_start
            
            # 显示相似实体检测性能统计
            projection_time = getattr(self.entity_detector, 'projection_time', 0)
            knn_time = getattr(self.entity_detector, 'knn_time', 0)
            wcc_time = getattr(self.entity_detector, 'wcc_time', 0)
            query_time = getattr(self.entity_detector, 'query_time', 0)
            similar_total = self.performance_stats["相似实体检测"]
            
            self.console.print(f"[blue]相似实体检测完成，找到 {len(duplicates)} 组候选实体，总耗时: {similar_total:.2f}秒[/blue]")
            self.console.print(f"[blue]其中: 投影创建: {projection_time:.2f}秒 ({projection_time/similar_total*100:.1f}%), "
                              f"KNN处理: {knn_time:.2f}秒 ({knn_time/similar_total*100:.1f}%), "
                              f"WCC处理: {wcc_time:.2f}秒 ({wcc_time/similar_total*100:.1f}%), "
                              f"查询处理: {query_time:.2f}秒 ({query_time/similar_total*100:.1f}%)[/blue]")
            
            # 3. 执行实体合并
            merge_start = time.time()
            self.console.print("[cyan]正在合并相似实体...[/cyan]")
            
            merged_count = self.entity_merger.process_duplicates(duplicates)
            
            self.performance_stats["实体合并"] = time.time() - merge_start
            
            # 显示实体合并性能统计
            llm_time = getattr(self.entity_merger, 'llm_time', 0)
            parse_time = getattr(self.entity_merger, 'parse_time', 0)
            db_time = getattr(self.entity_merger, 'db_time', 0)
            merge_total = self.performance_stats["实体合并"]
            
            self._display_results_table(
                "实体合并结果",
                {
                    "合并的实体组数": merged_count,
                    "总耗时": f"{merge_total:.2f}秒",
                    "LLM处理": f"{llm_time:.2f}秒 ({llm_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)",
                    "结果解析": f"{parse_time:.2f}秒 ({parse_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)",
                    "数据库操作": f"{db_time:.2f}秒 ({db_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00秒 (0.0%)"
                }
            )
            
            # 4. 社区检测
            # 社区检测是知识图谱分析的重要步骤，能够发现实体间的自然分组结构
            community_start = time.time()
            self.console.print("[cyan]正在执行社区检测...[/cyan]")

            # 使用工厂模式创建社区检测器，根据配置选择不同的算法
            # 这种设计模式提供了灵活性，允许轻松切换不同的社区检测算法
            detector = CommunityDetectorFactory.create(
                algorithm=community_algorithm,
                gds=self.gds,
                graph=self.graph
            )
            community_results = detector.process()

            # 记录社区检测阶段的时间消耗
            self.performance_stats["社区检测"] = time.time() - community_start

            self.console.print(f"[blue]社区检测完成，耗时: {self.performance_stats['社区检测']:.2f}秒[/blue]")
            if community_results and community_results.get('status') == 'success':
                community_count = community_results.get('details', {}).get('detection', {}).get('communityCount', 0)
                self.console.print(f"[blue]检测到 {community_count} 个社区[/blue]")

            # 5. 生成社区摘要
            # 社区摘要是为每个检测到的社区生成文本描述，增强知识图谱的可解释性
            summary_start = time.time()
            self.console.print("[cyan]正在生成社区摘要...[/cyan]")

            # 使用摘要工厂类创建适合当前社区检测算法的摘要生成器
            # 这确保了摘要生成器与社区检测算法的兼容性
            summarizer = CommunitySummarizerFactory.create_summarizer(
                community_algorithm,
                self.graph
            )
            summaries = summarizer.process_communities()

            # 记录社区摘要生成阶段的时间消耗
            self.performance_stats["社区摘要"] = time.time() - summary_start

            self._display_results_table(
                "社区摘要结果",
                {
                    "生成的摘要数量": len(summaries) if summaries else 0,
                    "耗时": f"{self.performance_stats['社区摘要']:.2f}秒"
                }
            )
            
            self.console.print("[green]索引和社区构建完成[/green]")
            
            # 显示性能统计摘要
            performance_table = Table(title="性能统计摘要")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")
            
            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]索引和社区构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """
        执行完整的索引和社区构建流程
        
        这是类的主要入口方法，负责协调整个处理流程。它记录开始时间，显示系统信息，
        调用build_index_and_communities方法执行核心处理，最后显示完成信息和总耗时。
        
        该方法还包含完善的异常处理机制，确保即使在处理出错时也能提供有意义的错误信息，
        并记录处理中断前的时间消耗。
        
        处理流程包括：
        1. 记录开始时间和显示系统资源信息
        2. 显示处理开始的可视化面板
        3. 执行核心的索引和社区构建过程
        4. 记录结束时间并计算总耗时
        5. 显示处理完成的可视化面板和性能统计
        
        Returns:
            bool: 如果处理成功返回True，否则抛出异常
        
        Raises:
            Exception: 如果处理过程中出现任何错误
        """
        try:
            # 记录开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息，帮助用户了解当前运行环境
            cpu_count = os.cpu_count() or "未知"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板，提供视觉化的处理开始提示
            start_text = Text("开始索引和社区构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 执行核心的索引和社区构建过程
            self.build_index_and_communities()
            
            # 记录结束时间并计算总耗时
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # 显示完成面板，提供视觉化的处理完成提示
            success_text = Text("索引和社区构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示总耗时信息，使用格式化的时间表示
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return True
            
        except Exception as e:
            # 异常处理：即使出错也记录结束时间
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            # 显示错误信息面板
            error_text = Text(f"处理过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise  # 重新抛出异常，允许上层调用者处理

if __name__ == "__main__":
    try:
        builder = IndexCommunityBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")