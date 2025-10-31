import os
import time
import psutil  # 用于获取系统资源使用情况
from typing import Dict, Any

# 富文本显示库，用于美化控制台输出
from rich.console import Console  # 高级终端文本处理
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn  # 进度条组件
from rich.table import Table  # 表格显示组件
from rich.panel import Panel  # 面板显示组件
from rich.text import Text  # 富文本处理组件
from graph import ChunkIndexManager  # 文本块索引管理器，负责创建和管理向量索引
from config.neo4jdb import get_db_manager  # 获取Neo4j数据库管理器实例
from config.settings import MAX_WORKERS, CHUNK_BATCH_SIZE  # 配置参数：并行工作线程数和批处理大小

# 用于抑制第三方库的警告信息
import shutup
shutup.please()  # 执行静音操作，避免不必要的警告信息

class ChunkIndexBuilder:
    """
    文本块索引构建器，负责在基础图谱构建后为Chunk节点创建向量索引，以支持naive RAG查询。
    
    该类实现了高效的文本块向量索引构建流程，支持大规模文本数据的向量化处理，并提供详细的性能统计信息。
    向量索引是实现语义搜索的关键组件，通过将文本转换为高维向量表示，支持基于相似度的高效检索。
    
    主要功能包括：
    1. Chunk节点索引的创建和管理 - 为文本块生成并存储向量表示
    2. 向量索引性能统计 - 跟踪并展示索引构建过程中的各项性能指标
    3. 进度可视化 - 提供实时的进度反馈和结果展示
    4. 错误处理和日志记录 - 确保构建过程的健壮性
    """
    
    def __init__(self):
        """初始化文本块索引构建器
        
        创建富文本控制台实例，初始化性能统计字典和计时器，
        并调用组件初始化方法设置必要的数据库连接和索引管理器。
        """
        # 初始化富文本控制台，用于美化输出
        self.console = Console()
        
        # 阶段性能统计字典，记录各个处理阶段的耗时
        self.performance_stats = {
            "初始化": 0,       # 组件初始化阶段耗时
            "索引创建": 0      # 索引构建阶段耗时
        }
        
        # 计时器变量，用于记录整个处理流程的开始和结束时间
        self.start_time = None  # 流程开始时间
        self.end_time = None    # 流程结束时间
        
        # 初始化必要的组件，包括数据库连接和索引管理器
        self._initialize_components()

    def _create_progress(self):
        """创建进度显示器
        
        返回一个配置好的Progress对象，用于显示任务进度。
        进度条包含动画旋转器、任务描述、进度条和百分比显示。
        
        Returns:
            Progress: 配置好的进度显示器实例
        """
        return Progress(
            SpinnerColumn(),           # 动画旋转器，提供视觉反馈
            TextColumn("[progress.description]{task.description}"),  # 任务描述文本
            BarColumn(),               # 进度条图形显示
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # 进度百分比
            console=self.console       # 使用已初始化的控制台
        )

    def _initialize_components(self):
        """初始化所有必要的组件
        
        初始化图数据库连接和文本块索引管理器，设置批处理大小和并行工作线程数，
        并记录初始化阶段的耗时。使用进度条展示初始化过程。
        """
        init_start = time.time()  # 记录初始化开始时间
        
        # 创建并显示进度条
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=2)  # 设置进度任务，共2个步骤
            
            # 步骤1: 初始化图数据库连接
            self.graph = get_db_manager().graph  # 获取图数据库对象
            progress.advance(task)  # 更新进度条
              
            # 步骤2: 初始化文本块索引管理器
            self.index_manager = ChunkIndexManager(
                batch_size=CHUNK_BATCH_SIZE,  # 批处理大小，控制每次处理的文本块数量
                max_workers=MAX_WORKERS       # 并行工作线程数，控制并发处理能力
            )
            
            # 输出使用的配置参数，帮助用户了解当前设置
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {CHUNK_BATCH_SIZE}[/blue]")
            progress.advance(task)  # 更新进度条到100%
        
        # 记录初始化阶段总耗时
        self.performance_stats["初始化"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题
        
        以醒目的格式显示当前处理阶段的标题，帮助用户跟踪处理进度。
        
        Args:
            title: 要显示的阶段标题文本
        """
        # 使用粗体青色文本显示阶段标题，添加换行确保视觉分隔
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格
        
        创建并显示一个格式化的表格，用于展示各种统计数据和结果。
        
        Args:
            title: 表格标题
            data: 要显示的键值对数据字典
        """
        # 创建表格对象，设置标题和显示表头
        table = Table(title=title, show_header=True)
        # 添加两列：指标名称（青色）和指标值（右对齐）
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")
        
        # 将字典中的每个键值对添加为表格的一行
        for key, value in data.items():
            table.add_row(key, str(value))
        
        # 输出表格到控制台
        self.console.print(table)
        
    def _format_time(self, seconds: float) -> str:
        """格式化时间为小时:分钟:秒.毫秒
        
        将秒数转换为易读的时间格式：HH:MM:SS.XXX
        
        Args:
            seconds: 要格式化的时间（秒）
            
        Returns:
            str: 格式化后的时间字符串
        """
        # 将总秒数分解为小时、分钟和秒
        hours, remainder = divmod(seconds, 3600)  # 3600秒 = 1小时
        minutes, seconds = divmod(remainder, 60)  # 60秒 = 1分钟
        
        # 格式化时间字符串，包括毫秒部分（精确到3位）
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_chunk_index(self):
        """
        构建文本块索引
        
        核心方法，负责调用索引管理器为数据库中的文本块创建向量索引。
        记录索引创建过程的性能指标，包括总耗时、嵌入计算时间和数据库操作时间，
        并显示已索引的节点数量等统计信息。
        
        Returns:
            bool: 处理是否成功
        """
        self._display_stage_header("构建文本块索引")
        
        try:
            # 创建Chunk索引
            index_start = time.time()  # 记录索引创建开始时间
            self.console.print("[cyan]正在创建文本块索引...[/cyan]")
            
            # 调用索引管理器创建文本块索引
            # 该过程会计算每个文本块的向量嵌入并存储到数据库中
            vector_store = self.index_manager.create_chunk_index()
            
            # 记录索引创建阶段总耗时
            self.performance_stats["索引创建"] = time.time() - index_start
            
            # 显示嵌入计算性能统计
            # 安全地获取索引管理器的嵌入计算时间（如果不存在则默认为0）
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            # 安全地获取索引管理器的数据库操作时间（如果不存在则默认为0）
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["索引创建"]
            
            # 仅当有实际耗时记录时显示性能统计
            if index_total > 0:
                self.console.print(f"[blue]索引创建完成，总耗时: {index_total:.2f}秒[/blue]")
                # 显示各阶段耗时及其占比，帮助用户了解性能瓶颈
                self.console.print(f"[blue]其中: 嵌入计算: {embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%), "
                                f"数据库操作: {db_time:.2f}秒 ({db_time/index_total*100:.1f}%)[/blue]")
            
            # 查询已成功创建索引的节点数量
            try:
                # 使用Cypher查询语言查询所有具有embedding属性的Chunk节点
                node_count = self.graph.query(
                    """
                    MATCH (c:`__Chunk__`)       // 匹配所有Chunk类型的节点
                    WHERE c.embedding IS NOT NULL  // 筛选出已成功创建嵌入向量的节点
                    RETURN count(c) as count       // 返回匹配节点的数量
                    """
                )
                
                # 显示索引创建结果统计表格
                self._display_results_table(
                    "索引创建结果",
                    {
                        "已索引节点数量": node_count[0]["count"] if node_count else 0,  # 成功索引的节点总数
                        "总耗时": f"{index_total:.2f}秒",  # 索引创建总耗时
                        # 嵌入计算耗时及其占比
                        "嵌入计算": f"{embedding_time:.2f}秒 ({embedding_time/index_total*100:.1f}%)" if index_total > 0 else "0.00秒",
                        # 数据库操作耗时及其占比
                        "数据库操作": f"{db_time:.2f}秒 ({db_time/index_total*100:.1f}%)" if index_total > 0 else "0.00秒"
                    }
                )
            except Exception as e:
                # 查询失败时显示警告，但不中断流程
                self.console.print(f"[yellow]查询索引状态时出错 (可忽略): {e}[/yellow]")
            
            self.console.print("[green]文本块索引构建完成[/green]")
            
            # 显示性能统计摘要表格
            performance_table = Table(title="性能统计摘要")
            performance_table.add_column("处理阶段", style="cyan")
            performance_table.add_column("耗时(秒)", justify="right")
            performance_table.add_column("占比(%)", justify="right")
            
            # 计算总耗时
            total_time = sum(self.performance_stats.values())
            
            # 添加每个处理阶段的耗时和占比
            for stage, elapsed in self.performance_stats.items():
                # 计算各阶段耗时占总耗时的百分比
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            # 添加总计行，使用粗体突出显示
            performance_table.add_row("总计", f"{total_time:.2f}", "100.0", style="bold")
            
            # 显示性能统计摘要
            self.console.print(performance_table)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]文本块索引构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行文本块索引构建流程
        
        主流程入口，协调整个文本块索引构建过程。记录系统资源信息，
        显示开始和结束面板，调用索引构建方法，并处理可能出现的异常。
        无论成功或失败，都会显示总耗时信息。
        
        Returns:
            bool: 处理是否成功
        """
        try:
            # 记录整个处理流程的开始时间
            self.start_time = time.time()
            
            # 显示系统资源信息，帮助用户了解运行环境
            cpu_count = os.cpu_count() or "未知"  # 获取CPU核心数
            # 计算内存总量（GB）
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"系统信息: CPU核心数 {cpu_count}, 内存 {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # 显示开始面板，提供视觉提示
            start_text = Text("开始文本块索引构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 执行核心的文本块索引构建操作
            self.build_chunk_index()
            
            # 记录整个处理流程的结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time  # 计算总耗时
            
            # 显示完成面板，提供成功的视觉反馈
            success_text = Text("文本块索引构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示格式化的总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return True
            
        except Exception as e:
            # 异常处理：记录结束时间（即使出错）
            self.end_time = time.time()
            
            # 如果有开始时间记录，则计算中断前的耗时
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            # 显示错误信息面板，使用红色突出显示错误
            error_text = Text(f"处理过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            
            # 重新抛出异常，允许上层调用者处理
            raise