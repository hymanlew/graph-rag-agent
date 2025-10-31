# 系统相关模块导入
import time  # 用于计时和性能监控
import os  # 用于文件系统操作和获取CPU核心数
import psutil  # 用于获取系统资源使用情况
from typing import Dict, Any, List, Tuple  # 用于类型注解

# 富文本显示库，用于美化控制台输出
from rich.console import Console  # 高级终端文本处理
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn  # 进度条组件
from rich.table import Table  # 表格显示组件
from rich.panel import Panel  # 面板显示组件
from rich.text import Text  # 富文本处理组件

# 项目核心模块导入
from model.get_models import get_llm_model, get_embeddings_model  # 获取LLM模型和嵌入模型
from config.prompt import (
    system_template_build_graph,  # 构建图谱的系统提示模板
    human_template_build_graph   # 构建图谱的人类提示模板
)
from config.settings import (
    entity_types,           # 定义的实体类型列表
    relationship_types,     # 定义的关系类型列表
    theme,                  # 主题/领域标签
    FILES_DIR,              # 文件存储目录
    CHUNK_SIZE,             # 文本分块大小
    OVERLAP,                # 文本块重叠大小
    MAX_WORKERS, BATCH_SIZE,  # 并行工作线程数和批处理大小
)
from config.neo4jdb import get_db_manager  # 获取Neo4j数据库管理器实例
from processor.document_processor import DocumentProcessor  # 文档处理器，负责文件读取和分块
from graph import GraphStructureBuilder  # 图结构构建器，负责创建基础图结构
from graph import EntityRelationExtractor  # 实体关系抽取器，负责从文本中提取实体和关系
from graph import GraphWriter  # 图写入器，负责将图数据写入数据库

# 用于抑制第三方库的警告信息
import shutup
shutup.please()  # 执行静音操作，避免不必要的警告信息


class KnowledgeGraphBuilder:
    """
    知识图谱构建器，负责图谱的基础构建流程。
    
    该类是整个图数据库构建的核心组件，实现了从原始文档到知识图谱的完整转换过程。
    构建过程采用流水线架构，包括文件处理、文本分块、图结构构建、实体关系抽取和数据库写入等阶段。
    
    主要功能包括：
    1. 文件读取和解析 - 支持多种格式文档的读取和解析
    2. 文本分块 - 将长文本切分为大小合适的文本块，平衡上下文完整性和处理效率
    3. 实体和关系抽取 - 使用LLM从文本中提取实体、关系和属性信息
    4. 构建基础图结构 - 创建文档和文本块的节点及关联关系
    5. 写入数据库 - 将构建的图结构持久化到Neo4j数据库中
    6. 性能统计和监控 - 记录各阶段的处理时间和资源消耗
    """
    
    def __init__(self):
        """初始化知识图谱构建器
        
        创建富文本控制台实例，初始化文档列表、计时器和性能统计字典，
        并调用组件初始化方法设置模型、数据库连接和各处理组件。
        """
        # 初始化富文本控制台，用于美化输出和进度显示
        self.console = Console()
        # 存储已处理文档的列表，每个文档包含文件名、内容、分块和处理结果
        self.processed_documents = []
        
        # 计时器变量，用于记录整个处理流程的开始和结束时间
        self.start_time = None  # 流程开始时间
        self.end_time = None    # 流程结束时间
        
        # 阶段性能统计字典，记录各个处理阶段的耗时
        self.performance_stats = {
            "初始化": 0,          # 组件初始化阶段耗时
            "文件处理": 0,        # 文件读取和分块阶段耗时
            "图结构构建": 0,       # 创建文档和文本块图结构的耗时
            "实体抽取": 0,         # 从文本中提取实体和关系的耗时
            "写入数据库": 0        # 将图数据写入数据库的耗时
        }
        
        # 初始化必要的组件，包括模型、数据库连接和各种处理器
        self._initialize_components()

    def _create_progress(self):
        """创建进度显示器
        
        返回一个配置好的Progress对象，用于显示任务进度。
        进度条包含动画旋转器、任务描述、进度条和百分比显示，提升用户体验。
        
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
        
        初始化LLM模型、嵌入模型、图数据库连接、文档处理器、图结构构建器和实体关系抽取器，
        并记录初始化阶段的耗时。使用进度条展示初始化过程。
        """
        init_start = time.time()  # 记录初始化开始时间
        
        # 创建并显示进度条
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]初始化组件...", total=4)  # 设置进度任务，共4个步骤
            
            # 步骤1: 初始化LLM模型和嵌入模型
            # 这些模型是知识图谱构建的核心，LLM用于实体关系抽取，嵌入模型用于向量表示
            self.llm = get_llm_model()  # 获取大语言模型实例
            self.embeddings = get_embeddings_model()  # 获取文本嵌入模型实例
            progress.advance(task)  # 更新进度条到25%
            
            # 步骤2: 初始化图数据库连接
            # 连接到Neo4j数据库，用于存储构建的知识图谱
            db_manager = get_db_manager()  # 获取数据库管理器实例
            self.graph = db_manager.graph  # 获取图数据库对象
            progress.advance(task)  # 更新进度条到50%
            
            # 步骤3: 初始化文档处理器
            # 配置文档处理器，设置文件目录、分块大小和重叠大小
            self.document_processor = DocumentProcessor(FILES_DIR, CHUNK_SIZE, OVERLAP)
            progress.advance(task)  # 更新进度条到75%
            
            # 步骤4: 初始化图结构构建器和实体关系抽取器
            # 图结构构建器负责创建基础图结构，实体关系抽取器负责从文本中提取知识
            self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)  # 批处理大小优化数据库写入性能
            self.entity_extractor = EntityRelationExtractor(
                self.llm,  # 用于文本理解和实体关系提取的大语言模型
                system_template_build_graph,  # 系统提示模板，指导模型如何进行实体关系抽取
                human_template_build_graph,   # 人类提示模板，提供具体的文本内容
                entity_types,    # 定义需要提取的实体类型
                relationship_types,  # 定义需要提取的关系类型
                max_workers=MAX_WORKERS,  # 并行处理线程数
                batch_size=5  # LLM批处理大小保持较小以确保提取质量
            )
            
            # 输出使用的配置参数，帮助用户了解当前设置
            self.console.print(f"[blue]并行处理线程数: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]数据库批处理大小: {BATCH_SIZE}[/blue]")
            
            progress.advance(task)  # 更新进度条到100%
        
        # 记录初始化阶段总耗时
        self.performance_stats["初始化"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """显示处理阶段的标题，以醒目的格式显示当前处理阶段的标题。
        
        Args:
            title: 要显示的阶段标题文本
        """
        # 使用粗体青色文本显示阶段标题，添加换行确保视觉分隔
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格，创建并显示一个格式化的表格，用于清晰展示各种统计数据和处理结果。
        
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
        将秒数转换为易读的时间格式：HH:MM:SS.XXX，用于显示处理耗时。
        
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

    def build_base_graph(self) -> List:
        """
        构建基础知识图谱
        
        核心方法，实现从原始文档到知识图谱的完整构建流程。
        包括文件处理、图结构构建、实体关系抽取和数据库写入等关键步骤。
        该方法是Graph-RAG系统的基础，构建的知识图谱将用于后续的语义检索和问答任务。
        
        Returns:
            List: 处理后的文件内容列表，包含文件名、原文、分块和处理结果
        """
        self._display_stage_header("构建基础知识图谱")
        
        try:
            # 1. 处理文件（读取和分块）
            # 这一步是知识图谱构建的起点，将原始文档转换为可处理的文本块
            process_start = time.time()  # 记录文件处理开始时间
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]处理文件...", total=1)
                
                # 使用DocumentProcessor处理目录中的所有文件
                # 该方法会读取文件内容、识别文件类型，并根据配置参数进行文本分块
                self.processed_documents = self.document_processor.process_directory()
                progress.update(task, completed=1)
                
                # 显示文件处理结果信息表格
                table = Table(title="文件信息")
                table.add_column("文件名")
                table.add_column("类型", style="cyan")
                table.add_column("内容长度", justify="right")
                table.add_column("分块数量", justify="right")
                
                # 为每个处理过的文档添加一行信息
                for doc in self.processed_documents:
                    file_type = self.document_processor.get_extension_type(doc["extension"])
                    chunks_count = doc.get("chunk_count", 0)
                    table.add_row(
                        doc["filename"],  # 文件名
                        file_type,         # 文件类型
                        str(doc["content_length"]),  # 文件内容长度（字符数）
                        str(chunks_count)   # 生成的文本块数量
                    )
                self.console.print(table)  # 输出文件信息表格
            
            # 记录文件处理阶段耗时
            self.performance_stats["文件处理"] = time.time() - process_start
            
            # 显示分块统计信息
            # 计算总文本块数量、总内容长度和平均块大小，帮助评估分块策略的有效性
            total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
            total_length = sum(doc["content_length"] for doc in self.processed_documents)
            # 计算平均块大小，处理除零情况
            avg_chunk_size = sum(sum(doc.get("chunk_lengths", [0])) for doc in self.processed_documents) / total_chunks if total_chunks else 0
            
            # 输出处理统计信息
            self.console.print(f"[blue]共处理 {len(self.processed_documents)} 个文件，总计 {total_length} 字符[/blue]")
            self.console.print(f"[blue]共生成 {total_chunks} 个文本块，平均每块 {avg_chunk_size:.1f} 字符[/blue]")
            
            # 3. 构建图结构
            # 这一步创建基础知识图谱的框架（写入图数据库），包括文档节点和文本块节点及其关系
            struct_start = time.time()  # 记录图结构构建开始时间
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]构建图结构...", total=3)
                
                # 步骤1: 清空数据库并创建Document节点
                self.struct_builder.clear_database()
                # 为每个成功分块的文档创建Document节点
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        self.struct_builder.create_document(
                            type="local",      # 文档类型为本地文件
                            uri=str(FILES_DIR),  # 文件存储目录路径
                            file_name=doc["filename"],  # 文件名
                            domain=theme        # 文档所属领域/主题
                        )
                progress.advance(task)  # 更新进度条到33%
                
                # 步骤2: 创建Chunk节点和关系，优化策略：根据文档大小选择不同的处理方法
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        chunks = doc["chunks"]
                        # 性能优化：对于大块数的文档使用并行处理
                        if doc.get("chunk_count", 0) > 100:
                            result = self.struct_builder.parallel_process_chunks(
                                doc["filename"],  # 文件名，用于关联
                                chunks,           # 文本块列表
                                max_workers=os.cpu_count() or 4  # 使用系统CPU核心数，至少4个
                            )
                        else:
                            # 对于小文件使用标准批处理，避免并行开销
                            result = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],  # 文件名，用于关联
                                chunks            # 文本块列表
                            )
                        # 保存处理结果到文档数据中
                        doc["graph_result"] = result
                progress.advance(task)  # 更新进度条到66%
                progress.advance(task)  # 更新进度条到100%
            
            # 记录图结构构建阶段耗时
            self.performance_stats["图结构构建"] = time.time() - struct_start
            
            # 4. 提取实体和关系
            # 这是知识图谱构建的核心步骤，使用LLM从文本中提取结构，实体与关系知识（缓存到文件中）
            extract_start = time.time()  # 记录实体关系提取开始时间
            with self._create_progress() as progress:
                # 计算总文本块数量，用于进度显示
                total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
                task = progress.add_task("[cyan]提取实体和关系...", total=total_chunks)
                
                # 定义进度回调函数，用于更新进度条
                def progress_callback(chunk_index):
                    progress.advance(task)
                
                # 准备处理的数据格式，转换为实体抽取器期望的格式
                file_contents_format = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        file_contents_format.append([
                            doc["filename"],  # 文件名
                            doc["content"],   # 原始内容
                            doc["chunks"]     # 文本块列表
                        ])
                
                # 提取实体和关系 性能优化：根据数据集大小选择不同的处理方法
                if total_chunks > 100:
                    # 对于大型数据集使用批处理模式，更高效地利用LLM API
                    processed_file_contents = self.entity_extractor.process_chunks_batch(
                        file_contents_format,
                        progress_callback
                    )
                else:
                    # 对于小型数据集使用标准并行处理，更简单且足够高效
                    processed_file_contents = self.entity_extractor.process_chunks(
                        file_contents_format,
                        progress_callback
                    )
                
                # 创建文件名到实体数据的映射，提高查找效率
                file_content_map = {}
                for processed_file in processed_file_contents:
                    if len(processed_file) >= 4:  # 确保有足够的元素，防止索引错误
                        filename = processed_file[0]      # 文件名作为键
                        entity_data = processed_file[3]   # 实体数据作为值
                        file_content_map[filename] = entity_data
                
                # 使用映射将结果放回到原始文档中，便于后续处理
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                        filename = doc["filename"]
                        if filename in file_content_map:
                            # 将实体提取结果关联到对应的文档
                            doc["entity_data"] = file_content_map[filename]
                        else:
                            # 错误处理：输出警告信息，但不中断流程
                            self.console.print(f"[yellow]警告: 文件 {filename} 的实体抽取结果未找到[/yellow]")
            
            # 记录实体关系提取阶段耗时
            self.performance_stats["实体抽取"] = time.time() - extract_start
            
            # 输出缓存统计信息，评估缓存系统的有效性
            # 安全地获取实体抽取器的缓存统计数据（如果不存在则默认为0）
            cache_hits = getattr(self.entity_extractor, 'cache_hits', 0)  # 缓存命中次数
            cache_misses = getattr(self.entity_extractor, 'cache_misses', 0)  # 缓存未命中次数
            total_requests = cache_hits + cache_misses  # 总请求数
            # 计算缓存命中率，处理除零情况
            cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            # 显示缓存统计信息，帮助评估性能优化效果
            self.console.print(f"[blue]LLM调用缓存命中率: {cache_rate:.1f}% ({cache_hits}/{total_requests})[/blue]")
            
            # 5. 写入数据库
            # 这是知识图谱构建的最后一步，将提取的结构化知识持久化到图数据库
            write_start = time.time()  # 记录数据库写入开始时间
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]写入数据库...", total=1)
                
                # 将处理数据转换为GraphWriter所需格式
                graph_writer_data = []
                for doc in self.processed_documents:
                    # 只处理有文本块和实体数据的文档
                    if "chunks" in doc and doc["chunks"] and "entity_data" in doc:
                        # 获取图构建结果（创建的chunk节点列表）和实体数据，包含 chunk_id, chunk Document
                        graph_result = doc.get("graph_result", [])
                        entity_data = doc.get("entity_data", [])
                        
                        # 数据完整性检查：确保必要数据存在且格式正确
                        if not graph_result:
                            self.console.print(f"[yellow]警告: 文件 {doc['filename']} 的图结构结果缺失[/yellow]")
                            continue
                            
                        if not entity_data or not isinstance(entity_data, list):
                            self.console.print(f"[yellow]警告: 文件 {doc['filename']} 的实体数据缺失或格式不正确[/yellow]")
                            continue
                            
                        # 调整数据格式以匹配GraphWriter期望的结构
                        graph_writer_data.append([
                            doc["filename"],    # 文件名
                            doc["content"],     # 原始内容
                            doc["chunks"],      # 文本块列表
                            graph_result,        # 图结构构建结果（chunks_with_hash数据）
                            entity_data,         # 实体关系提取结果
                        ])
                
                # 使用优化的GraphWriter将数据写入数据库
                graph_writer = GraphWriter(
                    self.graph,            # 图数据库连接
                    batch_size=50,         # 数据库操作批处理大小，优化写入性能
                    max_workers=os.cpu_count() or 4  # 并行工作线程数
                )
                # 执行数据库写入操作，将实体、关系和属性信息写入Neo4j
                graph_writer.process_and_write_graph_documents(graph_writer_data)
                progress.update(task, completed=1)  # 更新进度条到100%
            
            # 记录数据库写入阶段耗时
            self.performance_stats["写入数据库"] = time.time() - write_start
            
            # 显示完成信息
            self.console.print("[green]基础知识图谱构建完成[/green]")
            
            # 显示各阶段性能统计表格，帮助用户了解性能瓶颈
            performance_table = Table(title="性能统计")
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
            self.console.print(performance_table)
            
            # 返回处理好的文档列表，格式化为兼容的形式
            # 这个返回值主要用于后续处理或其他组件的调用
            file_contents_compat = []
            for doc in self.processed_documents:
                if "chunks" in doc and doc["chunks"]:  # 只处理成功分块的文档
                    # 构建标准格式的返回数据
                    content_list = [
                        doc["filename"],  # 文件名
                        doc["content"],   # 原始内容
                        doc["chunks"]     # 文本块列表
                    ]
                    # 如果有实体数据，则添加到返回列表中
                    if "entity_data" in doc:
                        content_list.append(doc["entity_data"])
                    file_contents_compat.append(content_list)
            
            return file_contents_compat
            
        except Exception as e:
            self.console.print(f"[red]基础图谱构建失败: {str(e)}[/red]")
            raise

    def process(self):
        """执行知识图谱构建流程，记录系统资源信息，显示开始和结束面板，调用基础图谱构建方法，并处理可能出现的异常。
        无论成功或失败，都会显示总耗时信息。
        
        Returns:
            List: 处理后的文档列表
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
            start_text = Text("开始知识图谱构建流程", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # 执行核心的基础知识图谱构建操作
            result = self.build_base_graph()
            
            # 记录整个处理流程的结束时间
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time  # 计算总耗时
            
            # 显示完成面板，提供成功的视觉反馈
            success_text = Text("知识图谱构建流程完成", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # 显示格式化的总耗时信息
            self.console.print(f"[bold green]总耗时：{self._format_time(elapsed_time)}[/bold green]")
            
            return result
            
        except Exception as e:
            # 异常处理：记录结束时间（即使出错）
            self.end_time = time.time()
            
            # 如果有开始时间记录，则计算中断前的耗时
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]中断前耗时：{self._format_time(elapsed_time)}[/bold yellow]")
                
            # 显示错误信息面板，使用红色突出显示错误
            error_text = Text(f"构建过程中出现错误: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            
            # 重新抛出异常，允许上层调用者处理
            raise

if __name__ == "__main__":
    """主程序入口，用于直接运行知识图谱构建器"""
    try:
        # 创建知识图谱构建器实例并执行构建流程
        builder = KnowledgeGraphBuilder()
        builder.process()
    except Exception as e:
        # 全局异常处理，确保错误信息被正确显示
        console = Console()
        console.print(f"[red]执行过程中出现错误: {str(e)}[/red]")