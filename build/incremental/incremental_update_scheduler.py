"""
增量更新调度模块

此模块负责Graph-RAG系统中增量更新任务的调度和管理，实现高效的定时更新策略。
通过智能调度不同组件的更新频率，可以在保证系统数据新鲜度的同时，最小化计算资源消耗，
避免频繁更新带来的性能开销。

在大规模知识图谱构建和维护过程中，不同的系统组件可能需要不同的更新频率。例如，
文件变更检测可以相对频繁地执行，而社区检测等计算密集型任务则应该降低频率。该模块
通过可配置的时间阈值管理，实现了组件级别的精细调度控制。

增量更新调度器是整个增量更新框架的协调中心，它连接了文件变更检测、实体嵌入更新、
图结构验证等各个组件，确保它们按照最优策略协同工作，既能及时反映数据变化，又能
最大化计算资源利用效率。

主要功能点：
- 基于时间阈值的组件更新调度 - 为不同组件设置不同的更新频率
- 支持定时更新和按需更新两种模式 - 灵活应对不同场景需求
- 防止任务重叠执行 - 使用线程锁确保任务安全执行
- 提供灵活的更新频率配置 - 通过配置文件支持自定义调度策略
- 支持完整增量更新流程和单独组件更新 - 满足不同规模的更新需求

设计特点：
- 非阻塞执行 - 采用线程池和异步机制，不阻塞主程序流程
- 智能决策 - 基于文件变更状态自动决定是否执行后续更新步骤
- 可视化监控 - 提供实时状态展示和性能指标输出
- 优雅停止 - 支持安全停止调度器，避免任务中断导致的数据不一致
"""

import time
import schedule
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime

# Rich库用于终端美化输出
from rich.console import Console
from rich.table import Table

# 导入配置参数
from config.settings import MAX_WORKERS

class IncrementalUpdateScheduler:
    """
    增量更新调度器，管理不同组件的更新频率，确保高效的增量更新策略。
    
    该类是Graph-RAG系统增量更新机制的核心调度器，负责协调各个组件的更新时机和频率。
    系统中的不同组件（如文件变更检测、实体嵌入更新、图结构完整性验证等）可能需要
    不同的更新频率，调度器通过配置化的时间阈值管理，确保资源高效利用的同时保持数据最新。
    
    架构设计：
    - 基于阈值的调度决策系统 - 维护各组件的更新时间记录和阈值配置
    - 任务管理系统 - 负责调度、执行和监控更新任务
    - 线程安全机制 - 使用锁和事件机制确保并发安全
    - 可视化状态报告 - 提供系统运行状态的实时展示
    
    主要功能：
    1. 根据配置管理不同组件的更新频率 - 通过可配置的时间阈值
    2. 实现智能调度，避免频繁更新 - 基于上次更新时间计算是否需要更新
    3. 支持按需更新和定时更新 - 提供手动触发和自动定时两种模式
    4. 防止任务重叠执行 - 使用线程锁确保同一时间只执行一个任务
    5. 提供完整的增量更新流程和单独组件更新功能
    
    该调度器是连接增量更新各个组件的中枢，它通过精心设计的调度策略，确保系统
    在保证数据质量的同时，最大程度地减少计算资源消耗。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增量更新调度器
        
        在初始化过程中，设置默认更新阈值配置，合并用户提供的配置，初始化运行状态记录
        和任务调度相关的数据结构。默认配置为常用更新频率，用户可以通过config参数覆盖特定组件的配置。
        
        Args:
            config: 调度配置字典，包含不同组件的更新阈值（秒），可选参数
        """
        # 初始化控制台输出对象，用于美化日志输出
        self.console = Console()
        
        # 默认配置 - 为不同组件设置合理的更新频率阈值（秒）
        self.default_config = {
            "file_change_threshold": 300,  # 文件变更检测：5分钟
            "entity_embedding_threshold": 1800,  # 实体嵌入更新：30分钟
            "chunk_embedding_threshold": 1800,  # 文本块嵌入更新：30分钟
            "graph_consistency_threshold": 86400,  # 图结构完整性检查：24小时
            "community_detection_threshold": 172800,  # 社区检测更新：48小时
            "manual_edit_check_threshold": 604800  # 手动编辑检查：7天
        }
        
        # 合并默认配置和用户配置
        self.config = {**self.default_config, **(config or {})}
        
        # 记录各组件上次运行时间的字典
        self.last_run = {}
        
        # 存储调度任务的字典，键为组件名称，值为schedule.Job对象
        self.scheduled_jobs = {}
        
        # 线程锁，防止同一组件的多个实例同时运行
        self.schedule_lock = threading.Lock()
        
        # 设置最大工作线程数，从配置中获取
        self.max_workers = MAX_WORKERS
    
    def should_update(self, component: str, force: bool = False) -> bool:
        """
        判断特定组件是否需要进行更新
        
        该方法实现了增量更新的智能决策逻辑，通过检查组件的上次更新时间与当前配置的阈值比较，
        决定是否需要执行更新操作。这种机制避免了不必要的频繁更新，节约计算资源。
        
        决策逻辑设计考虑了三种情况：
        1. 强制更新 - 由force参数控制，直接返回True，允许用户覆盖调度策略
        2. 首次更新 - 如果组件从未更新过(component not in last_run)，则返回True
        3. 定时更新 - 比较当前时间与上次更新时间的差值是否超过配置的阈值
        
        这种设计既保证了系统的灵活性，又确保了资源的高效利用，是增量更新调度的核心决策机制。
        
        Args:
            component: 组件名称，用于查找对应的配置阈值和上次运行时间
            force: 是否强制更新，设为True时将忽略时间阈值检查
            
        Returns:
            bool: True表示需要更新，False表示不需要更新
        """
        # 如果强制更新，则直接返回True
        if force:
            return True
            
        # 如果组件从未运行过，则需要更新
        if component not in self.last_run:
            return True
            
        # 计算距离上次运行的时间
        current_time = time.time()
        elapsed = current_time - self.last_run[component]
        
        # 根据配置获取该组件的更新阈值，如果没有配置则使用默认值（1小时）
        threshold = self.config.get(f"{component}_threshold", 3600)
        # 判断是否超过了更新阈值
        return elapsed > threshold
    
    def mark_updated(self, component: str):
        """
        标记组件已更新，记录当前时间
        
        当组件更新完成后，调用此方法记录更新时间，作为下次决定是否需要更新的依据。
        这是增量更新调度机制的关键环节，确保更新频率控制的准确性。
        
        Args:
            component: 组件名称，将在last_run字典中记录更新时间
        """
        # 记录组件的当前更新时间
        self.last_run[component] = time.time()
    
    def schedule_component(self, component: str, processor: Callable, interval: int = None):
        """
        调度单个组件的定期更新
        
        此方法使用schedule库为指定组件创建定期执行的任务。根据配置的时间间隔，
        设置任务按小时或分钟定期运行。这是实现自动化增量更新的基础，确保系统各组件
        按照设定频率自动更新，保持数据的及时性。
        
        该方法采用了智能时间单位转换机制，根据更新间隔的大小自动选择以小时或分钟为单位
        进行调度，使调度配置更加直观和易于管理。例如，间隔大于1小时的任务将按小时调度，
        较短的间隔则按分钟调度。
        
        调度的任务会被存储在scheduled_jobs字典中，便于后续管理和查看。任务调度信息
        会通过控制台输出，提供操作的可视化反馈。
        
        Args:
            component: 组件名称，用于标识和管理任务
            processor: 处理函数，当任务触发时将被执行
            interval: 更新间隔（秒），如果为None则使用配置文件中的值
        """
        # 获取组件更新间隔，如果未指定则从配置中读取
        if interval is None:
            interval = self.config.get(f"{component}_threshold", 3600)
        
        # 将秒数转换为小时和分钟，便于人类理解的调度
        hours = interval // 3600
        minutes = (interval % 3600) // 60
        
        # 根据间隔时间选择合适的调度单位（小时或分钟）
        if hours > 0:
            # 如果间隔大于1小时，按小时调度
            job = schedule.every(hours).hours.do(self._run_component, component, processor)
        else:
            # 否则按分钟调度，确保最小间隔为1分钟
            job = schedule.every(max(1, minutes)).minutes.do(self._run_component, component, processor)
        
        # 保存调度任务，便于后续管理
        self.scheduled_jobs[component] = job
        
        # 输出调度信息
        self.console.print(f"[blue]已调度组件 {component} 每 {hours}小时{minutes}分钟 更新一次[/blue]")
    
    def _run_component(self, component: str, processor: Callable):
        """
        运行组件更新的内部方法
        
        此方法是任务调度的核心执行单元，在任务触发时被调用。它实现了任务冲突检测、
        执行状态检查、更新执行和结果处理等功能。使用非阻塞锁确保同一时间只执行一个
        组件实例，避免并发导致的数据不一致或资源争用问题。
        
        该方法采用了几个关键设计原则：
        1. 非阻塞执行 - 使用acquire(blocking=False)避免任务等待，提高系统响应性
        2. 双重检查 - 任务触发后再次检查是否应该更新，确保执行时机正确性
        3. 异常隔离 - 捕获并记录异常，但不影响调度器的整体运行
        4. 资源释放保证 - 使用finally块确保锁的释放，防止死锁
        
        这种设计确保了即使在高负载或出现错误的情况下，调度器也能保持稳定运行，不会
        因单个组件的问题而影响整个系统。
        
        Args:
            component: 组件名称
            processor: 处理函数，执行实际的更新操作
            
        Returns:
            Any: 处理函数的返回结果，如果任务被跳过或出错则返回None
        """
        # 尝试获取锁，使用非阻塞模式确保任务不会等待
        if not self.schedule_lock.acquire(blocking=False):
            self.console.print(f"[yellow]组件 {component} 更新被跳过，因为另一个任务正在运行[/yellow]")
            return
            
        try:
            # 再次检查是否应该更新，防止任务触发时已经不是合适的更新时机
            if self.should_update(component):
                self.console.print(f"[cyan]开始更新组件: {component}[/cyan]")
                
                # 执行实际的更新操作
                result = processor()
                
                # 标记更新完成
                self.mark_updated(component)
                
                self.console.print(f"[green]组件 {component} 更新完成[/green]")
                return result
            else:
                # 如果不满足更新条件，则跳过
                self.console.print(f"[yellow]组件 {component} 更新被跳过，因为上次更新时间较近[/yellow]")
                
        except Exception as e:
            # 捕获并记录异常，但不会阻止调度器继续运行
            self.console.print(f"[red]组件 {component} 更新出错: {e}[/red]")
        finally:
            # 确保无论执行结果如何，都会释放锁
            self.schedule_lock.release()
    
    def schedule_update(self, processor):
        """
        调度完整的增量更新流程，设置所有组件的定期更新
        
        此方法一次性为系统中的所有关键组件设置定期更新任务，按照配置的频率调度各个组件。
        这些组件共同构成了完整的增量更新流程，从文件变更检测开始，到实体嵌入更新、
        图结构完整性验证等一系列处理。处理器对象需要实现所有必要的方法。
        
        该方法采用了组件化设计思想，将增量更新过程分解为多个独立的组件，每个组件负责
        特定的功能，并且可以独立调度。这种模块化设计带来了以下优势：
        1. 灵活配置 - 每个组件可以设置不同的更新频率
        2. 故障隔离 - 单个组件的问题不会影响其他组件
        3. 性能优化 - 计算密集型任务可以降低频率，减少资源消耗
        4. 可扩展性 - 便于添加新的更新组件
        
        调度的组件按照逻辑依赖关系配置，确保数据处理的正确性和一致性。例如，文件变更
        检测频率通常高于嵌入更新和图结构验证，因为它是增量更新的触发点。
        
        Args:
            processor: 处理对象，必须包含各种处理方法，如detect_file_changes、
                      update_entity_embeddings、verify_graph_consistency等
        """
        # 调度文件变更检测组件
        self.schedule_component("file_change", processor.detect_file_changes, 
                               self.config["file_change_threshold"])
        
        # 调度实体Embedding更新组件
        self.schedule_component("entity_embedding", processor.update_entity_embeddings,
                               self.config["entity_embedding_threshold"])
        
        # 调度文本块Embedding更新组件
        self.schedule_component("chunk_embedding", processor.update_chunk_embeddings,
                               self.config["chunk_embedding_threshold"])
        
        # 调度图结构完整性验证组件
        self.schedule_component("graph_consistency", processor.verify_graph_consistency,
                               self.config["graph_consistency_threshold"])
        
        # 调度社区检测更新组件
        self.schedule_component("community_detection", processor.detect_communities,
                               self.config["community_detection_threshold"])
        
        # 调度全量重建检查（低频率）
        self.schedule_component("full_rebuild", processor.rebuild_if_needed,
                               self.config["full_rebuild_threshold"])
    
    def start(self):
        """
        启动增量更新调度器
        
        创建并启动一个后台线程，用于定期检查和执行调度任务。使用线程事件(Event)机制，
        允许在需要时安全地停止调度线程。调度线程以守护线程方式运行，确保主程序退出时
        调度线程也会自动终止。
        
        Returns:
            threading.Event: 用于停止调度器的事件对象
        """
        self.console.print("[cyan]启动增量更新调度器...[/cyan]")
        
        # 创建停止事件，用于后续控制线程终止
        cease_continuous_run = threading.Event()
        
        # 定义调度线程类
        class ScheduleThread(threading.Thread):
            @classmethod
            def run(cls):
                # 循环执行，直到接收到停止信号
                while not cease_continuous_run.is_set():
                    # 执行所有待处理的调度任务
                    schedule.run_pending()
                    # 休眠一分钟，减少CPU占用
                    time.sleep(60)
        
        # 创建并启动调度线程
        continuous_thread = ScheduleThread()
        # 设为守护线程，确保主程序退出时自动终止
        continuous_thread.daemon = True
        continuous_thread.start()
        
        # 返回停止事件，供调用者控制调度器停止
        return cease_continuous_run
    
    def stop(self, cease_event):
        """
        停止增量更新调度器
        
        安全地停止调度线程并清理所有调度任务。使用传入的停止事件通知调度线程终止运行，
        同时清除schedule库中的所有任务，确保资源被正确释放，避免僵尸线程或未完成的任务。
        
        Args:
            cease_event: 停止事件对象，通过start()方法返回
        """
        self.console.print("[yellow]正在停止增量更新调度器...[/yellow]")
        # 设置停止事件，通知调度线程终止
        cease_event.set()
        
        # 清除所有调度任务
        schedule.clear()
        # 清空任务记录
        self.scheduled_jobs.clear()
        
        # 输出停止完成信息
        self.console.print("[green]增量更新调度器已停止[/green]")
    
    def run_once(self, processor):
        """
        立即执行一次完整的增量更新流程
        
        提供按需执行增量更新的功能，通常用于手动触发更新或系统启动时的初始化更新。
        执行流程首先检测文件变更，如果发现有新增、修改或删除的文件，则继续执行后续的
        嵌入更新和图结构验证等步骤。如果没有检测到文件变更，则跳过后续步骤，节约资源。
        
        该方法实现了智能执行策略，只有当有实际文件变更时才执行完整的更新流程，这种
        设计大大提高了系统的效率，避免了不必要的计算开销。同时，它使用锁机制确保
        同一时间只有一个run_once实例在执行，避免并发冲突。
        
        执行顺序遵循增量更新的逻辑依赖关系：
        1. 首先进行文件变更检测 - 确定哪些文件需要处理
        2. 然后更新实体嵌入 - 确保实体向量表示的更新
        3. 接着更新文本块嵌入 - 确保文本块向量表示的更新
        4. 最后验证图结构完整性 - 确保知识图谱的一致性
        
        每个步骤完成后，都会更新相应组件的最后运行时间记录。
        
        Args:
            processor: 处理对象，必须包含各种处理方法，如detect_file_changes、
                      update_entity_embeddings、verify_graph_consistency等
        """
        # 获取锁，确保同一时间只有一个run_once执行
        with self.schedule_lock:
            self.console.print("[bold cyan]开始执行一次完整增量更新...[/bold cyan]")
            
            try:
                # 第一步：检测文件变更
                file_changes = processor.detect_file_changes()
                # 标记文件变更检测已执行
                self.mark_updated("file_change")
                
                # 智能执行：只有当有文件变更时才执行后续步骤
                if file_changes and (file_changes.get("added") or 
                                    file_changes.get("modified") or 
                                    file_changes.get("deleted")):
                    # 更新实体Embedding
                    processor.update_entity_embeddings()
                    self.mark_updated("entity_embedding")
                    
                    # 更新文本块Embedding
                    processor.update_chunk_embeddings()
                    self.mark_updated("chunk_embedding")
                    
                    # 验证图结构完整性
                    processor.verify_graph_consistency()
                    self.mark_updated("graph_consistency")
                else:
                    # 如果没有检测到变更，跳过后续处理以节约资源
                    self.console.print("[yellow]没有检测到文件变更，跳过后续步骤[/yellow]")
                
                self.console.print("[green]完整增量更新执行完成[/green]")
                
            except Exception as e:
                # 捕获并记录任何异常
                self.console.print(f"[red]执行增量更新时出错: {e}[/red]")
    
    def force_update(self, component: str, processor: Callable):
        """
        强制更新指定组件，忽略时间阈值
        
        提供一种方式来手动触发特定组件的更新，而不考虑其配置的更新频率。
        这在需要立即更新特定组件时非常有用，例如在系统维护后或发现数据问题时。
        该方法确保更新操作的独占执行，并记录更新结果。
        
        Args:
            component: 组件名称
            processor: 处理函数，执行实际的更新操作
            
        Returns:
            Any: 处理函数的返回结果，如果执行出错则返回None
        """
        # 获取锁，确保更新操作的独占执行
        with self.schedule_lock:
            self.console.print(f"[bold cyan]强制更新组件: {component}[/bold cyan]")
            
            try:
                # 执行实际的更新操作
                result = processor()
                
                # 标记组件已更新，记录当前时间
                self.mark_updated(component)
                
                self.console.print(f"[green]组件 {component} 强制更新完成[/green]")
                # 返回处理结果
                return result
                
            except Exception as e:
                # 捕获并记录异常
                self.console.print(f"[red]强制更新组件 {component} 时出错: {e}[/red]")
                
    def print_status(self):
        """
        打印增量更新调度器的当前状态
        
        使用Rich库的表格组件，生成并显示各组件的更新状态信息，包括上次更新时间、
        下次计划更新时间和配置的更新间隔。这为管理员提供了系统更新状态的可视化概览，
        便于监控和管理。
        
        状态报告是系统可观测性的重要组成部分，提供了以下价值：
        1. 实时监控 - 直观了解各组件的更新状态和时间
        2. 问题排查 - 快速识别长期未更新或配置异常的组件
        3. 决策支持 - 帮助管理员评估调度策略是否合理
        4. 审计记录 - 记录系统更新活动的历史信息
        
        该方法使用Rich库的表格组件，生成格式美观、信息丰富的状态报告，提高了系统的
        用户体验和管理效率。
        """
        # 输出标题
        self.console.print("\n[bold cyan]增量更新调度器状态[/bold cyan]")
        
        # 创建状态表格
        status_table = Table(title="组件更新状态")
        status_table.add_column("组件", style="cyan")
        status_table.add_column("上次更新", style="magenta")
        status_table.add_column("下次更新", style="green")
        status_table.add_column("间隔", style="blue")
        
        # 获取当前时间
        current_time = time.time()
        
        # 组件名称和对应的配置键映射
        for component, threshold_key in [
            ("文件变更", "file_change_threshold"),
            ("实体Embedding", "entity_embedding_threshold"),
            ("Chunk Embedding", "chunk_embedding_threshold"),
            ("图结构完整性", "graph_consistency_threshold"),
            ("社区检测", "community_detection_threshold"),
            ("手动编辑检查", "manual_edit_check_threshold")
        ]:
            # 获取上次更新时间
            component_key = component.lower().replace(" ", "_")
            last_update = self.last_run.get(component_key, None)
            # 格式化上次更新时间显示
            last_update_str = "从未" if last_update is None else datetime.fromtimestamp(last_update).strftime("%Y-%m-%d %H:%M:%S")
            
            # 获取更新阈值并计算下次更新时间
            threshold = self.config.get(threshold_key, 3600)
            if last_update is not None:
                next_update = last_update + threshold
                next_update_str = datetime.fromtimestamp(next_update).strftime("%Y-%m-%d %H:%M:%S")
            else:
                next_update_str = "立即"  # 如果从未更新过，则下次更新为立即
            
            # 格式化更新间隔
            hours = threshold // 3600
            minutes = (threshold % 3600) // 60
            interval_str = f"{hours}小时{minutes}分钟"
            
            # 添加行到表格
            status_table.add_row(component, last_update_str, next_update_str, interval_str)
        
        # 打印完整表格
        self.console.print(status_table)