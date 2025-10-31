import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class FileChangeManager:
    """
    文件变更管理器，负责追踪文件的变更状态。
    
    该类是Graph-RAG增量更新系统的重要组成部分，通过维护文件注册表和计算文件哈希值，
    实现高效的文件变更检测。在知识图谱的增量构建过程中，只有识别出发生变化的文件，
    才能针对性地更新知识图谱，避免全量重建带来的性能开销。
    
    设计理念：
    - 基于内容的变更检测：使用SHA256哈希算法计算文件内容的唯一标识，相比文件元数据更可靠
    - 文件变更检测（新增、修改、删除）- 三向对比的高效变更识别
    - 持久化状态存储：通过JSON格式的文件注册表，保存文件状态信息，确保系统重启后能够恢复
    - 增量处理支持：精确识别新增、修改和删除的文件，为增量更新提供必要的信息
    - 处理历史追踪：记录每次文件处理的详细信息，便于性能分析和系统调试
    
    主要功能：
    1. 扫描文件目录，计算文件哈希值 - 用于精确识别文件内容变化
    2. 与历史记录比较，识别变更的文件 - 区分新增、修改和删除的文件
    3. 更新文件注册表 - 维护文件状态的持久化存储
    4. 记录文件处理历史 - 追踪每次处理的时间和结果
    
    该类在增量更新流程中通常是第一个被调用的组件，它的检测结果直接决定了后续处理哪些文件。
    """
    
    def __init__(self, files_dir: str, registry_path: str = "./file_registry.json"):
        """
        初始化文件变更管理器
        
        初始化过程中，设置监控目录路径，指定注册表文件位置，并加载现有的文件注册表。
        文件注册表是一个JSON文件，用于持久化存储文件的元数据和状态信息。
        
        Args:
            files_dir: 要监控的文件目录，通常指向Graph-RAG系统的文档存储目录
            registry_path: 文件注册表保存路径，默认为当前目录下的file_registry.json
        """
        # 将目录路径转换为Path对象，便于跨平台路径操作
        self.files_dir = Path(files_dir)
        # 设置注册表文件路径
        self.registry_path = Path(registry_path)
        # 加载现有的文件注册表
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        加载文件注册表
        
        从磁盘加载文件注册表JSON文件，解析为Python字典对象。文件注册表记录了所有监控文件
        的元数据信息，包括哈希值、文件大小、修改时间等。如果注册表文件不存在或无法解析，
        则返回空字典，表示没有历史记录。
        
        Returns:
            Dict: 文件注册表，键为文件相对路径，值为文件元数据字典
        """
        # 检查注册表文件是否存在
        if not self.registry_path.exists():
            return {}
            
        try:
            # 读取并解析注册表JSON文件
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # 错误处理：如果文件无法解析或不存在，返回空字典
            print(f"无法加载文件注册表，将创建新的注册表")
            return {}
    
    def _save_registry(self):
        """
        保存文件注册表到磁盘
        
        将内存中的文件注册表字典序列化为JSON格式，并写入指定路径的文件。使用ensure_ascii=False
        确保非ASCII字符（如中文文件名）能正确保存，indent=2使JSON文件格式化输出，便于阅读。
        这是持久化文件变更状态的关键方法，确保系统重启后能够恢复之前的文件状态信息。
        """
        # 序列化注册表字典并写入文件
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        计算文件的SHA256哈希值
        
        通过计算文件内容的SHA256哈希值，可以精确判断文件内容是否发生变化。这种方法比仅依赖
        文件修改时间更可靠，因为它直接基于文件内容而非文件系统元数据。使用分块读取的方式处理
        文件，避免一次性加载大文件导致内存溢出。
        
        Args:
            file_path: 文件路径对象
            
        Returns:
            str: 文件的SHA256哈希值，如计算失败则返回空字符串
        """
        # 创建SHA256哈希对象
        hash_obj = hashlib.sha256()
        try:
            # 以二进制模式打开文件
            with open(file_path, 'rb') as f:
                # 分块读取文件内容，每块4096字节
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            # 返回十六进制格式的哈希值
            return hash_obj.hexdigest()
        except Exception as e:
            # 错误处理：记录异常信息并返回空字符串
            print(f"计算文件哈希值失败: {file_path}, 错误: {e}")
            return ""
    
    def _scan_current_files(self) -> Dict[str, Dict[str, Any]]:
        """
        扫描当前文件目录中的所有文件，收集文件元数据
        
        此方法递归扫描监控目录中的所有文件，计算每个文件的哈希值并收集其他元数据信息。
        扫描结果以文件相对路径为键，文件元数据字典为值进行存储。这是变更检测的基础步骤，
        提供了当前文件系统状态的完整视图。
        
        扫描过程采用递归方式遍历所有子目录，确保不会遗漏任何文件。对于每个文件，
        计算其内容哈希值并收集关键元数据，这些数据将用于与历史状态比较，从而识别变更。
        该方法使用相对路径作为索引，避免了因监控目录位置变化而导致的路径不一致问题。
        
        Returns:
            Dict: 当前文件状态字典，键为文件相对路径，值为包含哈希值、大小、修改时间等的元数据
        """
        # 初始化当前文件状态字典
        current_files = {}
        
        # 遍历文件目录，递归访问子目录
        for root, _, files in os.walk(self.files_dir):
            for filename in files:
                # 构建完整文件路径
                file_path = Path(root) / filename
                # 计算相对路径，作为字典键
                rel_path = str(file_path.relative_to(self.files_dir))
                
                # 计算文件哈希值，用于内容比较
                file_hash = self._compute_file_hash(file_path)
                # 如果哈希值计算失败，跳过该文件
                if not file_hash:
                    continue
                
                # 记录文件元数据
                file_info = {
                    "hash": file_hash,               # 文件内容哈希值，用于检测内容变化
                    "size": file_path.stat().st_size,  # 文件大小（字节）
                    "last_modified": file_path.stat().st_mtime,  # 最后修改时间戳
                    "last_scanned": time.time()     # 本次扫描时间
                }
                current_files[rel_path] = file_info
        
        return current_files
    
    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变更，识别新增、修改和删除的文件
        
        这是FileChangeManager的核心方法，通过比较当前扫描到的文件状态与历史注册表，
        准确识别出三种类型的文件变更：新增的文件、内容被修改的文件以及被删除的文件。
        变更检测结果为增量更新系统提供了关键输入，系统可以据此决定哪些文件需要重新处理，
        哪些需要从知识图谱中移除。
        
        算法原理：
        1. 首先获取当前文件系统状态，包括所有文件的哈希值和元数据
        2. 通过三向比较策略识别变更：
           - 对于当前文件系统中的文件，如果注册表中不存在，则为新增文件
           - 对于当前文件系统中的文件，如果注册表中存在但哈希值不同，则为修改文件
           - 对于注册表中的文件，如果当前文件系统中不存在，则为删除文件
        
        这种方法确保了对文件变更的全面检测，同时避免了不必要的重复计算，提高了检测效率。
        
        Returns:
            Dict: 包含三种变更类型的文件列表：
                 - added: 新增的文件路径列表
                 - modified: 修改过的文件路径列表
                 - deleted: 删除的文件路径列表
        """
        # 获取当前文件状态
        current_files = self._scan_current_files()
        
        # 初始化变更类型列表
        added_files = []     # 新增文件列表
        modified_files = []  # 修改文件列表
        deleted_files = []   # 删除文件列表
        
        # 检测新增和修改的文件
        # 遍历当前所有文件，与注册表比较
        for file_path, file_info in current_files.items():
            if file_path not in self.registry:
                # 注册表中不存在的文件，视为新增
                added_files.append(file_path)
            elif file_info["hash"] != self.registry[file_path]["hash"]:
                # 文件存在但哈希值不同，视为修改
                modified_files.append(file_path)
        
        # 检测删除的文件
        # 遍历注册表，查找当前文件系统中不存在的文件
        for file_path in self.registry:
            if file_path not in current_files:
                # 当前文件系统中不存在的文件，视为删除
                deleted_files.append(file_path)
        
        # 返回变更结果字典
        return {
            "added": added_files,
            "modified": modified_files,
            "deleted": deleted_files
        }
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件的元数据信息
        
        根据文件相对路径从注册表中检索文件的元数据信息，包括哈希值、大小、修改时间等。
        如果文件不在注册表中，返回空字典。此方法常用于检查特定文件的状态或历史信息。
        
        Args:
            file_path: 文件相对路径
            
        Returns:
            Dict: 文件元数据字典，如果文件不存在于注册表中则返回空字典
        """
        # 从注册表中获取文件元数据，不存在则返回空字典
        return self.registry.get(file_path, {})
    
    def update_registry(self):
        """
        更新文件注册表，记录当前所有文件的状态
        
        扫描当前文件系统状态，更新注册表，并将新状态持久化保存到磁盘。这通常在增量更新
        完成后调用，确保下一次变更检测基于最新的文件状态。该方法会完全重写注册表，
        覆盖所有历史记录。
        
        该方法在增量更新流程中扮演着"提交"的角色，标志着一轮增量更新的完成。通过保存当前
        文件系统的完整状态，为下一轮增量更新提供基准。需要注意的是，调用此方法后，
        之前的文件状态信息将被完全替换，因此应确保在所有文件处理完成后再调用。
        """
        # 重新扫描文件系统并更新注册表
        self.registry = self._scan_current_files()
        # 保存更新后的注册表到磁盘
        self._save_registry()
        # 输出更新信息
        print(f"文件注册表已更新，共记录 {len(self.registry)} 个文件")
    
    def update_file_status(self, file_path: str, status: Dict[str, Any]):
        """
        更新单个文件的状态信息
        
        对已存在于注册表中的文件，更新其状态信息。这允许在不重新扫描整个文件系统的情况下，
        针对性地更新特定文件的元数据。更新后会自动保存注册表到磁盘。
        
        Args:
            file_path: 文件相对路径
            status: 要更新的文件状态字典，会与现有状态合并
        """
        # 仅当文件存在于注册表中时才更新
        if file_path in self.registry:
            # 更新文件状态信息
            self.registry[file_path].update(status)
            # 保存更新后的注册表
            self._save_registry()
    
    def register_file_processing(self, file_path: str, processing_info: Dict[str, Any]):
        """
        记录文件处理信息，追踪处理历史
        
        此方法用于记录文件的处理历史，包括处理时间和各种处理指标（如处理耗时、生成的节点数等）。
        通过维护处理历史，系统可以追踪文件的处理状态，分析处理性能，并在必要时进行调试。
        
        处理历史记录是系统可观测性的重要组成部分，它提供了以下价值：
        1. 性能分析 - 通过记录处理时间和资源消耗，识别性能瓶颈
        2. 审计追踪 - 追踪每个文件的处理历史，便于问题排查
        3. 变更影响评估 - 分析文件变更对知识图谱的影响
        4. 处理质量监控 - 跟踪处理结果的质量指标
        
        Args:
            file_path: 文件相对路径
            processing_info: 处理信息字典，包含处理相关的各种指标，如处理时间、生成的节点数等
        """
        # 仅当文件存在于注册表中时才记录处理信息
        if file_path in self.registry:
            # 确保处理历史列表存在
            if "processing_history" not in self.registry[file_path]:
                self.registry[file_path]["processing_history"] = []
            
            # 创建处理记录，添加时间戳和用户提供的处理信息
            processing_record = {
                "timestamp": datetime.now().isoformat(),  # 处理时间戳
                **processing_info  # 展开用户提供的处理信息
            }
            
            # 添加处理记录到历史列表
            self.registry[file_path]["processing_history"].append(processing_record)
            # 更新最后处理时间
            self.registry[file_path]["last_processed"] = processing_record["timestamp"]
            # 保存更新后的注册表
            self._save_registry()