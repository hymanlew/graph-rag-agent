import os
import time
import json
import threading
from typing import Any, Optional, List, Tuple, Dict
from collections import OrderedDict
from .base import CacheStorageBackend


class DiskCacheBackend(CacheStorageBackend):
    """
    磁盘缓存后端实现
    
    提供持久化的磁盘缓存存储，使用以下技术：
    1. 文件系统层次结构优化（前缀子目录）
    2. 批量写入机制减少I/O操作
    3. 索引文件维护缓存元数据
    4. 复合淘汰策略（访问频率+新近度+文件大小）
    5. 线程安全实现（使用RLock）
    """
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000, 
                 batch_size: int = 10, flush_interval: float = 30.0):
        """
        初始化磁盘缓存后端
        
        参数:
            cache_dir: 缓存文件存储目录
            max_size: 缓存最大项数
            batch_size: 批量写入的大小阈值
            flush_interval: 自动刷新间隔（秒）
        """
        # 配置参数
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # 内存数据结构，使用OrderedDict维护访问顺序
        self.metadata: OrderedDict[str, Dict[str, Any]] = OrderedDict()  # 保存缓存项元数据，用于LRU
        self.write_queue: List[Tuple[str, Any]] = []  # 批量写入队列，减少I/O
        self.last_flush_time = time.time()  # 上次刷新时间
        self._lock = threading.RLock()  # 可重入锁，确保线程安全
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载缓存索引，恢复之前的状态
        self._load_index()
    
    def _get_cache_path(self, key: str) -> str:
        """
        获取缓存文件路径
        
        使用两级目录结构，将键的前两个字符作为子目录名
        这种设计避免了单个目录下文件过多导致的性能问题
        
        参数:
            key: 缓存键
            
        返回:
            str: 缓存文件的完整路径
        """
        # 使用键的前两个字符作为子目录名
        subdir = key[:2]
        subdir_path = os.path.join(self.cache_dir, subdir)
        # 确保子目录存在
        os.makedirs(subdir_path, exist_ok=True)
        # 返回缓存文件路径
        return os.path.join(subdir_path, f"{key}.json")
    
    def _get_index_path(self) -> str:
        """
        获取索引文件路径
        
        索引文件存储所有缓存项的元数据信息
        
        返回:
            str: 索引文件的完整路径
        """
        return os.path.join(self.cache_dir, "index.json")
    
    def _load_index(self) -> None:
        """
        加载缓存索引
        
        从索引文件中恢复缓存元数据
        按照最后访问时间排序，维持LRU顺序
        加载后同步索引与文件系统状态
        """
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 保持访问顺序 - 按照最后访问时间排序
                    # 这样可以在重启后依然保持LRU淘汰策略
                    for key in sorted(data.keys(), key=lambda k: data[k].get('last_accessed', 0)):
                        self.metadata[key] = data[k]
            except Exception as e:
                print(f"加载缓存索引失败: {e}")
                # 索引文件损坏时，重置为空
                self.metadata = OrderedDict()
        
        # 验证磁盘上的文件并同步索引
        # 处理可能的文件丢失或额外文件情况
        self._sync_index_with_filesystem()
    
    def _sync_index_with_filesystem(self) -> None:
        """
        同步索引与文件系统
        
        确保内存中的元数据与实际磁盘文件一致
        处理以下情况：
        1. 索引中有但文件已删除的项
        2. 文件存在但索引中没有的项
        """
        existing_files = set()
        
        # 扫描所有子目录，收集所有实际存在的缓存文件键
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            # 只处理由键前缀创建的子目录（长度为2）
            if os.path.isdir(item_path) and len(item) == 2:
                for filename in os.listdir(item_path):
                    if filename.endswith(".json"):
                        key = filename[:-5]  # 移除.json后缀
                        existing_files.add(key)
        
        # 清理：移除索引中不存在的文件
        # 处理文件已被手动删除的情况
        keys_to_remove = [key for key in self.metadata if key not in existing_files]
        for key in keys_to_remove:
            del self.metadata[key]
        
        # 发现：添加文件系统中存在但索引中没有的文件
        # 处理索引损坏但文件仍然存在的情况
        for key in existing_files:
            if key not in self.metadata:
                file_path = self._get_cache_path(key)
                try:
                    # 从文件系统获取文件信息
                    stat = os.stat(file_path)
                    self.metadata[key] = {
                        "created_at": stat.st_ctime,     # 创建时间
                        "last_accessed": stat.st_atime, # 最后访问时间
                        "access_count": 0,              # 重置访问计数
                        "file_size": stat.st_size       # 文件大小
                    }
                except OSError:
                    continue  # 忽略无法访问的文件
    
    def _save_index(self) -> None:
        """
        保存缓存索引
        
        将内存中的元数据写入索引文件
        记录所有缓存项的访问信息和统计数据
        """
        try:
            with open(self._get_index_path(), 'w', encoding='utf-8') as f:
                # 转换OrderedDict为普通字典进行序列化
                json.dump(dict(self.metadata), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存索引失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项
        
        线程安全的缓存读取实现，同时更新访问统计
        
        参数:
            key: 缓存键
            
        返回:
            Optional[Any]: 缓存值，如果不存在或读取失败则返回None
        """
        with self._lock:  # 加锁确保线程安全
            cache_path = self._get_cache_path(key)
            
            # 检查键是否在元数据中且文件存在
            if key in self.metadata and os.path.exists(cache_path):
                try:
                    # 读取缓存文件
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        value = json.load(f)
                    
                    # 更新访问信息并移到末尾（LRU策略）
                    self.metadata[key]["last_accessed"] = time.time()
                    self.metadata[key]["access_count"] = self.metadata[key].get("access_count", 0) + 1
                    
                    # 移动到OrderedDict末尾，确保最近访问的项不会被优先淘汰
                    self.metadata.move_to_end(key)
                    
                    # 异步保存索引（延迟写入，避免频繁I/O）
                    self._schedule_index_save()
                    
                    return value
                except Exception as e:
                    print(f"读取缓存文件失败 ({key}): {e}")
                    # 如果文件损坏，从索引中删除
                    if key in self.metadata:
                        del self.metadata[key]
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存项
        
        实现了批量写入机制和缓存淘汰策略
        
        参数:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 缓存淘汰检查：如果缓存已满且是新键，执行淘汰
            if len(self.metadata) >= self.max_size and key not in self.metadata:
                self._evict_items()
            
            # 更新元数据
            current_time = time.time()
            if key in self.metadata:
                # 更新现有项的元数据
                self.metadata[key].update({
                    "last_accessed": current_time,
                    "access_count": self.metadata[key].get("access_count", 0)
                })
                # 移动到末尾，更新LRU顺序
                self.metadata.move_to_end(key)
            else:
                # 新增项的元数据
                self.metadata[key] = {
                    "created_at": current_time,    # 创建时间
                    "last_accessed": current_time, # 最后访问时间
                    "access_count": 0             # 访问计数
                }
            
            # 添加到写入队列（批量处理，减少I/O操作）
            self.write_queue.append((key, value))
            
            # 检查是否需要刷新写入队列
            # 条件1：队列大小达到阈值
            # 条件2：距离上次刷新时间过长
            if (len(self.write_queue) >= self.batch_size or 
                (time.time() - self.last_flush_time) > self.flush_interval):
                self._flush_write_queue()
    
    def _flush_write_queue(self) -> None:
        """
        刷新写入队列
        
        将队列中的数据批量写入磁盘
        跟踪成功和失败的写入操作
        更新元数据和索引
        """
        if not self.write_queue:
            return
        
        successful_writes = []  # 成功写入的键列表
        failed_writes = []      # 失败写入的键值对列表
        
        # 处理队列中的所有写入请求
        for key, value in self.write_queue:
            try:
                # 获取缓存文件路径
                cache_path = self._get_cache_path(key)
                # 写入JSON文件
                # default=str参数确保不可JSON序列化的对象也能被处理
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, indent=2, default=str)
                
                # 更新文件大小信息
                if key in self.metadata:
                    self.metadata[key]["file_size"] = os.path.getsize(cache_path)
                
                successful_writes.append(key)
            except Exception as e:
                print(f"写入缓存文件失败 ({key}): {e}")
                failed_writes.append((key, value))
        
        # 只保留失败的写入操作，供下次重试
        self.write_queue = failed_writes
        # 更新最后刷新时间
        self.last_flush_time = time.time()
        
        # 如果有成功的写入，保存索引更新
        if successful_writes:
            self._save_index()
    
    def _schedule_index_save(self) -> None:
        """
        调度索引保存
        
        避免过于频繁地保存索引，控制I/O操作频率
        实现索引更新的节流机制
        """
        current_time = time.time()
        # 每分钟最多保存一次索引，避免频繁I/O
        if current_time - self.last_flush_time > 60:  # 每分钟最多保存一次
            self._save_index()
            self.last_flush_time = current_time
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        从元数据、文件系统和写入队列中完全移除缓存项
        
        参数:
            key: 要删除的缓存键
            
        返回:
            bool: 删除是否成功
        """
        with self._lock:
            # 检查键是否存在
            if key not in self.metadata:
                return False
            
            # 从元数据中删除
            del self.metadata[key]
            
            # 删除文件
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    print(f"删除缓存文件失败 ({key}): {e}")
                    return False
            
            # 从写入队列中移除（避免被后续刷新操作重新写入）
            self.write_queue = [(k, v) for k, v in self.write_queue if k != key]
            
            # 保存更新后的索引
            self._save_index()
            return True
    
    def clear(self) -> None:
        """
        清空缓存
        
        完全清除所有缓存数据，包括：
        1. 写入队列中的待处理数据
        2. 内存中的元数据
        3. 磁盘上的所有缓存文件
        4. 索引文件
        """
        with self._lock:
            # 清空写入队列
            self.write_queue.clear()
            
            # 清空元数据
            self.metadata.clear()
            
            # 删除所有缓存文件（遍历目录结构）
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith(".json"):
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception as e:
                            print(f"删除缓存文件失败: {e}")
            
            # 保存空索引
            self._save_index()
    
    def flush(self) -> None:
        """
        强制刷新所有待写入的数据
        
        将写入队列中的所有数据立即写入磁盘
        用于程序退出或关键操作后的持久化
        """
        with self._lock:
            self._flush_write_queue()
    
    def _evict_items(self, num_to_evict: int = None) -> None:
        """
        淘汰缓存项
        
        使用复合评分策略选择要淘汰的缓存项：
        1. 访问频率（每小时访问次数）
        2. 新近度（距离最后访问的时间）
        3. 文件大小（较大的文件优先淘汰）
        
        参数:
            num_to_evict: 要淘汰的项目数量，默认为缓存大小的10%
        """
        if not self.metadata:
            return
        
        # 默认淘汰10%的缓存项
        if num_to_evict is None:
            num_to_evict = max(1, len(self.metadata) // 10)  # 淘汰10%
        
        # 计算每个缓存项的评分
        # 使用复合评分策略：访问频率 + 新近度 + 文件大小
        current_time = time.time()
        scores = {}
        
        for key, meta in self.metadata.items():
            # 计算项的年龄（从创建到现在）
            age = current_time - meta.get("created_at", current_time)
            # 获取访问次数
            access_count = meta.get("access_count", 0)
            # 计算距最后访问的时间
            last_accessed = meta.get("last_accessed", meta.get("created_at", current_time))
            recency = current_time - last_accessed
            # 获取文件大小（默认1KB）
            file_size = meta.get("file_size", 1000)  # 默认1KB
            
            # 计算复合分数（分数越低越容易被淘汰）
            # 1. 每小时访问频率：访问次数/年龄（小时）
            frequency_score = access_count / max(age / 3600, 1)  # 每小时访问频率
            # 2. 新近度分数：距离最后访问时间的倒数（小时）
            recency_score = 1 / max(recency / 3600, 1)  # 新近度分数
            # 3. 文件大小惩罚：较大的文件有更高的淘汰优先级
            size_penalty = file_size / 1024  # 大文件惩罚
            
            # 总分计算：访问频率 + 新近度 - 文件大小惩罚*0.1
            scores[key] = frequency_score + recency_score - size_penalty * 0.1
        
        # 选择分数最低的项目进行淘汰
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[:num_to_evict]
        
        # 执行淘汰
        for key in keys_to_evict:
            self.delete(key)