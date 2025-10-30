import time
import os
import pickle
import concurrent.futures
from typing import List, Tuple, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from graph.core import retry, generate_hash
from config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS, BATCH_SIZE as DEFAULT_BATCH_SIZE

class EntityRelationExtractor:
    """
    实体关系提取器
    
    功能：
    - 使用大语言模型进行自然语言理解和分析文本内容
    - 从文本中识别提取预定义类型的实体和关系
    - 支持并行处理多个文本块
    - 实现缓存机制减少重复计算
    - 提供批处理功能优化LLM调用效率
    
    实现思路：
    - 使用LangChain构建提示模板和处理链
    - 采用线程池实现并行处理
    - 通过哈希和文件存储实现高效缓存，避免重复处理相同内容
    - 设计动态批处理策略根据文本大小调整批次，减少LLM调用次数
    - 实现自动重试和错误处理机制

    大文件处理支持
    - 流式处理模式，避免内存溢出
    - 实时结果处理和图数据库写入

    架构特点：
    - 多层处理架构：缓存层、批处理层、并行处理层
    - 完整的错误处理：异常捕获、重试机制、降级策略
    - 详细的性能监控：处理时间、缓存命中率统计
    - 资源优化：动态批处理、内存管理、并行控制

    该模块采用异步并行处理架构，结合缓存机制，实现了高效的实体关系提取流程，为构建高质量的知识图谱提供了坚实基础。
    """
    def __init__(self, llm, system_template, human_template, 
             entity_types: List[str], relationship_types: List[str],
             cache_dir="./cache/graph", max_workers=4, batch_size=5):
        """
        初始化实体关系提取器
        
        参数：
            llm: 语言模型实例，用于执行实体关系提取任务
            system_template: 系统提示模板，指导模型如何执行提取任务
            human_template: 用户提示模板，定义输入格式和输出要求
            entity_types: 实体类型列表，定义要提取的实体类型
            relationship_types: 关系类型列表，定义要提取的关系类型
            cache_dir: 缓存目录，存储处理结果以避免重复计算
            max_workers: 并行工作线程数，控制并发处理能力
            batch_size: 批处理大小，控制批量提交给LLM的文本块数量
        
        初始化流程：
        1. 保存基本配置参数和模型实例
        2. 设置输出解析所需的分隔符
        3. 构建LangChain处理链
        4. 初始化缓存系统和统计计数器
        5. 创建并行处理配置
        """
        # 保存基本配置
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []
        
        # 设置特殊分隔符，用于解析模型输出
        self.tuple_delimiter = " : "     # 实体-关系-实体三元组中的分隔符
        self.record_delimiter = "\n"       # 不同记录之间的分隔符
        self.completion_delimiter = "\n\n"  # 完成标志分隔符

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        # 构建完整的对话提示模板，包含系统提示、历史消息和用户输入
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])
        self.chain = self.chat_prompt | self.llm
        
        # 缓存设置
        self.cache_dir = cache_dir
        self.enable_cache = True
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 并行处理配置，支持自定义或使用默认值
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        
        # 缓存统计计数器
        self.cache_hits = 0      # 缓存命中次数
        self.cache_misses = 0    # 缓存未命中次数
    
    def _generate_cache_key(self, text: str) -> str:
        """
        生成文本的缓存键
        
        参数：
            text: 输入文本内容
            
        返回：
            str: 基于文本内容的唯一哈希值，用于缓存查找
        """
        # 调用核心模块的哈希生成函数，创建文本内容的唯一标识
        return generate_hash(text)
    
    def _cache_path(self, cache_key: str) -> str:
        """
        获取缓存文件路径
        
        参数：
            cache_key: 缓存键
            
        返回：
            str: 缓存文件的完整路径
        
        实现说明：
            - 缓存文件存储在配置的缓存目录中
            - 使用.pkl扩展名标识pickle序列化的文件
            - 确保缓存文件与缓存键一一对应
        """
        # 构建缓存文件的完整路径：缓存目录 + 缓存键.pkl
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """
        保存处理结果到缓存，使用pickle序列化结果并存储到文件
        
        参数：
            cache_key: 缓存键
            result: 需要缓存的处理结果
        """
        # 检查缓存功能是否启用
        if not self.enable_cache:
            return
            
        # 获取缓存文件路径
        cache_path = self._cache_path(cache_key)
        try:
            # 使用pickle序列化并保存结果到二进制文件
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            # 错误处理：缓存保存失败只记录日志，不影响主流程
            print(f"缓存保存错误: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """
        从缓存加载处理结果，使用pickle反序列化加载缓存结果
        
        参数：
            cache_key: 缓存键
            
        返回：
            Optional[str]: 缓存的处理结果，如果缓存不存在则返回None
        """
        # 检查缓存功能是否启用
        if not self.enable_cache:
            return None
            
        # 获取缓存文件路径
        cache_path = self._cache_path(cache_key)
        
        # 检查缓存文件是否存在
        if os.path.exists(cache_path):
            try:
                # 反序列化加载缓存结果
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                    # 更新缓存命中统计
                    self.cache_hits += 1
                    return result
            except Exception as e:
                # 错误处理：缓存加载失败记录日志，不影响主流程
                print(f"缓存加载错误: {e}")
        
        # 更新缓存未命中统计
        self.cache_misses += 1
        return None
    
    def process_chunks(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        并行处理多个文件的文本块
        
        参数：
            file_contents: 文件内容列表，每个文件包含多个文本块
            progress_callback: 可选的进度回调函数
            
        返回：
            List[Tuple]: 更新后的文件内容列表，包含提取结果
            
        处理流程：
        1. 计算总任务量并开始计时
        2. 对每个文件分别处理
        3. 预检查缓存，只处理未缓存的文本块
        4. 使用线程池并行处理未缓存的文本块
        5. 实现错误重试机制确保稳定性
        6. 按原始顺序整理结果
        7. 输出性能和缓存统计信息
        
        优化特点：
        - 缓存优先策略，减少重复计算
        - 并行处理提高CPU利用率
        - 三级重试机制增强稳定性
        - 详细的性能监控和统计
        """
        # 开始计时，用于性能统计
        t0 = time.time()
        # 全局块索引，用于进度报告
        chunk_index = 0
        # 计算总文本块数量用于进度计算
        total_chunks = sum(len(file_content[2]) for file_content in file_contents)
        
        # 逐个处理文件
        for i, file_content in enumerate(file_contents):
            # 获取文件的文本块（假设在索引2位置）
            chunks = file_content[2]
            
            # 预检查缓存命中率，生成所有文本块的缓存键
            cache_keys = [self._generate_cache_key(''.join(chunk)) for chunk in chunks]
            # 批量加载缓存结果，构建缓存键到结果的映射
            cached_results = {key: self._load_from_cache(key) for key in cache_keys}
            # 找出需要处理的未缓存的索引，只有缓存未命中的块才需要处理
            non_cached_indices = [idx for idx, key in enumerate(cache_keys) if cached_results[key] is None]
            
            # 如果有未缓存的文本块需要处理
            if len(non_cached_indices) > 0:
                # 只为未缓存的chunks创建任务，避免重复处理
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 创建任务字典，将future映射到chunk索引，用于跟踪处理结果
                    future_to_chunk = {
                        executor.submit(self._process_single_chunk, ''.join(chunks[idx])): idx 
                        for idx in non_cached_indices
                    }
                    
                    # 处理完成的任务
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        # 获取对应的文本块索引
                        chunk_idx = future_to_chunk[future]
                        try:
                            # 获取处理结果
                            result = future.result()
                            # 存储结果到缓存结果映射中
                            cached_results[cache_keys[chunk_idx]] = result
                            
                            # 更新进度
                            if progress_callback:
                                progress_callback(chunk_index)
                            chunk_index += 1
                            
                        except Exception as exc:
                            print(f'Chunk {chunk_idx} 处理异常: {exc}')
                            # 三级重试逻辑，增强系统稳定性
                            retry_count = 0
                            while retry_count < 3:
                                try:
                                    print(f'尝试重试 Chunk {chunk_idx}, 第 {retry_count+1} 次')
                                    # 单独处理失败的文本块
                                    result = self._process_single_chunk(''.join(chunks[chunk_idx]))
                                    cached_results[cache_keys[chunk_idx]] = result
                                    break
                                except Exception as retry_exc:
                                    print(f'重试失败: {retry_exc}')
                                    retry_count += 1
                                    # 短暂延迟避免频繁请求导致的API限流
                                    time.sleep(1)
                            
                            # 安全措施：如果所有重试都失败，确保结果列表不为None，避免后续处理出错
                            if cached_results[cache_keys[chunk_idx]] is None:
                                cached_results[cache_keys[chunk_idx]] = ""
            
            # 整理结果，保持原始顺序
            ordered_results = [cached_results[key] for key in cache_keys]
            # 将处理结果附加到文件内容中
            file_content.append(ordered_results)
            
            # 计算并输出缓存统计信息
            cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            print(f"文件 {i+1}/{len(file_contents)} 处理完成, 缓存命中率: {cache_ratio:.1f}%")
        
        # 计算总处理时间和平均处理时间
        process_time = time.time() - t0
        print(f"所有chunks处理完成, 总耗时: {process_time:.2f}秒, 平均每chunk: {process_time/total_chunks:.2f}秒")
        return file_contents
    
    def process_chunks_batch(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        批量处理文本块，减少LLM调用次数
        
        参数：
            file_contents: 文件内容列表
            progress_callback: 进度回调函数
            
        返回：
            List[Tuple]: 处理结果
            
        实现特点：
        1. 智能动态批处理大小，根据文本长度自动调整
        2. 批量提交多个文本块到LLM，减少API调用次数
        3. 处理结果不匹配时自动降级到单个处理
        4. 对批处理错误实现优雅降级
        
        性能优化：
        - 根据平均文本长度动态调整批次大小
        - 优先检查缓存，避免不必要的批处理
        - 使用特殊分隔符确保批处理结果可以正确分割
        - 实现多级错误处理和降级策略
        """
        # 逐个处理文件
        for file_content in file_contents:
            # 获取文件的文本块
            chunks = file_content[2]
            # 存储处理结果
            results = []
            
            # 智能动态批处理大小计算
            # 计算每个文本块的长度
            chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
            # 计算平均文本块大小
            avg_chunk_size = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            
            # 动态调整批处理大小：文本块越大，批次越小
            # 避免超出LLM上下文窗口，同时保持处理效率
            dynamic_batch_size = max(1, min(self.batch_size, int(10000 / (avg_chunk_size + 1))))
            
            # 按批次处理文本块
            for i in range(0, len(chunks), dynamic_batch_size):
                # 获取当前批次的文本块
                batch_chunks = chunks[i:i+dynamic_batch_size]
                
                # 缓存检查
                batch_keys = [self._generate_cache_key(''.join(chunk)) for chunk in batch_chunks]
                cached_batch_results = [self._load_from_cache(key) for key in batch_keys]
                
                # 优化：如果批次中所有结果都已缓存，则直接使用缓存结果
                if None not in cached_batch_results:
                    # 将缓存结果添加到总结果中
                    results.extend(cached_batch_results)
                    # 更新进度
                    if progress_callback:
                        for j in range(len(batch_chunks)):
                            progress_callback(i + j)
                    # 跳过当前批次的处理
                    continue
                
                # 准备批处理输入
                batch_inputs = []
                for chunk in batch_chunks:
                    batch_inputs.append(''.join(chunk))
                
                # 使用特殊分隔符合并多个文本块，便于后续分割结果
                # 使用分隔线确保不同文本块之间有明显边界
                batch_text = f"\n{'-'*50}\n".join(batch_inputs)
                
                try:
                    # 使用原始提示模板处理批量输入
                    batch_response = self.chain.invoke({
                        "chat_history": self.chat_history,
                        "entity_types": self.entity_types,
                        "relationship_types": self.relationship_types,
                        "tuple_delimiter": self.tuple_delimiter,
                        "record_delimiter": self.record_delimiter,
                        "completion_delimiter": self.completion_delimiter,
                        "input_text": batch_text
                    })
                    
                    # 解析批量响应，分割成单独的结果
                    batch_results = self._parse_batch_response(batch_response.content)
                    
                    # 降级处理：检查结果数量是否匹配
                    if len(batch_results) != len(batch_chunks):
                        # 如果无法正确解析批处理响应，则单独处理每个chunk
                        batch_results = []
                        for idx, chunk in enumerate(batch_chunks):
                            # 检查缓存
                            cached_result = cached_batch_results[idx]
                            if cached_result is not None:
                                batch_results.append(cached_result)
                            else:
                                # 缓存未命中则单独处理
                                individual_result = self._process_single_chunk(''.join(chunk))
                                batch_results.append(individual_result)
                    else:
                        # 缓存批处理结果，只缓存之前未命中的结果
                        for idx, result in enumerate(batch_results):
                            if cached_batch_results[idx] is None:  # 只缓存未命中的结果
                                self._save_to_cache(batch_keys[idx], result)
                    
                    # 将当前批次的结果添加到总结果中
                    results.extend(batch_results)
                except Exception as e:
                    print(f"批处理错误，切换到单个处理: {e}")
                    # 错误处理：降级到单个处理
                    for idx, chunk in enumerate(batch_chunks):
                        try:
                            individual_result = self._process_single_chunk(''.join(chunk))
                            results.append(individual_result)
                        except Exception as e2:
                            print(f"单个chunk处理失败: {e2}")
                            results.append("")  # 安全措施：确保结果列表完整
                
                # 更新进度
                if progress_callback:
                    for j in range(len(batch_chunks)):
                        progress_callback(i + j)
            
            # 将处理结果附加到文件内容中
            file_content.append(results)
        
        return file_contents

    def _parse_batch_response(self, batch_content: str) -> List[str]:
        """
        解析批量响应，将其分割为单独的结果
        
        参数：
            batch_content: 批处理响应内容
            
        返回：
            List[str]: 分割后的结果列表，每个元素对应一个输入文本块
        
        实现说明：
            - 使用与合并时相同的分隔符进行分割
            - 对每个分割后的部分进行空白字符清理
            - 确保返回的结果列表与输入批次一一对应
        """
        # 使用预定义的分隔符分割响应，与合并时使用的分隔符保持一致
        parts = batch_content.split(f"\n{'-'*50}\n")
        # 清理每个部分的空白字符，确保结果干净
        return [part.strip() for part in parts]
    
    @retry(times=3, exceptions=(Exception,), delay=1.0)
    def _process_single_chunk(self, input_text: str) -> str:
        """
        处理单个文本块（带缓存机制）
        
        参数：
            input_text: 输入文本块
            
        返回：
            str: 提取的实体和关系结果

        装饰器说明：
            - 使用@retry装饰器实现自动重试机制
            - 最多重试3次，每次重试间隔1秒
            - 捕获所有异常类型
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(input_text)
        
        # 尝试从缓存加载，实现缓存优先策略
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # 未缓存时，调用LLM处理文本
        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
            "tuple_delimiter": self.tuple_delimiter,
            "record_delimiter": self.record_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "input_text": input_text
        })
        
        result = response.content
        
        # 保存结果到缓存，避免重复计算
        self._save_to_cache(cache_key, result)
        
        return result
    
    def stream_process_large_files(self, file_path: str, chunk_size: int = 5000, 
                                   structure_builder=None, graph_writer=None) -> None:
        """
        以流式方式处理大文件，避免一次性加载全部内容
        
        参数：
            file_path: 文件路径
            chunk_size: 块大小（字符数）
            structure_builder: 图结构构建器
            graph_writer: 图数据库写入器
        
        应用场景：
        - 处理超大文件（GB级别）
        - 内存受限环境
        - 需要实时进度反馈的场景
        - 长文档的增量处理
        """
        # 检查必要的组件
        if not structure_builder or not graph_writer:
            print("需要提供structure_builder和graph_writer才能进行流式处理")
            return
            
        # 定义文本块迭代器，流式读取文件内容，避免一次性加载全部文件到内存
        def text_chunks_iterator(file_path, chunk_size):
            # 打开文件并使用UTF-8编码
            with open(file_path, 'r', encoding='utf-8') as f:
                # 初始化块存储和字符计数
                chunk = []
                chars_count = 0
                # 逐行读取文件
                for line in f:
                    chunk.append(line)
                    chars_count += len(line)
                    # 当达到指定大小或遇到特殊标记时返回一个块
                    if chars_count >= chunk_size:
                        yield chunk
                        # 重置块和计数
                        chunk = []
                        chars_count = 0
                # 处理最后一个可能不完整的块，确保所有内容都被处理
                if chunk:  # 不要忘记最后一个可能不满的chunk
                    yield chunk
        
        # 处理文件的元数据
        file_name = os.path.basename(file_path)  # 获取文件名
        file_type = os.path.splitext(file_name)[1]  # 获取文件扩展名
        
        # 创建文档节点，将文件信息添加到图中
        structure_builder.create_document(
            type=file_type,
            uri=file_path,
            file_name=file_name,
            domain="document"
        )
        
        # 流式读取文件内容到列表中
        chunks = []
        for chunk in text_chunks_iterator(file_path, chunk_size):
            chunks.append(chunk)
        
        # 创建chunk之间的关系，构建文档结构
        chunks_with_hash = structure_builder.create_relation_between_chunks(
            file_name, chunks
        )
        
        # 并行处理所有chunks，结合实时写入
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务映射，用于跟踪异步处理结果
            future_to_chunk = {}
            
            # 遍历所有文本块数据
            for chunk_data in chunks_with_hash:
                # 获取文本块内容
                chunk_text = chunk_data['chunk_doc'].page_content
                # 生成缓存键
                cache_key = self._generate_cache_key(chunk_text)
                # 尝试从缓存加载
                cached_result = self._load_from_cache(cache_key)
                
                # 缓存优先处理
                if cached_result:
                    # 如果缓存命中，直接处理结果
                    try:
                        # 转换为图文档格式
                        graph_document = graph_writer.convert_to_graph_document(
                            chunk_data['chunk_id'],
                            chunk_data['chunk_doc'].page_content,
                            cached_result
                        )
                        
                        # 只有当有节点或关系时才写入图数据库，避免空操作优化
                        if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                            graph_writer.graph.add_graph_documents(
                                [graph_document],
                                baseEntityLabel=True,
                                include_source=True
                            )
                    except Exception as e:
                        # 错误处理：记录日志但不中断处理
                        print(f"处理缓存结果时出错: {e}")
                else:
                    # 如果缓存未命中，提交异步处理任务
                    future = executor.submit(self._process_single_chunk, chunk_text)
                    future_to_chunk[future] = chunk_data
            
            # 处理完成的异步任务并实时写入图数据库
            for future in concurrent.futures.as_completed(future_to_chunk):
                # 获取对应的文本块数据
                chunk_data = future_to_chunk[future]
                try:
                    # 获取处理结果
                    result = future.result()
                    
                    # 实时转换并写入一个chunk的结果到图数据库
                    graph_document = graph_writer.convert_to_graph_document(
                        chunk_data['chunk_id'],
                        chunk_data['chunk_doc'].page_content,
                        result
                    )
                    
                    # 优化：只有当有节点或关系时才写入，避免空操作
                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        graph_writer.graph.add_graph_documents(
                            [graph_document],
                            baseEntityLabel=True,
                            include_source=True
                        )
                        
                except Exception as exc:
                    # 错误处理：记录日志但继续处理其他任务
                    print(f"处理chunk {chunk_data['chunk_id']} 时发生错误: {exc}")