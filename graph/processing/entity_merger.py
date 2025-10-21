import re
import ast
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from model.get_models import get_llm_model
from config.prompt import system_template_build_index, user_template_build_index
from config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS
from graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

class EntityMerger:
    """
    实体合并管理器，负责基于LLM决策合并相似实体。
    
    主要功能包括：
    1. 使用LLM分析实体相似性并提供合并建议
    2. 解析和规范化LLM返回的合并建议
    3. 在图数据库中执行实体合并操作
    4. 清理合并后产生的重复关系
    5. 提供详细的性能统计信息
    
    设计特点：
    - 采用并行处理提高LLM分析效率
    - 批处理优化数据库操作性能
    - 多级降级策略确保操作稳定性
    - 完整的错误处理和重试机制
    """
    
    def __init__(self, batch_size: int = 20, max_workers: int = 4):
        """
        初始化实体合并管理器
        
        Args:
            batch_size: 批处理大小，影响数据库操作的内存使用和效率
            max_workers: 并行工作线程数，控制LLM处理的并发度
        """
        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()
        
        # 获取语言模型
        self.llm = get_llm_model()
        
        # 批处理和并行参数
        self.batch_size = batch_size or ENTITY_BATCH_SIZE  # 使用配置的默认值作为备选
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS  # 动态工作线程数
        
        # 设置LLM处理链
        self._setup_llm_chain()
        
        # 创建索引
        self._create_indexes()
        
        # 性能监控计数器
        self.llm_time = 0       # LLM处理总时间
        self.db_time = 0        # 数据库操作总时间
        self.parse_time = 0     # 结果解析总时间
    
    def _create_indexes(self) -> None:
        """
        创建必要的索引以优化查询性能
        
        在实体ID上创建索引，显著提升实体查询速度，尤其是在大规模图数据库中。
        这个索引对于后续的实体合并操作至关重要，可以避免全图扫描。
        """
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]
        
        connection_manager.create_multiple_indexes(index_queries)

    def _setup_llm_chain(self) -> None:
        """
        设置LLM处理链，用于实体合并决策
        
        设计思路：
        1. 检查模型能力确保兼容性
        2. 使用配置中的模板创建系统提示和人类提示
        3. 构建完整的对话链，包含消息占位符以支持上下文
        4. 创建管道将提示和LLM连接起来
        """
        # 检查模型能力
        if not hasattr(self.llm, 'with_structured_output'):
            print("当前LLM模型不支持结构化输出")

        # 创建提示模板 - 使用配置文件中的模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template_build_index
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            user_template_build_index
        )
        
        # 构建对话链，包含消息历史占位符
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),  # 支持多轮对话上下文
            human_message_prompt
        ])
        
        # 创建最终的处理链 - 使用LangChain的管道语法
        self.chain = self.chat_prompt | self.llm

    def _convert_to_list(self, result: str) -> List[List[str]]:
        """
        将LLM返回的实体列表文本转换为Python列表
        
        实现细节：
        1. 采用两级解析策略：
           - 首先尝试直接解析整个结果（更高效）
           - 失败时回退到正则表达式匹配方法
        2. 支持嵌套列表和单层列表的灵活解析
        3. 过滤无效结果，确保返回数据的质量
        4. 记录解析时间用于性能统计
        
        Args:
            result: LLM返回的文本结果，包含实体列表
            
        Returns:
            List[List[str]]: 二维列表，每个子列表包含一组可合并的实体
        """
        start_time = time.time()
        
        # 使用正则表达式匹配所有方括号包含的内容（备选方案）
        list_pattern = re.compile(r'\[.*?\]')
        entity_lists = []
        
        # 策略1：先尝试直接用ast.literal_eval解析整个结果（更高效）
        try:
            # 查找可能的列表开始位置
            list_start = result.find('[')
            if list_start >= 0:
                # 尝试找出完整列表部分，考虑嵌套层级
                nested_level = 0
                for i in range(list_start, len(result)):
                    if result[i] == '[':
                        nested_level += 1
                    elif result[i] == ']':
                        nested_level -= 1
                        if nested_level == 0:
                            # 提取出可能是列表的部分
                            list_portion = result[list_start:i+1]
                            try:
                                parsed_list = ast.literal_eval(list_portion)
                                if isinstance(parsed_list, list):
                                    # 检查是否是二维列表
                                    if all(isinstance(item, list) for item in parsed_list):
                                        entity_lists = parsed_list
                                    else:
                                        entity_lists = [parsed_list]
                                    break
                            except:
                                pass  # 如果解析失败，继续使用正则方法
        except:
            pass  # 如果上述方法失败，回退到正则表达式方法
        
        # 策略2：如果直接解析失败，使用正则表达式方法
        if not entity_lists:
            # 解析每个匹配的列表字符串
            for match in list_pattern.findall(result):
                try:
                    # 将字符串转换为Python列表
                    entity_list = ast.literal_eval(match)
                    # 只添加非空列表
                    if entity_list and isinstance(entity_list, list):
                        if all(isinstance(item, list) for item in entity_list):
                            # 如果是嵌套列表，扩展它
                            entity_lists.extend(entity_list)
                        else:
                            # 如果是单个列表，添加它
                            entity_lists.append(entity_list)
                except Exception as e:
                    print(f"解析实体列表时出错: {str(e)}, 原文本: {match}")
        
        # 过滤和规范化结果 - 确保数据质量
        valid_lists = []
        for entity_list in entity_lists:
            # 确保列表中的所有项目都是字符串
            if all(isinstance(item, str) for item in entity_list):
                # 去除重复项
                unique_list = list(dict.fromkeys(entity_list))
                if len(unique_list) > 1:  # 只保留至少有2个实体的组（可合并）
                    valid_lists.append(unique_list)
        
        # 累计解析时间
        self.parse_time += time.time() - start_time
        return valid_lists

    def get_merge_suggestions(self, duplicate_candidates: List[Any]) -> List[List[str]]:
        """
        使用LLM分析并提供实体合并建议 - 并行优化版本
        
        算法设计：
        1. 动态调整批处理大小，根据候选数量和工作线程数优化
        2. 对每个批次内部的候选组进行并行处理，提高CPU利用率
        3. 使用线程池管理并发任务，避免创建过多线程
        4. 精确记录LLM处理时间和解析时间
        5. 合并重叠的实体组，避免重复合并
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表
            
        Returns:
            List[List[str]]: 建议合并的实体分组列表
        """
        # 检查是否有候选实体
        if not duplicate_candidates:
            return []
        
        llm_start_time = time.time()
            
        # 收集LLM的合并建议
        merged_entities = []
        
        # 动态调整批处理大小 - 基于工作线程数和候选数量
        candidate_count = len(duplicate_candidates)
        optimal_batch_size = min(self.max_workers * 2, max(1, candidate_count // 5))
        
        print(f"处理 {candidate_count} 个候选实体组，批次大小: {optimal_batch_size}")
        
        # 分批处理，避免创建过多线程
        for batch_start in range(0, candidate_count, optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, candidate_count)
            batch = duplicate_candidates[batch_start:batch_end]
            
            # 使用线程池并行处理LLM请求 - 关键性能优化点
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务到线程池
                future_to_candidates = {
                    executor.submit(self._process_candidate_group, candidates): i
                    for i, candidates in enumerate(batch)
                }
                
                # 收集结果，处理完成一个就获取一个
                for future in concurrent.futures.as_completed(future_to_candidates):
                    try:
                        result = future.result()
                        if result:
                            merged_entities.append(result)
                    except Exception as e:
                        print(f"处理候选实体组时出错: {e}")
            
            # 报告进度
            print(f"已处理 {batch_end}/{candidate_count} 个候选实体组")
        
        # 累计LLM处理时间
        self.llm_time += time.time() - llm_start_time
        
        parse_start_time = time.time()
        # 解析并整理最终的合并建议
        results = []
        for candidates in merged_entities:
            # 将每个建议转换为列表格式
            temp = self._convert_to_list(candidates)
            results.extend(temp)
        
        # 累计解析时间
        self.parse_time += time.time() - parse_start_time
        
        # 合并具有相同实体的组，避免重复合并
        merged_results = self._merge_overlapping_groups(results)
        
        print(f"LLM分析完成，找到 {len(merged_results)} 组可合并实体")
        return merged_results
    
    def _merge_overlapping_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """
        合并有重叠的实体组
        
        算法原理：
        使用并查集（Disjoint Set Union，DSU）数据结构高效合并连通组件
        1. 首先构建实体到组索引的映射
        2. 然后使用并查集合并包含相同实体的组
        3. 最后收集合并后的结果
        
        时间复杂度：O(n α(n))，其中n是实体总数，α是阿克曼函数的反函数（近似常数）
        空间复杂度：O(n)
        
        Args:
            groups: 实体组列表，每个子列表代表一组应合并的实体
            
        Returns:
            List[List[str]]: 合并后的实体组列表，确保没有重叠的实体组
        """
        if not groups:
            return []
            
        # 创建实体到组的映射 - 记录每个实体出现在哪些组中
        entity_to_groups = {}
        for i, group in enumerate(groups):
            for entity in group:
                if entity not in entity_to_groups:
                    entity_to_groups[entity] = []
                entity_to_groups[entity].append(i)
        
        # 初始化并查集 - 每个组最初是独立的集合
        parent = list(range(len(groups)))
        
        # 查找函数 - 带路径压缩优化
        def find(x):
            if parent[x] != x:
                # 路径压缩：将x到根节点路径上的所有节点直接连接到根
                parent[x] = find(parent[x])
            return parent[x]
        
        # 合并函数
        def union(x, y):
            # 将x所在集合的根连接到y所在集合的根
            parent[find(x)] = find(y)
        
        # 合并存在共同实体的组 - 对每个实体，如果它出现在多个组中，合并这些组
        for entity, group_indices in entity_to_groups.items():
            if len(group_indices) > 1:  # 实体出现在多个组中
                # 以第一个组为基准，合并后续所有组
                for i in range(1, len(group_indices)):
                    union(group_indices[0], group_indices[i])
        
        # 收集合并后的组 - 使用根节点作为标识
        merged_groups = {}
        for i, group in enumerate(groups):
            root = find(i)  # 找到组i的根节点
            if root not in merged_groups:
                merged_groups[root] = set()  # 使用集合去重
            merged_groups[root].update(group)  # 合并实体
        
        # 转换回列表格式
        return [list(entities) for entities in merged_groups.values()]
    
    def _process_candidate_group(self, candidates: List[str]) -> Optional[str]:
        """
        处理单个候选实体组
        
        功能说明：
        对一组候选实体调用LLM进行分析，判断它们是否应该合并。
        包含重试机制以提高鲁棒性，应对LLM服务可能的临时故障。
        
        设计特点：
        - 健壮性：实现自动重试机制
        - 错误处理：详细的异常日志记录
        - 性能考量：重试前添加短暂延迟避免连续失败
        
        Args:
            candidates: 候选实体列表，需要分析是否应该合并
            
        Returns:
            str: LLM的分析结果，包含应该合并的实体列表
            None: 处理失败或无需合并
        """
        # 快速检查：确保至少有两个实体需要比较
        if not candidates or len(candidates) < 2:
            return None
            
        chat_history = []  # 初始化对话历史
        max_retries = 2   # 设置最大重试次数
        
        # 重试循环 - 提高LLM调用的健壮性
        for retry in range(max_retries + 1):
            try:
                # 调用LLM进行分析
                answer = self.chain.invoke({
                    "chat_history": chat_history,
                    "entities": candidates
                })
                return answer.content  # 返回LLM生成的合并建议
            except Exception as e:
                # 错误处理和重试逻辑
                if retry < max_retries:
                    print(f"LLM调用异常，尝试重试 ({retry+1}/{max_retries}): {e}")
                    time.sleep(1)  # 短暂延迟，避免频繁重试
                else:
                    print(f"LLM调用失败，最大重试次数已用尽: {e}")
                    return None

    def execute_merges(self, merge_groups: List[List[str]]) -> int:
        """
        执行实体合并操作 - 批处理优化版本
        
        核心功能：
        在Neo4j图数据库中执行实体合并操作，将多个相似实体合并为一个。
        采用多级降级策略确保操作成功率：
        1. 首选批量合并（最高效率）
        2. 批量失败时降级到单组合并
        
        性能优化：
        - 动态批处理大小根据数据量自动调整
        - 实时计算并显示进度和预计剩余时间
        - 精确记录数据库操作时间用于性能分析
        
        数据库操作：
        使用Neo4j的APOC库提供的mergeNodes函数执行合并
        属性策略设置为`'discard'`，确保合并后的节点保留所有原始关系
        
        Args:
            merge_groups: 要合并的实体分组列表，每个子列表包含一组要合并的实体ID
            
        Returns:
            int: 合并操作影响的节点总数
        """
        if not merge_groups:
            return 0
        
        # 开始计时数据库操作时间
        db_start_time = time.time()
            
        # 动态批处理大小：根据数据量自适应调整，避免批处理过大或过小
        group_count = len(merge_groups)
        optimal_batch_size = min(self.batch_size, max(5, group_count // 10))
        total_batches = (group_count + optimal_batch_size - 1) // optimal_batch_size
        
        print(f"开始执行 {group_count} 组实体合并，批次大小: {optimal_batch_size}")
        
        total_merged = 0      # 总合并实体计数
        batch_times = []      # 记录每个批次的处理时间
        
        # 批量处理合并操作
        for batch_index in range(total_batches):
            batch_start = time.time()
            
            # 计算当前批次的起止索引
            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, group_count)
            batch = merge_groups[start_idx:end_idx]
            
            try:
                # 执行Neo4j批量合并操作 - 使用UNWIND优化批量处理
                result = self.graph.query("""
                UNWIND $data AS candidates
                CALL {
                  WITH candidates
                  MATCH (e:__Entity__) WHERE e.id IN candidates
                  RETURN collect(e) AS nodes
                }
                CALL apoc.refactor.mergeNodes(nodes, {properties: {
                    `.*`: 'discard'
                }})
                YIELD node
                RETURN count(*) as merged_count
                """, params={"data": batch})
                
                if result:
                    batch_merged = result[0]["merged_count"]
                    total_merged += batch_merged
                    
                    # 记录本批次处理时间
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                    
                    # 计算平均时间和预计剩余时间 - 用户体验优化
                    avg_time = sum(batch_times) / len(batch_times)
                    remaining_batches = total_batches - (batch_index + 1)
                    estimated_remaining = avg_time * remaining_batches
                    
                    # 显示进度和性能信息
                    print(f"已处理合并批次 {batch_index+1}/{total_batches}, "
                          f"批次合并: {batch_merged} 实体, "
                          f"批次耗时: {batch_time:.2f}秒, "
                          f"预计剩余: {estimated_remaining:.2f}秒")
            except Exception as e:
                # 降级策略：批处理失败时尝试单个处理 - 提高容错性
                print(f"批量合并出错，尝试单个处理: {e}")
                batch_merged = 0
                
                # 逐个处理批次中的每个实体组
                for group in batch:
                    try:
                        # 单组合并操作
                        single_result = self.graph.query("""
                        MATCH (e:__Entity__) WHERE e.id IN $candidates
                        WITH collect(e) AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: {
                            `.*`: 'discard'
                        }})
                        YIELD node
                        RETURN count(*) as merged_count
                        """, params={"candidates": group})
                        
                        if single_result:
                            group_merged = single_result[0]["merged_count"]
                            total_merged += group_merged
                            batch_merged += group_merged
                    except Exception as e2:
                        print(f"单个组合并失败: {e2}")
                
                print(f"单个处理完成，本批次合并: {batch_merged} 实体")
        
        # 累计数据库操作时间
        self.db_time += time.time() - db_start_time
        
        return total_merged

    def clean_duplicate_relationships(self):
        """
        清除重复关系，包括：
        1. 相同方向的重复关系
        2. SIMILAR关系的双向冗余（保留一个方向）
        
        功能说明：
        实体合并后，图数据库中可能产生重复关系，需要进行清理以保持数据一致性。
        该方法分两步执行清理操作：先清理同向重复关系，再处理SIMILAR关系的双向冗余。
        
        数据库优化：
        - 只保留每个方向的一个关系实例
        - 对于SIMILAR关系，使用节点ID顺序确保每对节点只处理一次
        
        返回值:
            int: 总共删除的重复关系数量
        """
        print("开始清除重复关系...")
        
        # 第一步：清除相同方向的重复关系
        # 使用Cypher查询查找并删除同向重复关系，保留一个实例
        result1 = self.graph.query("""
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) as type, collect(r) as rels
        WHERE size(rels) > 1
        WITH a, b, type, rels[0] as kept, rels[1..] as rels
        UNWIND rels as rel
        DELETE rel
        RETURN count(*) as deleted
        """)
        
        deleted_count1 = result1[0]["deleted"] if result1 else 0
        print(f"已删除 {deleted_count1} 个相同方向的重复关系")
        
        # 第二步：清除SIMILAR关系的双向冗余（保留一个方向）
        # 处理相似关系的双向冗余，避免信息重复存储
        result2 = self.graph.query("""
        // 找出所有双向的SIMILAR关系
        MATCH (a)-[r1:SIMILAR]->(b)
        MATCH (b)-[r2:SIMILAR]->(a)
        WHERE a.id < b.id  // 确保每对节点只处理一次
        
        // 随机选择一个方向删除（这里选择删除b->a方向）
        DELETE r2
        
        RETURN count(*) as deleted_bidirectional
        """)
        
        deleted_count2 = result2[0]["deleted_bidirectional"] if result2 else 0
        print(f"已删除 {deleted_count2} 个双向SIMILAR关系的冗余方向")
        
        total_deleted = deleted_count1 + deleted_count2
        print(f"总共删除了 {total_deleted} 个重复关系")
        
        return total_deleted

    @timer
    def process_duplicates(self, duplicate_candidates: List[Any]) -> Tuple[int, Dict[str, Any]]:
        """
        处理重复实体的完整流程，包括获取合并建议和执行合并 - 性能优化版本
        
        工作流程：
        1. 数据预处理：标准化不同格式的候选数据
        2. 过滤无效候选组
        3. LLM分析：获取实体合并建议
        4. 执行合并：在图数据库中合并实体
        5. 关系清理：移除合并后产生的重复关系
        6. 性能统计：计算并返回详细的性能指标
        
        性能优化特点：
        - 输入数据灵活适配，支持不同格式的候选列表
        - 全流程计时，精确分析各阶段性能
        - 详细的进度和结果报告
        
        Args:
            duplicate_candidates: 潜在的重复实体候选列表，可以是多种格式
            
        Returns:
            Tuple[int, Dict[str, Any]]: 
                - 合并的实体数量
                - 包含详细性能统计的字典
        """
        start_time = time.time()
        
        # 数据预处理：确保duplicate_candidates是列表的列表，处理不同的数据结构
        fixed_candidates = []
        for candidates in duplicate_candidates:
            # 检查候选组是否是字典格式（包含combinedResult字段）
            if isinstance(candidates, dict) and "combinedResult" in candidates:
                candidate_list = candidates["combinedResult"]
                if isinstance(candidate_list, list) and len(candidate_list) > 1:
                    fixed_candidates.append(candidate_list)
            # 检查候选组是否已经是列表格式
            elif isinstance(candidates, list) and len(candidates) > 1:
                fixed_candidates.append(candidates)
        
        # 过滤处理数量过少的候选组（确保每组至少有2个实体）
        filtered_candidates = [
            candidates for candidates in fixed_candidates
            if len(candidates) > 1
        ]
        
        print(f"处理后候选实体组数: {len(filtered_candidates)}")
        print(f"开始处理 {len(filtered_candidates)} 组有效重复实体候选...")
        
        # 第一阶段：获取合并建议 - 使用LLM分析哪些实体应该合并
        merge_groups = self.get_merge_suggestions(filtered_candidates)
        
        suggestion_time = time.time()
        suggestion_elapsed = suggestion_time - start_time
        print(f"生成合并建议完成，用时 {suggestion_elapsed:.2f} 秒, "
            f"找到 {len(merge_groups)} 组可合并实体")
        print(f"其中: LLM处理时间: {self.llm_time:.2f}秒, 解析时间: {self.parse_time:.2f}秒")
        
        # 第二阶段：执行合并 - 在图数据库中实际执行实体合并操作
        merged_count = 0
        if merge_groups:
            merged_count = self.execute_merges(merge_groups)

        # 第三阶段：关系清理 - 合并实体后，清理可能产生的重复关系
        self.clean_duplicate_relationships()
                
        # 计算性能指标
        end_time = time.time()
        merge_elapsed = end_time - suggestion_time
        total_elapsed = end_time - start_time
        
        # 打印执行结果摘要
        print(f"实体合并完成，用时 {merge_elapsed:.2f} 秒, 合并了 {merged_count} 个实体")
        print(f"数据库操作时间: {self.db_time:.2f}秒")
        print(f"总耗时: {total_elapsed:.2f} 秒")
        
        # 构建性能统计摘要
        time_records = {
            "LLM处理时间": self.llm_time,
            "解析时间": self.parse_time,
            "数据库时间": self.db_time
        }
        
        # 获取详细性能统计
        performance_stats = get_performance_stats(total_elapsed, time_records)
        performance_stats.update({
            "候选实体组数": len(filtered_candidates),
            "识别出的合并组数": len(merge_groups),
            "合并的实体数": merged_count
        })
        
        # 打印性能统计
        print_performance_stats(performance_stats)
        
        return merged_count, performance_stats