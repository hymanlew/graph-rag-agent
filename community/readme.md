# 社区检测与摘要模块

## 文件结构

```
community/
├── __init__.py                    # 模块入口，导出工厂类和公共API
├── readme.md                      # 模块说明文档
├── detector/                      # 社区检测器目录
│   ├── __init__.py                # 检测器工厂类和注册机制
│   ├── base.py                    # 基础检测器抽象类，定义核心算法框架
│   ├── leiden.py                  # Leiden算法实现，基于层次聚类
│   ├── projections.py             # 图投影混入类，提供共享图数据处理功能
│   └── sllpa.py                   # SLLPA算法实现，基于标签传播
└── summary/                       # 社区摘要目录
    ├── __init__.py                # 摘要工厂类和注册机制
    ├── base.py                    # 基础摘要生成器抽象类，实现模板方法模式
    ├── leiden.py                  # Leiden社区摘要实现，适配Leiden算法结果
    └── sllpa.py                   # SLLPA社区摘要实现，适配SLLPA算法结果
```

## 模块概述

本模块为基于Neo4j图数据库的社区检测与摘要功能提供支持，是知识图谱项目的重要组成部分。该模块实现了从图数据中发现隐含社区结构，并为每个社区生成语义摘要的完整流程。

主要功能包括：

1. 识别图数据中的社区结构（社区检测）
   - 支持多种社区检测算法（Leiden和SLLPA）
   - 自适应参数优化，根据图规模和系统资源自动调整
   - 多级图投影策略，保证在不同规模图数据上的稳定性

2. 为每个社区生成摘要描述（社区摘要）
   - 社区重要性排序，优先处理核心社区
   - 基于LLM的社区语义摘要生成
   - 并行处理框架，提升摘要生成效率
   - 结果持久化到图数据库

## 设计思路与实现

### 设计模式

本模块采用多种设计模式确保代码的可维护性和可扩展性：

1. **工厂模式**：通过`CommunityDetectorFactory`和`CommunitySummarizerFactory`创建不同类型的检测器和摘要生成器，隐藏实现细节并支持动态注册新算法
2. **混入类（Mixin）**：使用`GraphProjectionMixin`提供共享的图投影功能，避免代码重复
3. **上下文管理器**：在`BaseCommunityDetector`中使用`_graph_projection_context`管理资源生命周期，确保资源正确释放
4. **模板方法模式**：在基类中定义算法骨架，由子类实现具体步骤，简化新算法的添加
5. **策略模式**：为不同的算法提供统一接口，允许在运行时切换实现

### 核心组件与流程

#### 1. 社区检测

**核心类**: `BaseCommunityDetector`  
**实现算法**: 
- Leiden算法 (`LeidenDetector`): 基于改进的Louvain算法，提供更高质量的社区划分和更好的模块化
- SLLPA算法 (`SLLPADetector`): 基于标签传播，适合大规模图的高效社区检测

**关键流程**:
1. **图投影**：通过`create_projection()`将Neo4j图数据投影到GDS库，支持三种投影模式：
   - 标准模式：保留所有节点和关系
   - 过滤模式：通过限制节点数量优化性能
   - 保守模式：使用最小数据集确保稳定性

2. **社区检测**：执行`detect_communities()`用特定算法识别社区结构：
   - Leiden：采用多层次优化策略，确保社区质量
   - SLLPA：使用标签传播，适合大规模图数据

3. **结果保存**：通过`save_communities()`将社区信息持久化到图数据库，创建社区节点和实体间关系

4. **资源清理**：使用`cleanup()`释放投影占用的资源，防止内存泄漏

**自适应优化**:
- 根据系统资源（内存、CPU）自动调整算法参数
- 多级错误处理和备用方案，确保在各种环境下稳定运行
- 完整的性能监控与统计，记录各阶段执行时间
- 针对大规模图的特殊优化，如批处理和增量处理

#### 2. 社区摘要

**核心类**: `BaseSummarizer`  
**辅助类**: 
- `BaseCommunityDescriber`: 负责生成社区的自然语言描述
- `BaseCommunityRanker`: 计算社区的重要性权重
- `BaseCommunityStorer`: 将社区摘要存储到图数据库

**关键流程**:
1. **社区排名**：通过`calculate_ranks()`计算社区重要性排名，使用以下因素：
   - 社区大小（节点数量）
   - 社区内部连接密度
   - 社区内部实体的重要性

2. **信息收集**：通过`collect_community_info()`获取社区内节点和关系信息，根据算法类型和规模采用不同策略：
   - 小规模：一次性查询所有社区信息
   - 大规模：分批查询，每批处理固定数量的社区
   - 针对不同算法（Leiden/SLLPA）的特殊优化

3. **摘要生成**：
   - 调用`_process_communities_parallel()`并行处理多个社区
   - 对每个社区使用LLM模型生成语义摘要
   - 支持不同的摘要生成策略和提示模板

4. **结果存储**：
   - 对少量摘要：单个存储
   - 对大量摘要：批量存储以提高性能
   - 更新社区节点的摘要属性

**性能优化**:
- 并行处理：利用`ThreadPoolExecutor`多线程生成摘要，大幅提升处理速度
- 分批处理：对大规模社区数据分批获取和处理，避免查询超时
- 自适应并发：根据系统资源自动调整线程池大小
- 完整的性能统计和监控，记录各阶段执行时间和处理数量

## 核心函数

### 社区检测模块

- **`BaseCommunityDetector.process()`**: 执行完整的社区检测流程，包括投影、检测和保存
  ```python
  def process(self) -> Dict[str, Any]:
      """执行完整的社区检测流程"""
      # 实现包括图投影、社区检测、结果保存和性能统计
  ```

- **`GraphProjectionMixin.create_projection()`**: 创建图投影，含多种降级策略
  ```python
  def create_projection(self) -> Tuple[Any, Dict]:
      """创建图投影，支持标准、过滤和保守多种模式"""
  ```

- **`LeidenDetector.detect_communities()`**: 执行Leiden算法社区检测
  ```python
  def detect_communities(self) -> Dict[str, Any]:
      """执行Leiden算法社区检测，含参数优化和失败降级"""
  ```

### 社区摘要模块

- **`BaseSummarizer.process_communities()`**: 处理所有社区的摘要生成流程
  ```python
  def process_communities(self) -> List[Dict]:
      """处理所有社区，包括权重计算、信息收集、摘要生成和存储"""
  ```

- **`BaseSummarizer._process_communities_parallel()`**: 并行处理社区摘要
  ```python
  def _process_communities_parallel(self, community_info: List[Dict], workers: int) -> List[Dict]:
      """利用多线程并行生成社区摘要"""
  ```

- **`LeidenSummarizer.collect_community_info()`**: 收集Leiden社区信息
  ```python
  def collect_community_info(self) -> List[Dict]:
      """收集社区信息，支持大规模批量处理"""
  ```

## 使用示例

### 社区检测

```python
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from community import CommunityDetectorFactory

# 初始化图连接
graph = Neo4jGraph(url="neo4j://localhost:7687", username="neo4j", password="password")
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "password"))

# 创建社区检测器（可选算法：'leiden'或'sllpa'）
detector = CommunityDetectorFactory.create('leiden', gds, graph)

# 执行社区检测
results = detector.process()
print(f"社区检测结果: {results}")
```

### 社区摘要生成

```python
from community import CommunitySummarizerFactory

# 创建对应的摘要生成器
summarizer = CommunitySummarizerFactory.create_summarizer('leiden', graph)

# 生成社区摘要
summaries = summarizer.process_communities()
print(f"已生成 {len(summaries)} 个社区摘要")
```

## 性能考量

- 内存使用量与图大小成正比，为大图分析提供多级降级策略
- 社区摘要生成通过多线程并行处理提高效率
- 自适应系统资源，自动调整并发度和算法参数
- 完善的错误处理和监控，提供详细的性能统计

## 扩展性

- 通过继承`BaseCommunityDetector`添加新的社区检测算法
- 通过继承`BaseSummarizer`实现自定义摘要生成逻辑
- 工厂类支持轻松注册和使用新的实现