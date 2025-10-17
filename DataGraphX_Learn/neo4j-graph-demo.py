"""
pip install "neo4j-graphrag[openai]"
neo4j 官方提供的 RAG 包

高层封装，提供更完整的端到端解决方案
控制粒度：较低，流程和交互方式相对固定
只适用于快速构建概念验证或标准GraphRAG应用

生产环境，是使用 neo4j + langchain-neo4j 组合
"""
import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# List the entities and relations the LLM should look for in the text
node_types = ["Person", "House", "Planet"]
relationship_types = ["PARENT_OF", "HEIR_OF", "RULES"]
patterns = [
    ("Person", "PARENT_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
]

# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    schema={
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns,
    },
    on_error="IGNORE",
    from_pdf=False,
)

# Run the pipeline on a piece of text
text = (
    "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
    "Atreides, an aristocratic family that rules the planet Caladan."
)
asyncio.run(kg_builder.run_async(text=text))
driver.close()


from neo4j_graphrag.indexes import create_vector_index
INDEX_NAME = "vector-index-name"

# Create the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Chunk",
    embedding_property="embedding",
    dimensions=3072,
    similarity_fn="euclidean",
)
driver.close()


from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType
vector = embedder.embed_query(text)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["1234"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)
driver.close()


from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "Who is Paul Atreides?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
print(response.answer)
driver.close()


"""
生产环境，是使用 neo4j + langchain-neo4j 组合
"""
import tempfile
import jieba
import jieba.analyse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from data_persistence_utils import (
    save_processed_data
)
from knowledge_graph_utils import load_graph_from_json, find_relevant_subgraph, create_knowledge_graph
from data_persistence_utils import generate_file_hash
from streamlit_plotly_events import plotly_events
from config import (
    GRAPH_CONFIG,
    DOC_CONFIG
)
from api_utils import (
    test_api_connection,
    test_embeddings
)
from langchain.embeddings.base import Embeddings


async def convert_and_add_to_graph(transformer, docs, graph, file_name, file_hash):
    print(f"开始处理 {len(docs)} 个文档")
    all_graph_documents = []
    total_docs = len(docs)
    for i, doc in enumerate(docs):
        graph_doc = await asyncio.to_thread(transformer.convert_to_graph_documents, [doc])
        all_graph_documents.extend(graph_doc)
        print(f"处理进度: {(i + 1)}/{total_docs}")

    # 将图数据转换为可序列化的格式
    serializable_data = {
        "nodes": [],
        "relationships": []
    }
    for doc in all_graph_documents:
        for node in doc.nodes:
            serializable_data["nodes"].append({
                "id": str(node.id),  # 确保id是字符串
                "type": node.type,
                "properties": {k: str(v) for k, v in node.properties.items()}  # 将所有属性值转换为字符串
            })
        for rel in doc.relationships:
            serializable_data["relationships"].append({
                "source": str(rel.source),  # 确保source是字符串
                "target": str(rel.target),  # 确保target是字符串
                "type": rel.type,
                "properties": {k: str(v) for k, v in rel.properties.items()}  # 将所有属性值转换为字符串
            })

    # 保存数据到 JSON 文件
    cache_file_name = f"{file_hash}_graph_data.json"
    save_processed_data(serializable_data, cache_file_name)
    print(f"图数据已保存到 {cache_file_name}")

    print(f"开始添加 {len(all_graph_documents)} 个图形文档到数据库")
    await asyncio.to_thread(graph.add_graph_documents, all_graph_documents, include_source=True)
    print("图形文档添加完成")


async def fulltext_query(graph: Neo4jGraph, question: str, max_nodes: int = 5) -> Dict[str, Any]:
    query = """
    CALL db.index.fulltext.queryNodes("entity_index", $question) YIELD node, score
    WHERE score > 0.5
    WITH node, score
    ORDER BY score DESC
    LIMIT $max_nodes
    MATCH (node)-[r]-(related)
    RETURN 
        labels(node) AS node_labels, 
        properties(node) AS node_properties,
        type(r) AS relationship_type,
        labels(related) AS related_labels
    """
    results = graph.query(query, {"question": question, "max_nodes": max_nodes})

    schema = {}
    for result in results:
        node_label = result["node_labels"][0]
        if node_label not in schema:
            schema[node_label] = {"properties": [], "relationships": []}

        # 添加属性
        for prop in result["node_properties"].keys():
            if prop not in schema[node_label]["properties"]:
                schema[node_label]["properties"].append(prop)

        # 添加关系
        related_label = result["related_labels"][0]
        relationship = {
            "type": result["relationship_type"],
            "target": related_label
        }
        if relationship not in schema[node_label]["relationships"]:
            schema[node_label]["relationships"].append(relationship)

    return schema


async def vector_query(question: str, graph: Neo4jGraph, graph_config: dict, embeddings: Embeddings, llm: ChatOpenAI):
    """改进的问题处理函数"""
    try:
        # 1. 提取关键词
        keywords = jieba.lcut(question)
        key_terms = [word for word in keywords if len(word) >= 2]  # 只保留长度>=2的词

        results = []
        # 2. 基于关键词的精确匹配
        for term in key_terms:
            term_query = """
            MATCH (n)
            WHERE n.text IS NOT NULL 
            AND toLower(n.text) CONTAINS toLower($term)
            RETURN n.text as content, 1.0 as score
            LIMIT 3
            """
            term_results = await asyncio.to_thread(
                graph.query,
                term_query,
                {"term": term}
            )
            results.extend(term_results)

        # 3. 基于整体语义的向量搜索，使用 Neo4j 向量索引进行语义搜索的查询
        # YIELD 用于从过程调用中提取并返回指定的字段，是必需的，配合 return。但如果不需要返回字段，则可以省略 YIELD 和 RETURN
        # 使用 MATCH 时，不需要 YIELD，直接使用 RETURN 来返回数据
        vector_query = """
        CALL db.index.vector.queryNodes(  -- 调用向量索引查询过程
            'vector_index',               -- 使用名为'vector_index'的向量索引
            5,                           -- 返回最相似的5个结果
            $embedding                   -- 传入查询向量（通常是问题文本的嵌入向量）
        ) 
        YIELD node, score                -- 获取返回的节点和相似度分数
        WHERE node.text IS NOT NULL      -- 过滤掉没有text属性的节点
        AND score > 0.3                  -- 只保留相似度高于0.3的结果
        RETURN node.text as content, score  -- 返回文本内容和相似度分数
        ORDER BY score DESC              -- 按相似度降序排列
        """
        question_embedding = await asyncio.to_thread(embeddings.embed_query, question)
        vector_results = await asyncio.to_thread(
            graph.query,
            vector_query,
            {"embedding": question_embedding}
        )
        results.extend(vector_results)

        # 4. 如果问题包含关系词,添加关系搜索
        relation_keywords = ["关系", "联系", "作用", "影响", "如何"]
        if any(word in question for word in relation_keywords):
            # 找出主要实体之间的关系
            terms_str = "|".join(key_terms)
            relation_query = f"""
            MATCH (n)-[r]-(m)
            WHERE n.text IS NOT NULL 
            AND m.text IS NOT NULL
            AND (
                ANY(term IN split($terms, '|') WHERE 
                    toLower(n.text) CONTAINS toLower(term)
                    OR toLower(m.text) CONTAINS toLower(term)
                )
            )
            RETURN n.text as source, type(r) as relation, m.text as target
            LIMIT 5
            """
            try:
                relation_results = await asyncio.to_thread(
                    graph.query,
                    relation_query,
                    {"terms": terms_str}
                )
                results.extend(relation_results)
            except Exception as e:
                logger.error(f"关系搜索出错: {str(e)}")

        # 5. 整理结果
        context = []
        seen_contents = set()

        # 处理文本结果
        for r in results:
            if 'content' in r and r['content']:
                content = r['content'].strip()
                if content and content not in seen_contents:
                    score_text = f"(相关度: {r['score']:.2f})" if 'score' in r else ""
                    context.append(f"- {content} {score_text}")
                    seen_contents.add(content)

        # 处理关系结果
        for r in results:
            if all(k in r for k in ['source', 'relation', 'target']):
                source = r['source'].strip()
                target = r['target'].strip()
                relation_text = f"- {source} --[{r['relation']}]--> {target}"
                if relation_text not in seen_contents:
                    context.append(relation_text)
                    seen_contents.add(relation_text)

        # 6. 生成回答
        if not context:
            return "未找到相关信息。", results, []

        """
        ASCII 字符代码参考
        chr(10)   # '\n' 换行符
        chr(9)   # '\t' 制表符
        chr(13)  # '\r' 回车符  
        chr(32)  # ' '  空格
        chr(34)  # '"'  双引号
        """
        llm_prompt = f"""
        基于以下检索到的信息回答问题：

        问题：{question}
        关键词：{', '.join(key_terms)}

        检索到的信息：
        {chr(10).join(context)}

        请按以下格式回答：
        1. 总结：总结检索到的信息要点
        2. 分析：分析信息的完整性和相关性
        3. 回答：基于检索到的信息回答问题
        """
        response = await llm.ainvoke(llm_prompt)
        return response.content, results, context

    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        return str(e), None, None


async def join_cypher_query(relevant_info: Dict[str, Any], question: str) -> str:
    nodes = relevant_info["nodes"]
    relations = relevant_info["relations"]

    # 基础查询：找到包含问题关键词的节点
    base_query = f"""
    MATCH (n)
    WHERE n.text CONTAINS '{question}'
    """

    # 扩展查询：探索相关节点和关系
    expand_query = """
    MATCH (n)-[r]-(related)
    """

    # 过滤条件
    filter_conditions = []
    if nodes:
        node_filter = " OR ".join([f"'{node}' IN labels(n) OR '{node}' IN labels(related)" for node in nodes])
        filter_conditions.append(f"({node_filter})")
    if relations:
        relation_filter = " OR ".join([f"type(r) = '{rel}'" for rel in relations])
        filter_conditions.append(f"({relation_filter})")

    filter_query = " AND ".join(filter_conditions)
    if filter_query:
        filter_query = f"WHERE {filter_query}"

    # 返回结果
    return_query = """
    RETURN DISTINCT n, r, related
    LIMIT 50
    """

    full_query = base_query + expand_query + filter_query + return_query
    return full_query


async def main():
    print("测试 API 连接")
    test_api_connection("openai", "api_key", "model_name")

    print("测试嵌入模型")
    test_embeddings("本地", base_url="embed_base_url", model="model_name")
    test_embeddings("OpenAI", api_key=api_key)

    llm = ChatOpenAI(model_name="model_name")
    embeddings = OpenAIEmbeddings()

    # 保持原有的 Neo4jGraph 连接
    graph = Neo4jGraph(
        url="neo4j_url",
        username="neo4j_username",
        password="neo4j_password"
    )

    uploaded_file = st.file_uploader("请选择一个PDF文件。", type="pdf")
    # 获取上传文件的名称和内容
    file_name = uploaded_file.name
    file_content = uploaded_file.getvalue()
    # 计算文件内容的MD5哈希值
    file_hash = generate_file_hash(file_content)

    # 检查Neo4j数据库中是否已存在此文件的处理结果
    check_query = f"MATCH (d:Document {{hash: '{file_hash}'}}) RETURN d"
    result = graph.query(check_query)
    if result:
        print(f"{file_name} 的处理结果已存在于数据库中。")

    # 若数据库中没有，则需要保存文件到图中
    # 保存上传的文件到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    # 清空数据库
    # MATCH (n)，匹配数据库中的所有节点
    # DETACH，断开关系。在删除节点之前，先自动断开该节点与其他节点的所有关系。如果不加DETACH，有关系的节点无法被直接删除
    # DELETE n，删除所有匹配到的节点
    await asyncio.to_thread(graph.query, "MATCH (n) DETACH DELETE n;")
    print("数据库已清空")

    # 加载PDF
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    print(f"PDF加载完成，页数: {len(pages)}")

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOC_CONFIG['chunk_size'],
        chunk_overlap=DOC_CONFIG['chunk_overlap']
    )
    with ThreadPoolExecutor() as executor:
        split_docs = list(executor.map(
            lambda page: text_splitter.split_text(
                page.page_content if hasattr(page, 'page_content') else page[1]),
            pages
        ))
    docs = [Document(page_content=split_doc, metadata={'source': file_name}) for page_splits in split_docs
            for split_doc in page_splits]
    print(f"文本分割完成，共 {len(docs)} 个文档片段")

    # 处理文档
    with ThreadPoolExecutor() as executor:
        lc_docs = list(executor.map(
            lambda doc: Document(page_content=doc.page_content.replace("\n", ""),
                                 metadata={'source': file_name}),
            docs
        ))
    print(f"文档处理完成，处理后共 {len(lc_docs)} 个文档")

    # 转换文档为图形，按照特定的提取约束，从文本中自动提取结构化知识图谱
    transformer = LLMGraphTransformer(
        llm=llm,
        # 节点提取约束，限制了LLM只能提取特定类型的知识
        allowed_nodes=GRAPH_CONFIG['allowed_nodes'],
        # 关系提取约束
        allowed_relationships=GRAPH_CONFIG['allowed_relationships'],
        # 属性设置
        node_properties=False,
        relationship_properties=False
    )
    await convert_and_add_to_graph(transformer, lc_docs, graph, file_name, file_hash)
    print("图形转换完成")

    """
    db.index.vector 是 neo4j 数据库内置的向量索引管理接口。它不是某个向量属性，而是表示一个管理向量索引的模块。
    - db: 表示数据库本身
    - index: 索引管理模块
    - vector: 向量索引的模块，也可以是 fulltext 全文检索
    - deleteIndex: 具体的操作函数，这个是向量索引删除（数据还在），drop 全文检索索引删除
    - createNodeIndex：创建索引
    - queryNodes：向量检索
    """
    # 先删除已存在的向量索引
    try:
        await asyncio.to_thread(
            graph.query,
            "CALL db.index.vector.deleteIndex('vector_index')"
        )
    except Exception as e:
        print(f"删除向量索引时出错（可能不存在）: {str(e)}")

    # 删除全文索引，删除了索引结构，而不会删除节点和属性
    try:
        await asyncio.to_thread(
            graph.query,
            "CALL db.index.fulltext.drop('entity_index')"
        )
    except Exception as e:
        print(f"删除全文索引时出错（可能不存在）: {str(e)}")

    # 创建新的向量索引
    """
    1. 创建向量索引
    CREATE VECTOR INDEX `vector_index` FOR (n:Document) ON n.embedding
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1536,
      `vector.similarity_function`: 'cosine'
    }}
    
    CALL db.index.vector.createNodeIndex(
        'vector_index',           -- 索引名称
        'Document',               -- 节点标签
        'embedding',              -- 向量属性名
        1536,                     -- 向量维度
        'cosine'                  -- 相似度算法
    )
    
    查看所有向量索引
    CALL db.index.vector.list()
    
    2. 为节点的 embedding 属性填充向量值
    MATCH (n:Document) 
    SET n.embedding = $embedding_vector  // 这里需要实际的向量数据
    
    Neo4jVector 是 LangChain 专门用于在 Neo4j 图数据库中集成向量搜索功能，实现图检索增强生成（GraphRAG）。
    """
    await asyncio.to_thread(
        Neo4jVector.from_existing_graph,  # 基于现有图数据结构创建向量索引属性，不改变结构，只添加属性
        embedding=embeddings,  # 使用的嵌入模型（如OpenAIEmbeddings）
        url="neo4j_url",  # Neo4j数据库连接信息
        username="neo4j_username",
        password="neo4j_password",
        database="neo4j",  # 目标数据库
        node_label="研究内容",  # 指定了具体的节点标签，只处理这个标签的节点
        text_node_properties=["id", "text"],  # 从这些属性构建文本
        embedding_node_property="embedding",  # 向量存储的属性名
        index_name="vector_index",  # 向量索引名称，用于向量检索
        keyword_index_name="entity_index",  # 全文检索索引名称，用于关键词检索
        search_type="hybrid"  # 搜索策略：混合搜索，使用混合搜索类型
    )
    print("向量索引创建完成")

    # 添加文档节点到Neo4j，标记处理完成
    doc_query = f"""
    CREATE (d:Document {{name: '{file_name}', hash: '{file_hash}', processed: true}})
    """
    await asyncio.to_thread(graph.query, doc_query)
    print(f"{file_name} 处理完成并已添加到数据库。")

    try:
        question = ""
        # 使用新的处理函数
        response, vector_results, query_results = await vector_query(
            question,
            graph,
            GRAPH_CONFIG,
            st.session_state['embeddings'],
            st.session_state['llm']
        )

        # 显示回答
        message_placeholder.markdown(response)

        # 显示知识图谱
        print("相关知识图谱")
        try:
            full_graph = load_graph_from_json('current_file_hash')
       
            # 先尝试找相关子图
            relevant_subgraph = find_relevant_subgraph(full_graph, question)

            # 如果相关子图太小或没找到，就使用完整图谱
            if not relevant_subgraph or relevant_subgraph.number_of_nodes() < 3:
                relevant_subgraph = full_graph
                print("显示完整知识图谱")
            else:
                print("显示相关知识子图")

            fig = create_knowledge_graph(relevant_subgraph)
            selected_points = plotly_events(fig, click_event=True)

            # 添加图谱统计信息
            print(f"知识图谱包含 {relevant_subgraph.number_of_nodes()} 个节点和 "
                    f"{relevant_subgraph.number_of_edges()} 个关系")

            # 显示选中节点的详细信息
            if selected_points:
                selected_node = selected_points[0]['pointNumber']
                node_info = list(relevant_subgraph.nodes(data=True))[selected_node]
                st.write(f"选中的节点: {node_info[0]}")
                st.write(f"节点内容: {node_info[1].get('text', '')}")
        except Exception as e:
            st.error(f"生成知识图谱时出错: {str(e)}")
    except Exception as e:
        error_message = f"处理问题时出错: {str(e)}"
        message_placeholder.error(error_message)


# 保持 main 函数的调用不变
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
