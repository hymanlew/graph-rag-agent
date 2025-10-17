# config.py

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
API_CONFIG = {
    'deepseek': {
        'base_url': "https://api.deepseek.com/v1",
        'default_model': "deepseek-chat"
    },
    'openai': {
        'base_url': "https://api.openai.com/v1",
        'default_model': "gpt-4o-mini"
    }
}

# Neo4j配置
NEO4J_CONFIG = {
    'url': "neo4j://localhost:7687",
    'username': "neo4j",
    'password': os.getenv('NEO4J_PASSWORD', '')
}

# 本地嵌入配置
EMBEDDING_CONFIG = {
    'local': {
        'base_url': "http://localhost:1234/v1",
        'model': "BAAI/BAAI_bge-large-zh-v1.5/bge-large-zh-v1.5-f32.gguf"
    }
}

# 图谱配置
GRAPH_CONFIG = {
    # 只能识别这6类节点，其他实体类型会被忽略
    'allowed_nodes': [
        "研究内容", "研究方法", "创新点", 
        "参考部分", "预期成果", "未来展望"
    ],
    # 只能使用这10种预定义关系
    'allowed_relationships': [
        "支持", "补充", "引用", "反映", "基于", 
        "达成", "依赖于", "借鉴", "指导", "产生"
    ]
}

# 文档处理配置
DOC_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 40
}

# 应用配置
APP_CONFIG = {
    'title': "DateGraphX：实时图谱RAG应用",
    'description': """
    此应用程序允许您上传PDF文件，将其内容提取到Neo4j图形数据库中，并使用自然语言执行查询。
    它利用LangChain和DeepSeek的模型生成Cypher查询，实时与Neo4j数据库交互。
    """,
    'logo_path': 'logo.png'
}
