# 快速开始指南

## One-API 部署

使用 Docker 启动 One-API：

```bash
docker run --name one-api -d --restart always \
  -p 13000:3000 \
  -e TZ=Asia/Shanghai \
  -v /home/ubuntu/data/one-api:/data \
  justsong/one-api
```

在 One-API 控制台中配置第三方 API Key。本项目的所有 API 请求将通过 One-API 转发。

One API 是一个开源的 LLM API 接口调用管理与分发系统
- 统一API接口，适配成标准的OpenAI API格式
- 管理与分发，密钥管理，负载均衡，TOKEN 额度控制，模型重定向（权重）
- 该项目的官方地址：https://github.com/songquanpeng/one-api

具体的填写方式可以看[这里](https://github.com/1517005260/graph-rag-agent/issues/7#issuecomment-2906770240)

**注意**：默认用管理员账号登录，用户名root，密码123456，进去之后可以改密码


## Neo4j 启动

```bash
cd graph-rag-agent/
docker compose up -d
```

默认账号密码：

```
用户名：neo4j
密码：12345678
```

## 环境搭建

```bash
conda create -n graphrag python==3.10
conda activate graphrag
cd graph-rag-agent/
pip install -r requirements.txt
```

注意：如需处理 `.doc` 格式（旧版 Word 文件），请根据操作系统安装相应依赖，详见 `requirements.txt` 中注释（Linux, windows）

## .env 配置

在项目根目录下创建 `.env` 文件（拷贝 .env.example 到 .env）

**注意**：全流程测试通过的只有deepseek（20241226版本）以及gpt-4o，剩下的模型，比如：
- deepseek（20250324版本）幻觉问题比较严重，有概率不遵循提示词，导致抽取实体失败；
- Qwen的模型可以抽取实体，但是好像不支持langchain/langgraph，所以问答的时候有概率报错，他们有自己的agent实现[Qwen-Agent](https://qwen.readthedocs.io/zh-cn/latest/framework/qwen_agent.html)

## 项目初始化并运行

```bash
pip install -e .
```

## 知识图谱原始文件放置

请将原始文件放入 `files/` 文件夹，支持有目录的存放。当前支持以下格式（采用简单分块，后续会优化处理方式）：

```
- TXT（纯文本）
- PDF（PDF 文档）
- MD（Markdown）
- DOCX（新版 Word 文档）
- DOC（旧版 Word 文档）
- CSV（表格）
- JSON（结构化文本）
- YAML/YML（配置文件）
```

## 知识图谱配置（`config/settings.py`）

## 构建知识图谱

```bash
cd graph-rag-agent/

# 初始全量构建
python build/main.py

# 单次变量（增量、减量）构建：
python build/incremental_update.py --once

# 后台守护进程，定期变量更新：
python build/incremental_update.py --daemon
```

**注意：** `main.py`是构建的全流程，如果需要单独跑某个流程，请先完成实体索引的构建，再进行 chunk 索引构建，否则会报错（chunk 索引依赖实体索引）。

## 知识图谱搜索测试

```bash
cd graph-rag-agent/test

# 查询前可以注释掉不想测试的Agent，防止运行过慢

# 非流式查询
python search_without_stream.py

# 流式查询
python search_with_stream.py
```

## 知识图谱评估

```bash
cd evaluator/test
# 查看对应 README 获取更多信息
```

## 示例问题配置（用于前端展示）

编辑 `config/settings.py` 中的 `examples` 字段：

```python
examples = [
    "《悟空传》的主要人物有哪些？",
    "唐僧和会说话的树讨论了什么？",
    "孙悟空跟女妖之间有什么故事？",
    "他最后的选择是什么？"
]
```

## 并发进程配置（`server/main.py`）

```python
# FastAPI 的并发进程数设置
workers = 2
```

## 深度搜索优化（建议禁用前端超时）

如需开启深度搜索功能，建议禁用前端超时限制，修改 `frontend/utils/api.py`：

```python
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": message,
        "session_id": st.session_state.session_id,
        "debug": st.session_state.debug_mode,
        "agent_type": st.session_state.agent_type
    },
    # timeout=120  # 建议注释掉此行
)
```

## 中文字体支持（Linux）

如需中文图表显示，可参考[字体安装教程](https://zhuanlan.zhihu.com/p/571610437)。默认使用英文绘图（`matplotlib`）。


## 启动前后端服务

```bash
# 启动后端
cd graph-rag-agent/
python server/main.py

# 启动前端
cd graph-rag-agent/
streamlit run frontend/app.py
```

**注意**：由于langchain版本问题，目前的流式是伪流式实现，即先完整生成答案，再分段返回。
