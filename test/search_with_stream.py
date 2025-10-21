import asyncio
import time
from datetime import datetime

# 导入各种智能代理类
from agent.deep_research_agent import DeepResearchAgent  # 深度研究代理
from agent.naive_rag_agent import NaiveRagAgent  # 基础RAG代理
from agent.graph_agent import GraphAgent  # 基于知识图谱的代理
from agent.hybrid_agent import HybridAgent  # 混合策略代理
from agent.fusion_agent import FusionGraphRAGAgent  # 融合图谱和RAG的代理


# 测试配置字典
TEST_CONFIG = {
    "queries": [
        "优秀学生的申请条件是什么？",         # 基础信息查询
        "学业奖学金有多少钱？",               # 具体数值查询
        "大学英语考试的标准是什么？",         # 标准/规范查询
        "小明同学旷课了30学时，又私藏了吹风机，他还殴打了同学，他还能评选国家奖学金吗？",  # 复杂多条件判断查询
    ],
    "max_wait_time": 300  # 每次测试的最大等待时间(秒)，防止测试无限期等待
}

async def test_agent_stream(agent, agent_name, query, thread_id, show_thinking=False, max_time=None):
    """测试特定Agent的流式响应
    
    参数:
        agent: 要测试的代理实例
        agent_name: 代理名称
        query: 测试查询内容
        thread_id: 线程唯一标识符
        show_thinking: 是否显示思考过程(仅DeepResearchAgent支持)
        max_time: 最大等待时间，默认使用配置中的值
    
    返回:
        包含测试结果和性能指标的字典
    """
    if max_time is None:
        max_time = TEST_CONFIG["max_wait_time"]
        
    print(f"\n[测试] {agent_name} - 流式 - 查询: '{query}'")
    
    try:
        # 检查代理是否支持流式输出
        if not hasattr(agent, 'ask_stream'):
            print(f"[错误] {agent_name} 不支持流式输出")
            return {
                "agent": agent_name,
                "query": query,
                "error": "不支持流式输出",
                "success": False
            }
        
        # 记录性能指标
        start_time = time.time()      # 记录开始时间
        chunk_count = 0               # 数据块计数器
        total_chars = 0               # 总字符数
        first_token_time = None       # 首个token响应时间
        collected_text = []           # 收集所有文本块
        
        # 设置超时时间点
        timeout = start_time + max_time
        
        # 打印流式测试开始提示
        print(f"开始接收流式输出 (最长等待 {max_time} 秒)...")
        
        # 执行流式查询
        # 注意：只有DeepResearchAgent支持show_thinking参数
        async for chunk in agent.ask_stream(query, thread_id=thread_id, **({"show_thinking": show_thinking} if agent_name.startswith("DeepResearchAgent") else {})):
            # 记录第一个token的时间（衡量首字节延迟）
            if chunk_count == 0:
                first_token_time = time.time()
            
            chunk_count += 1
            
            # 将字典类型的chunk转换为字符串
            # 不同代理可能返回不同类型的chunk
            if isinstance(chunk, dict):
                if 'answer' in chunk:
                    chunk_text = chunk['answer']
                    print("\n[接收到最终答案字典]")
                else:
                    chunk_text = str(chunk)
                    print("\n[接收到中间结果字典]")
            else:
                chunk_text = str(chunk)
            
            # 追加到收集的文本，用于后续分析
            collected_text.append(chunk_text)
            total_chars += len(chunk_text)
            
            # 每隔20个块显示一次进度
            if chunk_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"已接收 {chunk_count} 块，共 {total_chars} 字符，耗时 {elapsed:.2f} 秒")
            
            # 检查是否超时，如果超时则提前结束测试
            if time.time() > timeout:
                print(f"达到最大等待时间 {max_time} 秒，提前结束接收")
                break
        
        # 计算性能指标
        end_time = time.time()
        total_time = end_time - start_time
        time_to_first_token = (first_token_time - start_time) if first_token_time else None
        
        # 合并所有收集到的文本，得到完整响应
        full_text = "".join(collected_text)
        
        # 显示测试结果统计信息
        print(f"\n[完成] 流式查询完成")
        print(f"- 总耗时: {total_time:.2f}秒")     # 总响应时间
        if time_to_first_token:
            print(f"- 首块延迟: {time_to_first_token:.2f}秒")  # 首块到达时间，衡量用户感知延迟
        print(f"- 数据块数: {chunk_count}个")     # 数据块数量，衡量流式效果
        print(f"- 总字符数: {total_chars}字符")    # 总字符数，衡量回答长度
        
        # 显示结果预览或完整结果
        if len(full_text) > 300:
            preview_text = full_text[:300] + "..."
        else:
            preview_text = full_text
        
        print(f"\n结果:\n{full_text}\n")  # 打印完整结果，便于观察内容质量
        
        # 返回测试结果统计信息
        return {
            "agent": agent_name,              # 代理名称
            "query": query,                   # 查询内容
            "total_time": total_time,         # 总响应时间
            "time_to_first_token": time_to_first_token,  # 首块延迟
            "chunk_count": chunk_count,       # 数据块数量
            "total_chars": total_chars,       # 总字符数
            "success": True                   # 测试是否成功
        }
    
    except Exception as e:
        # 捕获并记录测试过程中的异常
        print(f"[错误] {agent_name} 流式处理查询时出错: {str(e)}")
        return {
            "agent": agent_name,
            "query": query,
            "error": str(e),
            "success": False
        }

async def run_stream_tests():
    """运行所有流式测试
    
    功能:
    - 创建所有代理实例
    - 遍历测试所有查询
    - 收集并汇总测试结果
    - 计算并显示性能指标
    """
    print("\n===== 开始流式Agent测试 =====\n")
    
    # 创建所有agent实例
    # 注意：默认只启用了FusionGraphRAGAgent，其他代理被注释掉
    agents = [
        # {"name": "DeepResearchAgent", "instance": DeepResearchAgent(use_deeper_tool=True)},
        # {"name": "NaiveRagAgent", "instance": NaiveRagAgent()},
        # {"name": "GraphAgent", "instance": GraphAgent()},
        # {"name": "HybridAgent", "instance": HybridAgent()},
        {"name": "FusionGraphRAGAgent", "instance": FusionGraphRAGAgent()}  # 融合图谱和RAG的代理
    ]
    
    # 存储所有测试结果
    results = []
    
    # 遍历所有测试查询
    for query in TEST_CONFIG["queries"]:
        print(f"\n===== 测试查询: {query} =====")
        
        for agent_info in agents:
            agent_name = agent_info["name"]
            agent = agent_info["instance"]
            
            # 为每个测试创建唯一的线程ID，避免冲突
            thread_id = f"stream_{agent_name}_{int(time.time())}"
            
            # 执行流式测试
            result = await test_agent_stream(agent, agent_name, query, thread_id)
            results.append(result)
            
            # 只有DeepResearchAgent支持思考过程测试
            # 如果启用了DeepResearchAgent，会额外测试其思考过程的流式输出
            if agent_name == "DeepResearchAgent":
                print("\n--- 测试思考过程流式输出 ---")
                thinking_result = await test_agent_stream(
                    agent, f"{agent_name}(思考模式)", query, 
                    f"{thread_id}_thinking", show_thinking=True
                )
                results.append(thinking_result)
    
    # 打印测试总结报告
    successful_tests = sum(1 for r in results if r.get("success", False))  # 计算成功测试数量
    total_tests = len(results)  # 总测试数量
    
    print("\n===== 测试总结 =====")
    print(f"成功测试: {successful_tests}/{total_tests}")  # 显示成功率
    
    # 计算并显示平均性能指标
    valid_results = [r for r in results if r.get("success", False)]  # 筛选成功的测试结果
    if valid_results:
        # 计算平均总耗时
        avg_total_time = sum(r.get("total_time", 0) for r in valid_results) / len(valid_results)
        # 计算平均首块延迟（只考虑有首块延迟数据的结果）
        avg_first_token = sum(r.get("time_to_first_token", 0) for r in valid_results if r.get("time_to_first_token")) / \
                         sum(1 for r in valid_results if r.get("time_to_first_token"))
        # 计算平均数据块数
        avg_chunks = sum(r.get("chunk_count", 0) for r in valid_results) / len(valid_results)
        
        print(f"平均总耗时: {avg_total_time:.2f}秒")      # 响应速度指标
        print(f"平均首块延迟: {avg_first_token:.2f}秒")   # 用户体验延迟指标
        print(f"平均数据块数: {avg_chunks:.1f}个")      # 流式输出流畅度指标
    
    # 显示失败的测试详情，便于调试
    failed = [r for r in results if not r.get("success", False)]
    if failed:
        print("失败的测试:")
        for f in failed:
            agent = f.get("agent", "未知")
            query = f.get("query", "未知")
            error = f.get("error", "未知错误")
            print(f"- {agent} 处理 '{query}' 时失败: {error}")

if __name__ == "__main__":
    # 程序入口，记录测试开始和结束时间
    print(f"开始测试: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 运行异步测试函数
    asyncio.run(run_stream_tests())
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")