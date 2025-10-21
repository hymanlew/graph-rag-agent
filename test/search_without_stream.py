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
    ]
}

def test_agent(agent, agent_name, query, thread_id, show_thinking=False):
    """测试特定Agent的非流式响应
    
    参数:
        agent: 要测试的代理实例
        agent_name: 代理名称
        query: 测试查询内容
        thread_id: 线程唯一标识符
        show_thinking: 是否显示思考过程(仅DeepResearchAgent支持)
    
    返回:
        包含测试结果和性能指标的字典
    """
    print(f"\n[测试] {agent_name} - 查询: '{query}'")
    
    try:
        # 记录开始时间，用于计算总响应时间
        start_time = time.time()
        
        # 判断是否需要显示思考过程
        if show_thinking and hasattr(agent, 'ask_with_thinking'):
            # 调用支持思考过程的接口
            result = agent.ask_with_thinking(query, thread_id=thread_id)
            # 解析返回结果
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
                # 打印思考过程中的关键信息，便于调试和分析
                thinking_keys = [k for k in result.keys() if k != 'answer']
                print(f"思考过程包含以下信息: {', '.join(thinking_keys)}")
            else:
                answer = str(result)
        else:
            # 调用标准的非流式询问接口
            answer = agent.ask(query, thread_id=thread_id)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 处理答案预览（用于长答案的简洁显示）
        if len(answer) > 300:
            answer_preview = answer[:300] + "..."
        else:
            answer_preview = answer
        
        # 显示测试完成信息和性能指标
        print(f"[完成] 用时 {execution_time:.2f}秒，结果长度 {len(answer)} 字符")
        print(f"\n结果:\n{answer}\n") # 打印完整结果，便于观察内容质量
        
        # 返回测试结果统计信息
        return {
            "agent": agent_name,              # 代理名称
            "query": query,                   # 查询内容
            "execution_time": execution_time, # 执行时间
            "result_length": len(answer),     # 结果长度
            "success": True                   # 测试是否成功
        }
    
    except Exception as e:
        # 捕获并记录测试过程中的异常
        print(f"[错误] {agent_name} 处理查询时出错: {str(e)}")
        return {
            "agent": agent_name,
            "query": query,
            "error": str(e),
            "success": False
        }

def run_tests():
    """运行所有非流式测试
    
    功能:
    - 创建所有代理实例
    - 遍历测试所有查询
    - 收集并汇总测试结果
    - 计算并显示性能指标
    """
    print("\n===== 开始非流式Agent测试 =====\n")
    
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
            thread_id = f"test_{agent_name}_{int(time.time())}"
            
            # 执行非流式测试
            result = test_agent(agent, agent_name, query, thread_id)
            results.append(result)
            
            # 只有DeepResearchAgent支持思考过程测试
            # 如果启用了DeepResearchAgent，会额外测试其思考过程
            if agent_name == "DeepResearchAgent":
                print("\n--- 测试思考过程 ---")
                thinking_result = test_agent(agent, f"{agent_name}(思考模式)", query, f"{thread_id}_thinking", show_thinking=True)
                results.append(thinking_result)
    
    # 打印测试总结报告
    successful_tests = sum(1 for r in results if r.get("success", False))  # 计算成功测试数量
    total_tests = len(results)  # 总测试数量
    
    print("\n===== 测试总结 =====")
    print(f"成功测试: {successful_tests}/{total_tests}")  # 显示成功率
    
    # 计算并显示平均性能指标
    # 筛选出有执行时间数据的结果
    execution_times = [r.get("execution_time", 0) for r in results if "execution_time" in r]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"平均执行时间: {avg_time:.2f}秒")  # 显示平均响应速度
    
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
    # 运行测试函数
    run_tests()
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")