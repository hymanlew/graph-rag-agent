import os
import sys
import json
import argparse
import time
from typing import List

# 添加父目录到路径，使得可以导入evaluator模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluator import set_debug_mode
from evaluator.utils.logging_utils import setup_logger
from evaluator.evaluator_config.evaluatorConfig import EvaluatorConfig
from evaluator.evaluators.composite_evaluator import CompositeGraphRAGEvaluator
from evaluator.evaluator_config.agent_evaluation_config import get_agent_metrics
from evaluator.utils.eval_utils import load_agent, load_dependencies, load_questions_and_answers

"""
多Agent评估比较脚本

此脚本用于对GraphRAG系统中的多个Agent进行综合评估和性能比较，支持以下功能：
- 批量评估不同类型的Agent（graph, hybrid, naive, fusion, deep等）
- 灵活配置评估指标和评估类型（全面评估、仅答案质量、仅检索性能）
- 支持使用标准答案进行质量评估和比较
- 生成详细的评估日志和性能对比报告
- 支持自定义评估参数和增强版Agent配置

评估结果将以JSON格式保存，并生成Markdown对比表格便于分析。
"""

def parse_args():
    """
    解析命令行参数
    
    设置评估脚本所需的所有配置参数，包括评估结果保存路径、问题文件、
    标准答案文件、评估指标、Agent类型等。支持灵活的评估配置，满足不同评估场景需求。
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="评估并比较所有Agent性能")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results/comparison",
                        help="评估结果保存目录")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="要评估的问题文件（JSON格式）")
    parser.add_argument("--golden_answers_file", type=str, default=None,
                        help="标准答案文件（JSON格式，可选）")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细评估过程")
    parser.add_argument("--metrics", type=str, default="",
                        help="要评估的指标，用逗号分隔，留空则使用所有Agent共有的指标")
    parser.add_argument("--agents", type=str, default="graph,hybrid,naive,fusion,deep",
                        help="要评估的Agent类型，用逗号分隔，默认评估所有Agent")
    parser.add_argument("--eval_type", type=str, default="all",
                        choices=["all", "answer", "retrieval"],
                        help="评估类型: all(全面评估), answer(仅答案质量), retrieval(仅检索性能)")
    parser.add_argument("--skip_missing", action="store_true",
                        help="如果某个Agent无法加载，跳过它而不是终止评估")
    parser.add_argument("--use_deeper", action="store_true",
                        help="是否为Deep Research Agent使用增强版工具")
    return parser.parse_args()

def get_common_metrics(agent_types: List[str], metric_type: str = None) -> List[str]:
    """
    获取所有指定Agent类型共有的评估指标
    
    为了公平比较不同Agent的性能，需要找到所有Agent都支持的评估指标。
    此函数通过计算多个Agent支持的评估指标的交集，确保比较的公平性和一致性。
    如果没有共有指标，则返回所有收集到的指标的并集。
    
    Args:
        agent_types: 要比较的Agent类型列表
        metric_type: 可选的指标类型过滤（answer或retrieval）
        
    Returns:
        List[str]: 所有Agent共有的评估指标列表
    """
    all_metrics = set()
    common_metrics = None
    
    for agent_type in agent_types:
        try:
            metrics = set(get_agent_metrics(agent_type, metric_type))
            if common_metrics is None:
                common_metrics = metrics
            else:
                common_metrics &= metrics
            all_metrics |= metrics
        except ValueError:
            continue
    
    if not common_metrics:
        # 如果没有共有指标，返回所有收集到的指标的并集
        return list(all_metrics)
    
    return list(common_metrics)

def main():
    """
    主函数：执行多Agent评估和比较的完整流程
    
    实现了多Agent评估的完整工作流：
    1. 解析命令行参数并配置评估环境
    2. 设置日志系统，记录评估过程和结果
    3. 加载依赖组件（Neo4j客户端和LLM模型）
    4. 加载问题和标准答案（如果有）
    5. 初始化需要评估的各种Agent
    6. 配置并创建复合评估器
    7. 根据是否有标准答案选择合适的评估方法
    8. 执行评估并获取各Agent的性能指标
    9. 保存评估结果和对比报告
    
    整个流程设计考虑了错误处理和日志记录，确保评估过程的稳定性和可追踪性。
    """
    args = parse_args()
    
    # 设置保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志记录
    logger = setup_logger("all_agents_evaluation", os.path.join(args.save_dir, "evaluation.log"))
    logger.info("开始评估并比较所有Agent性能")
    
    # 设置全局调试模式
    set_debug_mode(args.verbose)
    
    # 解析要评估的Agent类型
    agent_types = args.agents.split(',')
    logger.info(f"将评估以下Agent: {', '.join(agent_types)}")
    
    # 确定使用哪些指标
    metrics = []
    if args.metrics:
        # 使用用户指定的指标
        metrics = args.metrics.split(',')
        logger.info(f"使用用户指定的评估指标: {args.metrics}")
    else:
        # 获取所有待评估Agent共有的指标
        metric_type = None
        if args.eval_type == "answer":
            metric_type = "answer"
            logger.info("使用答案评估指标")
        elif args.eval_type == "retrieval":
            metric_type = "retrieval"
            logger.info("使用检索评估指标")
        
        metrics = get_common_metrics(agent_types, metric_type)
        logger.info(f"使用所有Agent共有的评估指标: {', '.join(metrics)}")
    
    # 加载依赖
    neo4j, llm = load_dependencies()
    
    # 加载问题和答案
    try:
        questions, golden_answers = load_questions_and_answers(
            args.questions_file, 
            args.golden_answers_file
        )
    except Exception as e:
        logger.error(f"加载问题和答案时出错: {e}")
        return
    
    # 加载所有Agent
    agents = {}
    for agent_type in agent_types:
        try:
            if agent_type == "deep" and args.use_deeper:
                # 特殊处理深度研究Agent，使用增强版
                from agent.deep_research_agent import DeepResearchAgent
                agents[agent_type] = DeepResearchAgent(use_deeper_tool=True)
                logger.info(f"成功加载{agent_type}Agent(增强版)")
            else:
                agents[agent_type] = load_agent(agent_type)
                logger.info(f"成功加载{agent_type}Agent")
        except Exception as e:
            logger.error(f"加载{agent_type}Agent失败: {e}")
            if args.skip_missing:
                logger.warning(f"已跳过{agent_type}Agent评估")
                continue
            else:
                return
    
    if not agents:
        logger.error("没有成功加载任何Agent，无法进行评估")
        return
    
    # 创建评估器
    config = {
        "save_dir": args.save_dir,
        "debug": args.verbose,
        "metrics": metrics,
        "neo4j_client": neo4j,
        "llm": llm
    }
    
    # 添加Agent到配置
    for agent_type, agent in agents.items():
        config[f"{agent_type}_agent"] = agent
    
    evaluator_config = EvaluatorConfig(config)
    evaluator = CompositeGraphRAGEvaluator(evaluator_config)
    
    start_time = time.time()
    all_results = {}
    
    # 有无标准答案决定评估方式
    try:
        if golden_answers:
            logger.info("使用标准答案进行比较评估...")
            all_results = evaluator.compare_agents_with_golden_answers(questions, golden_answers)
        else:
            logger.info("仅比较检索性能...")
            all_results = evaluator.compare_retrieval_only(questions)
    except Exception as e:
        logger.error(f"比较评估时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    end_time = time.time()
    logger.info(f"评估完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 打印比较结果
    if all_results:
        logger.info("\nAgent性能比较结果:")
        for agent_type, results in all_results.items():
            logger.info(f"\n{agent_type}Agent:")
            for metric, score in results.items():
                logger.info(f"  {metric}: {score:.4f}")
    
        # 保存比较结果
        results_path = os.path.join(args.save_dir, "agents_comparison.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n比较结果已保存至: {results_path}")
        
        # 创建对比表格
        try:
            comparison_table = evaluator.format_comparison_table(all_results)
            table_path = os.path.join(args.save_dir, "agents_comparison.md")
            
            with open(table_path, "w", encoding="utf-8") as f:
                f.write("# Agent性能对比表\n\n")
                f.write(comparison_table)
            
            logger.info(f"比较表格已保存至: {table_path}")
        except Exception as e:
            logger.error(f"生成比较表格时出错: {e}")
    else:
        logger.error("评估未生成任何结果")
    
    # 关闭所有Agent
    for agent_type, agent in agents.items():
        if hasattr(agent, 'close') and callable(agent.close):
            try:
                agent.close()
                logger.info(f"已关闭{agent_type}Agent")
            except Exception as e:
                logger.warning(f"关闭{agent_type}Agent时出错: {e}")

if __name__ == "__main__":
    """
    脚本入口点

    当脚本作为主程序运行时执行评估流程，支持命令行调用和参数配置。
    这样设计允许脚本被直接运行，也可以作为模块被导入使用。
    """
    main()