import os
import sys
import argparse

# 添加父目录到路径，使得可以导入evaluator模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluator import set_debug_mode
from evaluator.utils.logging_utils import setup_logger
from evaluator.evaluator_config.agent_evaluation_config import get_agent_metrics
from evaluator.utils.eval_utils import evaluate_agent, load_questions_and_answers

"""
Deep Research Agent评估脚本

此脚本专门用于评估系统中Deep Research Agent的性能表现。Deep Research Agent是一种具有
深度研究和推理能力的智能体，支持标准研究工具和增强版研究工具两种模式。

脚本提供了灵活的评估配置选项，包括：
- 选择不同的评估类型（全面评估、仅答案质量、仅检索性能、仅推理能力）
- 自定义评估指标
- 切换标准/增强版研究工具
- 详细的日志记录和结果保存
"""

def parse_args():
    """
    解析命令行参数
    
    此函数定义了Deep Research Agent评估脚本的所有命令行参数，提供灵活的评估配置选项。
    参数设计覆盖了评估流程中的关键配置点，包括结果保存、数据源、评估范围和详细程度等。
    
    Returns:
        argparse.Namespace: 包含所有解析后参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="评估Deep Research Agent性能")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results/deep_agent",
                        help="评估结果保存目录")
    parser.add_argument("--questions_file", type=str, required=True,
                        help="要评估的问题文件（JSON格式）")
    parser.add_argument("--golden_answers_file", type=str, default=None,
                        help="标准答案文件（JSON格式，可选）")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细评估过程")
    parser.add_argument("--metrics", type=str, default="",
                        help="要评估的指标，用逗号分隔，留空则使用默认指标")
    parser.add_argument("--eval_type", type=str, default="all",
                        choices=["all", "answer", "retrieval", "reasoning"],
                        help="评估类型: all(全面评估), answer(仅答案质量), retrieval(仅检索性能), reasoning(仅推理能力)")
    parser.add_argument("--use_deeper", action="store_true",
                        help="是否使用增强版深度研究工具(DeeperResearchTool)")
    return parser.parse_args()

def main():
    """
    主函数：执行Deep Research Agent的评估流程
    
    此函数实现了Deep Research Agent评估的完整工作流：
    1. 解析命令行参数并配置评估环境
    2. 设置日志系统，记录评估过程
    3. 根据评估类型和用户输入确定评估指标
    4. 初始化Deep Research Agent，并配置是否使用增强版工具
    5. 加载评估数据（问题和标准答案）
    6. 调用通用评估函数执行评估
    7. 异常处理确保评估过程的稳定性
    
    函数特别处理了不同评估类型的指标选择逻辑，以及标准/增强版工具的配置，
    并对整个过程进行了详细的日志记录。
    """
    args = parse_args()
    
    # 设置日志记录
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger("deep_evaluation", os.path.join(args.save_dir, "evaluation.log"))
    logger.info("开始评估Deep Research Agent")
    
    # 设置全局调试模式
    set_debug_mode(args.verbose)
    
    # 确定使用哪些指标
    metrics = []
    if args.metrics:
        # 使用用户指定的指标
        metrics = args.metrics.split(',')
        logger.info(f"使用用户指定的评估指标: {args.metrics}")
    else:
        # 根据评估类型使用默认指标
        if args.eval_type == "answer":
            metrics = get_agent_metrics("deep", "answer")
            logger.info(f"使用答案评估指标: {', '.join(metrics)}")
        elif args.eval_type == "retrieval":
            metrics = get_agent_metrics("deep", "retrieval")
            logger.info(f"使用检索评估指标: {', '.join(metrics)}")
        elif args.eval_type == "reasoning":
            metrics = get_agent_metrics("deep", "reasoning")
            # 如果使用增强版深度研究工具，添加对应指标
            if args.use_deeper:
                metrics.extend(get_agent_metrics("deep", "deeper"))
            logger.info(f"使用推理评估指标: {', '.join(metrics)}")
        else:
            metrics = get_agent_metrics("deep")
            logger.info(f"使用全部评估指标: {', '.join(metrics)}")
    
    try:
        # 初始化Deep Research Agent并配置是否使用增强版工具
        from agent.deep_research_agent import DeepResearchAgent
        agent = DeepResearchAgent()
        
        # 配置Agent使用标准或增强版研究工具
        if not args.use_deeper:
            agent = agent.is_deeper_tool(False)
            logger.info("使用标准深度研究工具(DeepResearchTool)")
        else:
            agent = agent.is_deeper_tool(True)
            logger.info("使用增强版深度研究工具(DeeperResearchTool)")
        
        # 加载问题和标准答案数据
        questions, golden_answers = load_questions_and_answers(
            args.questions_file, 
            args.golden_answers_file
        )
        
        # 调用通用评估函数执行评估
        evaluate_agent(
            agent_type="deep",
            questions=questions,
            golden_answers=golden_answers,
            save_dir=args.save_dir,
            metrics=metrics,
            verbose=args.verbose
        )
    except Exception as e:
        # 记录评估过程中的异常
        logger.error(f"评估过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    """
    脚本入口点
    
    当脚本作为主程序运行时，调用main()函数启动评估流程。
    这种标准的Python模式确保了当脚本被导入为模块时，不会自动执行评估逻辑。
    """
    main()