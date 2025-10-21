"""
代理管理服务模块

该模块提供了代理（Agent）的统一管理功能，负责创建、缓存和管理不同类型的代理实例。
实现了代理的会话隔离、资源管理和生命周期控制，是系统智能交互能力的核心组件之一。

主要功能：
- 支持多种类型代理的注册和管理
- 为每个会话维护独立的代理实例
- 提供代理资源的创建、复用和释放机制
- 支持会话历史的清除功能
- 提供日志格式化工具，便于前端展示

设计特点：
- 采用工厂模式创建代理实例
- 使用线程锁确保并发安全
- 实现会话隔离，防止会话间干扰
- 支持多种Agent类型的灵活切换
- 提供资源生命周期管理机制
"""
from typing import Dict, List
import threading
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, ToolMessage


# 创建Agent管理类
class AgentManager:
    """
    代理管理器类
    
    负责管理所有类型的代理实例，提供代理的创建、获取、会话管理和资源释放功能。
    实现了单例模式和线程安全的代理池管理，支持多种类型代理的统一访问接口。
    
    属性：
        agent_classes: 代理类型到代理类的映射字典
        agent_instances: 代理实例池，按会话和类型进行索引
        agent_lock: 线程锁，确保并发安全
    """
    
    def __init__(self):
        """
        初始化代理管理器
        
        注册系统支持的所有代理类型，初始化实例池和线程锁。
        这种懒加载方式确保只有在需要时才导入代理类，优化了资源使用。
        """
        # 导入各种Agent - 使用懒加载方式导入，减少初始加载时间
        from agent.graph_agent import GraphAgent
        from agent.hybrid_agent import HybridAgent
        from agent.naive_rag_agent import NaiveRagAgent
        from agent.deep_research_agent import DeepResearchAgent 
        from agent.fusion_agent import FusionGraphRAGAgent 
        
        # 初始化Agent类映射 - 将代理类型名称映射到对应的代理类
        self.agent_classes = {
            "graph_agent": GraphAgent,  # 知识图谱代理，基于知识图谱进行推理
            "hybrid_agent": HybridAgent,  # 混合代理，结合多种方法的综合代理
            "naive_rag_agent": NaiveRagAgent,  # 简单RAG代理，默认代理类型
            "deep_research_agent": DeepResearchAgent,  # 深度搜索代理，用于复杂问题研究
            "fusion_agent": FusionGraphRAGAgent,  # 融合代理，结合知识图谱和RAG的高级代理
        }
        
        # 保留Agent实例池 - 用于缓存和复用代理实例
        self.agent_instances = {}
        
        # 添加锁来保护实例访问 - 确保并发安全，避免多线程访问冲突
        self.agent_lock = threading.RLock()
    
    def get_agent(self, agent_type: str, session_id: str = "default"):
        """
        获取指定类型的代理实例，对每个会话使用独立实例
        
        采用单例模式和会话隔离的设计，确保每个会话获得独立的代理实例，
        避免不同会话之间的状态干扰，同时复用已有实例提高性能。
        
        Args:
            agent_type: 代理类型名称，必须是已注册的类型
            session_id: 会话ID，用于隔离不同用户的会话状态
            
        Returns:
            Agent实例：对应类型的代理实例，具有特定会话状态
            
        Raises:
            ValueError: 当指定的代理类型不存在时抛出
        """
        # 验证代理类型是否存在
        if agent_type not in self.agent_classes:
            raise ValueError(f"未知的agent类型: {agent_type}")
        
        # 为每个会话使用单独的Agent实例，避免资源争用
        # 使用代理类型和会话ID组合作为实例键
        instance_key = f"{agent_type}:{session_id}"
        
        # 使用线程锁确保并发安全
        with self.agent_lock:
            # 如果实例不存在，则创建新实例
            if instance_key not in self.agent_instances:
                # 创建新的Agent实例
                self.agent_instances[instance_key] = self.agent_classes[agent_type]()
            
            # 返回现有或新创建的实例
            return self.agent_instances[instance_key]
    
    def clear_history(self, session_id: str) -> Dict:
        """
        清除特定会话的聊天历史
        
        遍历并清除指定会话ID在所有代理类型中的聊天历史，保留必要的上下文信息。
        该功能用于重置对话状态，保护用户隐私，或开始新的对话主题。
        
        Args:
            session_id: 需要清除历史记录的会话ID
            
        Returns:
            Dict: 包含操作状态和消息的结果字典
                - status: 操作状态，"success"表示成功
                - remaining_messages: 操作结果的描述文本
        """
        remaining_text = ""
        
        try:
            # 清除对应会话的所有agent实例历史
            with self.agent_lock:
                for agent_type in self.agent_classes.keys():
                    instance_key = f"{agent_type}:{session_id}"
                    if instance_key in self.agent_instances:
                        agent = self.agent_instances[instance_key]
                        config = {"configurable": {"thread_id": session_id}}
                        
                        # 添加检查，防止None值报错
                        memory_content = agent.memory.get(config)
                        if memory_content is None or "channel_values" not in memory_content:
                            continue  # 跳过这个agent
                            
                        messages = memory_content["channel_values"]["messages"]
                        
                        # 如果消息少于2条，不进行删除操作
                        if len(messages) <= 2:
                            continue

                        i = len(messages)
                        for message in reversed(messages):
                            # 特殊处理工具消息情况
                            if isinstance(messages[2], ToolMessage) and i == 4:
                                break
                            # 从图状态中移除消息
                            agent.graph.update_state(config, {"messages": RemoveMessage(id=message.id)})
                            i = i - 1
                            if i == 2:  # 保留前两条消息（通常是系统提示和初始上下文）
                                break
            
            # 获取剩余消息
            remaining_text = "已清除会话历史"
        
        except Exception as e:
            print(f"清除聊天历史时出错: {str(e)}")
        
        return {
            "status": "success",
            "remaining_messages": remaining_text
        }
    
    def close_all(self):
        """
        关闭所有代理资源
        
        负责安全地释放所有代理实例占用的资源，包括模型连接、内存等。
        该方法在应用程序关闭时调用，确保资源正确释放，避免内存泄漏。
        """
        with self.agent_lock:
            # 遍历并关闭所有代理实例
            for instance_key, agent in self.agent_instances.items():
                try:
                    agent.close()  # 调用代理的close方法释放资源
                    print(f"已关闭 {instance_key} 资源")
                except Exception as e:
                    print(f"关闭 {instance_key} 资源时出错: {e}")
            
            # 清空实例池，确保资源完全释放
            self.agent_instances.clear()


# 创建全局实例
# 使用模块级别的单例模式，确保整个应用使用同一个代理管理器实例
agent_manager = AgentManager()


def format_messages_for_response(messages: List[Dict]) -> str:
    """
    将消息列表格式化为易于阅读的字符串
    
    处理不同类型的消息（用户消息和AI消息），为它们添加适当的前缀，
    然后将所有消息拼接成一个格式化的字符串，便于日志记录和调试。
    
    Args:
        messages: 消息对象列表，通常包含HumanMessage和AIMessage类型
    
    Returns:
        str: 格式化后的消息字符串，每条消息占一行，带有消息类型前缀
    """
    formatted = []
    # 遍历所有消息
    for msg in messages:
        # 只处理人类消息和AI消息
        if isinstance(msg, (HumanMessage, AIMessage)):
            # 根据消息类型添加不同的前缀
            prefix = "User: " if isinstance(msg, HumanMessage) else "AI: "
            formatted.append(f"{prefix}{msg.content}")
    # 使用换行符连接所有消息
    return "\n".join(formatted)


def format_execution_log(log: List[Dict]) -> List[Dict]:
    """
    格式化执行日志用于JSON响应
    
    将代理执行过程中产生的复杂日志对象转换为可序列化的字典列表，
    确保所有内容都可以安全地转换为JSON格式，便于前端展示和调试。
    
    Args:
        log: 原始执行日志，包含节点信息、输入和输出
    
    Returns:
        List[Dict]: 格式化后的执行日志，确保所有内容可序列化
    """
    formatted_log = []
    # 遍历每条日志条目
    for entry in log:
        # 创建格式化条目，包含节点名称
        formatted_entry = {"node": entry["node"]}
        
        # 处理输入部分
        if "input" in entry:
            # 根据输入类型进行不同的处理
            if isinstance(entry["input"], dict):
                input_str = {}
                # 递归处理字典中的每个键值对
                for k, v in entry["input"].items():
                    # 处理具有content属性的消息对象
                    if hasattr(v, "content"):
                        input_str[k] = {"content": v.content}
                    # 直接处理字符串
                    elif isinstance(v, str):
                        input_str[k] = v
                    else:
                        # 安全处理其他类型，确保可序列化
                        try:
                            import json
                            json.dumps(v)  # 测试是否可序列化
                            input_str[k] = v
                        except:
                            # 不可序列化的对象转换为字符串
                            input_str[k] = str(v)
            # 处理具有content属性的输入对象
            elif hasattr(entry["input"], "content"):
                input_str = {"content": entry["input"].content}
            # 其他类型直接转换为字符串
            else:
                input_str = str(entry["input"])
            formatted_entry["input"] = input_str
            
        # 处理输出部分 - 与输入处理逻辑类似
        if "output" in entry:
            if isinstance(entry["output"], dict):
                output_str = {}
                for k, v in entry["output"].items():
                    if hasattr(v, "content"):
                        output_str[k] = {"content": v.content}
                    elif isinstance(v, str):
                        output_str[k] = v
                    else:
                        try:
                            import json
                            json.dumps(v)  # 测试是否可序列化
                            output_str[k] = v
                        except:
                            output_str[k] = str(v)
            elif hasattr(entry["output"], "content"):
                output_str = {"content": entry["output"].content}
            else:
                output_str = str(entry["output"])
            formatted_entry["output"] = output_str
        
        # 添加格式化后的条目到结果列表
        formatted_log.append(formatted_entry)
    return formatted_log