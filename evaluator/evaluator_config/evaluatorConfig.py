from typing import Dict, Any, List, Optional

"""
评估器配置管理模块

提供评估器配置的加载、管理和访问功能，支持默认值设置、配置项获取和更新等操作。
此模块中的EvaluatorConfig类是评估系统配置的核心组件，负责存储和管理所有评估参数。
"""

class EvaluatorConfig:
    """
    评估器配置管理类
    
    负责管理GraphRAG评估系统的所有配置选项，提供配置访问、更新和验证功能。
    支持默认值设置，确保即使配置不完整也能正常运行。
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化评估器配置
        
        Args:
            config_dict: 配置字典，包含评估系统需要的各种参数
            
        初始化流程：
        1. 接收配置字典（为空时使用空字典）
        2. 设置默认配置值，确保关键配置有合理的默认值
        """
        self.config = config_dict or {}
        
        # 设置默认配置值，确保必要配置项有合理的默认值
        self._set_defaults()
    
    def _set_defaults(self):
        """
        设置默认配置值
        
        确保所有必要的配置项都有合理的默认值，只有在配置项不存在时才设置默认值，
        这样可以保留用户提供的自定义配置。
        """
        # 定义默认配置值字典
        defaults = {
            # 评估结果保存目录
            'save_dir': './evaluation_results',
            # 是否保存指标分数
            'save_metric_score': True,
            # 是否保存中间数据
            'save_intermediate_data': True,
            # 评估指标列表
            'metrics': [],
            # 调试模式开关
            'debug': True,
            # 数据集名称
            'dataset_name': 'default'
        }
        
        # 遍历默认配置，只设置不存在的配置项
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """
        获取指定配置项的值
        
        Args:
            key: 配置项键名
            default: 当配置项不存在时的默认返回值
            
        Returns:
            Any: 配置项的值或默认值
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        设置指定配置项的值
        
        Args:
            key: 配置项键名
            value: 配置项的值
        """
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        使用字典批量更新配置项
        
        Args:
            config_dict: 包含多个配置项的字典
        """
        self.config.update(config_dict)
    
    def get_metrics(self) -> List[str]:
        """
        获取配置的评估指标列表
        
        将所有指标名称转换为小写，便于后续统一比较和处理。
        
        Returns:
            List[str]: 评估指标名称列表（全部小写）
        """
        return [metric.lower() for metric in self.config.get('metrics', [])]
    
    def is_debug_enabled(self) -> bool:
        """
        判断是否开启调试模式
        
        调试模式下，系统会输出更详细的日志信息，有助于问题排查。
        
        Returns:
            bool: 是否开启调试模式
        """
        return self.config.get('debug', False)
    
    def get_save_dir(self) -> str:
        """
        获取评估结果保存目录路径
        
        评估结果、中间数据和日志将保存在此目录下。
        
        Returns:
            str: 保存目录路径
        """
        return self.config.get('save_dir', './evaluation_results')
    
    def get_agent(self, agent_type: str) -> Optional[Any]:
        """
        获取指定类型的Agent实例
        
        用于从配置中获取已初始化的Agent对象，支持不同类型的Agent：
        - naive: 基础RAG Agent
        - hybrid: 混合检索Agent
        - graph: 图检索Agent
        - deep: 深度研究Agent
        
        Args:
            agent_type: Agent类型标识符
            
        Returns:
            Any: Agent实例或None（如果未配置）
        """
        agent_key = f"{agent_type}_agent"
        return self.config.get(agent_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式
        
        返回配置字典的副本，避免外部代码意外修改内部状态。
        
        Returns:
            Dict[str, Any]: 配置字典的副本
        """
        return self.config.copy()