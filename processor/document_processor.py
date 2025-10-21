"""
文档处理模块

该模块实现了文档处理的核心功能，是知识库构建过程中的关键组件。
负责整合文件读取、文本分块和文档信息提取等操作，为后续的
向量化和图构建提供数据基础。
"""
import os
from typing import List, Dict, Optional, Any

from processor.file_reader import FileReader
from processor.text_chunker import ChineseTextChunker
from config.settings import FILES_DIR, CHUNK_SIZE, OVERLAP


class DocumentProcessor:
    """
    文档处理器类
    
    该类是文档处理流水线的核心控制器，负责协调文件读取器和文本分块器，
    提供完整的文档处理功能。设计采用组合模式，将不同功能委托给专用组件，
    同时提供统一的接口和丰富的统计信息。
    
    主要功能：
    - 批量处理目录中的文档文件
    - 将原始文档转换为结构化的文档对象
    - 对文档内容进行智能分块处理
    - 提供详细的文件统计信息
    - 支持多种文件格式的统一处理
    """
    
    def __init__(self, directory_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        """
        初始化文档处理器
        
        参数：
            directory_path: 文件目录路径，指定要处理的文档所在根目录
            chunk_size: 文本分块大小，默认为配置文件中定义的值
            overlap: 分块重叠大小，默认为配置文件中定义的值
            
        实现思路：
        - 保存目录路径供后续处理使用
        - 初始化文件读取器组件，负责实际的文件I/O操作
        - 初始化中文文本分块器组件，负责文本的智能分段
        - 使用默认参数从配置文件获取分块大小和重叠值
        """
        self.directory_path = directory_path
        # 组合模式：创建文件读取器实例
        self.file_reader = FileReader(directory_path)
        # 组合模式：创建文本分块器实例
        self.chunker = ChineseTextChunker(chunk_size, overlap)
        
    def process_directory(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        处理目录中的所有支持文件
        
        参数：
            file_extensions: 指定要处理的文件扩展名列表，如不指定则处理所有支持的类型
            recursive: 是否递归处理子目录，默认为True
            
        返回：
            List[Dict]: 处理结果列表，每个文件对应一个字典，包含详细的文件信息和分块结果
            
        实现思路：
        1. 调用文件读取器读取目录中的所有文件
        2. 对每个文件进行单独处理，提取元数据和内容
        3. 使用文本分块器对内容进行分块处理
        4. 计算分块统计信息（长度、数量等）
        5. 实现错误处理，确保单个文件处理失败不影响整体流程
        6. 提供详细的日志输出，便于调试和监控
        
        业务意义：
        - 作为文档处理的主要入口函数
        - 将原始文件转换为结构化的文档对象
        - 为后续的向量化和图构建提供数据基础
        - 生成详细的处理统计信息
        """
        # 调用文件读取器读取目录中的所有文件
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # 打印调试信息，帮助跟踪处理过程
        print(f"DocumentProcessor找到的文件数量: {len(file_contents)}")
        if len(file_contents) > 0:
            print(f"文件类型: {[os.path.splitext(f[0])[1] for f in file_contents]}")
        
        # 处理每个文件
        results = []
        for filepath, content in file_contents:
            # 提取文件扩展名，用于识别文件类型
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 创建文件处理结果字典，收集所有相关信息
            file_result = {
                "filepath": filepath,  # 相对路径，便于后续引用
                "filename": os.path.basename(filepath),  # 仅文件名，便于显示
                "extension": file_ext,  # 文件扩展名
                "content": content,  # 原始文件内容
                "content_length": len(content),  # 内容长度
                "chunks": None  # 初始化分块结果字段
            }
            
            # 对文本内容进行分块处理
            try:
                # 调用文本分块器进行分块
                chunks = self.chunker.chunk_text(content)
                file_result["chunks"] = chunks
                file_result["chunk_count"] = len(chunks)
                
                # 计算每个块的长度，用于质量评估
                chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
                file_result["chunk_lengths"] = chunk_lengths
                file_result["average_chunk_length"] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                
            except Exception as e:
                # 记录分块错误，但继续处理其他文件
                file_result["chunk_error"] = str(e)
                print(f"分块错误 ({filepath}): {str(e)}")
                
            results.append(file_result)
            
        return results
        
    def get_file_stats(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> Dict[str, Any]:
        """
        获取目录中文件的统计信息
        
        参数：
            file_extensions: 指定要统计的文件扩展名列表，如不指定则统计所有支持的类型
            recursive: 是否递归统计子目录，默认为True
            
        返回：
            Dict: 文件统计信息字典，包含详细的统计数据
            
        实现思路：
        1. 读取指定类型的所有文件
        2. 统计每种扩展名的文件数量
        3. 计算总内容长度和平均文件长度
        4. 收集子目录信息
        5. 处理可能的空内容情况
        
        业务意义：
        - 提供目录内容的整体概览
        - 帮助评估知识库的规模和组成
        - 为后续处理提供参考信息
        - 支持决策优化，如是否需要调整分块策略
        """
        # 读取指定类型的所有文件
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # 初始化统计计数器
        extension_counts = {}  # 扩展名计数
        total_content_length = 0  # 总内容长度
        
        # 统计子目录数量
        directories = set()  # 使用集合去重
        
        for filepath, content in file_contents:
            # 统计扩展名
            ext = os.path.splitext(filepath)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # 记录文件所在的子目录
            dirpath = os.path.dirname(filepath)
            if dirpath:  # 非空表示在子目录中
                directories.add(dirpath)
                
            # 累加内容长度，处理可能的None值
            if content is not None:
                total_content_length += len(content)
            else:
                print(f"警告: 文件 {filepath} 的内容为None")
            
        # 返回完整的统计信息
        return {
            "total_files": len(file_contents),  # 总文件数
            "extension_counts": extension_counts,  # 扩展名统计
            "total_content_length": total_content_length,  # 总字符数
            "average_file_length": total_content_length / len(file_contents) if file_contents else 0,  # 平均文件长度
            "directories": list(directories),  # 子目录列表
            "directory_count": len(directories)  # 子目录数量
        }
        
    def get_extension_type(self, extension: str) -> str:
        """
        获取文件扩展名对应的文档类型描述
        
        参数：
            extension: 文件扩展名（包括'.'，如'.pdf'）
            
        返回：
            str: 文档类型的中文描述
            
        实现思路：
        - 使用字典映射扩展名到类型描述
        - 支持常见的文档格式和数据格式
        - 对于未知类型返回默认描述
        - 不区分大小写（通过lower()实现）
        
        业务意义：
        - 提供更友好的文件类型显示
        - 帮助用户理解文档组成
        - 为不同类型的文档处理提供依据
        """
        # 扩展名到类型描述的映射表
        extension_types = {
            '.txt': '文本文件',
            '.pdf': 'PDF文档',
            '.md': 'Markdown文档',
            '.doc': 'Word文档',
            '.docx': 'Word文档',
            '.csv': 'CSV数据文件',
            '.json': 'JSON数据文件',
            '.yaml': 'YAML配置文件',
            '.yml': 'YAML配置文件',
        }
        
        # 根据扩展名获取类型描述，如果找不到则返回默认值
        return extension_types.get(extension.lower(), '未知类型')
        
        
if __name__ == "__main__":
    """
    测试文档处理器功能
    
    测试流程：
    1. 初始化文档处理器
    2. 列出目录中的所有文件
    3. 获取文件统计信息并打印
    4. 处理所有文件并输出处理结果摘要
    
    目的：
    - 验证文档处理器的基本功能
    - 提供使用示例
    - 展示处理结果的格式和内容
    """
    # 创建文档处理器实例
    processor = DocumentProcessor(FILES_DIR)
    
    # 列出目录中的所有文件
    print(f"目录 {FILES_DIR} 及其子目录中的所有文件:")
    all_files = processor.file_reader.list_all_files(recursive=True)
    for filepath in all_files:
        print(f"  {filepath}")
    
    # 获取并打印文件统计信息
    print("\n目录文件统计:")
    stats = processor.get_file_stats(recursive=True)
    print(f"总文件数: {stats['total_files']}")
    print(f"子目录数: {stats['directory_count']}")
    
    # 打印子目录列表
    if stats['directory_count'] > 0:
        print("子目录列表:")
        for directory in stats['directories']:
            print(f"  {directory}")
    
    # 打印文件类型分布
    print("\n文件类型分布:")
    for ext, count in stats["extension_counts"].items():
        print(f"  {ext} ({processor.get_extension_type(ext)}): {count}文件")
    print(f"总文本长度: {stats['total_content_length']}字符")
    print(f"平均文件长度: {stats['average_file_length']:.2f}字符")
    
    # 处理所有文件
    print("\n开始处理所有文件...")
    results = processor.process_directory(recursive=True)
    
    # 打印处理结果摘要
    for result in results:
        print(f"\n文件: {result['filepath']}")
        print(f"类型: {processor.get_extension_type(result['extension'])}")
        print(f"内容长度: {result['content_length']}字符")
        
        if result.get("chunks"):
            print(f"分块数量: {result['chunk_count']}")
            print(f"平均分块长度: {result['average_chunk_length']:.2f}字符")
        else:
            print(f"分块失败: {result.get('chunk_error', '未知错误')}")