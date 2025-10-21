"""
文本分块模块

该模块实现了针对中文文本的智能分块算法，能够在保持语义完整性的前提下，
将长文本分割成大小合适、带有重叠的文本块。这是知识库构建过程中的关键步骤，
合理的分块策略直接影响后续向量化和检索的效果。
"""
import hanlp
import re
from typing import List, Tuple

from config.settings import CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH

class ChineseTextChunker:
    """
    中文文本分块器类
    
    该类实现了基于HanLP分词的中文文本智能分块算法。设计考虑了中文的语言特性，
    通过以下核心策略实现高质量分块：
    1. 基于词和句子边界进行分块，避免截断语义
    2. 支持超长文本的预分割处理
    3. 实现相邻块之间的重叠，避免信息丢失
    4. 提供文本统计和诊断功能
    
    分块算法优势：
    - 保持句子完整性，提升语义连贯性
    - 针对超长文档的分层次处理
    - 鲁棒性设计，处理各种异常情况
    - 可配置的分块参数，适应不同场景
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, max_text_length: int = MAX_TEXT_LENGTH):
        """
        初始化中文文本分块器
        
        参数：
            chunk_size: 每个文本块的目标大小（分词数量）
            overlap: 相邻文本块的重叠大小（分词数量）
            max_text_length: HanLP处理的最大文本长度，超过此长度将进行预分割
            
        实现思路：
        1. 验证参数有效性（chunk_size必须大于overlap）
        2. 保存配置参数
        3. 加载HanLP分词器（使用中文预训练模型）
        4. 设置默认值来自配置文件
        
        设计考虑：
        - 使用Coarse-Electra-Small-ZH模型进行中文分词，平衡速度和准确性
        - 设置最大文本长度限制，避免内存溢出
        - 提供合理的默认参数，减少使用门槛
        """
        # 参数验证
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")
            
        # 保存配置参数
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        
        # 加载HanLP中文分词器
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        
    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        """
        批量处理多个文件的内容
        
        参数：
            file_contents: 文件名和内容的元组列表
            
        返回：
            包含文件名、内容和分块结果的元组列表
            
        实现思路：
        1. 遍历每个文件的内容
        2. 对每个文件分别调用chunk_text方法进行分块
        3. 将结果整合为统一的格式返回
        4. 保持原始内容，便于后续引用和验证
        
        业务意义：
        - 提供批量处理接口，简化多文件处理逻辑
        - 统一返回格式，便于后续处理
        - 保留原始文件名和内容，便于追踪
        """
        results = []
        for filename, content in file_contents:
            # 对每个文件调用分块方法
            chunks = self.chunk_text(content)
            # 保存结果
            results.append((filename, content, chunks))
        return results
    
    def _preprocess_large_text(self, text: str) -> List[str]:
        """
        预处理过大的文本，将其分割成较小的段落
        
        参数：
            text: 原始文本
            
        返回：
            分割后的文本段落列表
            
        实现思路：
        1. 首先检查文本长度是否超过处理上限
        2. 计算合适的段落目标大小
        3. 按段落（\n\n）分割文本
        4. 如果段落数过少，尝试按行（\n）分割
        5. 重新组合段落，确保大小合适
        6. 对超长段落进行特殊处理
        
        优化策略：
        - 目标段落大小设置为最大长度的一半，避免过于碎片化
        - 至少保持10000字符的最小段落大小
        - 保留段落结构，尽量不破坏文本的原始组织
        """
        # 文本长度检查
        if len(text) <= self.max_text_length:
            return [text]
        
        # 计算合适的段落大小
        target_segment_size = min(self.max_text_length, max(10000, self.max_text_length // 2))
        
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        
        # 如果段落数量很少，尝试按单个换行符分割
        if len(paragraphs) < 5:
            paragraphs = text.split('\n')
        
        # 重新组合段落，确保每个段落不超过目标大小
        processed_segments = []
        current_segment = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果当前段落本身就超长，需要进一步分割
            if len(para) > target_segment_size:
                # 先保存当前积累的内容
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                
                # 分割超长段落
                split_paras = self._split_long_paragraph(para, target_segment_size)
                processed_segments.extend(split_paras)
                
            else:
                # 检查添加当前段落是否会超长
                if len(current_segment) + len(para) + 2 > target_segment_size:  # +2 for \n\n
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para
        
        # 添加最后的segment
        if current_segment:
            processed_segments.append(current_segment)
        
        return processed_segments
    
    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        """
        分割超长段落的辅助方法
        
        参数：
            text: 超长段落文本
            max_size: 最大分割大小
            
        返回：
            分割后的段落列表
            
        实现思路：
        1. 首先检查文本长度
        2. 按句子边界（句号、感叹号、问号等）分割
        3. 重新组合句子和标点符号
        4. 如果无法找到句子边界，回退到固定长度分割
        5. 处理单个句子超长的特殊情况
        6. 确保最终生成的段落不超过最大长度
        
        算法亮点：
        - 尝试在自然语言边界分割，保持语义完整性
        - 处理标点符号，避免句子结构破坏
        - 多层回退机制，确保任何情况下都能完成分割
        """
        # 文本长度检查
        if len(text) <= max_size:
            return [text]
        
        # 按句子分割，保留标点符号
        sentences = re.split(r'([。！？.!?])', text)
        
        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)
        
        # 如果没有找到句子边界，按固定长度分割
        if not combined_sentences:
            result = []
            for i in range(0, len(text), max_size):
                result.append(text[i:i + max_size])
            return result
        
        # 重新组合句子，确保不超过最大长度
        segments = []
        current_segment = ""
        
        for sentence in combined_sentences:
            # 如果单个句子就超长，强制分割
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                
                # 按固定长度分割超长句子
                for i in range(0, len(sentence), max_size):
                    segments.append(sentence[i:i + max_size])
            else:
                # 检查添加当前句子是否会超长
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence
        
        # 添加最后的segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _safe_tokenize(self, text: str) -> List[str]:
        """
        安全的分词方法，处理可能的异常
        
        参数：
            text: 要分词的文本
            
        返回：
            分词结果列表
            
        实现思路：
        1. 检查文本长度是否在处理范围内
        2. 如果文本过长，直接按字符分割
        3. 尝试使用HanLP进行分词
        4. 异常捕获，确保程序稳定性
        5. 返回空列表作为无效输入的结果
        
        健壮性设计：
        - 多层防御措施，避免程序崩溃
        - 合理的默认行为，即使在异常情况下也能继续工作
        - 对输入进行验证，提前处理异常情况
        """
        try:
            # 检查文本长度
            if len(text) > self.max_text_length:
                return list(text)
            
            # 调用HanLP分词器
            tokens = self.tokenizer(text)
            return tokens if tokens else []
        except Exception:
            # 异常情况下按字符分割作为回退
            return list(text)
        
    def chunk_text(self, text: str) -> List[List[str]]:
        """
        将单个文本分割成块
        
        参数：
            text: 要分割的文本
            
        返回：
            分割后的文本块列表，每个块是token列表
            
        实现思路：
        1. 处理空文本或过短的文本
        2. 对超长文本进行预处理分割
        3. 对每个分割后的段落分别进行分块处理
        4. 整合所有段落的分块结果
        
        设计考虑：
        - 保持API简洁，内部处理复杂逻辑
        - 支持各种长度的文本处理
        - 确保语义完整性
        - 提供一致的返回格式
        """
        # 处理空文本或太短的文本
        if not text or len(text) < self.chunk_size / 10:
            tokens = self._safe_tokenize(text)
            return [tokens] if tokens else []
        
        # 预处理过大文本，分割成多个段落
        text_segments = self._preprocess_large_text(text)
        
        # 处理每个文本段落
        all_chunks = []
        for segment in text_segments:
            # 对每个段落单独进行分块
            segment_chunks = self._chunk_single_segment(segment)
            all_chunks.extend(segment_chunks)
        
        return all_chunks
    
    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        """
        处理单个文本段落的分块（核心算法）
        
        参数：
            text: 单个文本段落
            
        返回：
            分块结果列表
            
        实现思路：
        1. 首先对整个段落进行分词
        2. 使用滑动窗口算法进行分块
        3. 尝试在句子边界结束块，保持语义完整性
        4. 实现智能重叠，考虑句子边界
        5. 防止无限循环
        
        算法亮点：
        - 基于句子边界的智能分块，避免截断语义
        - 考虑重叠部分的语义连贯性
        - 自适应的块大小，略微超出目标大小以保持句子完整性
        - 健壮性设计，防止各种边界情况
        """
        if not text:
            return []
            
        # 先将整个文本分词
        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []
        
        chunks = []
        start_pos = 0
        
        # 滑动窗口算法
        while start_pos < len(all_tokens):
            # 确定当前块的结束位置
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))
            
            # 如果不是最后一块，尝试在句子边界结束
            if end_pos < len(all_tokens):
                # 寻找句子结束位置
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:  # 允许略微超出
                    end_pos = sentence_end
            
            # 提取当前块
            chunk = all_tokens[start_pos:end_pos]
            if chunk:  # 确保块不为空
                chunks.append(chunk)
            
            # 计算下一块的起始位置（考虑重叠）
            if end_pos >= len(all_tokens):
                break
                
            # 寻找重叠的起始位置
            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(all_tokens, overlap_start)
            
            # 如果找到合适的句子开始位置，使用它；否则使用计算的重叠位置
            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start
                
            # 防止无限循环
            if start_pos >= end_pos:
                start_pos = end_pos
        
        return chunks
    
    def _is_sentence_end(self, token: str) -> bool:
        """
        判断token是否为句子结束符
        
        参数：
            token: 要检查的token
            
        返回：
            bool: 是否为句子结束符
            
        实现思路：
        - 基于中文标点符号判断
        - 简单直接，高效运行
        """
        # 中文句号、感叹号、问号
        return token in ['。', '！', '？']
    
    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """
        从指定位置向后查找句子结束位置
        
        参数：
            tokens: 分词后的token列表
            start_pos: 起始搜索位置
            
        返回：
            int: 句子结束的位置索引
            
        实现思路：
        - 从起始位置向后遍历token
        - 遇到句子结束符就返回下一个位置
        - 如果遍历到列表末尾，返回末尾索引
        - 线性搜索，简单高效
        """
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1  # 返回结束符后面的位置
        return len(tokens)  # 未找到，返回末尾
    
    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """
        从指定位置向前查找句子结束位置
        
        参数：
            tokens: 分词后的token列表
            start_pos: 起始搜索位置
            
        返回：
            int: 句子结束的位置索引
            
        实现思路：
        - 从起始位置向前遍历token
        - 遇到句子结束符就返回下一个位置
        - 如果遍历到列表开头，返回0
        - 线性搜索，简单高效
        """
        # 从start_pos-1开始向前搜索
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1  # 返回结束符后面的位置
        return 0  # 未找到，返回开头
    
    def get_text_stats(self, text: str) -> dict:
        """
        获取文本统计信息
        
        参数：
            text: 输入文本
            
        返回：
            包含文本统计信息的字典
            
        实现思路：
        1. 计算基本统计信息：长度、段落数、行数
        2. 判断是否需要预处理
        3. 估算需要的分块数量
        4. 如果需要预处理，计算预处理后的段数和最大段长度
        
        业务意义：
        - 提供文本特征的量化描述
        - 帮助评估分块策略的合理性
        - 支持日志记录和性能监控
        - 为后续处理提供参考信息
        """
        # 基本统计信息
        stats = {
            'text_length': len(text),  # 文本总长度
            'needs_preprocessing': len(text) > self.max_text_length,  # 是否需要预处理
            'estimated_chunks': max(1, len(text) // self.chunk_size),  # 估算分块数量
            'paragraphs': len(text.split('\n\n')),  # 段落数
            'lines': len(text.split('\n'))  # 行数
        }
        
        # 如果需要预处理，计算预处理后的信息
        if stats['needs_preprocessing']:
            segments = self._preprocess_large_text(text)
            stats['preprocessed_segments'] = len(segments)  # 预处理后的段数
            stats['max_segment_length'] = max(len(seg) for seg in segments) if segments else 0  # 最大段长度
            
        return stats