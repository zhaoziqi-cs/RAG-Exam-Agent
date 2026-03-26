import re
from difflib import SequenceMatcher
from typing import List, Tuple

from src.config import settings
from src.schemas import AugmentedContext, ExamQuestion, RetrievedChunk

class QueryAugmentor:
    """
    Query Augmentation：
    把“题干 + 选项”整理成更适合向量检索的 query。

    第一版原则：
    1. 不做 LLM query rewrite
    2. 不引入额外 NLP 依赖
    3. 只做轻量清洗 + 题干/选项拼接
    """

    _OPTION_PREFIXE_PATTERN = re.compile(r"^\s*[A-Za-z]\s*[\.\、\)\）:：]\s*")

    def __init__(self, include_question_type: bool = True) -> None:
        self.include_question_type = include_question_type

    def build_query(self, exam_question: ExamQuestion) -> str:
        """
        输入结构化题目，输出检索 query。
        """
        exam_question.validate()

        question_text = self._clean_text(exam_question.question)
        option_texts = self._format_options(exam_question.options)

        parts: List[str] = []

        if self.include_question_type:
            qtype = "单选题" if exam_question.question_type == "single" else "多选题"
            parts.append(f"题型：{qtype}")
        
        parts.append(f"题目：{question_text}")

        if option_texts:
            parts.append("选项：" + "；".join(option_texts))

        return "\n".join(parts).strip()

    @classmethod
    def _format_options(cls, options: List[str]) -> List[str]:
        """
        把 options 统一格式化成：
        A. xxx
        B. xxx
        ...
        """
        formatted: List[str] = []

        for idx, option in enumerate(options):
            option = cls._clean_text(option)
            option = cls._strip_option_prefix(option)

            if not option:
                continue

            label = cls._index_to_label(idx)
            formatted.append(f"{label}. {option}")

        return formatted

    @classmethod
    def _strip_option_prefix(cls, text: str) -> str:
        """
        如果用户传入的 option 本身已经带了 A./B./C. 前缀，这里先去掉，
        避免后面变成 “A. A. xxx”。
        """
        return cls._OPTION_PREFIXE_PATTERN.sub("", text).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        轻量文本清洗：
        - 合并多余空白
        - 去除首尾空格
        """
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _index_to_label(index: int) -> str:
        if 0 <= index < 26:
            return chr(ord("A") + index)
        return f"OPTION_{index + 1}"


class ContextAugmentor:
    """
    Context Augmentation：
    把 retriever 返回的 RetrievedChunk 列表整理成给 LLM 的最终 context。

    第一版原则：
    1. 保留 retriever 的原始排序，不在这里重新排序
    2. 做轻量去重，减少重复 chunk 噪声
    3. 给每个 chunk 补来源标签
    4. 控制总上下文长度，避免 prompt 过长
    """

    def __init__(
            self,
            max_context_chars: int = settings.max_context_chars,
            max_reference_chunks: int = settings.max_reference_chunks,
            duplicate_similarity: float = 0.92,
    ) -> None:
        self.max_context_chars = max_context_chars
        self.max_reference_chunks = max_reference_chunks
        self.duplicate_similarity = duplicate_similarity

    def build_context(
            self,
            exam_question: ExamQuestion,
            retrieved_chunks: List[RetrievedChunk],
    ) -> AugmentedContext:
        """
        输入题目 + 检索结果，输出 AugmentedContext。
        """
        exam_question.validate()

        if not retrieved_chunks:
            return AugmentedContext(
                question=exam_question.question,
                context="未检索到可用的知识库片段。",
                references=[],
            )
        
        deduped_chunks = self._deduplicate_chunks(retrieved_chunks)
        candidate_chunks = deduped_chunks[: self.max_reference_chunks]

        context_text, used_references = self._compose_context(candidate_chunks)
        return AugmentedContext(
            question=exam_question.question,
            context=context_text,
            references=used_references,
        )
    
    def _deduplicate_chunks(
            self,
            chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        去重策略：
        1. chunk_id 重复，直接去掉
        2. 文本高度相似，也去掉
        3. 保留先出现的 chunk（默认认为 retriever 已按相关性排好序）
        """
        unique_chunks: List[RetrievedChunk] = [] #最终返回的去重后列表
        seen_chunk_ids = set() #记录已经出现过的 ID 硬过滤
        seen_normalized_texts: List[str] = [] #记录已经出现过的文本的轻量归一化结果，用于相似度比较

        for chunk in chunks:
            
            if chunk.chunk_id in seen_chunk_ids:
                continue

            normalized_text = self._normalize_text(chunk.text)
            if not normalized_text:
                continue
            
            is_duplicate = False
            for old_text in seen_normalized_texts:
                if self.is_similar(normalized_text, old_text):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            unique_chunks.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            seen_normalized_texts.append(normalized_text)
        return unique_chunks

    def _compose_context(
            self,
            chunks: List[RetrievedChunk],
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        把多个 chunk 按标准格式拼接成 context，
        同时严格限制最大字符数。
        """
        sections: List[str] = [] #每个 chunk 的格式化文本列表
        used_references: List[RetrievedChunk] = [] #最终被包含在 context 中的 chunk 列表
        current_length = 0

        for idx, chunk in enumerate(chunks):
            section = self._format_chunk_section(idx, chunk)
            section_length = len(section) #这里简单按字符数计算长度，实际可以更复杂一些，比如按 token 数
            if current_length + section_length > self.max_context_chars:
                # 如果连第一段都放不下，就硬截断第一段，避免直接返回空 context
                if not sections:
                    truncated = self._truncate_text(section, self.max_context_chars)
                    if truncated:
                        sections.append(truncated)
                        used_references.append(chunk)
                break
            # 未超过长度限制，正常添加
            sections.append(section)
            used_references.append(chunk)
            current_length += section_length + 2 #加上分隔符长度
            # 如果最后列表还是空的（说明 chunks 本身就是空的，或者截断也失败了），就返回默认提示语
            if not sections:
                return "未检索到可用的知识库片段。", []
        context_text = "\n\n".join(sections).strip()
        return context_text, used_references
    
    def _format_chunk_section(self,index: int, chunk: RetrievedChunk) -> str:
        """
        每个 chunk 的格式示例：
        [证据 1] 来源：制度手册.pdf | 第 3 页 | chunk_id=abc123
        xxxxxxxxxxxxx
        """
        header_parts = [f"证据 {index + 1}", f"来源：{chunk.source or 'unknown'}"]
        page = chunk.metadata.get("page")
        if page is not None:
            header_parts.append(f"第 {page} 页")
        if chunk.chunk_id:
            header_parts.append(f"chunk_id={chunk.chunk_id}")
        header = " | ".join(header_parts)
        body = self._clean_chunk_text(chunk.text)
        return f"{header}\n{body}"

    @staticmethod
    def _clean_chunk_text(text: str) -> str:
        """
        给 LLM 的 context 不适合带太多空行和杂乱空白。
        """
        text = text or ""
        text = text.replace("\u3000", " ")
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    
    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        用于去重比较的轻量归一化：
        - 转小写
        - 去掉多余空白
        """
        text = re.sub(r"\s+", " ", text or "").strip().lower()
        return text

    def is_similar(self, text_a: str, text_b: str) -> bool:
        """
        判断两个文本是否高度相似，超过设定的相似度阈值。
        这里用 SequenceMatcher 计算文本相似度，简单有效。
        """
        if not text_a or not text_b:
            return False
        if text_a == text_b:
            return True
        if abs(len(text_a) - len(text_b)) / max(len(text_a), len(text_b)) > 0.5:
            # 长度差距过大，直接认为不相似，避免不必要的相似度计算
            return False
        similarity = SequenceMatcher(None, text_a, text_b).ratio()
        return similarity >= self.duplicate_similarity


class ExamAugmentor:
    """
    对外统一入口：
    - build_retrieval_query: 给 retriever 用
    - build_augmented_context: 给 prompt / qa_pipeline 用
    """
    def __init__(
        self,
        query_augmentor: QueryAugmentor = None,
        context_augmentor: ContextAugmentor = None,
    ) -> None:
        self.query_augmentor = query_augmentor or QueryAugmentor()
        self.context_augmentor = context_augmentor or ContextAugmentor()
    
    def build_retrieval_query(self, exam_question: ExamQuestion) -> str:
        return self.query_augmentor.build_query(exam_question)
    
    def build_augmented_context(
        self,
        exam_question: ExamQuestion,
        retrieved_chunks: List[RetrievedChunk],
    ) -> AugmentedContext:
        return self.context_augmentor.build_context(exam_question, retrieved_chunks)
    
    def prepare(
        self,
        exam_question: ExamQuestion,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[str, AugmentedContext]:
        """
        便于 qa_pipeline 一步拿到：
        1. retrieval query
        2. augmented context
        """
        query = self.build_retrieval_query(exam_question)
        context = self.build_augmented_context(exam_question, retrieved_chunks)
        return query, context

if __name__ == "__main__":
    question = ExamQuestion(
        question='以下哪项不属于"资产基础质量"维度的筛选指标？',
        options=[
            "A、平衡债权与负债比例",
            "B、区域消费能力",
            "C、化债损失最小化",
            "D、可化债权与化债损失全覆盖"
        ],
        question_type="single",
        question_id="demo_001"
    )

    from src.retriever import VectorRetriever
    query_augmentor = QueryAugmentor()
    retriever = VectorRetriever()
    context_augmentor = ContextAugmentor()

    retrieved_query = query_augmentor.build_query(question)
    retrieved_chunks = retriever.retrieve(retrieved_query,top_k=5)
    print("===== Retrieved Chunks =====")
    print("chunk count:", len(retrieved_chunks))
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n--- chunk {i} ---")
        print("source:", chunk.source)
        print("chunk_id:", chunk.chunk_id)
        print("score:", chunk.score)
        print("page:", chunk.metadata.get("page"))
        print("text:", chunk.text[:150])

    augmented_context = context_augmentor.build_context(
        exam_question=question,
        retrieved_chunks=retrieved_chunks
    )

    print("\n===== AugmentedContext =====")
    print("question:", augmented_context.question)
    print("reference count:", len(augmented_context.references))
    print("context length:", len(augmented_context.context))
    print("\ncontext preview:\n")
    print(augmented_context.context[:1000])