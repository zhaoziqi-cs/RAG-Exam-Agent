from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DocumentChunk:
    """
    知识库切分后的最小检索单元。
    这是 ingestion 阶段的核心数据结构。
    """
    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedChunk:
    """
    检索阶段返回的结果。
    相比 DocumentChunk，多了来源source和相似度分数 score。
    """
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExamQuestion:
    """
    输入题目结构。
    用统一结构承接单选题 / 多选题，避免后面到处传裸 dict。
    """
    question: str
    options: List[str]
    question_type: str = "single"   # single / multiple
    question_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.question.strip():
            raise ValueError("question 不能为空")

        if self.question_type not in {"single", "multiple"}:
            raise ValueError("question_type 必须是 'single' 或 'multiple'")

        if not self.options:
            raise ValueError("options 不能为空")

        if len(self.options) < 2:
            raise ValueError("选择题至少需要 2 个选项")

@dataclass
class AugmentedContext:
    """
    Augmentation 阶段产物：
    把检索到的片段整理成最终给 LLM 的上下文。
    """
    question: str
    context: str
    references: List[RetrievedChunk] = field(default_factory=list)

@dataclass
class QAResult:
    """
    最终问答结果结构。
    对接你前面已经确定的输出格式。
    """
    answer: List[str]
    reason: str
    source: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "reason": self.reason,
            "source": self.source,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in self.retrieved_chunks
            ],
            "metadata": self.metadata,
        }
