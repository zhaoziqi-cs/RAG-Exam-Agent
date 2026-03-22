from dataclasses import dataclass,field
from pathlib import Path
from typing import Tuple
import os

@dataclass
class Settings:
    """
    项目全局配置。
    先把所有“后面一定会频繁调的参数”集中到这里。
    """

    # =========================
    # 1. 路径配置
    # =========================
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = field(init=False)
    vector_store_dir: Path = field(init=False)

    # =========================
    # 2. 向量库配置
    # =========================
    vector_store_backend: str = "chroma"
    collection_name: str = "exam_knowledge_base"
    
    # =========================
    # 3. 向量化模型配置
    # =========================
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    # =========================
    # 4. 文本切分配置
    # =========================
    chunk_size: int = 500
    chunk_overlap: int = 100
    separators: Tuple[str, ...] = (
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "：",
        "，",
        " ",
        "",
    )

    # =========================
    # 5. 检索配置
    # =========================
    top_k: int = 5
    score_threshold: float = 0.0

    # =========================
    # 6. Context Augmentation 配置
    # =========================
    max_context_chars: int = 3000
    max_reference_chunks: int = 5
    # =========================
    # 7. 题目配置
    # =========================
    allowed_question_types: Tuple[str, ...] = ("single", "multiple")

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.vector_store_dir = self.project_root / "vector_store"

        # 允许后续通过环境变量覆盖配置
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", self.embedding_model_name
        )
        self.embedding_device = os.getenv(
            "EMBEDDING_DEVICE", self.embedding_device
        )

    def make_dirs(self) -> None:
        """确保必要的目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()