from pathlib import Path
from typing import Any, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.schemas import RetrievedChunk

class VectorRetriever:
    """
    基础检索器：
    1. 加载本地 Chroma 向量库
    2. 接收 query
    3. 返回 top-k 检索结果

    注意：这里只做“检索”，不做增强和生成。
    """
    def __init__(
            self,
            persist_directory: str = settings.vector_store_dir,
            collection_name: str = settings.collection_name,
            embedding_model_name: str = settings.embedding_model_name,
            device: str = settings.embedding_device,
            normalize_embeddings: bool = settings.normalize_embeddings,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.embeddings = self._load_embeddings()
        self.vector_store = self._load_vector_store()

    def _load_embeddings(self):
        """
        加载和 ingestion 阶段相同的 embedding 模型。
        一定要和入库时保持一致，否则向量空间不一致，检索会出问题。
        """
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
        )
    
    def _load_vector_store(self):
        """
        加载 Chroma 向量库。
        """
        persist_path = Path(self.persist_directory)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"向量库目录不存在：{persist_path.resolve()}\n"
                f"请先运行 ingestion.py 建库。"
            )

        return Chroma(
            persist_directory=str(persist_path),
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    @staticmethod
    def _format_result(doc: Any, score: Optional[float] = None) -> RetrievedChunk:
        """
        将 LangChain 的 Document 统一转成 RetrievedChunk，
        方便后续 augmentor / pipeline 直接使用标准数据结构。
        """
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        source_name = Path(source).name if source else "unknown"

        return RetrievedChunk(
            chunk_id=metadata.get("chunk_id", ""),
            text=doc.page_content,
            source=source_name,
            score=float(score) if score is not None else 0.0,
            metadata=metadata,
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        执行语义检索，返回 top-k 结果。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空。")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        query = query.strip()
        
        #相关度分数检索
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query, k=top_k
        )
        results = [
            self._format_result(doc, score) for doc, score in docs_and_scores
        ]
        
        return results
    
    def print_results(self, results: List[RetrievedChunk]) -> None:
        """
        方便调试：把检索结果打印出来看看是否合理。
        """
        if not results:
            print("未检索到结果。")
            return

        for i, item in enumerate(results, start=1):
            print(f"\n===== Top {i} =====")
            print(f"  Source    : {item.source}")
            print(f"  Page      : {item.metadata.get('page', 'N/A')}")
            print(f"  Chunk ID  : {item.chunk_id or 'N/A'}")
            print(f"  Score     : {item.score:.4f}")
            print(f"  Content   : {item.text[:100]}...")
            print("=" * 50)

if __name__ == "__main__":
    retriever = VectorRetriever()

    query = f"2026年财务资金工作的年度主题是什么"
    results = retriever.retrieve(query=query, top_k=5)
    retriever.print_results(results)