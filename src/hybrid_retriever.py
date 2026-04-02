from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retriever import VectorRetriever
from src.schemas import RetrievedChunk


class BM25Retriever:
    """
    关键词检索器（BM25）：
    1. 直接复用现有 Chroma 向量库中的 documents + metadatas
    2. 在内存中构建 BM25 索引
    3. 返回和 VectorRetriever 一致的 RetrievedChunk 结构

    设计目标：
    - 不改 ingestion.py
    - 不新增第二套持久化索引
    - 优先补强短编号、数字边界、精确规则类题目
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        score_threshold: Optional[float] = None,
        k1: Optional[float] = None, # BM25 参数（词频权重）
        b: Optional[float] = None,  # BM25 参数（长度归一化）
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.vector_store = self.vector_retriever.vector_store

        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else float(getattr(settings, "score_threshold", 0.0))
        )
        self.k1 = k1 if k1 is not None else float(getattr(settings, "bm25_k1", 1.5))
        self.b = b if b is not None else float(getattr(settings, "bm25_b", 0.75))

        self.documents = self._load_documents_from_vector_store()
        if not self.documents:
            raise ValueError(
                "BM25Retriever 初始化失败：未从向量库中读取到任何文档。"
                "请先确认已运行 ingestion.py，并成功写入 Chroma。"
            )

        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def _load_documents_from_vector_store(self) -> List[Dict[str, Any]]:
        """
        从现有 Chroma 向量库读取全部 chunk。
        这样 BM25 与向量检索使用的是同一份知识库，避免对不齐。
        """
        payload = self.vector_store.get(include=["documents", "metadatas"])

        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        ids = payload.get("ids") or []

        results: List[Dict[str, Any]] = []

        for idx, text in enumerate(documents):
            metadata = (metadatas[idx] if idx < len(metadatas) else {}) or {}
            raw_source = metadata.get("source", "")
            source_name = Path(raw_source).name if raw_source else "unknown"

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = ids[idx] if idx < len(ids) else f"bm25_chunk_{idx:05d}"

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text or "",
                    "source": source_name,
                    "metadata": metadata,
                }
            )

        return results

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def _prepare_bm25_query(query: str) -> str:
        """
        把给向量检索用的结构化 query，压缩成更适合 BM25 的纯内容 query。
        """
        text = query or ""

        # 去掉包装词
        text = text.replace("题型：单选题", " ")
        text = text.replace("题型：多选题", " ")
        text = text.replace("题目：", " ")
        text = text.replace("选项：", " ")

        # 去掉 A. / B. / C. / D. 这类标签
        text = re.sub(r"\b[A-Z][\.\、\)\）:：]\s*", " ", text)

        # 合并空白
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """
        针对考试题做的轻量 tokenizer：
        - 保留数字串：246 / 1245 / 2026 / 1
        - 保留字母串
        - 保留中文串
        - 中文串再用 jieba 切词
        - 尝试保留“数字+中文”混合短语，增强边界规则题匹配
        """
        normalized = cls._normalize_text(text)
        if not normalized:
            return []

        tokens: List[str] = []

        # 1) 基础块提取：数字 / 字母 / 中文
        base_chunks = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", normalized)
        for chunk in base_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            tokens.append(chunk)

            # 中文串再切一遍
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                for word in jieba.lcut(chunk):
                    word = word.strip()
                    if word:
                        tokens.append(word)

        # 2) 保留压紧后的文本，增强短编号/短短语匹配
        compact = re.sub(r"\s+", "", normalized)
        if compact:
            # 适度保留，不要太长
            if len(compact) <= 32:
                tokens.append(compact)

            # 典型组合：246债权催收标准 / 逾期1年以上 / 4个抓手
            mixed_patterns = re.findall(
                r"\d+[\u4e00-\u9fff]{1,10}"
                r"|[\u4e00-\u9fff]{1,10}\d+"
                r"|[\u4e00-\u9fff]{1,10}\d+[\u4e00-\u9fff]{1,10}",
                compact,
            )
            tokens.extend(mixed_patterns)

        # 3) 去重但保序
        seen = set()
        deduped_tokens: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                deduped_tokens.append(token)

        return deduped_tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        执行 BM25 检索，返回 top-k 结果。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空。")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        clean_query = self._prepare_bm25_query(query)
        query_tokens = self._tokenize(clean_query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            raw_score = float(scores[idx])

            # 过滤完全无命中的 0 分结果
            if raw_score <= self.score_threshold:
                continue

            item = self.documents[idx]
            metadata = dict(item["metadata"] or {})
            metadata.update(
                {
                    "retrieval_method": "bm25",
                    "bm25_score": raw_score,
                }
            )

            results.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    source=item["source"],
                    score=raw_score,
                    metadata=metadata,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def print_results(self, results: List[RetrievedChunk]) -> None:
        if not results:
            print("BM25 未检索到结果。")
            return

        for i, item in enumerate(results, start=1):
            print(f"\n===== BM25 Top {i} =====")
            print(f"Source   : {item.source}")
            print(f"Page     : {item.metadata.get('page', 'N/A')}")
            print(f"Chunk ID : {item.chunk_id or 'N/A'}")
            print(f"Score    : {item.score:.4f}")
            print(f"Content  : {item.text[:120]}...")
            print("=" * 60)


class HybridRetriever:
    """
    混合检索器：
    - 向量检索：负责语义召回
    - BM25 检索：负责精确 token / 数字 / 短编号匹配
    - 融合策略：先独立召回，再按“排名分 + 双路命中奖励”重排

    第一版原则：
    - 简单稳定
    - 易调试
    - 不引入额外 rerank 模型
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
        fusion_bonus: Optional[float] = None,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever(
            vector_retriever=self.vector_retriever
        )

        default_top_k = int(getattr(settings, "top_k", 5))
        self.vector_top_k = (
            vector_top_k
            if vector_top_k is not None
            else int(getattr(settings, "vector_top_k", default_top_k))
        )
        self.bm25_top_k = (
            bm25_top_k
            if bm25_top_k is not None
            else int(getattr(settings, "bm25_top_k", default_top_k))
        )
        self.hybrid_top_k = (
            hybrid_top_k
            if hybrid_top_k is not None
            else int(getattr(settings, "hybrid_top_k", default_top_k))
        )

        # 双路都命中时的额外奖励；第一版给一个明显但不过分的加分
        self.fusion_bonus = (
            fusion_bonus
            if fusion_bonus is not None
            else float(getattr(settings, "hybrid_fusion_bonus", 3.0))
        )

    @staticmethod
    def _make_dedupe_key(chunk: RetrievedChunk) -> str:
        """
        优先用 chunk_id 去重。
        如果极端情况下 chunk_id 为空，再退化到 source + page + text 前缀。
        """
        if chunk.chunk_id:
            return chunk.chunk_id

        page = chunk.metadata.get("page", "N/A")
        text_prefix = (chunk.text or "")[:80]
        return f"{chunk.source}|{page}|{text_prefix}"

    def _collect_candidates(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        融合逻辑：
        1. 分别记录向量排名和 BM25 排名
        2. 用 rank 分作为主分
        3. 双路都命中则加 fusion_bonus
        4. 最终把融合信息回写进 metadata
        """
        candidates: Dict[str, Dict[str, Any]] = {}

        # 先收集向量结果
        for rank, chunk in enumerate(vector_results, start=1):
            key = self._make_dedupe_key(chunk)
            record = candidates.setdefault(
                key,
                {
                    "chunk": chunk,
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )
            record["vector_rank"] = rank
            record["vector_score"] = chunk.score
            # 默认以向量结果中的 chunk 作为主对象
            record["chunk"] = chunk

        # 再收集 BM25 结果
        for rank, chunk in enumerate(bm25_results, start=1):
            key = self._make_dedupe_key(chunk)
            record = candidates.setdefault(
                key,
                {
                    "chunk": chunk,
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )
            record["bm25_rank"] = rank
            record["bm25_score"] = chunk.score

            # 如果该 chunk 只来自 BM25，保留 BM25 版本
            # 如果已有向量版本，不覆盖 text/source，只补充 bm25 信息
            if record["chunk"] is None:
                record["chunk"] = chunk

        fused_results: List[RetrievedChunk] = []

        for record in candidates.values():
            chunk = record["chunk"]
            vector_rank = record["vector_rank"]
            vector_score = record["vector_score"]
            bm25_rank = record["bm25_rank"]
            bm25_score = record["bm25_score"]

            methods: List[str] = []
            hybrid_score = 0.0

            if vector_rank is not None:
                # 排名越靠前，加分越高
                hybrid_score += float(self.vector_top_k - vector_rank + 1)
                methods.append("vector")

            if bm25_rank is not None:
                hybrid_score += float(self.bm25_top_k - bm25_rank + 1)
                methods.append("bm25")

            if vector_rank is not None and bm25_rank is not None:
                hybrid_score += self.fusion_bonus

            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "retrieval_method": "hybrid",
                    "retrieval_methods": methods,
                    "vector_rank": vector_rank,
                    "vector_score": vector_score,
                    "bm25_rank": bm25_rank,
                    "bm25_score": bm25_score,
                    "hybrid_score": hybrid_score,
                }
            )

            fused_results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=hybrid_score,
                    metadata=metadata,
                )
            )

        # 先按 hybrid_score 排序；如果并列，则优先双路命中
        fused_results.sort(
            key=lambda x: (
                x.score,
                len(x.metadata.get("retrieval_methods", [])),
                -(x.metadata.get("vector_rank") or 9999),
                -(x.metadata.get("bm25_rank") or 9999),
            ),
            reverse=True,
        )

        return fused_results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        执行混合检索，返回 top-k 融合结果。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空。")

        final_top_k = top_k if top_k is not None else self.hybrid_top_k
        if final_top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=self.vector_top_k,
        )
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=self.bm25_top_k,
        )

        fused_results = self._collect_candidates(vector_results, bm25_results)
        return fused_results[:final_top_k]

    def print_results(self, results: List[RetrievedChunk]) -> None:
        if not results:
            print("HybridRetriever 未检索到结果。")
            return

        for i, item in enumerate(results, start=1):
            print(f"\n===== Hybrid Top {i} =====")
            print(f"Source    : {item.source}")
            print(f"Page      : {item.metadata.get('page', 'N/A')}")
            print(f"Chunk ID  : {item.chunk_id or 'N/A'}")
            print(f"Score     : {item.score:.4f}")
            print(
                "Methods   : "
                f"{', '.join(item.metadata.get('retrieval_methods', [])) or 'N/A'}"
            )
            print(f"V-Rank    : {item.metadata.get('vector_rank', 'N/A')}")
            print(f"BM25-Rank : {item.metadata.get('bm25_rank', 'N/A')}")
            print(f"Content   : {item.text[:120]}...")
            print("=" * 60)


if __name__ == "__main__":
    retriever = HybridRetriever()

    query = "246债权催收标准里，逾期几个月发律师函？"
    results = retriever.retrieve(query=query, top_k=5)
    retriever.print_results(results)