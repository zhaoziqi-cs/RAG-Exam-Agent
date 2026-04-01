from __future__ import annotations
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import DocumentChunk

def clean_text(
        text: str,
        backend: str | None = None
        ) -> str:
    """
    基础文本清洗：
    1. 去掉首尾空白
    2. 合并连续空格 / 制表符
    3. 合并过多空行
    """
    backend = (backend or settings.pdf_loader_backend).lower()
    if not text:
        return ""
    text = text.replace("\ufeff", "")    # 去掉字符串里的 BOM（字节顺序标记）字符
    text = text.replace("\u3000", " ")   # 全角空格替换为半角
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    if backend == "pypdfloader":
        return _clean_text_for_pypdf(text)

    if backend == "pymupdf4llm":
        return _clean_text_for_pymupdf4llm(text)

    raise ValueError(f"不支持的清洗 backend: {backend}")

def _clean_text_for_pypdf(text: str) -> str:
    """
    针对 PyPDFLoader 的脏文本：
    - 强清洗
    - 修复被空格拆开的中文/数字
    - 合并 PDF 硬换行
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 修复中文断裂：债 权 催 收 标 准 -> 债权催收标准
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    # 修复数字断裂：24 6 -> 246, 12 3 -> 123
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    # 普通硬换行并回句子（保留空段）
    text = re.sub(r"(?<![。！？；：.!?;\n])\n(?!\n)", "", text)

    return text.strip()

def _clean_text_for_pymupdf4llm(text: str) -> str:
    """
    针对 PyMuPDF4LLM 的结构化文本：
    - 轻清洗
    - 尽量保留 markdown / 段落 / 标题结构
    - 不强行合并中文之间的空格，不把单换行全部抹掉
    """
    # 去掉行尾空格
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 压缩过多空行，但保留段落
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 修复明显的短数字编号断裂：24 6 -> 246
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    # 对非常明显的“被空格拆开的中文标题”做轻修复
    # 债 权 催 收 标 准 -> 债权催收标准
    text = re.sub(
        r"((?:[\u4e00-\u9fff]\s+){2,}[\u4e00-\u9fff])",
        lambda m: re.sub(r"\s+", "", m.group(1)),
        text,
    )

    return text.strip()
    
def get_pdf_files(data_dir: Path | None = None) -> List[Path]:
    """
    获取指定目录下的所有 PDF 文件路径。
    """
    data_dir = data_dir or settings.data_dir
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到任何 PDF 文件。请确保数据目录正确且包含 PDF 文件。")
    
    return pdf_files

def _load_pdf_with_pypdfloader(pdf_path: Path) -> List[LCDocument]:
    """
    读取单个 PDF，按页返回 LangChain Document 列表。
    每页会保留 loader 给出的 metadata（通常包含 source/page）。
    """
    loader = PyPDFLoader(str(pdf_path))
    docs= loader.load()
    
    # 清洗文本内容
    cleaned_docs :List[LCDocument] = []
    for doc in docs:
        cleaned = clean_text(doc.page_content)
        if not cleaned:
            continue

        metadata = dict(doc.metadata or {})
        metadata["source_file"] = pdf_path.name
        metadata["source_path"] = str(pdf_path)
        cleaned_docs.append(LCDocument(page_content=cleaned, metadata=metadata))

    return cleaned_docs

def _load_pdf_with_pymupdf4llm(pdf_path: Path) -> List[LCDocument]:
    """
    使用 PyMuPDF4LLM 抽取 PDF。
    它更适合保留 Markdown / layout 结构，常常对表格、多栏、标题更友好。:contentReference[oaicite:2]{index=2}
    """
    loader = PyMuPDF4LLMLoader(str(pdf_path))
    docs = loader.load()

    cleaned_docs: List[LCDocument] = []
    for doc in docs:
        cleaned = clean_text(doc.page_content)
        if not cleaned:
            continue

        metadata = dict(doc.metadata or {})
        metadata["source_file"] = pdf_path.name
        metadata["source_path"] = str(pdf_path)
        metadata["loader_backend"] = "pymupdf4llm"

        cleaned_docs.append(
            LCDocument(page_content=cleaned, metadata=metadata)
        )

    return cleaned_docs

def load_pdf_documents(
    pdf_path: Path,
    backend: Optional[str] = None,
) -> List[LCDocument]:
    """
    读取单个 PDF，支持两种 loader backend：
    - pypdfloader
    - pymupdf4llm
    """
    backend = (backend or settings.pdf_loader_backend).lower()

    if backend == "pypdfloader":
        return _load_pdf_with_pypdfloader(pdf_path)

    if backend == "pymupdf4llm":
        return _load_pdf_with_pymupdf4llm(pdf_path)

    raise ValueError(
        f"不支持的 pdf loader backend: {backend}，"
        f"可选值为 pypdfloader / pymupdf4llm"
    )

def load_all_pdfs(
        data_dir: Path,
        backend: Optional[str] = None,
) -> List[LCDocument]:
    """
    读取指定目录下的所有 PDF 文件，返回一个包含所有文档的列表。
    """
    pdf_files = get_pdf_files(data_dir)
    all_docs: List[LCDocument] = []
    for pdf_file in pdf_files:
        docs = load_pdf_documents(pdf_file, backend=backend)
        all_docs.extend(docs)
    
    if not all_docs:
        raise ValueError("PDF 已找到，但解析后没有得到有效文本。")

    return all_docs

def build_text_splitter(backend: str | None = None,) -> RecursiveCharacterTextSplitter:
    """
    根据 backend 构建不同的切分器：
    - pypdfloader: 适合脏文本
    - pymupdf4llm: 优先按 markdown / 结构边界切
    """
    effective_backend = (backend or settings.pdf_loader_backend).lower()
    
    if effective_backend == "pymupdf4llm":
        separators = list(settings.markdown_separators)
    else:
        separators = list(settings.separators)

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

def split_documents(
        raw_docs: Sequence[LCDocument],
        backend: str | None = None,
        ) -> List[DocumentChunk]:
    """
    将原始文档切分为项目内部统一的 DocumentChunk。
    """
    splitter = build_text_splitter(backend=backend)
    split_docs = splitter.split_documents(list(raw_docs))
    chunks: List[DocumentChunk] = []
    for idx, doc in enumerate(split_docs):
        metadata = dict(doc.metadata or {})
        source_file = metadata.get("source_file", "unknown.pdf")
        raw_page = metadata.get("page", None)

        # PyPDFLoader 的 page 往往是从 0 开始，这里转成更直观的 1-based 页码
        if isinstance(raw_page, int):
            page_number = raw_page + 1
        else:
            page_number = None

        chunk_id = build_chunk_id(
            source_file=source_file,
            page_number=page_number,
            chunk_index=idx,
        )

        metadata["chunk_index"] = idx
        metadata["chunk_id"] = chunk_id
        metadata["source"] = source_file
        metadata["page_number"] = page_number
        metadata["char_count"] = len(doc.page_content)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                source=source_file,
                metadata=metadata,
            )
        )
    
    if not chunks:
        raise ValueError("文本切分后未生成任何 chunk。")
    
    return chunks



def build_chunk_id(source_file: str, page_number: int | None, chunk_index: int) -> str:
    """
    构造稳定且可读的 chunk_id。
    """
    stem = Path(source_file).stem
    page_part = f"p{page_number}" if page_number is not None else "pNA"
    return f"{stem}_{page_part}_chunk{chunk_index:04d}"

def build_embedding_model() -> HuggingFaceEmbeddings:
    """
    构建 embedding 模型。
    """
    encode_kwargs = {}
    if settings.normalize_embeddings:
        encode_kwargs["normalize_embeddings"] = True
    
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs=encode_kwargs,
    )

def chunks_to_lc_ducuments(chunks: Sequence[DocumentChunk]) -> Tuple[List[LCDocument], List[str]]:
    """
    将项目内部 DocumentChunk 转成 LangChain Document，并提取 ids。
    """
    lc_docs: List[LCDocument] = []
    ids: List[str] = []
    for chunk in chunks:
        lc_docs.append(
            LCDocument(
                page_content=chunk.text, 
                metadata={
                    **chunk.metadata
                },
            )
        )
        ids.append(chunk.chunk_id)

    return lc_docs, ids

def build_vector_store(
        chunks: Sequence[DocumentChunk], 
        reset_collection: bool = True,
    ) -> Chroma:
    """
    将切好的 chunks 写入 Chroma 向量库。
    """
    settings.make_dirs()
    embedding_model = build_embedding_model()
    
    if reset_collection:
        try:
            existing_store = Chroma(
                collection_name=settings.collection_name,
                persist_directory=str(settings.vector_store_dir),
                embedding_function=embedding_model,
            )
            existing_store.delete_collection()
        except Exception:
            # 首次创建 / collection 不存在时忽略
            pass
    
    vector_store = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.vector_store_dir),
        embedding_function=embedding_model,
    )

    lc_docs, ids = chunks_to_lc_ducuments(chunks)
    vector_store.add_documents(documents=lc_docs, ids=ids)
    return vector_store

def ingest_pdfs_to_vector_store(
        data_dir: Path | None = None,
        reset_collection: bool = True,
        backend: str | None = None,
) -> Tuple[Chroma, List[DocumentChunk]]:
    """
    一站式入库流程：
    读取 PDF -> 清洗 -> 切分 -> 写入向量库
    """
    raw_docs = load_all_pdfs(data_dir, backend=backend)
    chunks = split_documents(raw_docs)
    vector_store = build_vector_store(chunks, reset_collection = reset_collection)
    return vector_store, chunks


if __name__ == "__main__":
    vector_store, chunks = ingest_pdfs_to_vector_store()
    print(f"入库完成，共写入 {len(chunks)} 个 chunks。")
    print(f"向量库存储目录：{settings.vector_store_dir}")
    print(f"collection_name：{settings.collection_name}")
    print("Collection count:", vector_store._collection.count())

