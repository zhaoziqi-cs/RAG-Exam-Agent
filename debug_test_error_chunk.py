from pathlib import Path
from src.ingestion import load_pdf_documents, split_documents
from src.config import settings

pdf_path = settings.data_dir / "财务资金体系应知应会核心要点汇编2025年9月.pdf"

raw_docs = load_pdf_documents(pdf_path)
chunks = split_documents(raw_docs)

page2_chunks = [
    c for c in chunks
    if c.source == pdf_path.name and c.metadata.get("page_number") == 2
]

print(f"page 2 共 {len(page2_chunks)} 个 chunk\n")

for i, chunk in enumerate(page2_chunks, start=1):
    print("=" * 100)
    print(f"[{i}] chunk_id={chunk.chunk_id}")
    print(f"page={chunk.metadata.get('page_number')}")
    print(f"char_count={chunk.metadata.get('char_count')}")
    print("-" * 100)
    print(chunk.text)
    print()