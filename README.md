# RAG Exam Agent

一个面向**企业知识库自动答题 / 选择题问答**场景的 RAG 项目。  
当前版本已完成基础工程结构、PDF 入库与切块、向量化存储、基础语义检索模块，后续将继续补全 Query Augmentation、LLM Answering 和 End-to-End QA Pipeline。

---

## 1. 项目目标
本项目希望构建一个基础的知识库答题系统，适用于如下场景：

- 输入：题干 + 选项
- 检索：从企业制度、流程文档、知识手册中召回相关内容
- 生成：结合检索结果，由大模型输出答案、解析与来源
- 输出：结构化答题结果，支持后续评测与扩展

**RAG 基础链路** 搭建：

1. 文档入库（Ingestion）
2. 向量检索（Retrieval）
3. 后续补充增强（Augmentation）
4. 大模型生成（Generation）

---

## 📁**2. 项目架构**

```
           ┌──────────────┐
           │ PDF知识库     │
           └──────┬───────┘
                  ↓
        文本切分 + 向量化（embedding）
                  ↓
           FAISS向量数据库
                  ↓
       ┌──────── 查询流程 ────────┐
       ↓                         ↓
  题目解析                Query Rewrite
       ↓                         ↓
       └────→ 向量检索 TopK ←────┘
                    ↓
               Rerank（可选）
                    ↓
              Prompt构建
                    ↓
                 LLM
                    ↓
     结构化答案（A/B/C/D + 理由）
                    ↓
      自动化 or 人工点击
```
 ##  **📁 项目结构（第一版）**

```JSON
rag_exam_agent/
├── data/
├── src/
│   ├── ingestion.py        # 多PDF入库、切块、metadata
│   ├── retriever.py        # 基础检索
│   ├── augmentor.py        # 增强层：query/context增强
│   ├── llm.py              # LLM API 调用
│   ├── qa_pipeline.py      # 组装总流程
│   └── schemas.py          # 输入输出结构定义（后面可选）
├── vector_store/
├── app.py
├── requirements.txt
└── README.md
```

本项目不是朴素 RAG，而是加入了面向考试场景的 augmentation layer，对 query、context 与 answer schema 做了增强。

标准化数据流（schemas.py）
原始文档
-> DocumentChunk
-> RetrievedChunk
-> AugmentedContext
-> QAResult

ingestion.py
1. 读取 data/ 下的多个 PDF
2. 做文本清洗和切块
3. 补齐 metadata
4. 写入向量库 Chroma