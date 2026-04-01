# RAG Exam Agent

一个面向**企业知识库自动答题 / 选择题问答**场景的 RAG 项目。  

---

## 1. 项目简介

本项目聚焦于“**题干 + 选项 + 企业内部知识库 PDF**”的答题场景，目标是构建一个可扩展、可评测、可溯源的 RAG 系统。

与通用问答型 RAG 不同，本项目针对考试场景做了专门增强：

- 支持 **单选题 / 多选题**
- 支持 **题干 + 选项联合检索**
- 支持 **批量题目解析与批量评测**
- 支持 **检索证据、context、prompt、原始输出的调试导出**
- 支持 **不同 PDF 抽取 backend 的 A/B 对比**
- 为后续 **混合检索、Prompt 优化、自动化答题接入** 预留接口

---

## 2. 当前阶段成果

### 已完成主链路闭环

- 已完成多 PDF 入库与切块
- 已完成向量化存储与语义检索
- 已完成 Query / Context Augmentation
- 已完成 qa_pipeline 串联
- 已完成单题冒烟测试
- 已完成多题解析及批量答题
- 已完成批量结果导出
- 已完成错误回答的中间证据溯源

### 已完成本轮关键升级
- 新增 `parser.py`，支持 `input.txt -> List[ExamQuestion]`
- 新增 `batch_runner.py`，支持共享 pipeline 的批量答题
- 新增 `exporter.py`，支持批量结果导出
- 在 `qa_pipeline.py` 中补充调试信息：
  - `retrieval_query`
  - `retrieved_chunks_debug`
  - `used_references_debug`
  - `augmented_context`
  - `final_prompt`
  - `raw_llm_output`
- 对 PDF 抽取 backend 做了可切换改造：
  - `PyPDFLoader`
  - `PyMuPDF4LLM`

---

### 总结

在固定 30 道测试题上，项目已完成从“能跑通”到“能分析、能优化”的跨越：
- 初始批量正确率：**86.67%**
- 补充错误溯源能力后，定位到主要问题：
  - 未能召回正确知识块
  - PDF 表格/版式导致文本打散
  - 固定 size 切块导致 chunk 过杂
- 将 PDF 抽取 backend 改造成可切换后，验证：
  - `PyPDFLoader` 仍能稳定保持 **86.67%**
  - `PyMuPDF4LLM` 提升到 **93.33%**

剩余问题目标

后续将：

引入**混合检索（向量 + 关键词/BM25）**解决“**精确规则 / 边界定义**”类答错问题

逐选项判定 Prompt 再逐项判断 A/B/C/D 减少多选题漏选 / 误选

---

## 3. 项目目标

本项目希望构建一个基础的知识库答题系统，适用于如下场景：

- **输入**：题干 + 选项
- **检索**：从企业制度、流程文档、知识手册中召回相关内容
- **生成**：结合检索结果，由大模型输出答案、解析与来源
- **输出**：结构化答题结果，支持后续评测与扩展

---

## 4. 系统主链路

本项目采用的是**考试场景增强型 RAG**，而不是朴素问答式 RAG。

```text
题目输入
→ 输入规范化
→ Query Augmentation
→ Retriever
→ Context Augmentation
→ Prompt Builder
→ LLM
→ Output Parsing / Normalization
→ 最终答案
```

---

## 5. **项目架构**

```
           ┌──────────────┐
           │ PDF知识库     │
           └──────┬───────┘
                  ↓
        文本抽取（支持双 backend）
                  ↓
            文本清洗 + 切块
                  ↓
              Embedding
                  ↓
           Chroma 向量数据库
                  ↓
       ┌──────── 查询流程 ────────┐
       ↓                         ↓
  题目解析                Query Augmentation
       ↓                         ↓
       └────→ 向量检索 TopK ←────┘
                    ↓
          Context Augmentation
                    ↓
              Prompt构建
                    ↓
                 LLM
                    ↓
     结构化答案（A/B/C/D + reason + source）
                    ↓
      批量导出 / 人工核对 / 自动化扩展
```

---

## 6. **项目结构**

```
rag_exam_agent/
├── data/                  # 知识库 PDF
├── vector_store/          # 本地向量库
├── src/
│   ├── ingestion.py       # 多PDF入库、切块、metadata
│   ├── retriever.py       # 基础向量检索
│   ├── augmentor.py       # Query / Context 增强
│   ├── llm.py             # LLM API 调用
│   ├── qa_pipeline.py     # 单题总流程
│   ├── parser.py          # 多题解析
│   ├── batch_runner.py    # 批量答题执行器
│   ├── exporter.py        # 批量结果导出
│   ├── schemas.py         # 输入输出结构定义
│   └── config.py          # 全局配置
├── app.py
├── requirements.txt
└── README.md
```

---

## 7.**标准化数据流**
```text
原始文档
→ DocumentChunk
→ RetrievedChunk
→ AugmentedContext
→ QAResult
```

## 10. 核心模块说明

`ingestion.py`

负责知识库构建：

1. 读取 `data/` 下多个 PDF
2. 支持不同 PDF loader backend
3. 做文本清洗与切块
4. 补齐 metadata
5. 写入 Chroma 向量库

`retriever.py`

负责基础向量检索：

- 加载向量库
- 接收 query
- 返回 top-k 候选 chunk

`augmentor.py`

负责考试场景增强：

- Query Augmentation：将题干 + 选项拼成更适合检索的 query
- Context Augmentation：对检索结果去重、排序、拼接、控制长度

`qa_pipeline.py`

负责单题答题主流程：

- 调 retriever
- 调 augmentor
- 构造 prompt
- 调用 LLM
- 解析并归一化输出
- 保留完整调试信息用于错误分析

`parser.py`

负责把 `input.txt` 解析成结构化题目对象：

- 支持单选题 / 多选题
- 支持批量解析
- 输出 `ExamQuestion`

`batch_runner.py`

负责批量答题：

- 共享一个 pipeline
- 逐题调用 `answer()`
- 单题失败不影响全批次

`exporter.py`

负责导出结果：

- JSON 导出
- CSV 导出
- 支持保留调试字段，方便定位错误

---

## 11. 技术栈
Python
LangChain
Chroma
HuggingFace Embeddings
Sentence Transformers
BGE (BAAI/bge-small-zh-v1.5)
PDF Loader:
PyPDFLoader
PyMuPDF4LLM