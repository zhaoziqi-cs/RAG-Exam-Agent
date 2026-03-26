from src.augmentor import ExamAugmentor
from src.retriever import VectorRetriever
from src.qa_pipeline import RAGQAPipeline
from src.schemas import ExamQuestion
from src.llm import call_llm


def main():
    question = ExamQuestion(
        question="根据公司报销流程，第一步应该做什么？",
        options=[
            "领导审批",
            "提交申请",
            "财务打款",
            "归档"
        ],
        question_type="single",
        question_id="demo_001"
    )

    retriever = VectorRetriever()
    augmentor = ExamAugmentor()
    pipeline = RAGQAPipeline(
        retriever=retriever,
        augmentor=augmentor
    )

    # 1. 生成 prompt
    retrieval_query = augmentor.build_retrieval_query(question)
    retrieved_chunks = retriever.retrieve(retrieval_query, top_k=5)
    augmented_context = augmentor.build_augmented_context(
        exam_question=question,
        retrieved_chunks=retrieved_chunks
    )
    prompt = pipeline.build_prompt(
        exam_question=question,
        augmented_context=augmented_context
    )

    # 2. 保存 prompt，方便排查
    with open("debug_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    print("===== Prompt Ready =====")
    print("prompt length:", len(prompt))
    print("prompt 已写入 debug_prompt.txt")

    # 3. 调用 LLM，只看原始返回
    print("\n===== Calling LLM =====")
    raw_output = call_llm(prompt)

    print("\n===== Raw LLM Output =====")
    print(raw_output)

    with open("debug_llm_output.txt", "w", encoding="utf-8") as f:
        f.write(raw_output if raw_output else "")

    print("\nraw_output 已写入 debug_llm_output.txt")


if __name__ == "__main__":
    main()