from src.qa_pipeline import RAGQAPipeline
from src.schemas import ExamQuestion

def main():
    pipeline = RAGQAPipeline()

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

    result = pipeline.answer(question)

    print("===== FINAL QA RESULT =====")
    print("answer:", result.answer)
    print("reason:", result.reason)
    print("source:", result.source)
    print("retrieved_chunks:", len(result.retrieved_chunks))
    print("metadata keys:", list(result.metadata.keys()))

    print("\n===== RAW LLM OUTPUT =====")
    print(result.metadata.get("raw_llm_output", ""))

    print("\n===== REFERENCES =====")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        print(f"\n--- chunk {i} ---")
        print("source:", chunk.source)
        print("page:", chunk.metadata.get("page"))
        print("score:", chunk.score)
        print("text:", chunk.text[:150])

if __name__ == "__main__":
    main()