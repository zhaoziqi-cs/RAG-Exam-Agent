# src/batch_runner.py
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.parser import parse_input_file
from src.qa_pipeline import RAGQAPipeline
from src.schemas import ExamQuestion, QAResult

class BatchQARunner:
    """
    批量答题执行器：
    - 只初始化一次 pipeline
    - 逐题调用 pipeline.answer()
    - 单题失败不影响后续题目
    """

    def __init__(self, pipeline: Optional[RAGQAPipeline] = None) -> None:
        self.pipeline = pipeline or RAGQAPipeline()

    def run(
        self,
        questions: List[ExamQuestion],
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        对一组 ExamQuestion 进行批量答题。

        Args:
            questions: 已解析好的题目列表
            limit: 仅跑前 N 题，便于调试；None 表示全量
            verbose: 是否打印过程日志

        Returns:
            dict:
            {
                "total_questions": ...,
                "success_count": ...,
                "failed_count": ...,
                "results": [...],
                "errors": [...],
                "metadata": {...}
            }
        """
        target_questions = questions[:limit] if limit is not None else questions

        batch_result: Dict[str, Any] = {
            "total_questions": len(target_questions),
            "success_count": 0,
            "failed_count": 0,
            "results": [],
            "errors": [],
            "metadata": {
                "started_at": datetime.now().isoformat(),
                "limit": limit,
            },
        }

        if verbose:
            print("=" * 80)
            print(f"开始批量答题，共 {len(target_questions)} 题")
            print("=" * 80)

        for idx, question in enumerate(target_questions, start=1):
            if verbose:
                print(
                    f"[{idx}/{len(target_questions)}] "
                    f"question_id={question.question_id} "
                    f"type={question.question_type}"
                )

            try:
                qa_result = self.pipeline.answer(question)
                result_record = self._build_success_record(question, qa_result)
                batch_result["results"].append(result_record)
                batch_result["success_count"] += 1

                if verbose:
                    print(
                        f"  -> success | answer={qa_result.answer} | source={qa_result.source}"
                    )

            except Exception as exc:
                error_record = self._build_error_record(question, exc)
                batch_result["errors"].append(error_record)
                batch_result["failed_count"] += 1

                if verbose:
                    print(f"  -> failed  | error={exc}")

        batch_result["metadata"]["finished_at"] = datetime.now().isoformat()

        if verbose:
            print("=" * 80)
            print(
                f"批量答题结束：success={batch_result['success_count']} | "
                f"failed={batch_result['failed_count']}"
            )
            print("=" * 80)

        return batch_result

    def run_from_file(
        self,
        file_path: str,
        limit: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        从 input.txt 直接解析并批量答题。
        """
        questions = parse_input_file(file_path)
        result = self.run(questions=questions, limit=limit, verbose=verbose)
        result["metadata"]["source_file"] = file_path
        return result

    @staticmethod
    def _build_success_record(
        question: ExamQuestion,
        qa_result: QAResult,
    ) -> Dict[str, Any]:
        return {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "options": question.options,
            "status": "success",
            "answer": qa_result.answer,
            "reason": qa_result.reason,
            "source": qa_result.source,
            "retrieved_chunks_count": len(qa_result.retrieved_chunks),
            "qa_metadata": qa_result.metadata,
        }

    @staticmethod
    def _build_error_record(
        question: ExamQuestion,
        exc: Exception,
    ) -> Dict[str, Any]:
        return {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question": question.question,
            "options": question.options,
            "status": "failed",
            "error": str(exc),
        }


if __name__ == "__main__":
    runner = BatchQARunner()

    # 先跑前 3 题做冒烟测试
    batch_result = runner.run_from_file(
        file_path="input.txt",
        limit=3,
        verbose=True,
    )

    print("\n=== 批量结果摘要 ===")
    print(f"total_questions: {batch_result['total_questions']}")
    print(f"success_count : {batch_result['success_count']}")
    print(f"failed_count  : {batch_result['failed_count']}")

    if batch_result["results"]:
        print("\n=== 第一题结果示例 ===")
        first = batch_result["results"][0]
        print(f"question_id: {first['question_id']}")
        print(f"answer     : {first['answer']}")
        print(f"reason     : {first['reason']}")
        print(f"source     : {first['source']}")