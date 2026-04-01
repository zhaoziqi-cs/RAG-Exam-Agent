import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_export_prefix(base_name: Optional[str] = None) -> str:
    """
    生成导出文件前缀。
    例如:
    - batch_results_20260331_103015
    - exam_run_20260331_103015
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = base_name.strip() if base_name else "batch_results"
    return f"{prefix}_{timestamp}"

def safe_json_dumps(obj: Any) -> str:
    """
    把复杂对象尽量转成 JSON 字符串，便于写入 CSV。
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)
    
def normalize_result_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 success record 规范成适合写 CSV 的一行。
    """
    return {
        "question_id": record.get("question_id", ""),
        "question_type": record.get("question_type", ""),
        "question": record.get("question", ""),
        "options": " | ".join(record.get("options", [])),
        "answer": ",".join(record.get("answer", [])),
        "reason": record.get("reason", ""),
        "source": record.get("source", ""),
        "status": record.get("status", ""),
        "retrieved_chunks_count": record.get("retrieved_chunks_count", 0),
        "qa_metadata": safe_json_dumps(record.get("qa_metadata", {})),
    }

def normalize_error_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 error record 规范成适合写 CSV 的一行。
    """
    return {
        "question_id": record.get("question_id", ""),
        "question_type": record.get("question_type", ""),
        "question": record.get("question", ""),
        "options": " | ".join(record.get("options", [])),
        "status": record.get("status", ""),
        "error": record.get("error", ""),
    }

def export_summary_json(
    batch_result: Dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> str:
    summary = {
        "total_questions": batch_result.get("total_questions", 0),
        "success_count": batch_result.get("success_count", 0),
        "failed_count": batch_result.get("failed_count", 0),
        "metadata": batch_result.get("metadata", {}),
    }

    file_path = output_dir / f"{prefix}_summary.json"
    file_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(file_path)

def export_results_json(
    batch_result: Dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> str:
    payload = {
        "results": batch_result.get("results", []),
        "errors": batch_result.get("errors", []),
        "metadata": batch_result.get("metadata", {}),
    }

    file_path = output_dir / f"{prefix}_results.json"
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(file_path)

def export_results_csv(
    results: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
) -> str:
    file_path = output_dir / f"{prefix}_results.csv"

    fieldnames = [
        "question_id",
        "question_type",
        "question",
        "options",
        "answer",
        "reason",
        "source",
        "status",
        "retrieved_chunks_count",
        "qa_metadata",
    ]

    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in results:
            writer.writerow(normalize_result_row(record))

    return str(file_path)

def export_errors_csv(
    errors: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
) -> str:
    file_path = output_dir / f"{prefix}_errors.csv"

    fieldnames = [
        "question_id",
        "question_type",
        "question",
        "options",
        "status",
        "error",
    ]

    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in errors:
            writer.writerow(normalize_error_row(record))

    return str(file_path)

def export_batch_result(
    batch_result: Dict[str, Any],
    output_dir: str = "outputs",
    base_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    对外主入口：
    把 batch_runner 返回的结果导出到本地文件。
    """
    output_path = ensure_output_dir(output_dir)
    prefix = build_export_prefix(base_name)

    summary_json_path = export_summary_json(batch_result, output_path, prefix)
    results_json_path = export_results_json(batch_result, output_path, prefix)
    results_csv_path = export_results_csv(
        batch_result.get("results", []), output_path, prefix
    )
    errors_csv_path = export_errors_csv(
        batch_result.get("errors", []), output_path, prefix
    )

    return {
        "summary_json": summary_json_path,
        "results_json": results_json_path,
        "results_csv": results_csv_path,
        "errors_csv": errors_csv_path,
    }

if __name__ == "__main__":
    from src.batch_runner import BatchQARunner
    runner = BatchQARunner()
    batch_result = runner.run_from_file(
        file_path="../question/input.txt",
        limit=None,
        verbose=True,
    )

    paths = export_batch_result(
        batch_result=batch_result,
        output_dir="../question/outputs",
        base_name="exam_batch",
    )

    print(paths)