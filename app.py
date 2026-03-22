from src.qa_pipeline import build_prompt, parse_llm_output, normalize_answer
from src.llm import call_llm

question = "公司财务报销流程前两步是什么？"
options = {
    "A": "提交申请",
    "B": "领导审批",
    "C": "财务打款",
    "D": "归档"
}
q_type = "multiple"
context = "报销流程为：提交申请 -> 领导审批 -> 财务打款 -> 归档"

prompt = build_prompt(question, options, context, q_type)
raw_result = "答案是A和B"#call_llm(prompt)
result = parse_llm_output(raw_result)
result = normalize_answer(result)

print(f"解析后模型输出：{result}")