def build_prompt(question, options, context, q_type):
    return f"""
你是一个企业内部考试助手，请严格根据提供的知识库内容回答问题。

【题目类型】
{q_type}（single=单选, multiple=多选）

【题目】
{question}

【选项】
{chr(10).join([f"{k}. {v}" for k,v in options.items()])}

【知识库参考】
{context}

请遵守以下规则：
1. 必须基于知识库回答
2. 单选题只能选一个答案
3. 多选题可以选多个
4. 如果不确定，选择最合理答案，不要乱选
5. 必须返回JSON格式

输出格式如下：
{{
  "answer": ["A"],
  "reason": "简要解释",
  "source": "来源pdf名称+页码（如果有）"
}}
"""

import json
import re

def parse_llm_output(output):
    try:
        # 优先直接解析
        return json.loads(output)
    except:
        # 尝试从文本中提取 JSON
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    # fallback（兜底）
    return {
        "answer": [],
        "reason": "解析失败",
        "source": ""
    }

def normalize_answer(result):
    answer = result.get("answer", [])
    if isinstance(answer, str):
        answer = [answer]
    #强制转换为大写字母列表
    answer = [a.strip().upper() for a in answer]
    result["answer"] = answer
    return result