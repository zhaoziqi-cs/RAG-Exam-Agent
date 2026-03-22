from openai import OpenAI
import os

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="qwen-plus",   # 或 qwen-turbo
        messages=[
            {"role": "system", "content": "你是一个严谨的考试助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content
