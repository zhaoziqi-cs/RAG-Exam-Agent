from openai import OpenAI
import os

def call_llm(prompt: str) -> str:
    print("[LLM] 开始初始化客户端...")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")
    client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
    print("[LLM] 客户端初始化完成，开始请求模型...")
    response = client.chat.completions.create(
        model="qwen-plus",   # 或 qwen-turbo
        messages=[
            {"role": "system", "content": "你是一个严谨的考试助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    print("[LLM] 收到模型返回。")
    return response.choices[0].message.content
