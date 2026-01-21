import requests

BASE_URL = "http://10.112.110.111/v1"
API_KEY = "sk-XZrfiPGmZaGLZFPNUpy6ww"
MODEL_NAME = "GLM-4.5-Air-FP8"


headers = {
    "Authorization": f"Bearer {API_KEY}",
}

resp = requests.get(f"{BASE_URL}/models", headers=headers, timeout=10)
print(resp.status_code, resp.text)


import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    response = await client.models.list()

    for model in response.data:
        print(model.id)

asyncio.run(main())

exit()

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents import ModelSettings
from agents import ModelTracing  # 关键：引入 tracin

async def main():
    # 1) 创建 OpenAI 异步客户端（直连 OpenAI 平台）
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)  # 或从环境变量读取

    # 2) 初始化 Chat Completions 模型封装
    #    model 可以用具体模型名，如 "gpt-4o"、"gpt-4o-mini"（以你账户可用为准）
    model = OpenAIChatCompletionsModel(
        model="GLM-4.5-Air-FP8",
        openai_client=client,
    )

    # 3) 构造模型设置（温度/最大tokens等）
    settings = ModelSettings(
        temperature=0.7,
        max_output_tokens=512,
    )

    # 4) 以 system + user 输入调用
    system_instructions = "You are a helpful assistant."
    user_input = "请用 3 点总结 Transformer 的核心思想。"

    # tracing = ModelTracing()  # 或 ModelTracing.noop()
    tracing =  ModelTracing.noop()

    resp = await model.get_response(
        system_instructions=system_instructions,
        input=user_input,            # 也可传入结构化的 input 列表（带角色/内容）
        model_settings=settings,
        tools=[],                    # 无工具调用
        output_schema=None,          # 若需结构化输出可传 schema
        handoffs=[],                 # 高级场景
        tracing=tracing,
    )

    print(resp.output_text)  # 输出模型文本

asyncio.run(main())
