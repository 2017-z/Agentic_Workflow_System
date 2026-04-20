# File: tests/test_engine.py
# Author: baomofan
# Description: Integration smoke test for AsyncLLMEngine and Function Calling.

import asyncio
import json
import sys
import os
from dotenv import load_dotenv

# 必须在初始化任何模块前加载环境变量（包含代理和 API Key）
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engine.llm_client import AsyncLLMEngine
from tools.arxiv_tool import ArxivSearchTool


async def main():
    print("--- 初始化 Agent 核心组件 ---")
    try:
        engine = AsyncLLMEngine()
        arxiv_tool = ArxivSearchTool()
    except Exception as e:
        print(f"[❌ 初始化失败] 检查环境变量是否正确配置: {e}")
        return

    # 1. 获取工具的 JSON Schema
    tools = [arxiv_tool.get_input_schema()]

    # 2. 构造对话上下文
    messages = [
        {
            "role": "system",
            "content": "You are a professional academic research agent. You must use the provided tools to fetch real data before answering."
        },
        {
            "role": "user",
            "content": "帮我检索两篇关于 DPO (Direct Preference Optimization) 算法的最新论文。"
        }
    ]

    print("\n--- 发送请求至大模型引擎 (等待推理) ---")
    try:
        response_message = await engine.generate_response(messages=messages, tools=tools)

        print("\n--- 引擎响应解析 ---")
        # 工业级断言：检查模型是否触发了 tool_calls
        if response_message.tool_calls:
            print("[✅ 测试通过] 大模型成功理解意图并触发了 Tool Call！")
            for tool_call in response_message.tool_calls:
                print(f"  - 目标工具名称: {tool_call.function.name}")
                print(f"  - 提取的参数: {tool_call.function.arguments}")
        else:
            print("[❌ 测试失败] 大模型没有调用工具，而是直接回复了文本（产生幻觉或未命中工具）：")
            print(f"  - 回复内容: {response_message.content}")

    except Exception as e:
        print(f"\n[❌ 测试抛出异常] {e}")


if __name__ == "__main__":
    asyncio.run(main())