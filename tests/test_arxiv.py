# File: tests/test_arxiv.py
# Author: baomofan
# Description: Smoke test for ArxivSearchTool async execution and schema generation.

import asyncio
import json
import sys
import os

# 将 src 目录临时加入环境变量，确保模块可被导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tools.arxiv_tool import ArxivSearchTool


async def main():
    print("--- 正在初始化 ArxivSearchTool ---")
    tool = ArxivSearchTool()

    print("\n--- 验证 LLM Function Calling Schema ---")
    schema = tool.get_input_schema()
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    print("\n--- 执行异步并发检索测试 (Query: 'DPO algorithm') ---")
    result = await tool.execute_async(query="all:DPO algorithm", max_results=2)

    if result.success:
        print("\n[✅ 测试通过] 成功抓取论文数据：")
        print(json.dumps(result.data, indent=2, ensure_ascii=False))
    else:
        print(f"\n[❌ 测试失败] 异常信息：{result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())