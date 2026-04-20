# File: tests/test_scraper.py
# Author: baomofan
# Description: Smoke test for WebScraperTool async execution and text extraction.

import asyncio
import json
import sys
import os
from dotenv import load_dotenv

# 工业级规范：在初始化任何网络组件前加载环境变量，包括代理配置
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tools.web_scraper_tool import WebScraperTool


async def main():
    print("--- 正在初始化 WebScraperTool ---")
    tool = WebScraperTool()

    test_url = "https://en.wikipedia.org/wiki/Direct_preference_optimization"
    print(f"\n--- 执行网页抓取测试 (URL: {test_url}) ---")

    result = await tool.execute_async(url=test_url)

    if result.success:
        print("\n[✅ 测试通过] 成功抓取网页正文 (截取前500字符)：")
        print(result.data["content"][:500] + "...")
        print(f"\n[数据统计] 总字符数: {len(result.data['content'])}")
    else:
        print(f"\n[❌ 测试失败] 异常信息：{result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())