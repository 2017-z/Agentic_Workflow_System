# File: tests/test_workflow.py
# Author: baomofan
# Description: Full integration test for Agent Workflow including ReAct loop and CoT logging.

import asyncio
import sys
import os
from dotenv import load_dotenv

# 必须在最顶层加载环境变量
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from engine.llm_client import AsyncLLMEngine
from tools.arxiv_tool import ArxivSearchTool
from tools.web_scraper_tool import WebScraperTool
from orchestrator.workflow import AgentWorkflow


async def main():
    print("--- 正在装载 Agent 核心模块 ---")
    try:
        engine = AsyncLLMEngine()
        tools = [ArxivSearchTool(), WebScraperTool()]
        workflow = AgentWorkflow(engine=engine, tools=tools)
    except Exception as e:
        print(f"[❌ 初始化失败] {e}")
        return

    # 设定一个复杂的复合型任务，强制大模型进行多步推理
    test_query = (
        "帮我检索关于 DPO (Direct Preference Optimization) 算法的2篇最新论文，"
        "并根据检索到的摘要信息，生成一份简短的 Markdown 格式的研究速报。"
    )

    print(f"\n--- 启动 Agent 自动工作流 ---")
    print(f"Task Query: {test_query}\n")

    # 执行主循环
    final_answer, cot_logs = await workflow.execute_task(user_query=test_query)

    print("\n==================================================")
    print("              [ 智能体 CoT 思考链路 ]")
    print("==================================================")
    for log in cot_logs:
        step = log.get("step", "Unknown")
        if "Tool Execution" in step:
            status = log.get("status", "")
            print(f"-> [{step}] 调用工具: {log.get('tool')}")
            print(f"   注入参数: {log.get('args')}")
            print(f"   执行状态: {status}")
            if "Success" in status:
                print(f"   数据预览: {log.get('result_preview', '')}")
        else:
            print(f"-> [{step}] {log.get('message', '')}")
        print("-" * 50)

    print("\n==================================================")
    print("                 [ 最终生成报告 ]")
    print("==================================================")
    print(final_answer)


if __name__ == "__main__":
    asyncio.run(main())