# File: src/orchestrator/workflow.py
# Author: baomofan
# Description: Core Agent Orchestrator implementing ReAct loop and CoT logging.

import json
import logging
import time
from typing import List, Dict, Any, Tuple
from engine.llm_client import AsyncLLMEngine
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class AgentWorkflow:
    def __init__(self, engine: AsyncLLMEngine, tools: List[BaseTool]):
        self.m_engine = engine
        self.m_tools = {t.name: t for t in tools}
        self.m_max_steps = 5
        self.m_system_prompt = (
            "你是一个资深人工智能研究员与前沿科技分析专家。\n\n"
            "【工具调用路由指南】（核心法则）：\n"
            "1. 学术论文检索 (search_arxiv_papers)：仅当用户问题明确属于计算机科学(CS)、数学、高能物理领域的学术文献调研时调用。\n"
            "2. 通用网络搜索 (search_web)：如果你判断问题属于医学、生物学、金融商业、日常科普或实时资讯（如实时票价、新闻），必须直接跳过学术检索，优先调用此通用搜索工具。\n"
            "3. 网页正文抓取 (scrape_web_content)：仅用于提取搜索结果中具体 URL 的长文本详情，严禁凭空捏造或盲猜商业应用的 URL。\n\n"
            "【操作规范】：\n"
            "1. 面对问题，必须严格遵守上述路由指南进行任务规划，避免在错误领域进行无效重试。\n"
            "2. 综合所获数据，输出结构化、严谨的 Markdown 分析报告。\n"
            "3. 如果所有可用工具均无法获取有效数据，请直接向用户说明系统当前的能力限制，绝对禁止产生事实性幻觉。"
        )

    async def execute_task(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        执行主循环，返回 (最终回答文本, CoT思考链路日志)
        """
        messages = [
            {"role": "system", "content": self.m_system_prompt},
            {"role": "user", "content": user_query}
        ]

        cot_logs = []
        tools_schema = [tool.get_input_schema() for tool in self.m_tools.values()]

        start_time = time.time()
        cot_logs.append({"step": "Init", "message": f"接收到任务: {user_query}"})

        for step in range(self.m_max_steps):
            logger.info(f"--- Workflow Step {step + 1}/{self.m_max_steps} ---")

            response_message = await self.m_engine.generate_response(messages=messages, tools=tools_schema)

            if not response_message.tool_calls:
                final_content = response_message.content or ""
                cot_logs.append({
                    "step": f"Final Answer (Step {step + 1})",
                    "message": "生成最终报告",
                    "content": final_content
                })
                return final_content, cot_logs

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments

                log_entry = {
                    "step": f"Tool Execution (Step {step + 1})",
                    "tool": func_name,
                    "args": func_args_str,
                    "status": "Pending"
                }

                if func_name not in self.m_tools:
                    err_msg = f"Tool {func_name} not found."
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": err_msg})
                    log_entry["status"] = f"Failed: {err_msg}"
                    cot_logs.append(log_entry)
                    continue

                try:
                    kwargs = json.loads(func_args_str)
                    logger.info(f"Executing tool {func_name} with args: {kwargs}")

                    tool_start = time.time()
                    result = await self.m_tools[func_name].execute_async(**kwargs)
                    tool_cost = time.time() - tool_start

                    if result.success:
                        result_str = json.dumps(result.data, ensure_ascii=False)
                        log_entry["status"] = f"Success ({tool_cost:.2f}s)"
                        log_entry["result_preview"] = result_str[:200] + "..."
                    else:
                        result_str = f"Error: {result.error_message}"
                        log_entry["status"] = f"Failed ({tool_cost:.2f}s): {result.error_message}"

                except Exception as e:
                    result_str = f"Execution Exception: {repr(e)}"
                    log_entry["status"] = result_str

                cot_logs.append(log_entry)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

        err_msg = "Task aborted: Maximum reasoning steps reached."
        cot_logs.append({"step": "Timeout", "message": err_msg})
        return err_msg, cot_logs