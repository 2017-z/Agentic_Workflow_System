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
        self.m_max_steps = 10
        self.m_system_prompt = (
            "你是一个资深人工智能研究员与前沿科技分析专家。\n\n"
            "【工具调用路由指南】（核心法则）：\n"
            "1. 学术论文检索 (search_arxiv_papers)：仅当问题属于计算机科学(CS)、数学、高能物理领域的文献调研时调用。\n"
            "2. 通用网络搜索 (search_web)：当问题属于运动医学、生物学、金融商业、日常科普或实时资讯时，必须优先调用此工具。\n"
            "3. 使用学术检索工具时，必须直接从返回结果的 url 字段获取链接，严禁自行拼凑 arXiv ID。\n"
            "4. 网页正文抓取 (scrape_web_content)：仅用于提取搜索结果中确切 URL 的长文本详情，严禁盲猜或捏造 URL。\n\n"
            "【动态终止与降级策略】（防死循环核心法则）：\n"
            "1. 快速失败阈值：如果连续 2 次使用搜索工具均未获取到足以支撑完整结论的核心数据，必须立即停止调用该工具，绝对禁止进行第 3 次无意义的关键词穷举。\n"
            "2. 状态透明化：当终止外部检索时，你必须在回答的开头明确向用户声明：“经过外部检索，未能获取到足够直接的数据支撑。”\n"
            "3. 自主决策与知识降级：声明检索失败后，你需要评估该问题。如果属于基础科学或普遍常识（如运动生理学机制），请明确标示“基于内部专业知识为您分析：”并给出解答；如果属于强时效性或强事实性问题（如今日实时票价、突发新闻），必须直接拒绝回答，严禁产生幻觉。"
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