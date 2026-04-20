# File: src/app.py
# Author: baomofan
# Description: 智能体工作流系统 Web 交互界面。

import asyncio
import json
import os
import sys
import gradio as gr
from dotenv import load_dotenv

# 加载环境变量与路径配置
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.llm_client import AsyncLLMEngine
from tools.arxiv_tool import ArxivSearchTool
from tools.web_scraper_tool import WebScraperTool
from orchestrator.workflow import AgentWorkflow

# 初始化核心后端组件
engine = AsyncLLMEngine()
tools = [ArxivSearchTool(), WebScraperTool()]
workflow = AgentWorkflow(engine=engine, tools=tools)


def format_cot_logs(cot_logs: list) -> str:
    """
    将内部日志流格式化为结构化的 Markdown 文本。
    """
    formatted_str = ""
    for log in cot_logs:
        step = log.get("step", "Unknown")
        if "Tool Execution" in step:
            status = log.get("status", "")
            formatted_str += f"### 步骤：{step}\n"
            formatted_str += f"- **目标工具：** {log.get('tool')}\n"
            formatted_str += f"- **调用参数：** {log.get('args')}\n"
            formatted_str += f"- **执行状态：** {status}\n\n"
        else:
            formatted_str += f"### 步骤：{step}\n"
            formatted_str += f"{log.get('message', '')}\n\n"
    return formatted_str


async def process_user_input(user_message: str, history: list):
    """
    核心回调处理函数。
    遵循报错信息的绝对要求：构造并返回仅包含 role 和 content 的字典列表。
    避免对历史列表进行就地追加（append），采用深拷贝机制防止状态污染。
    """
    # 1. 重建全新的上下文列表
    current_history = list(history) if history else []

    # 2. 压入用户消息与系统占位符（严格字典格式）
    current_history.append({"role": "user", "content": user_message})
    current_history.append({"role": "assistant", "content": "系统正在处理任务，请稍候..."})

    yield current_history, "状态：正在处理"

    try:
        # 3. 执行 ReAct 异步工作流
        final_answer, cot_logs = await workflow.execute_task(user_query=user_message)

        # 4. 渲染思考链路与最终回答
        cot_markdown = format_cot_logs(cot_logs)
        current_history[-1]["content"] = final_answer
        yield current_history, cot_markdown

    except Exception as e:
        error_msg = f"工作流异常：{repr(e)}"
        current_history[-1]["content"] = error_msg
        yield current_history, error_msg


# 构建 UI 布局（移除所有 type 参数）
with gr.Blocks(title="智能体工作流系统") as demo:
    gr.Markdown("# 智能体工作流系统")
    gr.Markdown("基于函数调用技术构建的前沿科研与互联网信息分析平台。")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=650,
                show_label=False
            )
            msg = gr.Textbox(
                label="指令输入",
                placeholder="请输入您的研究任务...",
                lines=3,
                show_label=False
            )
            with gr.Row():
                submit_btn = gr.Button("执行任务", variant="primary")
                clear_btn = gr.Button("清空会话")

        with gr.Column(scale=1):
            gr.Markdown("### 思考链路 (Chain of Thought)")
            cot_display = gr.Markdown(
                value="此处将展示智能体的推理逻辑与工具调用过程。",
                elem_id="cot_panel"
            )

    # 事件绑定
    submit_btn.click(
        fn=process_user_input,
        inputs=[msg, chatbot],
        outputs=[chatbot, cot_display]
    )

    # 清空功能逻辑
    clear_btn.click(
        fn=lambda: ([], "此处将展示智能体的推理逻辑与工具调用过程。"),
        inputs=None,
        outputs=[chatbot, cot_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )