# File: src/app.py
# Author: baomofan
# Standard: Industrial Application Grade | Strict Dict State Management | Fluent UX

import asyncio
import os
import sys
import json
import gradio as gr
from dotenv import load_dotenv, set_key

# 动画与样式定制
CSS = """
.gradio-container { background-color: #f9f9f9; }
#cot_panel { 
    transition: all 0.5s ease-in-out; 
    border-left: 3px solid #007bff; 
    padding-left: 15px; 
}
.thinking-state {
    animation: pulse 1.5s infinite;
    color: #666;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}
"""

# 基础路径配置
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    EXE_DIR = BUNDLE_DIR

sys.path.append(os.path.join(BUNDLE_DIR, "src"))
sys.path.append(BUNDLE_DIR)
ENV_PATH = os.path.join(EXE_DIR, ".env")


def save_config(api_key, proxy_port):
    """持久化配置到 .env。严格对齐底层引擎所需的 LLM_API_KEY"""
    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "w", encoding="utf-8") as f: f.write("")
    # 修复点 1：将变量名修正为底层期望的 LLM_API_KEY
    set_key(ENV_PATH, "LLM_API_KEY", api_key)
    set_key(ENV_PATH, "PROXY_HTTP", f"http://127.0.0.1:{proxy_port}")
    return "配置已保存，系统已准备就绪。"


async def process_step_by_step(user_message: str, history: list):
    """
    流式日志分步展示逻辑
    """
    load_dotenv(ENV_PATH, override=True)
    current_history = list(history) if history else []

    # 1. 验证配置
    if not os.getenv("LLM_API_KEY"):
        current_history.append({"role": "user", "content": user_message})
        current_history.append({"role": "assistant", "content": "配置缺失：请先在系统设置中配置 API Key。"})
        yield current_history, "状态：配置缺失"
        return

    # 延迟加载核心组件
    from engine.llm_client import AsyncLLMEngine
    from tools.arxiv_tool import ArxivSearchTool
    from tools.web_scraper_tool import WebScraperTool
    from tools.web_search_tool import WebSearchTool
    from orchestrator.workflow import AgentWorkflow

    engine = AsyncLLMEngine()
    tools = [ArxivSearchTool(), WebScraperTool(), WebSearchTool()]
    workflow = AgentWorkflow(engine=engine, tools=tools)

    # 2. 初始化界面状态
    current_history.append({"role": "user", "content": user_message})
    current_history.append({"role": "assistant", "content": "正在分析任务..."})

    current_cot_md = "### 任务启动\n系统正在规划执行路径..."
    yield current_history, current_cot_md

    # 3. 构造系统级上下文
    messages = [
        {"role": "system", "content": workflow.m_system_prompt},
        {"role": "user", "content": user_message}
    ]
    tools_schema = [tool.get_input_schema() for tool in tools]
    tool_map = {t.name: t for t in tools}

    # 4. 执行主推理循环
    for step in range(workflow.m_max_steps):
        yield current_history, current_cot_md + f"\n\n<div class='thinking-state'>正在进行第 {step + 1} 步推理...</div>"

        try:
            response_message = await engine.generate_response(messages=messages, tools=tools_schema)
        except Exception as e:
            current_history[-1]["content"] = f"大模型引擎调用异常：{repr(e)}"
            yield current_history, current_cot_md + "\n\n### 引擎故障"
            return

        if not response_message.tool_calls:
            final_answer = response_message.content or "任务已完成，但未生成可见输出。"
            current_history[-1]["content"] = final_answer
            current_cot_md += f"\n\n### 任务完成\n已生成最终分析报告。"
            yield current_history, current_cot_md
            break

        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments

            current_cot_md += f"\n\n#### 动作：调用 `{func_name}`\n参数：`{func_args_str}`"
            yield current_history, current_cot_md

            try:
                kwargs = json.loads(func_args_str)
                if func_name in tool_map:
                    result = await tool_map[func_name].execute_async(**kwargs)
                    if result.success:
                        result_str = json.dumps(result.data, ensure_ascii=False)
                        current_cot_md += f"\n执行成功，获取到关键信息。"
                    else:
                        result_str = f"Error: {result.error_message}"
                        current_cot_md += f"\n执行失败：{result.error_message}"
                else:
                    result_str = f"Error: Tool {func_name} not found."
                    current_cot_md += f"\n工具不存在。"
            except Exception as e:
                result_str = f"Execution Exception: {repr(e)}"
                current_cot_md += f"\n执行异常：{repr(e)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })
            yield current_history, current_cot_md

    else:
        err_msg = "已达到最大思考步数限制，强制终止任务。"
        current_history[-1]["content"] = err_msg
        current_cot_md += f"\n\n### 任务终止\n{err_msg}"
        yield current_history, current_cot_md


# UI 布局重构
with gr.Blocks(title="Agentic Research Pro") as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.Markdown("# 智能体深度调研分析系统")

    with gr.Tabs():
        with gr.TabItem("分析终端"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="交互历史", height=600)
                    with gr.Row():
                        msg = gr.Textbox(placeholder="输入调研任务...", scale=5, show_label=False)
                        btn = gr.Button("开始执行", variant="primary", scale=1)
                        # 修复点 2：恢复清空会话按钮
                        clear_btn = gr.Button("清空会话", scale=1)
                with gr.Column(scale=1):
                    gr.Markdown("### 推理链路 (CoT)")
                    cot_panel = gr.Markdown(value="等待任务输入...", elem_id="cot_panel")

        with gr.TabItem("系统设置"):
            gr.Markdown("### 基础环境配置")
            api_input = gr.Textbox(label="API Key ", type="password")
            proxy_input = gr.Number(label="本地代理端口 ", value=7890)
            save_btn = gr.Button("保存并应用配置")
            status_msg = gr.Markdown()

    # 事件绑定
    save_btn.click(save_config, [api_input, proxy_input], status_msg)
    btn.click(process_step_by_step, [msg, chatbot], [chatbot, cot_panel])

    # 修复点 3：绑定清空按钮的状态重置逻辑
    clear_btn.click(
        fn=lambda: ([], "等待任务输入..."),
        inputs=None,
        outputs=[chatbot, cot_panel]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)