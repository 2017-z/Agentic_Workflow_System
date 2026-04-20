# Agent Research Pro 智能体深度调研系统

## 项目概述

Agent Research Pro 是一个基于大语言模型（LLM）与函数调用（Function Calling）技术构建的工业级科研调研与互联网信息分析平台。

不同于传统的单次问答交互，本系统采用智能体工作流（Agentic Workflow）架构。系统通过 ReAct（Reasoning and Acting）逻辑框架，赋予 AI 自主规划、工具调度以及多步推理的能力，能够针对复杂指令生成具有数据支撑的深度分析报告。

## 技术原理

### 1. 核心架构：ReAct 推理循环
系统不直接输出答案，而是进入“思考-行动-观察”的闭环。Agent 会根据任务目标自主决定是否需要检索论文、搜索网页或深入抓取特定 URL 的内容。

### 2. 工具链系统
* **学术检索 (search_arxiv_papers)**：对接 Arxiv API，支持计算机科学、数学及物理领域的精准文献调研。
* **通用搜索 (search_web)**：集成 DuckDuckGo 引擎，覆盖生物学、金融、时事新闻等非学术领域信息。
* **网页解析 (scrape_web_content)**：对检索到的目标 URL 进行深度正文提取，为模型提供一手原始文本。

### 3. 策略优化
* **动态路由逻辑**：系统根据意图识别自动分发工具调用，CS 领域问题直通学术库，常识或时效问题路由至通用搜索。
* **快速失败与截断 (Fail-Fast)**：为防止无效搜索导致死循环，系统设定了硬性重试阈值（Max Steps）与检索质量校验。
* **知识降级协议**：当外部数据获取受阻时，系统会明确声明数据来源缺失，并转由内部专业知识提供辅助分析，杜绝事实性幻觉。

## 安装与配置

### 开发环境配置
1.  **克隆仓库**：
    ```bash
    git clone https://github.com/2017-z/Agentic_Workflow_System.git
    cd Agentic_Workflow_System
    ```
2.  **创建环境**：
    ```bash
    conda env create -f environment.yml
    conda activate agentic_env
    ```
3.  **配置文件**：
    将 `.env.template` 重命名为 `.env`，并填写以下必要参数：
    * `LLM_API_KEY`: 您的 API 密钥。
    * `PROXY_HTTP`: 本地代理地址（如 [http://127.0.0.1:7890](http://127.0.0.1:7890)）。

### 运行应用
```bash
python src/app.py
```

## 发行版打包

项目支持使用 PyInstaller 打包为独立的 Windows 可执行文件（.exe）。

1.  **执行构建**：
    ```bash
    python build.py
    ```
2.  **分发使用**：
    运行 `dist/AgentResearchPro.exe`。首次启动请在“系统设置”选项卡中配置您的 API 密钥与代理端口。

## 目录结构说明

```plaintext
├── src/
│   ├── app.py                # 界面入口与流式状态管理
│   ├── engine/
│   │   └── llm_client.py     # 异步大模型驱动引擎
│   ├── orchestrator/
│   │   └── workflow.py       # 智能体决策中枢与系统提示词
│   └── tools/                # 插件化工具库
├── build.py                  # 自动化打包脚本
├── .env.template             # 配置文件模板
├── environment.yml           # Conda 环境定义
└── .gitignore                # Git 忽略规则
```

## 注意事项

* **API 消耗**：深度调研任务可能涉及多次工具调用，请注意监控您的 Token 使用情况。
* **代理环境**：由于 Arxiv 等数据源位于海外，建议确保本地代理服务（如端口 7890）处于开启状态。