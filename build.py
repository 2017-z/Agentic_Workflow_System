# File: build.py
# Author: baomofan
# Description: 自动化构建脚本，将 Agent 系统打包为 Windows 可执行文件。
# 已修复 safehttpx 缺失元数据导致的闪退问题。

import PyInstaller.__main__
import os
import shutil

# 获取当前绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))

def run_build():
    print("--- 启动生产环境构建进程 ---")

    # 1. 确保环境模板文件存在
    template_path = os.path.join(base_path, ".env.template")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("DEEPSEEK_API_KEY=\nPROXY_HTTP=http://127.0.0.1:7890\n")
        print("已自动创建缺失的 .env.template 文件")

    # 2. 清理旧的构建缓存（非常重要，防止旧报错被缓存）
    for folder in ['build', 'dist']:
        path = os.path.join(base_path, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"已清理旧的 {folder} 目录")

    # 3. 构建配置参数
        # 3. 构建配置参数
    args = [
        'src/app.py',
        '--name=AgentResearchPro',
        '--onefile',
        '--noconfirm',
        '--clean',
            # 资源文件注入
        f'--add-data={os.path.join(base_path, "src")};src',
        f'--add-data={template_path};.',

            # 核心框架包全量搜集
        '--collect-all=gradio',
        '--collect-all=gradio_client',  # 防御性补充：Gradio 客户端库
        '--collect-all=uvicorn',
        '--collect-all=anyio',
        '--collect-all=fastapi',

            # 碎片化底层依赖包全量搜集 (解决 version.txt 缺失问题)
        '--collect-all=safehttpx',
        '--collect-all=groovy',  # 本次报错的关键修复点

            # 显式导入内部模块
        '--hidden-import=engine.llm_client',
        '--hidden-import=tools.arxiv_tool',
        '--hidden-import=tools.web_scraper_tool',
        '--hidden-import=tools.web_search_tool',
        '--hidden-import=orchestrator.workflow',
    ]

    # 4. 执行打包动作
    PyInstaller.__main__.run(args)
    print("\n--- 构建完成，请查看 dist/ 目录 ---")

if __name__ == "__main__":
    run_build()