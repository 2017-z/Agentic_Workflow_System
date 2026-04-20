# File: src/tools/web_search_tool.py
# Author: baomofan
# Description: General web search tool for Agentic Workflow using DuckDuckGo.

import logging
import asyncio
from typing import Dict, Any
from duckduckgo_search import DDGS
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    def __init__(self):
        self.m_name = "search_web"
        self.m_description = "Perform a general web search to find real-time information, news, or cross-domain knowledge (e.g., biology, medicine, daily life) that is not covered by academic databases."
        super().__init__(name=self.m_name, description=self.m_description)
        self.m_max_results = 5

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.m_name,
                "description": self.m_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute_async(self, query: str, **kwargs) -> ToolResult:
        try:
            # DuckDuckGo search is synchronous by default in this library,
            # wrap it in asyncio to prevent blocking the event loop.
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, query)

            if not results:
                return ToolResult(success=True, data={"message": "No relevant search results found."})

            return ToolResult(success=True, data={"results": results})

        except Exception as e:
            logger.error(f"Error in WebSearchTool: {repr(e)}")
            return ToolResult(success=False, error_message=repr(e))

    def _sync_search(self, query: str) -> list:
        with DDGS() as ddgs:
            # 获取前 5 个搜索结果的标题、摘要和链接
            results = list(ddgs.text(query, max_results=self.m_max_results))
            return results