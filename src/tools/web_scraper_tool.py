# File: web_scraper_tool.py
# Author: baomofan
# Description: Asynchronous web page extraction tool utilizing LLM-friendly Reader API.

import aiohttp
import logging
from typing import Dict, Any
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebScraperTool(BaseTool):
    def __init__(self):
        self.m_name = "scrape_web_content"
        self.m_description = "Scrape and extract the main text content from a given URL. Returns highly structured Markdown optimized for LLMs."
        super().__init__(name=self.m_name, description=self.m_description)
        self.m_timeout = aiohttp.ClientTimeout(total=25)
        self.m_reader_endpoint = "https://r.jina.ai/"
        self.m_headers = {
            "X-Return-Format": "markdown"
        }
        self.m_max_chars = 12000

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.m_name,
                "description": self.m_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full HTTP/HTTPS URL of the web page to scrape."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    async def execute_async(self, url: str, **kwargs) -> ToolResult:
        target_url = f"{self.m_reader_endpoint}{url}"

        try:
            async with aiohttp.ClientSession(timeout=self.m_timeout, headers=self.m_headers, trust_env=True) as session:
                async with session.get(target_url) as response:
                    response.raise_for_status()
                    markdown_content = await response.text()

                    if len(markdown_content) > self.m_max_chars:
                        markdown_content = markdown_content[
                                           :self.m_max_chars] + "\n\n[Content truncated due to length limits]"

                    return ToolResult(success=True, data={"url": url, "content": markdown_content})

        except aiohttp.ClientError as e:
            logger.error(f"Network error scraping {url}: {repr(e)}")
            return ToolResult(success=False, error_message=repr(e))
        except Exception as e:
            logger.error(f"Unexpected error in WebScraperTool: {repr(e)}")
            return ToolResult(success=False, error_message=repr(e))