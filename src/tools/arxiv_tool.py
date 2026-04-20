# File: arxiv_tool.py
# Author: baomofan
# Description: Asynchronous Arxiv literature search tool with URL extraction.

import aiohttp
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

class ArxivSearchTool(BaseTool):
    def __init__(self):
        self.m_name = "search_arxiv_papers"
        self.m_description = "Search for academic papers on Arxiv. Returns paper titles, authors, URLs, and abstracts."
        super().__init__(name=self.m_name, description=self.m_description)
        self.m_base_url = "http://export.arxiv.org/api/query"
        self.m_timeout = aiohttp.ClientTimeout(total=15)

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
                            "description": "The search query (e.g., 'all:BitNet' or 'ti:Large Language Models')."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of papers to return. Default is 3.",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute_async(self, query: str, max_results: int = 3, **kwargs) -> ToolResult:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results
        }

        try:
            async with aiohttp.ClientSession(timeout=self.m_timeout) as session:
                async with session.get(self.m_base_url, params=params) as response:
                    response.raise_for_status()
                    xml_data = await response.text()
                    papers = self._parse_arxiv_xml(xml_data)
                    return ToolResult(success=True, data={"papers": papers})

        except Exception as e:
            logger.error(f"Error in Arxiv tool: {repr(e)}")
            return ToolResult(success=False, error_message=str(e))

    def _parse_arxiv_xml(self, xml_string: str) -> List[Dict[str, str]]:
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_string)
        papers = []

        for entry in root.findall('atom:entry', namespace):
            # 新增：提取 id 作为论文的 URL
            id_elem = entry.find('atom:id', namespace)
            title_elem = entry.find('atom:title', namespace)
            summary_elem = entry.find('atom:summary', namespace)
            published_elem = entry.find('atom:published', namespace)

            paper_url = id_elem.text.strip() if id_elem is not None else ""
            title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else ""
            summary = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else ""
            published = published_elem.text if published_elem is not None else ""

            authors = [
                author.find('atom:name', namespace).text
                for author in entry.findall('atom:author', namespace)
                if author.find('atom:name', namespace) is not None
            ]

            papers.append({
                "title": title,
                "url": paper_url,  # 确保返回 URL 字段
                "authors": ", ".join(authors),
                "published": published,
                "abstract": summary[:800] + "..." if len(summary) > 800 else summary # 略微增加摘要长度减少二跳抓取
            })

        return papers