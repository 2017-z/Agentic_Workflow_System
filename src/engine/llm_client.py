# File: llm_client.py
# Author: baomofan
# Description: Asynchronous LLM engine wrapper supporting Function Calling and exponential backoff.

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)


class AsyncLLMEngine:
    def __init__(self):
        self.m_api_key = os.getenv("LLM_API_KEY")
        self.m_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.m_model_name = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

        if not self.m_api_key:
            raise ValueError("Critical Error: LLM_API_KEY environment variable is missing.")

        self.m_client = AsyncOpenAI(api_key=self.m_api_key, base_url=self.m_base_url)
        self.m_max_retries = 3
        self.m_base_delay = 2.0

    async def generate_response(self, messages: List[Dict[str, Any]],
                                tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        attempt = 0
        while attempt < self.m_max_retries:
            try:
                kwargs = {
                    "model": self.m_model_name,
                    "messages": messages,
                    "temperature": 0.1,
                }

                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = await self.m_client.chat.completions.create(**kwargs)
                return response.choices[0].message

            except RateLimitError as e:
                attempt += 1
                delay = self.m_base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit. Retrying in {delay}s... (Attempt {attempt}/{self.m_max_retries})")
                await asyncio.sleep(delay)
            except APIError as e:
                logger.error(f"LLM API Error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected LLM Engine error: {e}")
                raise

        raise RuntimeError("Max retries exceeded for LLM API call.")