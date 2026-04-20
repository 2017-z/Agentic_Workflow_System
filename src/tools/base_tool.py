# File: base_tool.py
# Author: baomofan
# Description: Abstract base class and standard data structures for Agent Action Tools.

import abc
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    success: bool = Field(..., description="Execution status flag")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Successful execution payload")
    error_message: Optional[str] = Field(default=None, description="Error trace context")

class BaseTool(abc.ABC):
    def __init__(self, name: str, description: str):
        self.m_name = name
        self.m_description = description

    @property
    def name(self) -> str:
        return self.m_name

    @property
    def description(self) -> str:
        return self.m_description

    @abc.abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def execute_async(self, **kwargs) -> ToolResult:
        pass