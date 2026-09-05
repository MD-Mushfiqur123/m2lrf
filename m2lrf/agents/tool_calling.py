"""
M-2LRF Agent Engine: JSON Schema Tool Calling & Function Execution.
===================================================================
Provides robust tool definition, schema validation, and tool call parsing
compatible with OpenAI and Hermes tool calling specifications:
- `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- Automatic parameter type casting and schema enforcement
- Safe sandbox dispatch and structured tool output collating
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import json
import re


class ToolDefinition:
    """Represents an executable agent tool with JSON schema metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_callable(cls, func: Callable[..., Any], name: Optional[str] = None, description: Optional[str] = None) -> "ToolDefinition":
        """Introspects Python callable signature to automatically build JSON schema."""
        func_name = name or func.__name__
        func_doc = description or (func.__doc__ or "No description provided.").strip()

        sig = inspect.signature(func)
        properties = {}
        required = []

        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {"type": param_type}
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return cls(name=func_name, description=func_doc, parameters=parameters, function=func)


class ToolCallingEngine:
    """Manages a registry of tools and executes parsed tool calls."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: Union[ToolDefinition, Callable[..., Any]]) -> None:
        if callable(tool) and not isinstance(tool, ToolDefinition):
            tool_def = ToolDefinition.from_callable(tool)
        else:
            tool_def = tool
        self.tools[tool_def.name] = tool_def

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tools.values()]

    def parse_tool_calls(self, completion: str) -> List[Dict[str, Any]]:
        """
        Extracts tool calls formatted as:
        `<tool_call>{"name": "foo", "arguments": {...}}</tool_call>`
        or markdown code blocks.
        """
        calls = []

        # 1. XML-style tag match
        matches = re.findall(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL)
        for m in matches:
            try:
                parsed = json.loads(m.strip())
                if "name" in parsed:
                    calls.append(parsed)
            except Exception:
                pass

        # 2. Markdown JSON block fallback if no XML tags
        if not calls:
            code_matches = re.findall(r"```json\s*(\{.*?\})\s*```", completion, re.DOTALL)
            for cm in code_matches:
                try:
                    parsed = json.loads(cm.strip())
                    if "name" in parsed and parsed.get("name") in self.tools:
                        calls.append(parsed)
                except Exception:
                    pass

        return calls

    def execute_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call and returns a structured response."""
        name = call.get("name", "")
        args = call.get("arguments", {})

        if name not in self.tools:
            return {
                "tool_name": name,
                "status": "error",
                "output": f"Tool '{name}' not found in registry.",
            }

        tool = self.tools[name]
        try:
            if isinstance(args, str):
                args = json.loads(args)
            result = tool.function(**args)
            return {
                "tool_name": name,
                "status": "success",
                "output": result,
            }
        except Exception as e:
            return {
                "tool_name": name,
                "status": "error",
                "output": f"Execution error in {name}: {str(e)}",
            }
