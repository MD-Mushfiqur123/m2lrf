"""
M-2LRF Universal Prompt Formatters (Axolotl-Inspired)
======================================================
Formatters converting structured datasets into standardized instruction/chat templates.
"""

from typing import Dict, List, Any, Optional, Tuple


class PromptFormatter:
    """Base prompt formatter."""
    def format(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError


class AlpacaFormatter(PromptFormatter):
    """Formats Alpaca-style instruction/input/output examples."""
    def format(self, example: Dict[str, Any]) -> str:
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")

        if inp and len(inp.strip()) > 0:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def split_prompt_response(self, example: Dict[str, Any]) -> Tuple[str, str]:
        instruction = example.get("instruction", "")
        inp = example.get("input", "")
        output = example.get("output", "")
        if inp and len(inp.strip()) > 0:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt, output


class ChatMLFormatter(PromptFormatter):
    """Formats ChatML conversation turns (<|im_start|>system...<|im_end|>)."""
    def format(self, example: Dict[str, Any]) -> str:
        messages = example.get("messages", example.get("conversations", []))
        text = ""
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            if role in ["human", "user"]:
                role_tag = "user"
            elif role in ["gpt", "assistant"]:
                role_tag = "assistant"
            elif role in ["system"]:
                role_tag = "system"
            else:
                role_tag = role

            content = msg.get("content", msg.get("value", ""))
            text += f"<|im_start|>{role_tag}\n{content}<|im_end|>\n"
        return text


class Llama3Formatter(PromptFormatter):
    """Formats LLaMA-3 header conversations (<|start_header_id|>...<|end_header_id|>)."""
    def format(self, example: Dict[str, Any]) -> str:
        messages = example.get("messages", example.get("conversations", []))
        text = "<|begin_of_text|>"
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            if role in ["human", "user"]:
                role_tag = "user"
            elif role in ["gpt", "assistant"]:
                role_tag = "assistant"
            elif role in ["system"]:
                role_tag = "system"
            else:
                role_tag = role

            content = msg.get("content", msg.get("value", ""))
            text += f"<|start_header_id|>{role_tag}<|end_header_id|>\n\n{content}<|eot_id|>"
        return text


class DPOFormatter:
    """Formats preference data into (prompt, chosen, rejected)."""
    def format(self, example: Dict[str, Any]) -> Tuple[str, str, str]:
        prompt = example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        return prompt, chosen, rejected


FORMATTER_MAP = {
    "alpaca": AlpacaFormatter,
    "chatml": ChatMLFormatter,
    "llama3": Llama3Formatter,
    "dpo": DPOFormatter
}


def get_formatter(template_name: str) -> Any:
    name_clean = template_name.lower().strip()
    if name_clean in FORMATTER_MAP:
        return FORMATTER_MAP[name_clean]()
    return AlpacaFormatter()
