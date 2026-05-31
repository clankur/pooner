"""Parse tool calls from raw model output.

Supports both JSON and Qwen3.5 XML formats inside <tool_call> tags.
Replace this module if a library (vLLM, qwen-agent, etc.) covers parsing.
"""

import json
import re

_TAG_OPEN = "<tool_call>"
_TAG_CLOSE = "</tool_call>"

_FUNC_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def extract_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract (name, arguments) from model output, or None if no tool call found.

    Handles two formats:
      JSON:  <tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>
      XML:   <tool_call><function=foo><parameter=x>1</parameter></function></tool_call>
    """
    start = text.find(_TAG_OPEN)
    if start == -1:
        return None
    end = text.find(_TAG_CLOSE, start)
    if end == -1:
        return None

    raw = text[start + len(_TAG_OPEN) : end].strip()

    # JSON format
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "name" in obj:
            return obj["name"], obj.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        pass

    # XML format
    m = _FUNC_RE.search(raw)
    if m:
        name = m.group(1)
        args: dict[str, str] = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            args[pm.group(1)] = pm.group(2)
        return name, args

    return None
