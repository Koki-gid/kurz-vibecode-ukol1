#!/usr/bin/env python3
"""Simple OpenAI Responses API demo with tool calling."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI


def calculate(operation: str, a: float, b: float) -> str:
    """Run a basic math operation requested by the model."""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return json.dumps({"error": "Division by zero is not allowed."})
        result = a / b
    else:
        return json.dumps({"error": f"Unsupported operation: {operation}"})

    return json.dumps(
        {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result,
        }
    )


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Route the model's tool call to the local Python function."""
    if name == "calculate":
        return calculate(**arguments)
    return json.dumps({"error": f"Unknown tool: {name}"})


def get_final_text(response: Any) -> str:
    """Return text output in a way that survives small SDK response changes."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    collected: list[str] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                collected.append(text)
    return "\n".join(collected).strip()


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python llm_tool_call.py "What is (15 + 7) * 3?"')
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY environment variable.")
        return 1

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    user_prompt = sys.argv[1]

    tools = [
        {
            "type": "function",
            "name": "calculate",
            "description": "Perform a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Arithmetic operation to execute.",
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number.",
                    },
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    response = client.responses.create(
        model=model,
        input=user_prompt,
        tools=tools,
        instructions=(
            "If arithmetic is needed, call the calculate tool. "
            "After receiving the tool result, answer clearly in Czech."
        ),
    )

    tool_outputs: list[dict[str, str]] = []
    for item in response.output:
        if item.type != "function_call":
            continue

        arguments = json.loads(item.arguments)
        result = run_tool(item.name, arguments)
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            }
        )

    if tool_outputs:
        response = client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
        )

    print(get_final_text(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
