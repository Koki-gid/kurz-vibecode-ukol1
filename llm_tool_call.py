#!/usr/bin/env python3
"""Simple OpenAI Responses API demo with tool calling."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI


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


def format_tool_result(tool_result: str) -> str:
    """Convert JSON tool output into a plain Czech summary for the model."""
    parsed = json.loads(tool_result)
    if "error" in parsed:
        return f"Nastroj vratil chybu: {parsed['error']}"
    return (
        "Nastroj calculate vratil tento vysledek: "
        f"operace {parsed['operation']}, cislo A {parsed['a']}, "
        f"cislo B {parsed['b']}, vysledek {parsed['result']}."
    )


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

    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    client = OpenAI(timeout=timeout)
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

    try:
        response = client.responses.create(
            model=model,
            input=user_prompt,
            tools=tools,
            instructions=(
                "If arithmetic is needed, call the calculate tool. "
                "Use the tool result to produce the final answer in Czech. "
                "Do not stop after the tool call."
            ),
        )

        while True:
            tool_outputs: list[dict[str, str]] = []
            tool_summaries: list[str] = []
            for item in response.output:
                if item.type != "function_call":
                    continue

                arguments = json.loads(item.arguments)
                result = run_tool(item.name, arguments)
                tool_summaries.append(format_tool_result(result))
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    }
                )

            if not tool_outputs:
                break

            response = client.responses.create(
                model=model,
                previous_response_id=response.id,
                input=tool_outputs
                + [
                    {
                        "type": "message",
                        "role": "user",
                        "content": (
                            "Pouzij vysledek nastroje a odpovez uzivateli cesky "
                            "jednou kratkou vetou bez JSONu. "
                            + " ".join(tool_summaries)
                        ),
                    }
                ],
            )
    except APITimeoutError:
        print(
            "Request timed out while connecting to the OpenAI API. "
            "Try again, check your internet connection, or set a longer timeout "
            "for example: export OPENAI_TIMEOUT=120"
        )
        return 1
    except APIConnectionError:
        print(
            "Could not connect to the OpenAI API. "
            "Check your internet connection and API access, then try again."
        )
        return 1

    print(get_final_text(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
