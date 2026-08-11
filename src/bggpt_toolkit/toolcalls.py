"""A tool-calling loop that works around two live BgGPT reliability gaps.

Both were found live-testing BgGPT against api.bggpt.ai with a real MCP tool wired in:

1. Given a question that clearly needs a tool, and only a soft instruction to use it, BgGPT
   sometimes narrates a *fake* call as literal answer text (e.g. `"[get_status()]"`) instead of
   emitting a real structured `tool_calls` delta — there is no tool result to parse, just a string
   that looks like one. Forcing `tool_choice="required"` on the first round, once the caller has
   already decided a tool is needed, produces a real structured call instead.

2. A model that keeps calling tools every round can exhaust the round budget and leave the caller
   with no text at all — total silence, not even a partial answer. Appending one bonus final round
   with `tools` omitted entirely guarantees a real text answer grounded in whatever tool results
   were already gathered, because the model physically cannot return another `tool_calls` delta.

gpt-5.5-class models handle both cases fine on soft prompting alone; BgGPT does not (yet) — hence
enforcing both in code rather than relying on prompt wording.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


async def run_tool_loop(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute_tool: ToolExecutor,
    max_rounds: int = 3,
    temperature: float = 0.2,
    rate_limit: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a chat-completions tool-calling turn, mutating `messages` in place with each
    assistant/tool step so the caller can inspect the final conversation afterward.

    Args:
        client: an `AsyncOpenAI` client (see `bggpt_toolkit.client.async_client`).
        tools: chat-completions tool definitions; pass `[]` for a plain no-tools turn.
        execute_tool: called as `await execute_tool(name, arguments)`, must return the tool
            result as a string (including any error message it wants BgGPT to see).
        max_rounds: number of rounds that may call a tool. One extra, tools-omitted round always
            runs after that budget is spent, guaranteeing a real text answer (see module docstring).
        rate_limit: optional `await rate_limit()` called before every request, e.g.
            `bggpt_toolkit.ratelimit.bggpt_rate_limiter.acquire_async`.

    Yields:
        `{"type": "delta", "text": str}` for each chunk of the model's final answer text.
        `{"type": "tool_call", "name": str}` when a tool call has been received and is about to run.
    """
    for round_idx in range(max_rounds + 1):
        if rate_limit is not None:
            await rate_limit()
        kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": temperature, "stream": True,
        }
        if tools and round_idx < max_rounds:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "required" if round_idx == 0 else "auto"
        stream = await client.chat.completions.create(**kwargs)
        buffer = ""
        tool_calls: dict[int, dict[str, str | None]] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
                continue  # tool-call deltas carry no .content
            text = delta.content
            if not text:
                continue
            buffer += text
            yield {"type": "delta", "text": text}

        if not tool_calls:
            return  # final answer already streamed above

        messages.append(
            {
                "role": "assistant",
                "content": buffer or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in tool_calls.values()
                ],
            }
        )
        for c in tool_calls.values():
            yield {"type": "tool_call", "name": c["name"]}
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                result_text = f"Invalid arguments from model: {c['arguments']!r}"
            else:
                result_text = await execute_tool(c["name"], args)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result_text})
