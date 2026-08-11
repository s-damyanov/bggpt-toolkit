"""Offline tests for run_tool_loop, against a fake OpenAI-shaped client — no network."""

from __future__ import annotations

from types import SimpleNamespace

from bggpt_toolkit.toolcalls import run_tool_loop


def _tool_call_delta(index, id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments) if (name or arguments) else None
    return SimpleNamespace(index=index, id=id, function=function)


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _fake_stream(chunks):
    for c in chunks:
        yield c


class FakeCompletions:
    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_stream(self._responses.pop(0))


class FakeClient:
    def __init__(self, responses: list[list]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


TOOLS = [{"type": "function", "function": {"name": "get_status", "parameters": {}}}]


async def test_forces_tool_choice_required_on_first_round_only() -> None:
    responses = [
        [_chunk(tool_calls=[_tool_call_delta(0, id="call1", name="get_status", arguments="{}")])],
        [_chunk(content="Done.")],
    ]
    client = FakeClient(responses)
    calls: list[tuple[str, dict]] = []

    async def execute_tool(name: str, arguments: dict) -> str:
        calls.append((name, arguments))
        return "OK"

    messages: list[dict] = [{"role": "user", "content": "status?"}]
    events = [
        e
        async for e in run_tool_loop(
            client, model="bggpt-27b", messages=messages, tools=TOOLS,
            execute_tool=execute_tool, max_rounds=1,
        )
    ]

    round0_kwargs = client.chat.completions.calls[0]
    assert round0_kwargs["tool_choice"] == "required"
    assert round0_kwargs["tools"] == TOOLS

    # bonus round omits tools entirely, so the model structurally cannot fabricate another call
    round1_kwargs = client.chat.completions.calls[1]
    assert "tools" not in round1_kwargs
    assert "tool_choice" not in round1_kwargs

    assert calls == [("get_status", {})]
    assert {"type": "tool_call", "name": "get_status"} in events
    assert {"type": "delta", "text": "Done."} in events

    # messages mutated with the assistant tool-call turn and the tool result
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "get_status"
    assert messages[-1] == {"role": "tool", "tool_call_id": "call1", "content": "OK"}


async def test_no_tools_returns_after_one_round() -> None:
    responses = [[_chunk(content="Hi"), _chunk(content=" there.")]]
    client = FakeClient(responses)

    async def execute_tool(name: str, arguments: dict) -> str:
        raise AssertionError("should not be called when there are no tools")

    events = [
        e
        async for e in run_tool_loop(
            client, model="bggpt-27b", messages=[{"role": "user", "content": "hi"}], tools=[],
            execute_tool=execute_tool, max_rounds=3,
        )
    ]

    assert len(client.chat.completions.calls) == 1
    assert "tools" not in client.chat.completions.calls[0]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hi there."


async def test_invalid_tool_arguments_reported_without_calling_executor() -> None:
    responses = [
        [_chunk(tool_calls=[_tool_call_delta(0, id="call1", name="get_status", arguments="{bad")])],
        [_chunk(content="Fallback answer.")],
    ]
    client = FakeClient(responses)

    async def execute_tool(name: str, arguments: dict) -> str:
        raise AssertionError("should not be called with malformed arguments")

    messages: list[dict] = [{"role": "user", "content": "status?"}]
    async for _ in run_tool_loop(
        client, model="bggpt-27b", messages=messages, tools=TOOLS,
        execute_tool=execute_tool, max_rounds=1,
    ):
        pass

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["content"].startswith("Invalid arguments")


async def test_rate_limit_called_before_every_request() -> None:
    responses = [[_chunk(content="Hi")]]
    client = FakeClient(responses)
    calls = 0

    async def rate_limit() -> None:
        nonlocal calls
        calls += 1

    async def execute_tool(name: str, arguments: dict) -> str:
        return "OK"

    async for _ in run_tool_loop(
        client, model="bggpt-27b", messages=[{"role": "user", "content": "hi"}], tools=[],
        execute_tool=execute_tool, max_rounds=1, rate_limit=rate_limit,
    ):
        pass

    assert calls == 1
