"""A scripted stand-in for a chat model, so agent tests are deterministic."""

import json

from langchain_core.messages import AIMessageChunk


def text(*parts):
    """One model turn that just answers, streamed in pieces."""
    return [AIMessageChunk(content=part) for part in parts]


def call(name, args, call_id="call-1"):
    """One model turn that asks for a tool."""
    return [AIMessageChunk(content="", tool_call_chunks=[
        {"name": name, "args": json.dumps(args), "id": call_id, "index": 0}
    ])]


class ScriptedLLM:
    """
    Plays back `turns` in order, one per model call. Records every message
    list it was given, which tool names were bound, and whether each call
    was made with tools bound.
    """

    def __init__(self, *turns, bind_error=None, stream_error=None):
        self.turns = list(turns)
        self.calls = []
        self.with_tools = []
        self.bound_names = None
        self.bind_error = bind_error
        self.stream_error = stream_error

    def bind_tools(self, tools):
        if self.bind_error:
            raise self.bind_error
        self.bound_names = {tool.name for tool in tools}
        return _Bound(self)

    def stream(self, messages, with_tools=False):
        if self.stream_error:
            raise self.stream_error
        self.calls.append(list(messages))
        self.with_tools.append(with_tools)
        yield from self.turns.pop(0)


class _Bound:
    def __init__(self, parent):
        self.parent = parent

    def stream(self, messages):
        return self.parent.stream(messages, with_tools=True)
