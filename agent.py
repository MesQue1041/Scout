import json
from datetime import date
from typing import Iterator

from openai import OpenAI, BadRequestError

import config
import guardrail
from tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = """You are a research assistant with two tools:
- search_documents: the user's own files (resume, notes, project write-ups, PDFs)
- search_web: the live web

Today's date is {today}.

How to choose:
- About the user ("my", "I", their projects, meetings, skills) -> search_documents.
- Current events, latest versions, prices, or public facts -> search_web.
- Comparing the user's material with the outside world -> call BOTH.
- Answerable from the conversation so far, or small talk -> no tool.
- If a search comes back weak, rephrase and search again, or try the other tool.

How to answer:
- Base answers on tool results, not memory. Cite inline: [filename] for documents,
  [title](url) for web pages.
- If the documents don't contain the answer, say so plainly. Never invent details
  about the user.
- Be concise."""


class Agent:
    def __init__(self, client: OpenAI | None = None, use_guardrail: bool = True):
        self.client = client or OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.use_guardrail = use_guardrail
        self.history: list[dict] = []  # user / assistant / tool messages

    def reset(self) -> None:
        self.history = []

    # memory 
    def _trim_history(self) -> None:    # keep only the last N user turns
        user_idx = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_idx) > config.MAX_HISTORY_TURNS:
            self.history = self.history[user_idx[-config.MAX_HISTORY_TURNS]:]

    def _last_assistant_text(self) -> str:   # so guardrail can allow follow-ups even if short or vague
        for m in reversed(self.history):
            if m["role"] == "assistant" and m.get("content"):
                return m["content"]
        return ""

    # one streamed model call 
    def _stream_once(self, allow_tools: bool) -> Iterator[dict]:
        kwargs = dict(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())}]
                     + self.history,
            temperature=config.TEMPERATURE,
            stream=True,
        )
        if allow_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"

        text_parts: list[str] = []
        calls: dict[int, dict] = {}  # tool calls arrive in fragments, keyed by index

        for chunk in self.client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                yield {"type": "token", "text": delta.content}

            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        message = {"role": "assistant", "content": "".join(text_parts) or None}
        if calls:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for _, c in sorted(calls.items())
            ]
        elif message["content"] is None:
            message["content"] = ""
        yield {"type": "_message", "message": message}

    # one user turn 
    def chat(self, user_message: str) -> Iterator[dict]:
        tools_used: list[str] = []

        if self.use_guardrail:
            allowed, reason = guardrail.check(user_message, self._last_assistant_text(), self.client)
            if not allowed:
                yield {"type": "blocked", "reason": reason}
                yield {"type": "token", "text": guardrail.REFUSAL}
                yield {"type": "done", "tools_used": [], "blocked": True}
                return  # blocked turns are not added to memory

        turn_start = len(self.history)
        self.history.append({"role": "user", "content": user_message})

        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            # On the last round, withhold tools so the model must answer
            allow_tools = round_no < config.MAX_TOOL_ROUNDS
            message = None
            try:
                for event in self._stream_once(allow_tools):
                    if event["type"] == "_message":
                        message = event["message"]
                    else:
                        yield event
            except BadRequestError as e:
                yield {"type": "error", "message": f"Model request failed: {e}"}
                del self.history[turn_start:]  # rolls back this whole turn so history stays valid
                yield {"type": "done", "tools_used": tools_used}
                return

            self.history.append(message)

            if not message.get("tool_calls"):
                break  # final answer already streamed

            for call in message["tool_calls"]:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                yield {"type": "tool_call", "name": name, "args": args}
                result = run_tool(name, args)
                tools_used.append(name)
                yield {"type": "tool_result", "name": name, "preview": result[:300]}
                self.history.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        self._trim_history()
        yield {"type": "done", "tools_used": tools_used}
