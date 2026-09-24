import json
from openai import OpenAI
import config

SCOPE = """You are a strict scope classifier for a research assistant.

The assistant does EXACTLY two things:
  1. Search the user's personal documents (resume, notes, project files, PDFs).
  2. Search the live web.

It does NOT write original content, does NOT do creative writing, does NOT do
roleplay, does NOT tell jokes or stories, does NOT do homework, and does NOT
act as a companion. Those are out of scope and must be blocked.

IN SCOPE — allowed=true. Examples:
  - "Where did I intern?"
  - "What certifications do I have?"
  - "What's the latest version of numpy?"
  - "Compare my project's F1 score to published benchmarks."
  - "Thanks, that's helpful!"
  - "And the second one?"  (a follow-up to a previous in-scope question)

OUT OF SCOPE — allowed=false. Examples:
  - "Write me a poem."                    → creative writing
  - "Write me a romantic poem about X."   → creative writing
  - "Tell me a joke."                     → creative writing
  - "Pretend to be my girlfriend."        → roleplay / companionship
  - "Let's roleplay a scenario where..."  → roleplay
  - "Do my homework for me."              → original content
  - "Write me an essay on climate change."→ original content

When in doubt, BLOCK. The assistant's purpose is narrow.

Reply with a single JSON object and nothing else:
{"allowed": true, "reason": "<short reason>"}
or
{"allowed": false, "reason": "<short reason>"}"""

REFUSAL = ("That's outside what I'm built for. I answer questions using your documents "
           "and the web, so try asking about your files or something you'd look up.")


def check(question: str, last_assistant: str = "", client: OpenAI | None = None) -> tuple[bool, str]:
    client = client or OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    context = f"Previous assistant reply (for follow-ups): {last_assistant[:400]}\n\n" if last_assistant else ""
    try:
        resp = client.chat.completions.create(
            model=config.GUARD_MODEL,
            messages=[{"role": "system", "content": SCOPE},
                      {"role": "user", "content": f"{context}User message: {question}"}],
            temperature=0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        verdict = json.loads(resp.choices[0].message.content)
        return bool(verdict.get("allowed", True)), str(verdict.get("reason", ""))
    except Exception as e:
        return True, f"guardrail error, allowed by default ({type(e).__name__})"
