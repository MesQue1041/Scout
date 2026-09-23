import json
from openai import OpenAI
import config

SCOPE = """You are a gatekeeper for a research assistant. The assistant can search
the user's personal documents (resume, notes, project files, PDFs) and the web.

ALLOW:
- questions about the user's own documents, projects, career, or notes
- factual or research questions that could be answered by searching the web
- follow-ups to the previous turn, even if short or vague ("and the second one?")
- brief pleasantries ("thanks", "hi")

BLOCK:
- creative writing requests (poems, stories, jokes on demand)
- role-play or companionship ("pretend to be my girlfriend")
- requests to write code, do homework, or produce long original content
- anything harmful or illegal

Reply with JSON only: {"allowed": true|false, "reason": "<short reason>"}"""

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
