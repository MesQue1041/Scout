# Research assistant: documents + web, with a real agent loop

A chat assistant pointed at a folder of your own files that can also search the
live web. Per question, the model decides which source to use, it can be your documents,
the web, both, or neither. It remembers the conversation, streams its answers,
refuses off-topic requests, and comes with an eval suite that measures whether
it's choosing the right source and answering correctly.

## Quick start

    pip install -r requirements.txt
    cp .env.example .env
    python ingest.py              # index sample_docs/
    python cli.py                 # terminal chat
    uvicorn server:app --reload   # web UI at http://localhost:8000
    python evals/run_evals.py     # score the agent
    python tests/test_offline.py  # offline tests

Edit `.env` and add a free Groq key from console.groq.com. A Tavily key is
optional; without one, `search_web` falls back to DuckDuckGo.

The first `ingest.py` run downloads the embedding model (~90 MB) from Hugging
Face and caches it locally. Groq retires models over time. If you get a
"model not found" error, pick a current tool-calling model from Groq's model
list and update `LLM_MODEL` in `.env`. Any OpenAI-compatible endpoint works
(OpenAI, Ollama, etc.) by changing `LLM_BASE_URL`.

## How it works

    question ─► guardrail (small model, JSON verdict) ──blocked──► refusal
                    │ allowed
                    ▼
            ┌─► LLM (streaming, with 2 tool schemas) ──text──► streamed to user
            │       │ tool calls
            │       ▼
            │   search_documents ─► Chroma (MiniLM embeddings, cosine)
            │   search_web       ─► Tavily, or DuckDuckGo fallback
            └────── results appended to history, loop (max 4 rounds)

| File | Job |
|---|---|
| `ingest.py` | Load .md/.txt/.pdf, paragraph-aware chunking, embed, store in Chroma |
| `tools.py` | The two tools and their JSON schemas |
| `agent.py` | The agent loop: streaming, tool-call reassembly, memory |
| `guardrail.py` | Off-topic classifier that runs before the agent |
| `cli.py` / `server.py` / `static/index.html` | Three front ends over the same event stream |
| `evals/` | 14 test cases and the harness that scores them |

## Design decisions

**The agent is a generator of events.** `Agent.chat()` yields `token`,
`tool_call`, `tool_result`, `blocked`, and `done` events. The CLI prints them,
the server forwards them as Server-Sent Events, and the eval harness records
them. One loop, three consumers, and the evals test exactly the code the demo
runs.

**Streaming with tools is the hard part.** When a streamed response contains
tool calls, the function name and JSON arguments arrive in fragments across
many chunks, and parallel calls are interleaved by `index`. `_stream_once`
concatenates fragments per index and only parses the JSON after the stream
ends. A large part of `test_offline.py` exists to exercise this.

**Memory is trimmed at user-message boundaries.** Keeping "the last N messages"
can cut between an assistant's tool call and its tool result, and the API
rejects a tool result with no matching call. Trimming by user turn keeps every
turn whole. If a turn fails midway, the whole turn is rolled back for the same
reason.

**The last loop round withholds tools,** so a model stuck re-searching is
forced to answer instead of looping forever. On some providers (Groq's
`gpt-oss` models, for example) this final round is non-streaming and drops the
`tools` field entirely, because otherwise the model tries to emit a tool call
mid-stream and the API rejects it.

**Tool errors are returned to the model, not raised.** If the web search fails,
the model sees the error string and can say so, or try the other tool, or
answer from what it already has.

**The guardrail is a separate, small model and fails open.** It is cheap,
independently testable, and gets the previous reply as context so short
follow-ups ("and the second one?") aren't mistaken for off-topic. If it errors,
the question goes through, because blocking a real question costs more than
answering a silly one. Worth knowing: whether it blocks a given category of
request depends on the guard model. With the current default it catches
roleplay reliably and creative writing less so.

**The evals score routing and content separately.** A failure tells you whether
the agent picked the wrong source or picked the right source and answered
badly. The cases cover each routing path, a question whose answer isn't in the
documents (it should say so, not invent), a multi-turn memory check, and a
guardrail false-positive check (a plain "thanks" must not be blocked). Each run
writes full answers to `evals/results_*.json` so you can diff before and after
a prompt or chunk-size change.

On `sample_docs/` with `openai/gpt-oss-120b` as the LLM and
`openai/gpt-oss-20b` as the guard, the suite scores 12/14 (86%), with the two
misses being a keyword-phrasing issue in one case and the guardrail's
creative-writing gap in the other. Different models move the number by a few
points in either direction.

## Known limits

Keyword matching is a blunt measure of answer quality; an LLM-as-judge scorer
would be the next step. Web-routing cases have no content check because the
right answer changes over time. Sessions live in memory, so restarting the
server clears them, and two tabs opening `/chat` on the same session ID can
interleave turns (there's no per-session lock). There's no reranker, so
retrieval quality depends entirely on the MiniLM embeddings and chunk size.

## Using your own files

Point `ingest.py` at a folder:

    python ingest.py --docs path/to/your/files

Then rewrite the document cases in `evals/cases.yaml` so the `must_include`
and `any_of` keywords match your content. Otherwise the eval suite is scoring
your files against someone else's expected answers.
