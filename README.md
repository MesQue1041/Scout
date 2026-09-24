# Scout: a document + web research assistant with a real agent loop

A chat assistant that answers questions from a folder of your own files and
can also search the live web when it needs to. For each question the model
decides where to look, your documents, the web, both, or neither. It keeps
track of the conversation, streams its answers as they're generated, refuses
requests that are outside its scope, and ships with an eval suite that checks
whether it picked the right source and got the content right.

## Quick start

    pip install -r requirements.txt
    cp .env.example .env
    python ingest.py              # index sample_docs/
    python cli.py                 # terminal chat
    uvicorn server:app --reload   # web UI at http://localhost:8000
    python evals/run_evals.py     # score the agent
    python tests/test_offline.py  # offline tests

Add a free Groq key from console.groq.com to .env. A Tavily key is optional,
without one search_web falls back to DuckDuckGo.

The first ingest.py run downloads the embedding model (about 90 MB) from
Hugging Face and caches it locally. Groq retires models over time, so if you
get a "model not found" error, grab a current tool-calling model from Groq's
model list and update LLM_MODEL in .env. Any OpenAI-compatible endpoint works
in its place, OpenAI, Ollama, whatever, just change LLM_BASE_URL.

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
| ingest.py | Load .md/.txt/.pdf, paragraph-aware chunking, embed, store in Chroma |
| tools.py | The two tools and their JSON schemas |
| agent.py | The agent loop, streaming, tool-call reassembly, memory |
| guardrail.py | Off-topic classifier that runs before the agent |
| cli.py / server.py / static/index.html | Three front ends over the same event stream |
| evals/ | 14 test cases and the harness that scores them |

## Design decisions

**The agent is a generator of events.** Agent.chat() yields token, tool_call,
tool_result, blocked, and done events. The CLI prints them, the server
forwards them as Server-Sent Events, and the eval harness records them. One
loop, three consumers, and the evals test exactly the code the demo runs.

**Streaming with tools is the hard part.** When a streamed response contains
tool calls, the function name and JSON arguments arrive in fragments spread
across many chunks, and parallel calls get interleaved by index.
_stream_once concatenates fragments per index and only parses the JSON once
the stream ends. A big chunk of test_offline.py exists just to exercise this.

**Memory is trimmed at user-message boundaries.** Keeping "the last N
messages" can cut between an assistant's tool call and its tool result, and
the API rejects a tool result that has no matching call. Trimming by user
turn keeps every turn intact. If a turn fails partway through, the whole
turn gets rolled back for the same reason.

**The last loop round withholds tools**, so a model that's stuck re-searching
is forced to actually answer instead of looping forever. On some providers
(Groq's gpt-oss models did this) that final round runs non-streaming and
drops the tools field entirely, because otherwise the model tries to emit a
tool call mid-stream and the API rejects it.

**Tool errors get returned to the model, not raised.** If web search fails,
the model sees the error string and can say so, try the other tool, or
answer with what it already has.

**The guardrail is a separate, small model, and it fails open.** It's cheap,
testable on its own, and gets the previous reply as context so a short
follow-up like "and the second one?" doesn't get mistaken for off-topic. If
it errors out, the question goes through anyway, blocking a real question
costs more than answering a silly one. Worth knowing that whether it catches
a given category depends entirely on the guard model. With the current
default it catches roleplay reliably and creative writing less so.

**The evals score routing and content separately.** A failure tells you
whether the agent picked the wrong source, or picked the right source and
answered badly. The cases cover each routing path, a question whose answer
genuinely isn't in the documents (it should say so instead of inventing
something), a multi-turn memory check, and a guardrail false-positive check
(a plain "thanks" must not get blocked). Every run writes full answers to
evals/results_*.json so you can diff before and after a prompt or
chunk-size change.

## A real run

Offline tests, which check the streaming/tool-parsing plumbing without
touching any real API:

    chunks: 4
    Error: arguments were not valid JSON: 'not json'
    Error: unknown tool 'nope'.
    ['tool_call', 'tool_result', 'tool_call', 'tool_result', 'token', 'token', 'done']
    history roles: ['user', 'assistant', 'tool', 'tool', 'assistant', 'user', 'assistant']
    after trim: ['user', 'assistant']
    ALL OK

Full eval suite, with openai/gpt-oss-120b as the LLM and openai/gpt-oss-20b
as the guard:

    PASS internship            26.8s
    PASS lstm_f1                2.7s
    PASS deadline                2.5s
    PASS certs                  3.6s
    PASS supervisor              2.2s
    FAIL not_in_docs             6.1s   content: expected keywords missing
    PASS web_version             5.2s
    PASS web_news                8.4s
    PASS both_compare           27.3s
    PASS memory_followup        12.9s
    PASS smalltalk_allowed      17.5s
    FAIL offtopic_poem           1.8s   routing: expected blocked, got none
    PASS offtopic_roleplay       1.1s

    Routing accuracy: 93%   Content: 93%   Overall pass rate: 86% (12/14)

The two failures are the known gaps mentioned above rather than anything
surprising. not_in_docs failed on a keyword technicality, the agent said the
salary figure wasn't in the documents, just not using one of the exact
phrases the eval was checking for, so this is a scoring issue, not a made-up
answer. offtopic_poem failed because the guardrail's current guard model
lets creative-writing requests through more often than roleplay requests, so
"write me a romantic poem about my cat" got answered instead of blocked.
Different guard models move this number a few points in either direction.
Different runs with the same setup can also swing a point or two since
nothing here is deterministic.

## Known limits

Keyword matching is a blunt way to measure answer quality, an LLM-as-judge
scorer would be the natural next step. Web-routing cases have no content
check because the right answer changes over time. Sessions live in memory,
so restarting the server clears them, and two tabs opening /chat on the same
session ID can interleave turns since there's no per-session lock. There's
no reranker, so retrieval quality depends entirely on the MiniLM embeddings
and chunk size.

## Using your own files

Point ingest.py at a folder:

    python ingest.py --docs path/to/your/files

Then rewrite the document cases in evals/cases.yaml so the must_include and
any_of keywords match your content. Otherwise the eval suite is scoring your
files against someone else's expected answers.
