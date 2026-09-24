import sys, json, hashlib, types
from pathlib import Path; ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np
import chromadb
from chromadb.api.types import EmbeddingFunction
import config
config.CHROMA_DIR = "/tmp/chroma_test"

#  fake embedding
class FakeEmbed(EmbeddingFunction):
    def __init__(self): pass
    def __call__(self, input):
        out = []
        for t in input:
            v = np.zeros(256)
            for w in t.lower().split():
                v[int(hashlib.md5(w.strip(".,?:").encode()).hexdigest(), 16) % 256] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out
    @staticmethod
    def name(): return "fake"
    def get_config(self): return {}
    @staticmethod
    def build_from_config(c): return FakeEmbed()

import ingest
ingest.SentenceTransformerEmbeddingFunction = lambda model_name: FakeEmbed()

# chunking
ch = ingest.chunk_text("a"*2000, size=800, overlap=150)
assert all(len(c) <= 800 for c in ch) and len(ch) == 3, [len(c) for c in ch]
print("chunks:", ingest.build_index(str(ROOT / "sample_docs")))

import tools; tools.MIN_SCORE = 0.0
print(tools.search_documents("LSTM autoencoder F1 score")[:200].replace("\n"," | "))
print(tools.run_tool("search_documents", "not json"))
print(tools.run_tool("nope", "{}"))

# fake streaming client
def D(content=None, tool_calls=None):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content, tool_calls=tool_calls))])
def TC(index, id=None, name=None, args=None):
    return types.SimpleNamespace(index=index, id=id, function=types.SimpleNamespace(name=name, arguments=args))

class FakeClient:
    def __init__(self, scripts): self.scripts = scripts; self.calls = []
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("response_format"):  # guardrail call
            q = kw["messages"][-1]["content"]
            ok = "poem" not in q
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({"allowed": ok, "reason": "test"})))])
        return iter(self.scripts.pop(0))

scripts = [
    # turn 1: two parallel tool calls, args fragmented
    [D(tool_calls=[TC(0, "c1", "search_documents", '{"que')]), D(tool_calls=[TC(0, args='ry": "LSTM F1"}')]),
     D(tool_calls=[TC(1, "c2", "search_web", '{"query": "HDFS benchmark"}')]), types.SimpleNamespace(choices=[])],
    [D("Your LSTM got "), D("0.86 [lighthouse_project.md].")],
    # turn 2: no tools
    [D("You're welcome!")],
]
import agent as A
A.run_tool = lambda n, a: f"RESULT for {n} {a}"
fc = FakeClient(scripts)
ag = A.Agent(client=fc)
evs = list(ag.chat("What F1 did my LSTM get vs published?"))
print([e["type"] for e in evs])
assert evs[-1]["tools_used"] == ["search_documents", "search_web"]
assert json.loads(ag.history[1]["tool_calls"][0]["function"]["arguments"]) == {"query": "LSTM F1"}
list(ag.chat("thanks"))
print("history roles:", [m["role"] for m in ag.history])
b = list(ag.chat("write a poem"))
assert b[0]["type"] == "blocked" and len(ag.history) == 7

# trimming never orphans tool messages
config.MAX_HISTORY_TURNS = 1
ag._trim_history(); print("after trim:", [m["role"] for m in ag.history])
assert ag.history[0]["role"] == "user"
print("ALL OK")
