import json
import config

MIN_SCORE = 0.2  # cosine similarity below this is almost certainly noise

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the user's personal files (resume, notes, PDFs). "
                "Use for anything about the user themselves, their projects, work, "
                "meetings, or content they've saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "A focused search query, rephrased as a standalone question if needed."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the live web. Use for current events, recent releases or versions, "
                "prices, and public facts that are not about the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A concise web search query."}
                },
                "required": ["query"],
            },
        },
    },
]


def search_documents(query: str, k: int = config.TOP_K) -> str:
    from ingest import get_collection  # avoids loading the embedding model until needed

    col = get_collection()
    if col.count() == 0:
        return "The document index is empty. Tell the user to run `python ingest.py`."
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - dist  # cosine distance is similarity
        if score < MIN_SCORE:
            continue
        label = meta["source"] + (f" p.{meta['page']}" if meta.get("page") else "")
        hits.append(f"[{label}] (relevance {score:.2f})\n{text}")
    if not hits:
        return "No relevant passages found in the user's documents."
    return "\n\n---\n\n".join(hits)


def search_web(query: str, n: int = config.WEB_RESULTS) -> str:
    results = []
    if config.TAVILY_API_KEY:
        from tavily import TavilyClient
        resp = TavilyClient(api_key=config.TAVILY_API_KEY).search(query, max_results=n)
        results = [{"title": r["title"], "url": r["url"], "snippet": r["content"]}
                   for r in resp.get("results", [])]
    else:
        try:
            from ddgs import DDGS
        except ImportError:  
            from duckduckgo_search import DDGS
        results = [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                   for r in DDGS().text(query, max_results=n)]
    if not results:
        return "No web results found."
    return "\n\n---\n\n".join(f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results)


TOOL_FUNCS = {"search_documents": search_documents, "search_web": search_web}


def run_tool(name: str, arguments: str) -> str:
    if name not in TOOL_FUNCS:
        return f"Error: unknown tool '{name}'."
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: arguments were not valid JSON: {arguments!r}"
    try:
        return TOOL_FUNCS[name](**args)
    except Exception as e: 
        return f"Error running {name}: {type(e).__name__}: {e}"
