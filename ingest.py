import argparse
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import config

SUPPORTED = {".md", ".txt", ".pdf"}


def load_documents(folder: str) -> list[dict]:
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            for i, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"source": path.name, "page": i, "text": text})
        else:
            docs.append({"source": path.name, "page": None,
                         "text": path.read_text(encoding="utf-8", errors="ignore")})
    return docs


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(p) > size:
            if current:
                chunks.append(current)
                current = ""
            step = size - overlap
            chunks.extend(p[i:i + size] for i in range(0, max(len(p) - overlap, 1), step))
            continue
        if len(current) + len(p) + 2 <= size:
            current = f"{current}\n\n{p}" if current else p
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    return client.get_or_create_collection(
        name=config.COLLECTION,
        embedding_function=embed_fn,   # text to vectors happen here
        metadata={"hnsw:space": "cosine"},
    )


def build_index(folder: str) -> int:
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # didn't exist yet
    get_collection.cache_clear()
    col = get_collection()

    ids, texts, metas = [], [], []
    for doc in load_documents(folder):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            page = doc["page"] or 0
            ids.append(f"{doc['source']}::p{page}::c{i}")
            texts.append(chunk)
            metas.append({"source": doc["source"], "page": page, "chunk": i})

    batch = 256
    for start in range(0, len(ids), batch):
        col.add(ids=ids[start:start + batch],
                documents=texts[start:start + batch],
                metadatas=metas[start:start + batch])
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=config.DOCS_DIR)
    args = ap.parse_args()
    n = build_index(args.docs)
    print(f"Indexed {n} chunks from {args.docs} into '{config.COLLECTION}' at {config.CHROMA_DIR}/")
