import os
from dotenv import load_dotenv

load_dotenv()

# LLM 
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GUARD_MODEL = os.getenv("GUARD_MODEL", "llama-3.1-8b-instant")  

# Retrieval
DOCS_DIR = os.getenv("DOCS_DIR", "sample_docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")    # vector database that stores text as number-lists ("embeddings")
COLLECTION = os.getenv("COLLECTION", "my_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")   # turns a chunk of text into a list of around 384 numbers
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # only used when splitting long paragraphs
TOP_K = int(os.getenv("TOP_K", "4"))

# Web search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_RESULTS = int(os.getenv("WEB_RESULTS", "5"))

# Agent loop
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))  # user turns kept in memory
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "4"))      
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
