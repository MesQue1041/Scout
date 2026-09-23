import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import Agent

app = FastAPI(title="Research assistant")

sessions: dict[str, Agent] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


def get_agent(session_id: str) -> Agent:
    if session_id not in sessions:
        sessions[session_id] = Agent()
    return sessions[session_id]


@app.post("/chat")
def chat(req: ChatRequest):
    agent = get_agent(req.session_id)

    def event_stream():
        for ev in agent.chat(req.message):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/reset")
def reset(req: ChatRequest):
    sessions.pop(req.session_id, None)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
