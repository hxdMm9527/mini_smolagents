"""mini_smolagents backend — FastAPI + SSE。

启动：uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import agents, chat, memory

app = FastAPI(title="mini_smolagents", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat")
app.include_router(agents.router, prefix="/api/agents")
app.include_router(memory.router, prefix="/api/memory")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
