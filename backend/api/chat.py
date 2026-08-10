"""POST /api/chat/stream — SSE 流式 Agent 执行。"""

import json
import uuid
from typing import Generator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from backend.agents_config import MEMORY, REGISTRY, build_agents

router = APIRouter()

_AGENTS = build_agents()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _event_gen(agent_id: str, message: str, session_id: str) -> Generator[str, None, None]:
    # 同步 generator：StreamingResponse 会在线程池中迭代，避免阻塞事件循环
    yield sse({"type": "session", "session_id": session_id})
    agent = _AGENTS.get(agent_id)
    if agent is None:
        yield sse({"type": "error", "content": f"Agent '{agent_id}' 不存在"})
        return
    agent.session_id = session_id
    for event in agent.run_stream(message):
        if event["type"] == "done":
            event["session_id"] = session_id
        yield sse(event)
    yield sse({"type": "end"})


@router.post("/stream")
async def chat_stream(request: Request):
    body = await request.json()
    agent_id = body.get("agent_id", "PM")
    message = body.get("message", "")
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not message.strip():
        return StreamingResponse(iter([sse({"type": "error", "content": "消息不能为空"})]),
                                 media_type="text/event-stream")

    return StreamingResponse(
        _event_gen(agent_id, message, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
