"""会话管理 API — 历史聊天记录的列表 / 加载 / 删除。"""

from fastapi import APIRouter, HTTPException

from backend.agents_config import CHECKPOINT

router = APIRouter()


@router.get("")
async def list_sessions():
    return {"sessions": CHECKPOINT.list()}


@router.get("/{session_id}")
async def get_session(session_id: str):
    messages = CHECKPOINT.load(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    clean = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return {"session_id": session_id, "messages": clean}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    CHECKPOINT.delete(session_id)
    return {"ok": True}
