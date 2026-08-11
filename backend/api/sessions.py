"""会话管理 API — 历史聊天记录的列表 / 加载 / 删除。"""

from fastapi import APIRouter, HTTPException

from backend.agents_config import CHECKPOINT

router = APIRouter()


@router.get("")
async def list_sessions():
    return {"sessions": CHECKPOINT.list()}


@router.get("/{session_id}")
async def get_session(session_id: str):
    data = CHECKPOINT.load_full(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    turns = data.get("turns", [])
    if turns:
        return {"session_id": session_id, "turns": turns}
    # 旧数据无 turns：回退为纯文本消息
    messages = data.get("messages", [])
    clean = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return {"session_id": session_id, "turns": [], "messages": clean}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    CHECKPOINT.delete(session_id)
    return {"ok": True}
