"""POST /api/memory/search — RAG 语义检索。"""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.agents_config import MEMORY

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_memory(req: SearchRequest):
    hits = MEMORY.search(req.query, top_k=req.top_k)
    return {"hits": [{"task": h.task, "document": h.document, "score": h.score} for h in hits]}
