"""GET /api/agents — 可用 Agent 角色列表。"""

from fastapi import APIRouter

from backend.agents_config import REGISTRY

router = APIRouter()


@router.get("")
async def list_agents():
    cards = REGISTRY.list_cards()
    return [
        {
            "name": card.name,
            "description": card.description,
            "capabilities": card.capabilities,
            "tools": card.tools,
        }
        for card in cards
    ]
