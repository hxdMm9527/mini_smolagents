"""GET /api/agents — 可对话的 Agent 角色列表（仅主 Agent）。"""

from fastapi import APIRouter

from backend.agents_config import MAIN_AGENT_NAME, REGISTRY

router = APIRouter()


@router.get("")
async def list_agents():
    cards = [c for c in REGISTRY.list_cards() if c.name == MAIN_AGENT_NAME]
    return [
        {
            "name": card.name,
            "description": card.description,
            "capabilities": card.capabilities,
            "tools": card.tools,
        }
        for card in cards
    ]
