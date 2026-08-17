import uuid
from dataclasses import dataclass, field


@dataclass
class AgentCard:
    """Agent 的"名片"：描述它是什么、能做什么。"""
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class Task:
    """标准化的委托任务。"""
    description: str
    target_agent: str = ""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: str = ""
    parent_agent: str = ""


@dataclass
class Artifact:
    """标准化的委托结果。status: success / fail / partial"""
    task_id: str
    status: str
    content: str
    error: str = ""


class AgentRegistry:
    """注册 / 查找 / 委托 Agent。

    delegate_fn: 可选执行回调 (agent, description) -> str。
    未注入时默认调用 agent.run(description)；注入后由调用方决定执行方式（同步或流式），
    使 registry 与 Agent 的具体执行方式解耦。
    """

    def __init__(self, delegate_fn=None):
        self._agents: dict[str, object] = {}
        self._cards: dict[str, AgentCard] = {}
        self._delegate_fn = delegate_fn

    def register(self, agent, capabilities: list[str] | None = None) -> AgentCard:
        card = AgentCard(
            name=agent.name,
            description=agent.description or "",
            capabilities=capabilities or [],
            tools=[t for t in (getattr(agent, "tools", {}) or {})],
        )
        self._agents[agent.name] = agent
        self._cards[agent.name] = card
        return card

    def find(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def get_agent(self, name: str):
        return self._agents.get(name)

    def list_capabilities(self) -> dict[str, list[str]]:
        return {name: card.capabilities for name, card in self._cards.items()}

    def list_cards(self) -> list[AgentCard]:
        return list(self._cards.values())

    def delegate(self, task: Task) -> Artifact:
        return self._invoke(task.target_agent, task.description, task.task_id)

    def delegate_to(self, target: str, description: str) -> Artifact:
        return self._invoke(target, description, "")

    def _invoke(self, target: str, description: str, task_id: str) -> Artifact:
        agent = self._agents.get(target)
        if agent is None:
            return Artifact(
                task_id=task_id,
                status="fail",
                content="",
                error=f"Agent '{target}' not registered",
            )
        try:
            if self._delegate_fn is not None:
                result = self._delegate_fn(agent, description)
            else:
                result = agent.run(description)
            return Artifact(task_id=task_id, status="success", content=result)
        except Exception as e:
            return Artifact(
                task_id=task_id,
                status="fail",
                content="",
                error=f"{type(e).__name__}: {e}",
            )
