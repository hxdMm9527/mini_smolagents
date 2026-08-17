"""上下文组装与会话元数据（L1 元数据 + 组装器）。"""
from dataclasses import dataclass, field
from datetime import datetime

from .config import DEFAULT_TOKEN_BUDGET


@dataclass
class SessionMetadata:
    """会话元数据（L1）：纯内存对象，会话结束即弃，不进入任何持久层。"""
    session_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    topic: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1 字/token，英文约 4 字符/token，取 len//2 折中。"""
    return len(text or "") // 2


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    if max_tokens <= 0:
        return ""
    return text[: max_tokens * 2].rstrip() + "…"


class ContextComposer:
    """按固定顺序组装各层上下文，并施加总 token 预算。

    顺序：system 指令 → 档案卡 → 召回 → 摘要 → 窗口。
    超预算时优先截断摘要，其次召回；system 与窗口原文保留。
    档案卡/召回 MVP 阶段可为空（插槽保留）。
    """

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget

    def compose(self, system_prompt: str, profile: str = "", recall: str = "",
                summary: str = "", window: list | None = None) -> list[dict]:
        window = window or []

        window_tokens = sum(estimate_tokens(m.get("content") or "") for m in window)
        fixed = estimate_tokens(system_prompt) + estimate_tokens(profile) + window_tokens
        flexible = max(0, self.token_budget - fixed)

        summary = _truncate_tokens(summary, flexible)
        recall = _truncate_tokens(recall, max(0, flexible - estimate_tokens(summary)))

        system_content = system_prompt
        if profile:
            system_content += f"\n\n[用户档案]\n{profile}"
        if recall:
            system_content += f"\n\n[相关历史记忆，供参考：]\n{recall}"

        messages = [{"role": "system", "content": system_content}]
        if summary:
            messages.append({"role": "system", "content": f"[历史对话摘要]\n{summary}"})
        messages.extend(window)
        return messages