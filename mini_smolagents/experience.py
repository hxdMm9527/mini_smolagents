"""经验库：重复成功 → 可复用通用经验。

薄包装 EpisodicMemory（复用去重/删除/更新/衰减/client 复用能力），
文档格式为 `Task: 经验\n\nResult: <经验文本>`，search 返回时剥离前缀。
"""
from .memory import EpisodicMemory

_PREFIX = "Task: 经验\n\nResult: "


class ExperienceMemory:
    def __init__(self, collection_name="experience_memory", persist_dir="./chroma_db", embedding_fn=None):
        self._mem = EpisodicMemory(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_fn=embedding_fn,
        )

    def add(self, experience: str) -> bool:
        return self._mem.add("经验", experience)

    def search(self, query: str, top_k: int = 3):
        hits = self._mem.search(query, top_k=top_k)
        for h in hits:
            if h.document.startswith(_PREFIX):
                h.document = h.document[len(_PREFIX):]
        return hits

    def delete(self, query: str) -> int:
        return self._mem.delete(query)

    def update(self, query: str, new_experience: str) -> bool:
        return self._mem.update(query, _PREFIX + new_experience)

    def count(self) -> int:
        return self._mem.count()

    def clear(self) -> None:
        return self._mem.clear()