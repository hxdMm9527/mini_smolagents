import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .config import TRUNC_MEDIUM

logger = logging.getLogger(__name__)

_FAILURE_MARKERS = ("error", "timed out", "traceback", "失败", "异常")


@dataclass
class MemoryHit:
    """记忆检索命中。score 越大越相关。"""
    task: str
    document: str
    score: float


class Memory(Protocol):
    """记忆后端协议：add / search / clear / count。"""
    def add(self, task: str, result: str) -> None: ...
    def search(self, query: str, top_k: int = 3) -> list[MemoryHit]: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...


class StorePolicy:
    """存储决策：判断一次运行结果是否值得写入长期记忆。"""

    def __init__(self, min_length: int = 10):
        self.min_length = min_length

    def should_store(self, task: str, result: str) -> bool:
        result = result or ""
        if len(result.strip()) < self.min_length:
            return False
        lower = result.lower()
        if any(marker in lower for marker in _FAILURE_MARKERS):
            return False
        return True


def should_store(task: str, result: str, min_length: int = 10) -> bool:
    """向后兼容的模块级函数（默认 StorePolicy）。"""
    return StorePolicy(min_length=min_length).should_store(task, result)


class EpisodicMemory:
    """ChromaDB-backed vector memory for semantic retrieval of past agent runs."""

    def __init__(self, collection_name="agent_memory", persist_dir="./chroma_db", embedding_fn=None):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "EpisodicMemory 需要 chromadb，请先安装：pip install chromadb"
            ) from e
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn
        )

    def add(self, task: str, result: str):
        doc_id = str(uuid.uuid4())
        document = f"Task: {task}\n\nResult: {result}"
        self.collection.add(
            documents=[document],
            metadatas=[{"task": task[:TRUNC_MEDIUM], "timestamp": datetime.now().isoformat()}],
            ids=[doc_id],
        )

    def search(self, query: str, top_k: int = 3) -> list[MemoryHit]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        hits = []
        docs = results.get("documents")
        if docs and docs[0]:
            for i, doc in enumerate(docs[0]):
                meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 / (1.0 + distance)
                hits.append(MemoryHit(
                    task=meta.get("task", ""),
                    document=doc,
                    score=score,
                ))
        return hits

    def clear(self):
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()


class Checkpoint:
    """JSON file-based session state persistence."""

    def __init__(self, base_dir=".memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_session_id(self, session_id: str) -> None:
        if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError(f"非法 session_id: {session_id!r}（不能含路径分隔符或 '..'）")

    def _path(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        return self.base_dir / f"{session_id}.json"

    def save(self, session_id: str, messages: list[dict], turns: list[dict] | None = None):
        data = {
            "session_id": session_id,
            "messages": messages,
            "turns": turns or [],
            "saved_at": datetime.now().isoformat(),
        }
        path = self._path(session_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def load(self, session_id: str) -> list[dict] | None:
        data = self.load_full(session_id)
        return data["messages"] if data else None

    def load_full(self, session_id: str) -> dict | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("turns", [])
        return data

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.json")]

    def list(self) -> list[dict]:
        """返回会话摘要列表（按保存时间倒序）：[{session_id, saved_at, title}]"""
        items = []
        for p in self.base_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.debug("读取会话文件失败: %s", e)
                continue
            title = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    title = m["content"][:50]
                    break
            items.append({
                "session_id": data.get("session_id", p.stem),
                "saved_at": data.get("saved_at", ""),
                "title": title,
            })
        items.sort(key=lambda x: x["saved_at"], reverse=True)
        return items

    def delete(self, session_id: str):
        path = self._path(session_id)
        if path.exists():
            path.unlink()
