import json
import uuid
from datetime import datetime
from pathlib import Path

_FAILURE_MARKERS = ("error", "timed out", "traceback", "失败", "异常")


def should_store(task: str, result: str, min_length: int = 10) -> bool:
    """判断一段 run() 结果是否值得存入长期记忆（方案 B 启发式规则）。"""
    result = result or ""
    if len(result.strip()) < min_length:
        return False
    lower = result.lower()
    if any(marker in lower for marker in _FAILURE_MARKERS):
        return False
    return True


class EpisodicMemory:
    """ChromaDB-backed vector memory for semantic retrieval of past agent runs."""
    def __init__(self, collection_name="agent_memory", persist_dir="./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, task: str, result: str):
        doc_id = str(uuid.uuid4())
        document = f"Task: {task}\n\nResult: {result}"
        self.collection.add(
            documents=[document],
            metadatas=[{"task": task[:500], "timestamp": datetime.now().isoformat()}],
            ids=[doc_id],
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        items = []
        docs = results.get("documents")
        if docs and docs[0]:
            for i, doc in enumerate(docs[0]):
                meta = results.get("metadatas", [[{}]])[0][i] if results.get("metadatas") else {}
                items.append({
                    "document": doc,
                    "task": meta.get("task", ""),
                    "score": results["distances"][0][i] if results.get("distances") else 0,
                })
        return items

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

    def _path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save(self, session_id: str, messages: list[dict], turns: list[dict] | None = None):
        data = {
            "session_id": session_id,
            "messages": messages,
            "turns": turns or [],
            "saved_at": datetime.now().isoformat(),
        }
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
            except Exception:
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
