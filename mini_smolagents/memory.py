import json
import logging
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ._chroma import get_client
from .config import MEMORY_DEDUP_THRESHOLD, MEMORY_HALF_LIFE_DAYS, MEMORY_MATCH_THRESHOLD, TRUNC_MEDIUM
from .embedding import get_embedding_function

logger = logging.getLogger(__name__)

_FAILURE_MARKERS = ("error", "timed out", "traceback", "失败", "异常")


@dataclass
class MemoryHit:
    """记忆检索命中。score 为有效分（相似度 × 时效衰减），越大越相关。"""
    task: str
    document: str
    score: float
    id: str | None = None
    count: int = 1
    timestamp: str = ""


class Memory(Protocol):
    """记忆后端协议：add / search / delete / update / clear / count。"""
    def add(self, task: str, result: str) -> bool: ...
    def search(self, query: str, top_k: int = 3) -> list[MemoryHit]: ...
    def delete(self, query: str) -> int: ...
    def update(self, query: str, new_doc: str) -> bool: ...
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
        self.client = get_client(persist_dir)
        self.embedding_fn = embedding_fn if embedding_fn is not None else get_embedding_function()
        self.collection = self._ensure_collection(collection_name)
        self.dedup_threshold = MEMORY_DEDUP_THRESHOLD
        self.half_life_days = MEMORY_HALF_LIFE_DAYS

    def _ensure_collection(self, name: str):
        """探测既有 collection 维度，与当前 embedding 不符（含空库）则重建。"""
        fn = self.embedding_fn
        if fn is None:
            try:
                existing = self.client.get_collection(name)
            except Exception:
                existing = None
            if existing is not None and (existing.metadata or {}).get("embedding_dim"):
                logger.warning("记忆库 %s 由中文 embedding 构建但模型不可用，降级重建为内置 embedding", name)
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
            return self.client.get_or_create_collection(name, embedding_function=None)
        dim = fn.dim
        try:
            existing = self.client.get_collection(name)
        except Exception:
            existing = None
        if existing is not None:
            same_space = (existing.metadata or {}).get("embedding_dim") == dim
            if not same_space:
                try:
                    sample = existing.get(limit=1, include=["embeddings"])
                    emb = (sample.get("embeddings") or [None])[0]
                except Exception:
                    emb = None
                if emb is None:
                    logger.info("记忆库 %s 为空，重建以使用中文 embedding (dim=%d)", name, dim)
                else:
                    logger.warning(
                        "记忆库 %s 向量维度 %d 与当前 embedding %d 不符，已重建（旧向量空间不兼容）",
                        name, len(emb), dim,
                    )
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
            else:
                return existing
        return self.client.create_collection(name, embedding_function=fn, metadata={"embedding_dim": dim})

    def add(self, task: str, result: str) -> bool:
        document = f"Task: {task}\n\nResult: {result}"
        existing = self.collection.query(query_texts=[document], n_results=1)
        docs = existing.get("documents")
        if docs and docs[0]:
            distance = existing["distances"][0][0] if existing.get("distances") else 0.0
            score = 1.0 / (1.0 + distance)
            if score > self.dedup_threshold:
                hit_id = (existing.get("ids") or [[]])[0][0]
                meta = (existing.get("metadatas") or [[{}]])[0][0]
                new_meta = {
                    **meta,
                    "count": int(meta.get("count", 1)) + 1,
                    "last_seen": datetime.now().isoformat(),
                }
                self.collection.update(ids=[hit_id], metadatas=[new_meta])
                return False
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[document],
            metadatas=[{
                "task": task[:TRUNC_MEDIUM],
                "timestamp": datetime.now().isoformat(),
                "count": 1,
            }],
            ids=[doc_id],
        )
        return True

    def _locate(self, query: str):
        existing = self.collection.query(query_texts=[query], n_results=1)
        ids = existing.get("ids")
        if not ids or not ids[0]:
            return None
        distance = (existing.get("distances") or [[0.0]])[0][0]
        if 1.0 / (1.0 + distance) >= MEMORY_MATCH_THRESHOLD:
            return ids[0][0]
        doc = (existing.get("documents") or [[]])[0][0]
        q = query.strip()
        if len(q) >= 2 and q in doc:
            return ids[0][0]
        return None

    def delete(self, query: str) -> int:
        hit_id = self._locate(query)
        if hit_id is None:
            return 0
        self.collection.delete(ids=[hit_id])
        return 1

    def update(self, query: str, new_doc: str) -> bool:
        hit_id = self._locate(query)
        if hit_id is None:
            return False
        meta = (self.collection.query(query_texts=[query], n_results=1).get("metadatas") or [[{}]])[0][0]
        new_meta = {**meta, "timestamp": datetime.now().isoformat(), "count": 1}
        self.collection.update(ids=[hit_id], documents=[new_doc], metadatas=[new_meta])
        return True

    def search(self, query: str, top_k: int = 3) -> list[MemoryHit]:
        candidates = max(top_k * 3, top_k)
        results = self.collection.query(query_texts=[query], n_results=candidates)
        hits = []
        docs = results.get("documents")
        if docs and docs[0]:
            metas = results.get("metadatas", [[{}]])[0] if results.get("metadatas") else [{}] * len(docs[0])
            ids = results.get("ids", [[]])[0] if results.get("ids") else []
            dists = results.get("distances", [[0.0]])[0] if results.get("distances") else [0.0] * len(docs[0])
            for i, doc in enumerate(docs[0]):
                meta = metas[i] if i < len(metas) else {}
                ts = meta.get("timestamp", "")
                raw = 1.0 / (1.0 + (dists[i] if i < len(dists) else 0.0))
                hits.append(MemoryHit(
                    task=meta.get("task", ""),
                    document=doc,
                    score=self._decayed_score(raw, ts),
                    id=ids[i] if i < len(ids) else None,
                    count=int(meta.get("count", 1)),
                    timestamp=ts,
                ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def _decayed_score(self, raw_score: float, timestamp_iso: str) -> float:
        if not timestamp_iso:
            return raw_score
        try:
            age_days = (datetime.now() - datetime.fromisoformat(timestamp_iso)).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return raw_score
        return raw_score * math.exp(-max(0.0, age_days) / self.half_life_days)

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

    def save(self, session_id: str, messages: list[dict], turns: list[dict] | None = None, summary: str = "", summarized_upto: int = 0):
        data = {
            "session_id": session_id,
            "messages": messages,
            "turns": turns or [],
            "summary": summary,
            "summarized_upto": summarized_upto,
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
