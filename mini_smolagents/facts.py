"""用户零散事实的向量存储与语义召回（档案卡 facts 层）。"""
import uuid
from datetime import datetime

from ._chroma import get_client
from .config import FACT_DEDUP_THRESHOLD, MEMORY_MATCH_THRESHOLD, TRUNC_MEDIUM
from .embedding import get_embedding_function


class FactsMemory:
    """chromadb-backed storage for user facts, with semantic recall."""

    def __init__(self, collection_name="user_facts", persist_dir="./chroma_db", embedding_fn=None):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "FactsMemory 需要 chromadb，请先安装：pip install chromadb"
            ) from e
        self.client = get_client(persist_dir)
        self.embedding_fn = embedding_fn if embedding_fn is not None else get_embedding_function()
        self.collection = self._ensure_collection(collection_name)
        self.dedup_threshold = FACT_DEDUP_THRESHOLD

    def _ensure_collection(self, name: str):
        """探测既有 collection 维度，与当前 embedding 不符（含空库）则重建。"""
        fn = self.embedding_fn
        if fn is None:
            try:
                existing = self.client.get_collection(name)
            except Exception:
                existing = None
            if existing is not None and (existing.metadata or {}).get("embedding_dim"):
                logger.warning("事实库 %s 由中文 embedding 构建但模型不可用，降级重建为内置 embedding", name)
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
                    logger.info("事实库 %s 为空，重建以使用中文 embedding (dim=%d)", name, dim)
                else:
                    logger.warning(
                        "事实库 %s 向量维度 %d 与当前 embedding %d 不符，已重建（旧向量空间不兼容）",
                        name, len(emb), dim,
                    )
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
            else:
                return existing
        return self.client.create_collection(name, embedding_function=fn, metadata={"embedding_dim": dim})

    def add(self, fact: str) -> bool:
        fact = (fact or "").strip()
        if not fact:
            return False
        existing = self.collection.query(query_texts=[fact], n_results=1)
        docs = existing.get("documents")
        if docs and docs[0]:
            distance = existing["distances"][0][0] if existing.get("distances") else 0.0
            score = 1.0 / (1.0 + distance)
            if score > self.dedup_threshold:
                return False
        self.collection.add(
            documents=[fact[:TRUNC_MEDIUM]],
            metadatas=[{"timestamp": datetime.now().isoformat()}],
            ids=[str(uuid.uuid4())],
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

    def update(self, query: str, new_fact: str) -> bool:
        hit_id = self._locate(query)
        if hit_id is None:
            return False
        self.collection.update(
            ids=[hit_id],
            documents=[new_fact[:TRUNC_MEDIUM]],
            metadatas=[{"timestamp": datetime.now().isoformat()}],
        )
        return True

    def search(self, query: str, top_k: int = 3) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents")
        if docs and docs[0]:
            return list(docs[0])
        return []

    def clear(self):
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()