"""用户零散事实的向量存储与语义召回（档案卡 facts 层）。"""
import uuid
from datetime import datetime

from .config import FACT_DEDUP_THRESHOLD, TRUNC_MEDIUM


class FactsMemory:
    """chromadb-backed storage for user facts, with semantic recall."""

    def __init__(self, collection_name="user_facts", persist_dir="./chroma_db", embedding_fn=None):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "FactsMemory 需要 chromadb，请先安装：pip install chromadb"
            ) from e
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        self.dedup_threshold = FACT_DEDUP_THRESHOLD

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