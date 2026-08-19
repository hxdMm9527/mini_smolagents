"""chromadb client 复用（按持久化路径缓存）。

EpisodicMemory / FactsMemory 共享同一路径的 PersistentClient，
避免每次实例化都创建新 client（后台资源累积、sqlite 锁竞争）。
"""
import threading

import chromadb

_clients: dict[str, chromadb.ClientAPI] = {}
_lock = threading.Lock()


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    with _lock:
        client = _clients.get(persist_dir)
        if client is None:
            client = chromadb.PersistentClient(path=persist_dir)
            _clients[persist_dir] = client
        return client