"""中文语义 embedding（bge-small-zh-v1.5），失败回退 chromadb 内置。

- 懒加载单例：首次使用才加载模型。
- 本地已有模型缓存时强制离线加载（local_files_only），
  避免每次启动都联网检查 huggingface 导致国内直连超时卡顿。
- HF_ENDPOINT 环境变量走镜像（国内：https://hf-mirror.com），首次下载用。
- 加载失败（缺依赖/离线无模型）返回 None，由调用方回退内置 embedding。
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("MINI_SMOLAGENTS_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


def model_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _model_cached() -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (hub / model_dir_name(EMBEDDING_MODEL)).is_dir()


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    if _model_cached():
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return SentenceTransformer(EMBEDDING_MODEL)


class ChineseEmbeddingFunction:
    """chromadb 兼容的 embedding function（bge 512 维）。"""

    @property
    def dim(self) -> int:
        model = _load_model()
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(dim_fn())

    def __call__(self, input):
        texts = [input] if isinstance(input, str) else list(input)
        model = _load_model()
        emb = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, e)) for e in emb]

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)

    def name(self) -> str:
        return EMBEDDING_MODEL.rsplit("/", 1)[-1]


def get_embedding_function():
    """加载中文 embedding 函数；失败返回 None（调用方回退内置 embedding）。"""
    try:
        fn = ChineseEmbeddingFunction()
        logger.debug("中文 embedding 就绪: %s (dim=%d)", fn.name(), fn.dim)
        return fn
    except Exception as e:
        logger.warning("中文 embedding 加载失败（%s），回退 chromadb 内置 embedding", e)
        return None