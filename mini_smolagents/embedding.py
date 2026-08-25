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

ONNX_MODEL = Path(__file__).resolve().parent.parent / "models" / "bge-small-zh-v1.5.onnx"
ONNX_TOKENIZER = Path(__file__).resolve().parent.parent / "models" / "tokenizer.json"


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


class OnnxChineseEmbedding:
    """ONNX 版 bge（onnxruntime + tokenizers，启动 <0.5s，无需 torch/sentence_transformers）。"""

    _dim = 512

    def __init__(self):
        self._session = None
        self._tok = None

    def _ensure_loaded(self):
        if self._session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            self._session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
            self._tok = Tokenizer.from_file(str(ONNX_TOKENIZER))
            self._tok.enable_truncation(max_length=512)

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts):
        if not texts:
            return []
        self._ensure_loaded()
        encs = self._tok.encode_batch(texts, add_special_tokens=True)
        max_len = max(len(e.ids) for e in encs) or 1
        ids = [[0] * max_len for _ in encs]
        mask = [[0] * max_len for _ in encs]
        for i, e in enumerate(encs):
            n = len(e.ids)
            ids[i][:n] = e.ids
            mask[i][:n] = [1] * n
        out = self._session.run(["embedding"], {
            "input_ids": ids, "attention_mask": mask,
        })[0]
        return [list(map(float, v)) for v in out]

    def __call__(self, input):
        texts = [input] if isinstance(input, str) else list(input)
        return self._encode(texts)

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)

    def name(self) -> str:
        return "bge-small-zh-v1.5-onnx"


def _onnx_ready() -> bool:
    return ONNX_MODEL.is_file() and ONNX_TOKENIZER.is_file()


def get_embedding_function():
    """加载中文 embedding 函数（ONNX 优先，失败回退 sentence_transformers）；最终失败返回 None。"""
    if _onnx_ready():
        try:
            fn = OnnxChineseEmbedding()
            logger.debug("中文 embedding 就绪: %s (dim=%d)", fn.name(), fn.dim)
            return fn
        except Exception as e:
            logger.warning("ONNX embedding 加载失败（%s），回退 sentence_transformers", e)
    try:
        fn = ChineseEmbeddingFunction()
        logger.debug("中文 embedding 就绪: %s (dim=%d)", fn.name(), fn.dim)
        return fn
    except Exception as e:
        logger.warning("中文 embedding 加载失败（%s），回退 chromadb 内置 embedding", e)
        return None