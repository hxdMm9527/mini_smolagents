"""embedding 模型加载策略测试。"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_smolagents.embedding import EMBEDDING_MODEL, _model_cached, get_embedding_function, model_dir_name


def test_model_dir_name():
    assert model_dir_name("BAAI/bge-small-zh-v1.5") == "models--BAAI--bge-small-zh-v1.5"


def test_model_cached_detects_local_cache():
    cached = _model_cached()
    assert isinstance(cached, bool)
    if cached:
        assert (Path.home() / ".cache" / "huggingface" / "hub" / model_dir_name(EMBEDDING_MODEL)).is_dir()


def test_model_cached_respects_hf_home(monkeypatch):
    with patch("pathlib.Path.is_dir", return_value=True):
        monkeypatch.setenv("HF_HOME", r"C:\fake\cache")
        assert _model_cached() is True


def test_load_model_without_env_still_works(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    fn = get_embedding_function()
    assert fn is not None
    emb = fn("测试 文本")
    assert len(emb) == 1
    assert len(emb[0]) == 512