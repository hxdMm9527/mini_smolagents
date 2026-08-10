import tempfile
import os

from mini_smolagents.memory import Checkpoint, EpisodicMemory, should_store


def test_should_store():
    assert should_store("写函数", "def validate_email(): ... 完整代码") is True
    assert should_store("写函数", "太短") is False
    assert should_store("写函数", "Error: 执行超时 timed out") is False
    assert should_store("写函数", "Traceback: 异常") is False
    assert should_store("写函数", "") is False


def test_episodic_memory_create(tmpdir):
    mem = EpisodicMemory(collection_name="test_create", persist_dir=str(tmpdir))
    assert mem.collection.name == "test_create"
    assert mem.count() == 0


def test_episodic_memory_add_and_search(tmpdir):
    mem = EpisodicMemory(collection_name="test_add", persist_dir=str(tmpdir))
    mem.add("写一个邮箱验证函数", "返回 validate_email() 代码")
    mem.add("写一个斐波那契函数", "返回 fibonacci() 代码")

    assert mem.count() == 2

    results = mem.search("正则验证邮箱", top_k=1)
    assert len(results) == 1
    assert "邮箱" in results[0]["document"]

    results_all = mem.search("函数", top_k=3)
    assert len(results_all) == 2


def test_episodic_memory_clear(tmpdir):
    mem = EpisodicMemory(collection_name="test_clear", persist_dir=str(tmpdir))
    mem.add("test task", "test result")
    assert mem.count() == 1

    mem.clear()
    assert mem.count() == 0


def test_checkpoint_save_and_load(tmpdir):
    cp = Checkpoint(base_dir=str(tmpdir))
    messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]
    cp.save("session_a", messages)

    loaded = cp.load("session_a")
    assert loaded == messages

    assert cp.load("nonexistent") is None


def test_checkpoint_list_and_delete(tmpdir):
    cp = Checkpoint(base_dir=str(tmpdir))
    cp.save("s1", [{"role": "user", "content": "a"}])
    cp.save("s2", [{"role": "user", "content": "b"}])

    sessions = cp.list_sessions()
    assert set(sessions) == {"s1", "s2"}

    cp.delete("s1")
    assert cp.list_sessions() == ["s2"]


def test_integration(tmpdir):
    mem = EpisodicMemory(collection_name="test_integ", persist_dir=str(tmpdir))
    cp = Checkpoint(base_dir=str(tmpdir))

    task = "写一个排序函数"
    result = "def quicksort(arr): ..."

    mem.add(task, result)
    found = mem.search("排序算法", top_k=1)
    assert len(found) == 1
    assert "quicksort" in found[0]["document"]

    messages = [
        {"role": "system", "content": "test agent"},
        {"role": "user", "content": task},
        {"role": "assistant", "content": result},
    ]
    cp.save("session_1", messages)
    loaded = cp.load("session_1")
    assert loaded == messages

    print("All tests passed!")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as base:
        print(f"Temp dir: {base}")
        base = Path(base)

        test_episodic_memory_create(str(base / "create"))
        test_episodic_memory_add_and_search(str(base / "add"))
        test_episodic_memory_clear(str(base / "clear"))
        test_checkpoint_save_and_load(str(base / "checkpoint_a"))
        test_checkpoint_list_and_delete(str(base / "checkpoint_b"))
        test_integration(str(base / "integ"))
        test_should_store()
        print("\n=== ALL TESTS PASSED ===")
