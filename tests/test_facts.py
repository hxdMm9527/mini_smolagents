from mini_smolagents.facts import FactsMemory
from mini_smolagents.memory import EpisodicMemory


def test_facts_memory_create(tmpdir):
    mem = FactsMemory(collection_name="test_facts_create", persist_dir=str(tmpdir))
    assert mem.collection.name == "test_facts_create"
    assert mem.count() == 0


def test_facts_memory_add_and_search(tmpdir):
    mem = FactsMemory(collection_name="test_facts_add", persist_dir=str(tmpdir))
    assert mem.add("用户喜欢 Python") is True
    assert mem.add("用户在杭州工作") is True
    assert mem.count() == 2

    results = mem.search("Python", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0]


def test_facts_memory_clear(tmpdir):
    mem = FactsMemory(collection_name="test_facts_clear", persist_dir=str(tmpdir))
    mem.add("用户喜欢 Python")
    assert mem.count() == 1
    mem.clear()
    assert mem.count() == 0


def test_facts_memory_dedup(tmpdir):
    mem = FactsMemory(collection_name="test_facts_dedup", persist_dir=str(tmpdir))
    assert mem.add("用户喜欢 Python") is True
    assert mem.add("用户喜欢 Python") is False
    assert mem.count() == 1


def test_facts_memory_empty_add(tmpdir):
    mem = FactsMemory(collection_name="test_facts_empty", persist_dir=str(tmpdir))
    assert mem.add("") is False
    assert mem.add("   ") is False
    assert mem.count() == 0


def test_facts_memory_independent_collection(tmpdir):
    facts = FactsMemory(collection_name="test_facts_ind", persist_dir=str(tmpdir))
    epi = EpisodicMemory(collection_name="test_epi_ind", persist_dir=str(tmpdir))
    facts.add("用户喜欢 Python")
    epi.add("写个排序函数", "def quicksort(): ...")
    assert facts.count() == 1
    assert epi.count() == 1
    results = facts.search("Python", top_k=3)
    assert "Python" in results[0]