import json

from mini_smolagents.config import DEFAULT_MAX_FACTS, TRUNC_MEDIUM
from mini_smolagents.profile import Profile


def test_profile_roundtrip(tmp_path):
    p = Profile()
    p.set_name("小明")
    p.set_role("开发者")
    p.set_preference("语言", "中文")
    p.append_facts(["喜欢 Python", "在杭州"])
    p.save(str(tmp_path))

    loaded = Profile.load(str(tmp_path))
    assert loaded.data["name"] == "小明"
    assert loaded.data["role"] == "开发者"
    assert loaded.data["preferences"] == {"语言": "中文"}
    assert loaded.data["facts"] == ["喜欢 Python", "在杭州"]


def test_load_missing_returns_empty(tmp_path):
    p = Profile.load(str(tmp_path / "nonexistent"))
    assert p.data["name"] == ""
    assert p.data["facts"] == []
    assert p.data["preferences"] == {}


def test_load_corrupted_degrades(tmp_path):
    d = tmp_path / "user_profile.json"
    d.write_text("{not valid json", encoding="utf-8")
    p = Profile.load(str(tmp_path))
    assert p.data["name"] == ""


def test_load_wrong_types_reset(tmp_path):
    d = tmp_path / "user_profile.json"
    d.write_text(json.dumps({"facts": "not-a-list", "name": "小明"}), encoding="utf-8")
    p = Profile.load(str(tmp_path))
    assert p.data["facts"] == []
    assert p.data["name"] == "小明"


def test_set_name_first_write_only():
    p = Profile()
    assert p.set_name("小明") is True
    assert p.set_name("小红") is False
    assert p.data["name"] == "小明"


def test_set_role_first_write_only():
    p = Profile()
    assert p.set_role("经理") is True
    assert p.set_role("员工") is False
    assert p.data["role"] == "经理"


def test_append_constraints_dedupe():
    p = Profile()
    assert p.append_constraints(["不用 emoji", "简洁"]) is True
    assert p.append_constraints(["简洁", "用中文"]) is True
    assert p.data["constraints"] == ["不用 emoji", "简洁", "用中文"]


def test_set_preference_overwrites():
    p = Profile()
    p.set_preference("语言", "中文")
    p.set_preference("语言", "英文")
    assert p.data["preferences"]["语言"] == "英文"


def test_set_style_pref():
    p = Profile()
    p.set_style_pref("verbosity", "concise")
    assert p.data["style_prefs"]["verbosity"] == "concise"


def test_append_facts():
    p = Profile()
    p.append_facts(["a", "b"])
    p.append_facts(["c"])
    assert p.data["facts"] == ["a", "b", "c"]


def test_append_facts_accepts_string():
    p = Profile()
    p.append_facts("单一事实")
    assert p.data["facts"] == ["单一事实"]


def test_append_feedback():
    p = Profile()
    p.append_feedback(["上次太啰嗦"])
    assert p.data["feedback"] == ["上次太啰嗦"]


def test_item_truncated():
    p = Profile()
    p.append_facts(["x" * (TRUNC_MEDIUM + 100)])
    assert len(p.data["facts"][0]) == TRUNC_MEDIUM


def test_size_limit_rejects():
    p = Profile()
    big = ["x" * TRUNC_MEDIUM] * 40
    assert p.append_facts(big) is False
    assert p.data["facts"] == []


def test_empty_items_noop():
    p = Profile()
    assert p.append_facts([]) is False
    assert p.append_facts("") is False
    assert p.data["facts"] == []


def test_set_facts_replaces():
    p = Profile()
    p.set_facts(["a", "b", "c"])
    assert p.data["facts"] == ["a", "b", "c"]


def test_to_text():
    p = Profile()
    p.set_name("小明")
    p.set_style_pref("verbosity", "concise")
    p.append_feedback(["上次太啰嗦"])
    p.append_facts(["在杭州"])
    text = p.to_text()
    assert "小明" in text
    assert "concise" in text
    assert "上次太啰嗦" in text
    assert "在杭州" in text


def test_to_text_empty():
    assert Profile().to_text() == ""


def test_dirty_flag(tmp_path):
    p = Profile()
    assert p.dirty is False
    p.set_name("小明")
    assert p.dirty is True
    p.save(str(tmp_path))
    assert p.dirty is False