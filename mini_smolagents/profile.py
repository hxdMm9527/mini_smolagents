"""用户档案卡（L2）：单 JSON 文件 + 固定 schema + 受限写操作。"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_MAX_FACTS, DEFAULT_PROFILE_MAX_BYTES, PROFILE_FILENAME, TRUNC_MEDIUM

logger = logging.getLogger(__name__)

_SCHEMA = {
    "name": str,
    "role": str,
    "preferences": dict,
    "constraints": list,
    "facts": list,
    "feedback": list,
    "style_prefs": dict,
}

_EMPTY = {
    "name": "",
    "role": "",
    "preferences": {},
    "constraints": [],
    "facts": [],
    "feedback": [],
    "style_prefs": {},
}


def _coerce_items(items) -> list[str]:
    """把字符串或可迭代输入规范化为截断后的字符串列表。"""
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    result = []
    for it in items:
        text = str(it).strip()
        if text:
            result.append(text[:TRUNC_MEDIUM])
    return result


class Profile:
    """用户档案卡存储。数据为纯 dict，落盘为单 JSON 文件。"""

    def __init__(self, data=None):
        merged = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in _EMPTY.items()}
        if data:
            for k, default in merged.items():
                v = data.get(k, default)
                if isinstance(v, type(default)):
                    merged[k] = v
        self.data = merged
        self.data.setdefault("updated_at", "")
        self.dirty = False

    @classmethod
    def load(cls, base_dir=".memory", filename=PROFILE_FILENAME):
        path = Path(base_dir) / filename
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return cls()
            return cls(data)
        except Exception as e:
            logger.debug("加载档案卡失败: %s", e)
            return cls()

    def save(self, base_dir=".memory", filename=PROFILE_FILENAME):
        path = Path(base_dir)
        path.mkdir(parents=True, exist_ok=True)
        full = path / filename
        tmp = full.with_suffix(full.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, full)
        self.dirty = False

    def set_name(self, name: str) -> bool:
        if self.data["name"]:
            return False
        return self._commit({**self.data, "name": str(name).strip()})

    def set_role(self, role: str) -> bool:
        if self.data["role"]:
            return False
        return self._commit({**self.data, "role": str(role).strip()})

    def set_preference(self, key: str, value) -> bool:
        prefs = {**self.data["preferences"], key: value}
        return self._commit({**self.data, "preferences": prefs})

    def set_style_pref(self, key: str, value) -> bool:
        prefs = {**self.data["style_prefs"], key: value}
        return self._commit({**self.data, "style_prefs": prefs})

    def append_facts(self, items) -> bool:
        items = _coerce_items(items)
        if not items:
            return False
        return self._commit({**self.data, "facts": self.data["facts"] + items})

    def append_feedback(self, items) -> bool:
        items = _coerce_items(items)
        if not items:
            return False
        return self._commit({**self.data, "feedback": self.data["feedback"] + items})

    def append_constraints(self, items) -> bool:
        items = _coerce_items(items)
        if not items:
            return False
        merged = list(self.data["constraints"])
        for it in items:
            if it not in merged:
                merged.append(it)
        return self._commit({**self.data, "constraints": merged})

    def set_facts(self, items) -> bool:
        """合并治理用：整体替换 facts 列表（仍走校验 + 上限）。"""
        items = _coerce_items(items)
        return self._commit({**self.data, "facts": items})

    def to_text(self) -> str:
        d = self.data
        lines = []
        if d["name"]:
            lines.append(f"- 姓名：{d['name']}")
        if d["role"]:
            lines.append(f"- 角色：{d['role']}")
        if d["preferences"]:
            lines.append(f"- 偏好：{json.dumps(d['preferences'], ensure_ascii=False)}")
        if d["style_prefs"]:
            lines.append(f"- 风格偏好：{json.dumps(d['style_prefs'], ensure_ascii=False)}")
        if d["constraints"]:
            lines.append("- 约束：" + "；".join(d["constraints"]))
        if d["facts"]:
            lines.append("- 已知事实：" + "；".join(d["facts"]))
        if d["feedback"]:
            lines.append("- 历史反馈：" + "；".join(d["feedback"]))
        return "\n".join(lines)

    def _commit(self, new_data: dict) -> bool:
        for key, typ in _SCHEMA.items():
            if not isinstance(new_data.get(key), typ):
                logger.debug("档案卡 schema 校验失败: %s 类型非法", key)
                return False
        candidate = {**new_data, "updated_at": datetime.now().isoformat()}
        payload = json.dumps(candidate, ensure_ascii=False)
        if len(payload.encode("utf-8")) > DEFAULT_PROFILE_MAX_BYTES:
            logger.debug("档案卡超过大小上限，拒绝本次更新")
            return False
        self.data = candidate
        self.dirty = True
        return True