from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
