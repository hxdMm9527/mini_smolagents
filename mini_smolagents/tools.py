import inspect
from typing import Any

from .types import Tool

TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


def tool(func) -> Tool:
    name = func.__name__
    doc = inspect.getdoc(func)
    description = doc.split("\n")[0].strip() if doc else ""

    sig = inspect.signature(func)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = param.annotation
        json_type = TYPE_MAP.get(annotation, "string")
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(name=name, description=description, parameters=parameters, func=func)
