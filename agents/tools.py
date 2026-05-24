"""
工具系统 —— 将 Python 函数包装为 LLM 可调用的 Tool，并通过 ToolRegistry 统一管理。

核心组件：
- Tool: 单个工具的封装，自动根据函数签名生成 Anthropic 格式的 tool spec
- @tool 装饰器: 便捷地将函数转为 Tool
- ToolRegistry: 工具注册表，支持按名查找、批量导出所有工具的 spec
- tool_registry: 全局单例，方便各模块共享同一套工具
"""

import inspect
from typing import Any, Callable


class Tool:
    """将普通 Python 函数包装为 LLM 可调用的工具。

    Anthropic API 需要每个工具提供 name / description / input_schema 三个字段，
    本类通过 inspect 自动从函数签名中提取参数名、类型和必填信息。
    """

    def __init__(self, func: Callable, name: str | None = None, description: str | None = None):
        self.func = func                                              # 原始 Python 函数
        self.name = name or func.__name__                             # 工具名，默认用函数名
        self.description = description or (func.__doc__ or "").strip() # 工具描述，默认用 docstring

    def call(self, **kwargs) -> Any:
        """实际执行工具函数，自动根据类型注解转换参数类型。

        LLM 有时会把数字参数传成字符串（如 "40" 而非 40），
        这里根据函数签名的类型注解自动做转换：
        - int 注解 → int(value)
        - float 注解 → float(value)
        - bool 注解 → bool(value)
        """
        sig = inspect.signature(self.func)
        converted = {}
        for param_name, value in kwargs.items():
            if param_name in sig.parameters:
                expected_type = sig.parameters[param_name].annotation
                if expected_type is not inspect.Parameter.empty:
                    try:
                        if expected_type is int:
                            value = int(value)
                        elif expected_type is float:
                            value = float(value)
                        elif expected_type is bool and not isinstance(value, bool):
                            value = str(value).lower() in ("true", "1", "yes")
                    except (ValueError, TypeError):
                        pass  # 转换失败则保持原值
            converted[param_name] = value
        return self.func(**converted)

    @property
    def anthropic_spec(self) -> dict:
        """生成符合 Anthropic tool use 协议的工具规格。

        自动检测参数类型注解：
        - int / float  → "number"
        - bool         → "boolean"
        - 其他(默认)    → "string"

        没有默认值的参数自动标记为 required。
        """
        sig = inspect.signature(self.func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # 类型映射：Python 类型 → JSON Schema 类型
            param_type = "string"
            if param.annotation is int or param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"

            properties[param_name] = {"type": param_type}

            # 没有默认值 → 必填参数
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def tool(name: str | None = None, description: str | None = None):
    """装饰器：将普通函数转为 Tool 对象。

    用法：
        @tool(name="add", description="两数相加")
        def add(a: int, b: int) -> int:
            return a + b

    参数：
        name: 工具名称，默认使用函数名
        description: 工具描述，默认使用函数 docstring
    """

    def decorator(func: Callable) -> Tool:
        return Tool(func, name=name, description=description)

    return decorator


class ToolRegistry:
    """工具注册表 —— 管理一组 Tool，提供按名查找和批量导出接口。

    典型用法：
        registry = ToolRegistry()
        registry.register(Tool(my_func, "my_tool"))
        specs = registry.get_specs()  # 传给 LLM 的 tools 参数
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}  # name → Tool 映射

    def register(self, tool: Tool) -> Tool:
        """注册一个工具（可链式调用），同名工具会被覆盖。"""
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        """按名称查找工具，不存在返回 None。"""
        return self._tools.get(name)

    def get_specs(self) -> list[dict]:
        """批量导出所有工具的 Anthropic spec，用于 API 调用的 tools 参数。"""
        return [t.anthropic_spec for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """支持 'tool_name' in registry 语法。"""
        return name in self._tools


# 全局单例 —— 整个应用共享同一套工具注册表
tool_registry = ToolRegistry()
