"""
BaseAgent —— 所有 Agent 的抽象基类。

所有 Agent（Planner / Supervisor / Worker）都必须：
1. 有一个唯一名称 (name)
2. 实现 run(query) 方法
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Agent 抽象基类 —— 定义多 Agent 系统中所有角色的公共接口。

    所有子类必须实现 run() 方法。
    """

    def __init__(self, name: str):
        self.name = name  # Agent 的唯一标识名

    @abstractmethod
    def run(self, query: str) -> str:
        """执行 Agent 的核心逻辑。

        参数：
            query: 输入查询 / 指令
        返回：
            执行结果字符串
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
