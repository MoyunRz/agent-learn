"""
多 Agent 系统的消息与任务模型。

核心数据类型：
- MessageType: 消息类型枚举（请求/响应/计划/任务/结果/错误）
- AgentMessage: Agent 之间的消息，携带发送者、接收者、内容和时间戳
- Task: Planner 分解出的单个子任务，含类型、描述、输入和依赖
- ExecutionPlan: Planner 产出的完整执行计划，支持依赖感知的任务调度
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any
import time


class MessageType(Enum):
    """Agent 间消息的类型标记。"""
    REQUEST = "request"    # 请求
    RESPONSE = "response"  # 响应
    PLAN = "plan"          # 执行计划
    TASK = "task"          # 任务委派
    RESULT = "result"      # 任务完成结果
    ERROR = "error"        # 错误


@dataclass
class AgentMessage:
    """多 Agent 系统中传递的消息。

    字段：
        sender:   发送方 Agent 名称
        receiver: 接收方 Agent 名称
        type:     消息类型
        content:  消息内容（任意类型）
        context:  附加上下文信息
        timestamp: 时间戳（Unix 秒）
    """
    sender: str
    receiver: str
    type: MessageType
    content: Any
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create_task(cls, sender: str, receiver: str, task_data: dict) -> "AgentMessage":
        """快捷方法：创建一条 TASK 类型消息。Supervisor 用此方法向 Worker 派发任务。"""
        return cls(
            sender=sender,
            receiver=receiver,
            type=MessageType.TASK,
            content=task_data,
            context={},
            timestamp=time.time(),
        )

    @classmethod
    def create_result(cls, sender: str, result: Any) -> "AgentMessage":
        """快捷方法：创建一条 RESULT 类型消息。Worker 用此方法将结果返回给 Supervisor。"""
        return cls(
            sender=sender,
            receiver="supervisor",
            type=MessageType.RESULT,
            content=result,
            context={},
            timestamp=time.time(),
        )


@dataclass
class Task:
    """Planner 分解出的单个子任务。

    字段：
        id:              任务唯一标识（如 "task_1"）
        type:            任务类型（calculation / coding / research / general）
        description:     任务描述文本
        input_data:      执行任务所需的输入数据
        expected_output: 期望输出的描述
        dependencies:    依赖的任务 ID 列表（这些任务完成后本任务才能开始）
    """
    id: str
    type: str
    description: str
    input_data: Any = None
    expected_output: str = ""
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Planner 产出的完整执行计划，包含一组 Task 和它们之间的依赖关系。

    核心方法 get_ready_tasks() 实现了「依赖感知」调度：
    只有所有依赖任务都已完成的任务，才会被标记为 ready。
    这使得 Supervisor 可以安全地并行执行互不依赖的任务。
    """
    tasks: list[Task] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)  # task_id → [依赖的 task_id ...]

    def add_task(self, task: Task, depends_on: list[str] = None):
        """向计划中添加一个任务及其依赖。"""
        self.tasks.append(task)
        if depends_on:
            self.dependencies[task.id] = depends_on

    def get_ready_tasks(self, completed: set[str]) -> list[Task]:
        """获取当前可以执行的任务列表。

        一个任务「就绪」的条件：
        1. 它的所有依赖任务都已在 completed 集合中
        2. 它自身尚未完成

        参数：
            completed: 已完成任务 ID 的集合
        返回：
            所有就绪任务的列表（这些任务之间互不依赖，可并行执行）
        """
        ready = []
        for task in self.tasks:
            deps = self.dependencies.get(task.id, [])
            if all(d in completed for d in deps) and task.id not in completed:
                ready.append(task)
        return ready
