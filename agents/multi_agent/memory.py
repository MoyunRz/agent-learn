"""
AgentMemory —— 跨会话长期记忆存储与检索。

支持：
- remember(): 存储记忆条目（带标签，方便分类检索）
- recall(): 按关键词 / 标签检索历史
- get_recent(): 获取最近 N 条记忆
- auto_store(): 自动将任务结果写入记忆

数据持久化到 JSON 文件，程序重启后历史记忆不丢失。
"""

import json
import os
import time
from datetime import datetime
from typing import Any


class AgentMemory:
    """长期记忆存储 —— JSON 文件持久化，支持关键词和标签检索。

    用法：
        memory = AgentMemory(".agent_memory.json")
        memory.remember("计算结果", "15+25=40", tags=["math", "calculator"])
        results = memory.recall(keyword="25")
    """

    def __init__(self, persist_file: str = ".agent_memory.json"):
        self.file = persist_file
        self._ensure_file()

    # ---------- 文件初始化 ----------

    def _ensure_file(self):
        """确保 JSON 文件存在且为合法数组。"""
        if not os.path.exists(self.file):
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_all(self) -> list[dict]:
        """读取全部记忆条目。"""
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_all(self, data: list[dict]):
        """覆盖写入全部记忆条目。"""
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- 存储 ----------

    def remember(self, key: str, value: Any, tags: list[str] = None) -> dict:
        """存储一条记忆。

        参数：
            key: 记忆标识（如任务名）
            value: 记忆内容
            tags: 分类标签（如 ["math", "task_1"]）

        返回：
            写入的完整条目
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "key": key,
            "value": value,
            "tags": tags or [],
        }
        data = self._read_all()
        data.append(entry)
        self._write_all(data)
        return entry

    # ---------- 检索 ----------

    def recall(self, keyword: str = None, tag: str = None, limit: int = 5) -> list[dict]:
        """检索记忆 —— 在 key / value / tags 三个字段中搜索。

        参数：
            keyword: 在 key + value + tags 中模糊匹配
            tag: 按标签精确过滤
            limit: 最多返回条数（从最新开始）

        返回：
            匹配的记忆条目列表
        """
        data = self._read_all()
        results = []
        for entry in reversed(data):  # 最新的排前面
            if keyword:
                # 在 key、value、tags 三个字段中搜索
                searchable = " ".join([
                    str(entry.get("key", "")),
                    str(entry.get("value", "")),
                    " ".join(entry.get("tags", [])),
                ])
                if keyword.lower() not in searchable.lower():
                    continue
            if tag and tag not in entry.get("tags", []):
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def get_recent(self, limit: int = 5) -> list[dict]:
        """获取最近 N 条记忆。"""
        data = self._read_all()
        return list(reversed(data[-limit:]))

    # ---------- 工具方法 ----------

    def auto_store(self, task_id: str, task_type: str, result: str):
        """自动存储任务执行结果 —— Supervisor 在每个任务完成后调用。

        参数：
            task_id: 任务 ID
            task_type: 任务类型（calculation / coding / research）
            result: Worker 返回的结果
        """
        return self.remember(
            key=task_id,
            value=result,
            tags=[task_type, task_id],
        )

    def to_context_text(self, keyword: str = None, task_type: str = None, limit: int = 3) -> str:
        """检索记忆并格式化为可注入 prompt 的文本。

        两阶段检索：
        1. 按 keyword 在 key/value/tags 中搜索
        2. 若结果不够，按 task_type 标签补充最近的同类记忆

        参数：
            keyword: 检索关键词（如 task.description）
            task_type: 任务类型标签（如 "calculation"），用于补充同类型记忆
            limit: 最多返回条数

        返回：
            格式化的上下文文本，无结果时返回空字符串
        """
        # 第一阶段：关键词搜索
        entries = self.recall(keyword=keyword, limit=limit)

        # 第二阶段：关键词搜不够，用同类型记忆补充
        if len(entries) < limit and task_type:
            type_entries = self.recall(tag=task_type, limit=limit)
            seen_keys = {e["key"] for e in entries}
            for e in type_entries:
                if e["key"] not in seen_keys and len(entries) < limit:
                    entries.append(e)
                    seen_keys.add(e["key"])

        # 如果还是没有，取最近几条
        if not entries:
            entries = self.get_recent(limit=limit)

        if not entries:
            return ""

        lines = ["\n[历史记忆 / Relevant History]"]
        for e in entries:
            lines.append(f"  [{e['key']}] {e['value']}")
        return "\n".join(lines)

    def clear(self):
        """清空全部记忆。"""
        self._write_all([])
