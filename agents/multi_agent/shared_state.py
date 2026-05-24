"""
共享状态存储 —— 多 Agent 间的线程安全键值对存储，带 JSON 文件持久化。

用途：
- Supervisor 将各 Worker 的执行结果写入 SharedState
- 其他 Worker / 后续流程可以从 SharedState 读取前置任务的结果
- 所有操作都会记录到磁盘（JSON 文件），支持事后审计和恢复

线程安全：
- 所有写操作（store / clear）使用 threading.Lock 保护
- 读操作（get / get_all / get_history）不加锁，允许高并发读取
"""

import json
import os
import threading
from datetime import datetime
from typing import Any
from pathlib import Path


class SharedState:
    """线程安全的共享 KV 存储，带 JSON 持久化。

    每次 store() 调用会：
    1. 更新内存中的值
    2. 追加到操作历史
    3. 写入磁盘 JSON 文件（完整的操作日志）

    参数：
        persist_file: JSON 持久化文件路径，默认 .agent_state.json
    """

    def __init__(self, persist_file: str = ".agent_state.json"):
        self._store: dict[str, Any] = {}       # 当前状态，key → value
        self._history: list[dict] = []          # 完整操作历史，每项含 timestamp/key/value
        self._lock = threading.Lock()           # 写操作锁
        self._persist_file = persist_file

    def store(self, key: str, value: Any):
        """存储一个键值对（线程安全，自动持久化）。

        参数：
            key: 键名（通常用 task_id）
            value: 值（通常用 Worker 返回的结果字符串）
        """
        with self._lock:
            self._store[key] = value
            entry = {
                "timestamp": datetime.now().isoformat(),
                "key": key,
                "value": value,
            }
            self._history.append(entry)
            self._persist_append(entry)  # 写入磁盘

    def get(self, key: str) -> Any:
        """读取单个键的值，不存在返回 None。"""
        return self._store.get(key)

    def get_all(self) -> dict[str, Any]:
        """返回当前所有键值对的浅拷贝。"""
        return self._store.copy()

    def get_history(self) -> list[dict]:
        """返回完整的操作历史列表。"""
        return self._history

    def clear(self):
        """清空所有状态和历史（线程安全）。"""
        with self._lock:
            self._store.clear()
            self._history.clear()

    def update(self, updates: dict[str, Any]):
        """批量写入多个键值对。"""
        for key, value in updates.items():
            self.store(key, value)

    def _persist_append(self, entry: dict):
        """将单条操作记录追加到磁盘 JSON 文件。

        采用「读取→追加→写回」的方式，确保文件内容是一个 JSON 数组。
        写入失败时静默忽略（不影响核心流程）。
        """
        try:
            # 如果文件已存在，读取原有数组；否则初始化为空数组
            if os.path.exists(self._persist_file):
                with open(self._persist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(entry)

            with open(self._persist_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 持久化失败不影响主流程
