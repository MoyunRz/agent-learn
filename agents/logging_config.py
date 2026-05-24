"""
集中化日志配置 —— 为所有 agent 模块提供统一的日志输出。

特性：
- 单例初始化，重复调用 setup_logging() 不会创建重复 handler
- 所有日志写入 stdout，格式为「时间 [级别] 消息」
- 通过 get_logger("agents") 获取同一个 logger 实例
"""

import logging
import sys

_initialized = False  # 防止重复初始化


def get_logger(name: str = "agents") -> logging.Logger:
    """获取名为 'agents' 的 logger。

    所有 agent 模块通过此函数获取同一个 logger，
    确保日志格式和级别统一控制。
    """
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """初始化日志系统（全局只执行一次）。

    参数：
        level: 日志级别，默认 INFO。调试时传入 logging.DEBUG。

    输出格式示例：
        14:32:05 [INFO ] Run started | query='计算 3+5'
    """
    global _initialized
    if _initialized:
        return  # 已经初始化过，跳过
    _initialized = True

    logger = logging.getLogger("agents")
    logger.setLevel(level)

    # 只输出到 stdout，不传播到 root logger（避免重复输出）
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # handler 放开到 DEBUG，由 logger 级别控制实际输出
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
