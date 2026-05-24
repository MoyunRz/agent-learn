"""
ReAct Agent —— 单 Agent 推理-行动循环的核心实现。

ReAct = Reasoning (推理) + Acting (行动)

流程：
1. 用户输入问题
2. LLM 分析问题，可能输出：
   - thinking 块：内部思考过程
   - tool_use 块：决定调用某个工具
   - text 块：直接给出最终答案
3. 如果有 tool_use → 执行工具 → 将结果返回 LLM → 回到步骤 2
4. 如果有 text → 这就是最终答案，结束循环
5. 超过 max_steps 仍未得到答案 → 终止并返回错误
"""

import json
import os
import time
import logging
import anthropic
from .tools import ToolRegistry
from .logging_config import get_logger

logger = get_logger(__name__)


class ReActAgent:
    """ReAct Agent —— 通过 MiniMax M2.7 模型实现思考-工具调用-回答的循环。

    核心参数：
        tools: 工具注册表，Agent 可以调用的所有工具
        client: Anthropic SDK 客户端（默认自动创建，指向 MiniMax 兼容 API）
        model: 模型名称
        max_steps: 最大推理步数，防止死循环
    """

    def __init__(
        self,
        tools: ToolRegistry,
        client: anthropic.Anthropic | None = None,
        model: str = "MiniMax-M2.7",
        max_steps: int = 10,
        system_rules: str = None,
    ):
        self.tools = tools
        # 默认创建指向 MiniMax 的 Anthropic 客户端
        self.client = client or anthropic.Anthropic(
            base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
            api_key=os.getenv("MINIMAX_API_KEY", "<api key>"),
        )
        self.model = model
        self.max_steps = max_steps
        self.system_rules = system_rules  # 约束规则，注入到每次 LLM 调用的第一条消息

    def _make_assistant_block(self, content: list) -> dict:
        """构建 assistant 角色的消息块，用于追加到对话历史中。"""
        return {"role": "assistant", "content": content}

    def run(self, user_query: str, verbose: bool = False) -> str:
        """执行 ReAct 循环，返回 LLM 的最终文本回答。

        参数：
            user_query: 用户问题
            verbose: 是否输出 DEBUG 级别日志

        返回：
            LLM 的最终文本回答，或超时后的错误信息
        """
        if verbose:
            logging.getLogger("agents").setLevel(logging.DEBUG)

        tool_names = list(self.tools._tools.keys())
        logger.info("=" * 50)
        logger.info("Run started | query=%r", user_query)
        logger.info("Tools: %s | model=%s | max_steps=%d",
                     tool_names, self.model, self.max_steps)

        # 消息历史：规则约束作为第一条消息（存在时），然后是用户输入
        messages = []
        if self.system_rules:
            messages.append({"role": "user", "content": [{"type": "text", "text": self.system_rules}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_query}]})
        tool_specs = self.tools.get_specs()  # 一次性获取所有工具规格
        t_start = time.time()

        # ==================== ReAct 主循环 ====================
        for step in range(self.max_steps):
            logger.info("--- Step %d/%d | messages=%d ---", step + 1, self.max_steps, len(messages))

            # --- 第 1 步：调用 LLM ---
            t_call = time.time()
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=messages,
                tools=tool_specs if tool_specs else None,  # API 要求空列表时传 None
            )
            elapsed = time.time() - t_call

            # 统计 response 中各类型 block 的数量
            block_types = {}
            for block in resp.content:
                block_types[block.type] = block_types.get(block.type, 0) + 1
            logger.debug("API call | elapsed=%.2fs | blocks=%s | usage=%s",
                         elapsed, block_types,
                         getattr(resp, "usage", None))

            # 打印 thinking 块（截断长文本）
            for block in resp.content:
                if block.type == "thinking":
                    thinking_preview = block.thinking[:200] + "..." if len(block.thinking) > 200 else block.thinking
                    logger.debug("[think] %s", thinking_preview)

            # --- 第 2 步：解析响应，分类 text 和 tool_use ---
            text_parts = []
            tool_use_blocks = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            # --- 第 3 步：有文本 → 这是最终答案，直接返回 ---
            if text_parts:
                answer = "\n".join(text_parts)
                logger.info("[answer] %s", answer)
                total_elapsed = time.time() - t_start
                logger.info("Run complete | steps=%d | duration=%.2fs",
                            step + 1, total_elapsed)
                logger.info("=" * 50)
                return answer

            # --- 第 4 步：有工具调用 → 逐个执行，将结果追加到消息历史 ---
            if tool_use_blocks:
                # 先把完整的 assistant 响应（含 tool_use 块）追加到历史
                messages.append(self._make_assistant_block(resp.content))

                tool_results = []
                for block in tool_use_blocks:
                    name = block.name    # 工具名
                    args = block.input   # LLM 给出的参数（dict）

                    tool = self.tools.get(name)

                    if tool is None:
                        # 工具不存在 → 返回错误信息给 LLM，让它自行纠正
                        result = f"Error: tool '{name}' not found"
                        logger.error("[call] %s(%s) → TOOL NOT FOUND", name, args)
                    else:
                        logger.info("[call] %s(%s)", name, args)
                        t_tool = time.time()
                        try:
                            result = str(tool.call(**args))
                            tool_elapsed = time.time() - t_tool
                            result_preview = result[:200] + "..." if len(result) > 200 else result
                            logger.debug("[result] %s → %s (%.3fs)",
                                        name, result_preview, tool_elapsed)
                        except Exception as e:
                            # 工具执行异常 → 也返回给 LLM，让它知道出错了
                            result = f"Error: {e}"
                            logger.error("[result] %s → FAILED: %s", name, e)

                    # 构建 tool_result 块（Anthropic 协议格式）
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                # 工具结果以 user 角色追加
                messages.append({"role": "user", "content": tool_results})
                logger.debug("Tool results appended | messages now=%d", len(messages))
                continue  # 回到循环开头，再次调用 LLM

            # --- 异常情况：既没 text 也没 tool_use ---
            logger.warning("Step %d: no text or tool_use in response | blocks=%s",
                           step + 1, list(block_types.keys()))

        # --- 超过最大步数 ---
        total_elapsed = time.time() - t_start
        logger.warning("Max steps (%d) exceeded | duration=%.2fs", self.max_steps, total_elapsed)
        logger.info("=" * 50)
        return "Max steps exceeded"
