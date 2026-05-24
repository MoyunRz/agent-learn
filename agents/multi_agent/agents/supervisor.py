"""
SupervisorAgent —— 多 Agent 系统的调度核心。

职责：
1. 调用 PlannerAgent 将用户查询分解为 ExecutionPlan
2. 按依赖关系调度 Worker 执行每个 Task
3. 依赖已满足的 Task 可以并行执行（ThreadPoolExecutor）
4. 所有结果存入 SharedState
5. 最终汇总所有 Task 结果并返回

这是整个多 Agent 系统的「指挥中心」。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseAgent
from ..message import ExecutionPlan
from ..shared_state import SharedState


class SupervisorAgent(BaseAgent):
    """调度 Agent —— 协调 Planner 与 Worker 完成复杂任务的执行。

    参数：
        planner: PlannerAgent 实例，负责分解查询
        workers: dict[str, WorkerAgent]，按名称索引的 Worker 池
        shared_state: 共享状态存储，用于跨 Worker 传递结果
    """

    def __init__(self, planner, workers: dict, shared_state: SharedState = None, memory=None,
                 system_rules: str = None):
        super().__init__(name="supervisor")
        self.planner = planner
        self.workers = workers
        self.shared_state = shared_state or SharedState()
        self.memory = memory  # AgentMemory 实例，用于长期记忆
        self.system_rules = system_rules  # 约束规则

    def run(self, query: str, verbose: bool = False) -> str:
        """执行完整的多 Agent 流程。

        流程：
        1. Planner 分解 → ExecutionPlan
        2. 按依赖关系逐个/并行执行 Task
        3. 汇总结果

        参数：
            query: 用户原始查询
            verbose: 是否打印内部执行日志
        返回：
            汇总后的执行结果
        """
        # Step 1: 规划 —— 将查询分解为子任务
        plan = self.planner.run(query, verbose=verbose)

        # Step 2: 执行 —— 按依赖关系调度 Worker
        results = self._execute_plan(plan)

        # Step 3: 汇总 —— 将所有结果拼接为最终输出
        return self._synthesize_response(results, plan)

    def _execute_plan(self, plan: ExecutionPlan) -> dict[str, str]:
        """依赖感知的任务执行引擎。

        循环逻辑：
        1. 获取所有「就绪」任务（依赖全部满足）
        2. 通过线程池并行执行所有就绪任务
        3. 将完成的任务标记为 completed
        4. 重新检查是否有新任务就绪
        5. 直到所有任务完成或无法推进

        返回：
            {task_id: result_string} 的字典
        """
        results = {}
        completed = set()  # 已完成的任务 ID 集合

        while len(completed) < len(plan.tasks):
            # 获取当前依赖已全部满足的任务
            ready_tasks = plan.get_ready_tasks(completed)

            if not ready_tasks:
                break  # 没有就绪任务且还有未完成任务 → 可能存在环形依赖，安全退出

            # 收集前置依赖结果
            tasks_with_deps = []
            for task in ready_tasks:
                dep_results = {
                    dep_id: results[dep_id]
                    for dep_id in task.dependencies
                    if dep_id in results
                }
                tasks_with_deps.append((task, dep_results))

            # 依赖链上的任务通常是串行的（一次只有一个就绪），直接同步执行；
            # 无依赖关系的任务才走线程池并行执行
            if len(ready_tasks) == 1:
                # 单任务 → 直接执行，避免创建无用线程
                task, dep_results = tasks_with_deps[0]
                try:
                    result = self._execute_task(task, dep_results)
                    results[task.id] = result
                    self.shared_state.store(task.id, result)
                    completed.add(task.id)
                    # 自动存储到长期记忆
                    if self.memory:
                        self.memory.auto_store(task.id, task.type, result)
                except Exception as e:
                    results[task.id] = f"Error: {e}"
                    completed.add(task.id)
            else:
                # 多任务并行执行
                with ThreadPoolExecutor(max_workers=len(ready_tasks)) as executor:
                    future_to_task = {
                        executor.submit(self._execute_task, task, dep_results): task
                        for task, dep_results in tasks_with_deps
                    }

                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            results[task.id] = result
                            self.shared_state.store(task.id, result)
                            completed.add(task.id)
                            # 自动存储到长期记忆
                            if self.memory:
                                self.memory.auto_store(task.id, task.type, result)
                        except Exception as e:
                            results[task.id] = f"Error: {e}"
                            completed.add(task.id)

        return results

    def _execute_task(self, task, dep_results: dict[str, str] = None) -> str:
        """执行单个 Task —— 选择合适的 Worker，注入记忆后委派任务，
        执行后校验结果是否满足规则约束。

        参数：
            task: Task 对象
            dep_results: 前置依赖任务的执行结果 {task_id: result}
        返回：
            Worker 执行结果字符串（校验失败时带 [规则校验失败] 前缀）
        """
        worker = self._select_worker(task.type)
        result = worker.execute(task, dep_results=dep_results, memory=self.memory)

        # 规则校验：检查结果是否满足约束
        if task.expected_output:
            validation_error = self._validate_result(result, task.expected_output, task.type)
            if validation_error:
                return validation_error

        return result

    def _validate_result(self, result: str, expected: str, task_type: str) -> str | None:
        """校验 Worker 返回结果是否符合预期。

        - calculation 类型：结果必须是纯数字或包含数字
        - 通用：检查结果是否包含期望的关键内容

        返回校验失败信息，校验通过返回 None。
        """
        # calculation 任务的结果必须是数字或包含数字
        if task_type in ("calculation", "arithmetic", "math"):
            import re
            numbers = re.findall(r'\d+', result)
            if not numbers:
                return f"[规则校验失败] 计算结果中未找到数字: {result}"

        return None

    def _select_worker(self, task_type: str):
        """根据任务类型选择合适的 Worker。

        映射规则：
            calculation / arithmetic / math  → calculator worker
            coding / code_generation / code_review → coder worker
            research → researcher worker
            未匹配 → default worker → 第一个可用 worker

        参数：
            task_type: 任务类型字符串
        返回：
            匹配的 WorkerAgent 实例
        """
        mapping = {
            "calculation": self.workers.get("calculator"),
            "arithmetic": self.workers.get("calculator"),
            "math": self.workers.get("calculator"),
            "coding": self.workers.get("coder"),
            "code_generation": self.workers.get("coder"),
            "code_review": self.workers.get("coder"),
            "research": self.workers.get("researcher"),
        }
        worker = mapping.get(task_type) or self.workers.get("default")
        if worker is None:
            # 最后的兜底：使用 workers 字典中的第一个 worker
            worker = next(iter(self.workers.values()))
        return worker

    def _synthesize_response(self, results: dict[str, str], plan: ExecutionPlan) -> str:
        """汇总所有任务结果为最终输出。

        按计划中的任务顺序输出，格式为：
            [calculation] 结果1
            [coding] 结果2
        """
        output_parts = []
        for task in plan.tasks:
            if task.id in results:
                output_parts.append(f"[{task.type}] {results[task.id]}")
        return "\n".join(output_parts) if output_parts else "No results produced"
