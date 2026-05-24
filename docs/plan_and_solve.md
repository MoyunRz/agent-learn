# Plan-and-Solve (P&S)

## 概述

Plan-and-Solve 是 Wang et al. (2023) 提出的两阶段推理策略。核心思想：**先看清全貌再动手**。

与 ReAct（边想边做）不同，P&S 强制 LLM 先生成完整的步骤计划，然后按序执行，每步都能看到完整计划和已完成步骤的结果。

## 架构

```mermaid
flowchart TD
    Q["用户问题"] --> P1["Phase 1: Plan<br/>生成步骤列表"]

    P1 --> P1A["LLM 分析问题<br/>分解为编号步骤"]
    P1A --> PLAN["plan = [<br/>  1. 计算 100/5<br/>  2. 将结果 *3<br/>  3. 然后 +20<br/>]"]

    PLAN --> P2["Phase 2: Solve<br/>按序执行每个步骤"]

    P2 --> LOOP{"还有未执行的步骤?"}
    LOOP -->|是| BUILD["构建上下文 prompt:<br/>- 原始问题<br/>- 完整计划<br/>- 已完成步骤+结果<br/>- 当前要执行的步骤"]

    BUILD --> EXEC["ReActAgent.run()<br/>LLM 推理 + 工具调用"]
    EXEC --> STORE["存入 history:<br/>(步骤, 结果)"]
    STORE --> LOOP

    LOOP -->|否| SYNTH["_synthesize()<br/>汇总所有结果 → 最终答案"]
    SYNTH --> OUT["输出: 答案"]

    style P1 fill:#4a90d9,color:#fff
    style P2 fill:#50b86c,color:#fff
    style OUT fill:#e67e22,color:#fff
```

## 完整执行时序

```mermaid
sequenceDiagram
    actor 用户
    participant ps as PlanAndSolveAgent
    participant plan_llm as LLM-规划
    participant step_agent as ReActAgent
    participant step_llm as LLM-执行
    participant tools as ToolRegistry

    用户->>ps: run("计算 100/5, 然后 *3, 加上 20")

    rect rgb(74, 144, 217)
        Note over ps,plan_llm: Phase 1: Plan
        ps->>plan_llm: "请分解为编号步骤"
        plan_llm-->>ps: "1. 计算 100/5\n2. 乘以 3\n3. 加上 20"
        ps->>ps: 解析为 plan = [step1, step2, step3]
    end

    rect rgb(80, 184, 108)
        Note over ps,step_agent: Phase 2: Solve - Step 1
        ps->>ps: _build_step_prompt(step=1)
        ps->>step_agent: run(上下文 prompt)
        step_agent->>step_llm: "请执行: 计算 100/5"
        step_llm-->>step_agent: tool_use(divide, 100, 5)
        step_agent->>tools: divide(100, 5)
        tools-->>step_agent: 20
        step_llm-->>step_agent: "20"
        step_agent-->>ps: "20"
        ps->>ps: history.append(("计算100/5", "20"))
    end

    rect rgb(80, 184, 108)
        Note over ps,step_agent: Step 2 (能看见 Step 1 的结果)
        ps->>step_agent: run("已完成: [100/5=20]\n请执行: *3")
        step_agent->>step_llm: tool_use(multiply, 20, 3)
        step_agent->>tools: multiply(20, 3)
        tools-->>step_agent: 60
        step_agent-->>ps: "60"
    end

    rect rgb(80, 184, 108)
        Note over ps,step_agent: Step 3 (能看见 Step 1+2 的结果)
        ps->>step_agent: run("已完成: [100/5=20, *3=60]\n请执行: +20")
        step_agent->>step_llm: tool_use(add, 60, 20)
        step_agent->>tools: add(60, 20)
        tools-->>step_agent: 80
        step_agent-->>ps: "80"
    end

    rect rgb(230, 126, 34)
        Note over ps: Synthesize
        ps->>ps: _synthesize()
        ps-->>用户: "最终答案: 80"
    end
```

## 关键设计

### Step Prompt 结构

每一步发给 LLM 的 prompt 包含四部分：

```
Original problem: 计算 100/5, 然后 *3, 加上 20

Full plan:
  Step 1: 计算 100/5
  Step 2: 将结果乘以 3
  Step 3: 将结果加上 20

Completed steps and their results:
  Step 1 [计算 100/5]: 20

Now execute ONLY Step 2: 将结果乘以 3

Remaining steps (do NOT execute these yet):
  Step 3: 将结果加上 20
```

关键点：LLM 知道前面做了什么、下一步做什么、还有什么没做。避免了多 Agent 系统中前置结果丢失的问题。

### 与 Multi-Agent 的对比

| 维度 | Plan-and-Solve | Multi-Agent (Supervisor) |
|------|---------------|--------------------------|
| 规划方式 | LLM 文本列表 | Planner 生成 JSON Task |
| 执行单元 | 同一 Agent 循环 | 不同 Worker Agent |
| 上下文传递 | 累积 history | dep_results 注入 |
| 并行执行 | 不支持（串行） | 支持（依赖图） |
| 适合场景 | 线性推理 | 复杂依赖 + 并行 |

### 容错

- `_generate_plan()` 解析失败时，按行拆分 LLM 响应，每个非空行当作一个步骤
- 每步执行使用独立的 `ReActAgent` 实例，避免消息历史混乱
