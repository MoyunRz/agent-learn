# 多 Agent 协作执行流程

## 架构总览

```mermaid
flowchart TD
    USER["用户查询"] --> SUP["SupervisorAgent<br/>调度中心"]

    SUP -->|"1. 规划"| PLANNER["PlannerAgent<br/>任务分解"]
    PLANNER -->|"LLM 生成 JSON"| PLAN["ExecutionPlan<br/>子任务 + 依赖图"]

    PLAN -->|"2. 调度"| LOOP{"依赖感知<br/>调度循环"}

    LOOP -->|"就绪任务"| DISPATCH{"就绪数量?"}

    DISPATCH -->|"单个"| SYNC["同步执行<br/>无线程创建"]
    DISPATCH -->|"多个"| PARALLEL["ThreadPoolExecutor<br/>并行执行"]

    SYNC --> WORKER_SEL["_select_worker()<br/>按任务类型匹配 Worker"]
    PARALLEL --> WORKER_SEL

    WORKER_SEL -->|"calculation"| CALC["CalculatorWorker<br/>工具: add, multiply"]
    WORKER_SEL -->|"coding"| CODE["CoderWorker<br/>工具: execute_code"]
    WORKER_SEL -->|"research"| RES["ResearchWorker"]
    WORKER_SEL -->|"兜底"| DEF["第一个可用 Worker"]

    CALC --> EXEC["worker.execute(task, dep_results)"]
    CODE --> EXEC
    RES --> EXEC
    DEF --> EXEC

    EXEC --> REACT["ReActAgent.run()<br/>LLM 推理 + 工具调用"]
    REACT --> RESULT["返回结果字符串"]

    RESULT --> STORE["存入 results & SharedState"]
    STORE --> LOOP

    LOOP -->|"全部完成"| SYNTH["_synthesize_response()<br/>汇总所有结果"]
    SYNTH --> OUTPUT["最终输出"]

    style SUP fill:#fff3e0
    style PLANNER fill:#e3f2fd
    style PLAN fill:#f3e5f5
    style LOOP fill:#e8f5e9
    style CALC fill:#ff9800,color:#fff
    style CODE fill:#2196f3,color:#fff
    style OUTPUT fill:#4caf50,color:#fff
```

## 完整执行流程（以 "先计算 15+25，然后乘以 3，最后用代码验证结果" 为例）

```mermaid
sequenceDiagram
    actor 用户
    participant main as main.py
    participant sup as Supervisor
    participant planner as Planner
    participant llm_p as LLM-规划
    participant calc as Calculator
    participant code as Coder
    participant llm as LLM-执行
    participant tools as ToolRegistry
    participant state as SharedState

    用户->>main: main_multi_agent.py
    main->>main: 注册工具 + 创建 Agent
    main->>sup: run(query)

    Note over sup,llm_p: ══ Phase 1: 任务规划 ══
    activate sup
    sup->>planner: run(query)
    activate planner
    planner->>planner: _build_plan_prompt()
    planner->>llm_p: ReActAgent 调用 LLM
    llm_p-->>planner: JSON 执行计划
    planner->>planner: _parse_plan_response()
    deactivate planner
    planner-->>sup: ExecutionPlan (task_1, task_2, task_3)
    deactivate sup

    Note over sup,llm: ══ Phase 2: 执行 task_1 ══
    activate sup
    sup->>sup: ready=[task_1] / 同步执行
    sup->>calc: execute(task_1, dep={})
    activate calc
    calc->>calc: prompt="Task: 计算 15+25"
    calc->>llm: ReActAgent.run()
    llm-->>calc: tool_use(add, a=15, b=25)
    calc->>tools: add.call(15, 25)
    tools-->>calc: 40
    calc->>llm: tool_result: 40
    llm-->>calc: text: "40"
    deactivate calc
    calc-->>sup: "40"
    sup->>state: store(task_1, "40")
    deactivate sup

    Note over sup,llm: ══ Phase 3: 执行 task_2 (依赖 task_1) ══
    activate sup
    sup->>sup: ready=[task_2] / dep={"task_1":"40"}
    sup->>calc: execute(task_2, dep={"task_1":"40"})
    activate calc
    calc->>calc: prompt="乘以 3 / [task_1]: 40"
    calc->>llm: ReActAgent.run()
    llm-->>calc: tool_use(multiply, a=40, b=3)
    calc->>tools: multiply.call(40, 3)
    tools-->>calc: 120
    calc->>llm: tool_result: 120
    llm-->>calc: text: "120"
    deactivate calc
    calc-->>sup: "120"
    sup->>state: store(task_2, "120")
    deactivate sup

    Note over sup,llm: ══ Phase 4: 执行 task_3 (依赖 task_2) ══
    activate sup
    sup->>sup: ready=[task_3] / dep={"task_2":"120"}
    sup->>code: execute(task_3, dep={"task_2":"120"})
    activate code
    code->>code: prompt="代码验证 / [task_2]: 120"
    code->>llm: ReActAgent.run()
    code->>tools: execute_code("120==(15+25)*3")
    tools-->>code: True
    llm-->>code: text: "验证通过"
    deactivate code
    code-->>sup: "验证通过"
    sup->>state: store(task_3, "验证通过")
    deactivate sup

    Note over sup: ══ Phase 5: 汇总输出 ══
    activate sup
    sup->>sup: _synthesize_response()
    sup-->>main: 汇总结果
    main->>main: client.close()
    main-->>用户: 最终输出
    deactivate sup
```


## 依赖感知调度算法

```mermaid
flowchart TD
    START["_execute_plan(plan)"] --> INIT["results = 空字典<br/>completed = 空集合"]

    INIT --> CHECK{"未完成任务 > 0 ?"}
    CHECK -->|否| DONE["return results"]

    CHECK -->|是| READY["get_ready_tasks(completed)"]
    READY --> EMPTY{"就绪任务为空?"}
    EMPTY -->|是| DONE
    EMPTY -->|否| INJECT["注入依赖任务结果<br/>→ dep_results"]

    INJECT --> COUNT{"就绪任务 == 1 ?"}

    COUNT -->|是| SYNC["同步执行<br/>无线程创建"]
    COUNT -->|否| PARALLEL["ThreadPoolExecutor<br/>并行提交 N 个任务"]

    SYNC --> COLLECT1["收集结果, 标记完成"]
    PARALLEL --> WAIT["as_completed 等待完成"]
    WAIT --> COLLECT2["收集结果, 标记完成"]

    COLLECT1 --> STORE["写入 SharedState"]
    COLLECT2 --> STORE

    STORE --> CHECK

    style START fill:#e3f2fd
    style DONE fill:#4caf50,color:#fff
    style EMPTY fill:#f44336,color:#fff
    style SYNC fill:#ff9800,color:#fff
    style PARALLEL fill:#2196f3,color:#fff
```

## 消息与数据流

```mermaid
flowchart LR
    subgraph 输入
        Q["用户查询"]
    end

    subgraph Planner
        P1["_build_plan_prompt()<br/>构造 JSON 格式要求"]
        P2["ReActAgent.run()<br/>调用 LLM 生成计划"]
        P3["_parse_plan_response()<br/>正则提取 JSON"]
        P4["回退: 单 task 兜底"]
    end

    subgraph 计划
        T1["Task(id='task_1')<br/>type: calculation<br/>deps: []"]
        T2["Task(id='task_2')<br/>type: calculation<br/>deps: [task_1]"]
        T3["Task(id='task_3')<br/>type: coding<br/>deps: [task_2]"]
    end

    subgraph 数据传递
        S1["results['task_1'] = '40'"]
        S2["dep_results = {<br/>  'task_1': '40'<br/>}"]
        S3["Worker prompt:<br/>Task + dep_results"]
    end

    subgraph 输出
        O1["[calculation] 40"]
        O2["[calculation] 120"]
        O3["[coding] 验证通过"]
    end

    Q --> P1 --> P2 --> P3
    P3 -.->|解析失败| P4
    P3 --> T1
    T1 -->|依赖| T2
    T2 -->|依赖| T3

    T1 --> S1 --> S2 --> S3
    S3 -->|task_2 执行| O2
    S3 -->|task_3 执行| O3
    T1 -->|直接执行| O1
```

## Worker 类型映射

```mermaid
flowchart LR
    TASK["task.type"] --> MAP{"匹配规则"}

    MAP -->|"calculation<br/>arithmetic<br/>math"| CALC["CalculatorWorker<br/>工具: add, multiply"]
    MAP -->|"coding<br/>code_generation<br/>code_review"| CODE["CoderWorker<br/>工具: execute_code"]
    MAP -->|"research"| RES["ResearchWorker"]
    MAP -->|"default worker<br/>存在?"| DEF{"检查 workers['default']"}
    DEF -->|是| DEF_W["workers['default']"]
    DEF -->|否| FALLBACK["next(iter(workers.values()))<br/>第一个可用 Worker"]

    style MAP fill:#f3e5f5
    style CALC fill:#ff9800,color:#fff
    style CODE fill:#2196f3,color:#fff
```

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **前置结果注入** | `_execute_plan` 在提交任务前从 `results` 提取依赖任务结果，通过 `dep_results` 传入 Worker，写入 prompt 让 LLM 看到上游输出 |
| **类型自动转换** | `Tool.call()` 根据函数签名的类型注解（`a: int`）自动将 LLM 传来的字符串参数转为对应类型，防止 `"40" * "3"` 之类的错误 |
| **单任务不走线程池** | `ready_tasks == 1` 时直接同步执行，避免创建无用 daemon 线程，同时规避 PyCharm + Python 3.13 线程清理异常 |
| **JSON 解析容错** | Planner 用正则提取 LLM 响应中的 JSON，解析失败时回退为单个通用 task，保证系统不崩溃 |
| **共享 Anthropic Client** | 所有组件共用同一个 client 实例，减少 httpx 连接池和 daemon 线程数量 |
| **优雅关闭** | 脚本结束前调用 `client.close()` 显式关闭连接池、join 所有线程 |
