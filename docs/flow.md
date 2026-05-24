# ReAct Agent 执行流程图

## 总览

```mermaid
flowchart TD
    A["用户输入: 计算 (3+5)*2"] --> B["main.py: agent.run()"]
    B --> C["ReActAgent.run() 初始化"]
    C --> D["组装 messages<br/>[user: '计算 (3+5)*2']"]
    D --> E["提取 tool_specs<br/>[add, multiply]"]
    E --> F["进入主循环 (max_steps=10)"]

    F --> G{"Step N"}
    G --> H["POST MiniMax M2.7<br/>携带 messages + tools"]
    H --> I["解析 response.content"]

    I --> J{"block.type?"}

    J -->|thinking| K["[think] 打印日志<br/>不参与逻辑"]
    J -->|text| L["收集到 text_parts"]
    J -->|tool_use| M["收集到 tool_use_blocks"]

    K --> N{"还有更多 block?"}
    L --> N
    M --> N

    N -->|是| J
    N -->|否| O{"text_parts 有内容?"}

    O -->|是| P["✅ 返回答案<br/>结束循环"]
    O -->|否| Q{"tool_use_blocks 有内容?"}

    Q -->|是| R["完整回传 assistant 消息<br/>(含 thinking + tool_use)"]
    Q -->|否| G

    R --> S["遍历每个 tool_use block"]
    S --> T["从 registry 查找工具"]
    T --> U{"工具存在?"}

    U -->|是| V["tool.call(args)<br/>执行 Python 函数"]
    U -->|否| W["返回错误信息"]

    V --> X["构造 tool_result"]
    W --> X
    X --> Y["拼到 messages 末尾<br/>role=user, content=tool_result"]
    Y --> Z{"step < max_steps?"}
    Z -->|是| G
    Z -->|否| AA["❌ Max steps exceeded"]

    style P fill:#4caf50,color:#fff
    style AA fill:#f44336,color:#fff
    style K fill:#2196f3,color:#fff
    style V fill:#ff9800,color:#fff
```

## 以 "计算 (3+5)*2" 为例的完整追踪

```mermaid
sequenceDiagram
    actor 用户
    participant main as main.py
    participant agent as ReActAgent
    participant llm as MiniMax M2.7
    participant tools as ToolRegistry

    用户->>main: python main.py
    main->>tools: register(add)
    main->>tools: register(multiply)
    main->>agent: agent.run("计算 (3+5)*2")

    rect rgb(240, 248, 255)
        Note over agent,llm: Step 1
        agent->>llm: POST messages=[user:"计算(3+5)*2"]<br/>tools=[add, multiply]
        llm-->>agent: content=[thinking, tool_use(name="add", a=3, b=5)]
        agent->>tools: get("add")
        tools-->>agent: Tool(add)
        agent->>tools: add.call(a=3, b=5)
        tools-->>agent: 8
        agent->>llm: assistant=完整content + user=[tool_result: 8]
    end

    rect rgb(255, 248, 240)
        Note over agent,llm: Step 2
        agent->>llm: POST messages=[user, assistant, tool_result]<br/>tools=[add, multiply]
        llm-->>agent: content=[thinking, tool_use(name="multiply", a=8, b=2)]
        agent->>tools: get("multiply")
        tools-->>agent: Tool(multiply)
        agent->>tools: multiply.call(a=8, b=2)
        tools-->>agent: 16
        agent->>llm: assistant=完整content + user=[tool_result: 16]
    end

    rect rgb(240, 255, 240)
        Note over agent,llm: Step 3
        agent->>llm: POST messages=[user, assistant, tool, assistant, tool_result]
        llm-->>agent: content=[thinking, text="16"]
        agent-->>main: return "16"
        main-->>用户: 结果: 16
    end
```

## run() 内部决策树

```mermaid
flowchart TD
    START["run(user_query, verbose)"] --> INIT["messages = [user]<br/>tool_specs = registry.get_specs()"]
    INIT --> LOOP{"for step in range(max_steps)"}
    
    LOOP --> CALL["client.messages.create(<br/>  model, max_tokens=4096,<br/>  messages, tools<br/>)"]
    
    CALL --> SCAN["扫描 response.content"]
    
    SCAN --> TEXT_CHECK{"存在 text 块?"}
    TEXT_CHECK -->|是| RETURN["return text<br/>✅ 任务完成"]
    TEXT_CHECK -->|否| TOOL_CHECK{"存在 tool_use 块?"}
    
    TOOL_CHECK -->|否| NEXT["continue 下一轮"]
    TOOL_CHECK -->|是| SAVE["保存 assistant 消息<br/>(完整 content 列表)"]
    
    SAVE --> ITER["遍历 tool_use 块"]
    ITER --> FIND["tool = registry.get(block.name)"]
    FIND --> EXEC["result = tool.call(**block.input)"]
    EXEC --> TCONV["result → str"]
    TCONV --> APPEND["追加 tool_result 到 messages"]
    
    APPEND --> MORE{"还有更多 tool_use?"}
    MORE -->|是| ITER
    MORE -->|否| LOOP
    
    NEXT -.-> LOOP

    LOOP -->|超过步数| MAX["return 'Max steps exceeded'<br/>❌ 异常终止"]

    style RETURN fill:#4caf50,color:#fff
    style MAX fill:#f44336,color:#fff
```

## 消息结构演变

```mermaid
flowchart LR
    subgraph S1["Step 1 请求"]
        M1["role: user<br/>content: [{type: text, text: '计算(3+5)*2'}]"]
    end

    S1 -->|LLM 返回| R1["role: assistant<br/>content: [thinking, tool_use(add)]"]

    R1 -->|"tool_registry 执行"| T1["role: user<br/>content: [{type: tool_result, tool_use_id: xx, content: '8'}]"]

    subgraph S2["Step 2 请求 — 3 条消息"]
        M1
        R1
        T1
    end

    S2 -->|LLM 返回| R2["role: assistant<br/>content: [thinking, tool_use(multiply)]"]

    R2 -->|"tool_registry 执行"| T2["role: user<br/>content: [{type: tool_result, tool_use_id: yy, content: '16'}]"]

    subgraph S3["Step 3 请求 — 5 条消息"]
        M1
        R1
        T1
        R2
        T2
    end

    S3 -->|LLM 返回| R3["role: assistant<br/>content: [thinking, text('16')]"]

    R3 -->|"有 text → return"| DONE["✅ 答案: 16"]
```

## 文件关系

```mermaid
flowchart TD
    subgraph "入口"
        MAIN["main.py<br/>注册工具 + 启动 agent"]
        ENV[".env<br/>MINIMAX_API_KEY"]
    end

    subgraph "agents/tools.py"
        TOOL["Tool<br/>包装 Python 函数<br/>生成 anthropic_spec"]
        DECO["@tool 装饰器"]
        REG["ToolRegistry<br/>注册表: 按名存取"]
    end

    subgraph "agents/react.py"
        RACT["ReActAgent<br/>主循环 + 消息管理"]
    end

    ENV -->|load_dotenv| MAIN
    MAIN -->|register| REG
    MAIN -->|ReActAgent(tools=)| RACT

    TOOL -.->|"@tool 调用"| DECO
    TOOL -->|存储| REG
    REG -->|get_specs| RACT
    REG -->|get(name)| RACT
    RACT -->|anthropic.Anthropic| LLM["MiniMax M2.7<br/>api.minimaxi.com/anthropic"]

    style MAIN fill:#e3f2fd
    style RACT fill:#fff3e0
    style REG fill:#f3e5f5
    style LLM fill:#e8f5e9
```
