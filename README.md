# py-agents — 模块化多智能体框架

基于 Anthropic SDK 构建的多智能体推理框架，支持 ReAct、Plan-and-Solve、Tree of Thoughts、Supervisor 和 Network 五种推理范式。

## 项目架构

```
BaseAgent (抽象基类)
  ├── PlannerAgent          # 任务分解
  ├── SupervisorAgent       # 中心调度器
  └── WorkerAgent           # 多继承 ReActAgent + BaseAgent
        ├── CalculatorWorker
        ├── CoderWorker
        ├── ResearchWorker
        ├── PlanAndSolveWorker
        └── NetworkAgent    # P2P 网络节点
```

所有 Agent 共享 `ToolRegistry` 工具系统，通过 ReAct 循环（思考 → 工具调用 → 观察 → 重复）进行推理。

## 五种推理范式

| 范式 | 描述 | 入口文件 |
|------|------|---------|
| **ReAct** | 思考-行动循环，LLM 自主调用工具直至得出结论 | `main.py` |
| **Plan-and-Solve** | 先规划步骤，再逐步执行 | `demo_plan_and_solve.py` |
| **Tree of Thoughts** | BFS 搜索推理树，剪枝保留最优路径 | `demo_tree_of_thoughts.py` |
| **Supervisor** | 中心调度：Planner 分解任务 → Supervisor 分配给 Worker 执行 | `main_multi_agent.py` |
| **Network** | 去中心化 P2P：链式传递、多轮辩论、广播 | `demo_network.py` |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 MINIMAX_API_KEY
```

### 3. 运行示例

```bash
python main.py                     # 基础 ReAct 推理
python main_multi_agent.py         # Supervisor 多智能体协作
python demo_network.py             # Network 链式 + 辩论
python demo_pas_worker.py          # Supervisor + PlanAndSolve Worker
python demo_plan_and_solve.py      # 独立 Plan-and-Solve
python demo_tree_of_thoughts.py    # Tree of Thoughts BFS 搜索
```

## Network 模式详解

Network 模式是去中心化的 P2P 多智能体网络，支持三种执行模式：

### 链式（Chain）

消息沿有向边依次传递，每步精炼：

```
用户查询 → 分析师 → 评审专家 → 汇总专家 → 结果
```

### 辩论（Debate）

所有 Agent 多轮发表观点，最后由 moderator 汇总：

```
Round 0: 所有 Agent 独立给出初始观点
Round 1: 看到其他人的观点，更新自己观点
Round 2: 再次更新...
最终: moderator 综合所有观点给出结论
```

关键设计：
- **编排是程序化的**：通过 for 循环控制，不是 AI 自主决策
- **忽略邻接表**：遍历所有已注册 Agent
- **同步回合制**：Agent 每轮只看到上一轮的观点
- **Moderator 不参与辩论**：只在最终汇总阶段介入

### 广播（Broadcast）

从起始节点 BFS 遍历，每个节点处理后转发给未访问的邻居。

## 辩论执行流程

```
all_opinions = {}   # {轮次号: {agent_name → opinion_text}}

# Round 0 — 初始观点
for agent_name in agent_names:
    opinion = agent.receive(msg_type="query")
    all_opinions[0][agent_name] = opinion

# Round 1..N — 辩论回合
for round in range(1, rounds+1):
    for agent_name in agent_names:
        # 取出其他人上一轮的观点
        others = all_opinions[round-1] 中排除自己
        opinion = agent.receive(others)  # 基于他人观点更新
        all_opinions[round][agent_name] = opinion

# 汇总
final = moderator.receive(all_opinions[N])
```

## 目录结构

```
py-agents/
├── agents/
│   ├── tools.py              # 工具系统（Tool, ToolRegistry）
│   ├── react.py              # ReActAgent 推理引擎
│   ├── plan_and_solve.py     # PlanAndSolveAgent
│   ├── tree_of_thoughts.py   # TreeOfThoughts BFS 搜索
│   ├── logging_config.py     # 日志系统
│   ├── multi_agent/
│   │   ├── network.py        # P2P 网络（NetworkAgent, AgentNetwork）
│   │   ├── message.py        # 消息模型（Task, ExecutionPlan）
│   │   ├── shared_state.py   # 线程安全 KV 存储
│   │   ├── memory.py         # 跨会话长期记忆
│   │   └── agents/
│   │       ├── base.py       # BaseAgent 抽象基类
│   │       ├── planner.py    # 任务分解器
│   │       ├── supervisor.py # 中心调度器
│   │       └── worker.py     # Worker 及其子类
│   └── workers/
│       └── __init__.py       # Worker 工厂函数
├── docs/                     # Mermaid 流程图文档
├── main.py                   # ReAct 示例
├── main_multi_agent.py       # Supervisor 示例
├── demo_network.py           # Network 示例（链式 + 辩论）
├── demo_pas_worker.py        # Supervisor + P&S Worker 示例
├── demo_plan_and_solve.py    # Plan-and-Solve 示例
├── demo_tree_of_thoughts.py  # Tree of Thoughts 示例
├── requirements.txt          # anthropic + python-dotenv
└── .env.example              # API Key 模板
```

## 依赖

- `anthropic` — LLM API 调用（兼容任意 Anthropic 兼容端点）
- `python-dotenv` — 加载 .env 环境变量

默认使用 MiniMax M2.7 模型，可通过 `model` 参数切换。
