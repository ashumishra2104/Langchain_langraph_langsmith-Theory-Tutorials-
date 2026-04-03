# 🔗 Langgraph

> **Instructor:** [Ashu Mishra](https://www.linkedin.com/in/ashumish/)  
> **Part of:** AI Product Management Cohort — LangChain, LangGraph & LangSmith Tutorial Series

This folder contains everything related to **LangGraph** — theory, teaching guides, and a full multi-version Blog Writer Agent built progressively from scratch.

---

## 📁 Folder Structure

```
Langgraph/
├── Theory/
│   ├── README.md
│   └── LangGraph_Teaching_Guide_new.pdf
│
└── Blog Agent/
    ├── Basic Agent/          ← V1: LangGraph fundamentals, Fan-Out
    ├── Improved Agent/       ← V2: Orchestrator-Worker, Pydantic, Map-Reduce
    ├── Research Agent/       ← V3: Tavily web research integration
    └── V4 - Image & Research Agent/  ← V4: AI images + Streamlit UI
```

---

## 🧠 Learning Path

| Step | Folder | What You Learn |
|------|--------|----------------|
| 1 | `Theory/` | Why LangGraph exists, state management, graph concepts |
| 2 | `Blog Agent/Basic Agent/` | Nodes, shared state, Fan-Out with `Send()` |
| 3 | `Blog Agent/Improved Agent/` | Orchestrator-Worker pattern, Pydantic, ordered reduce |
| 4 | `Blog Agent/Research Agent/` | Real-time web research with Tavily |
| 5 | `Blog Agent/V4 - Image & Research Agent/` | AI image generation + Streamlit frontend |

---

## 🔑 Key Concepts Covered

- **StateGraph** — shared memory across agent steps
- **Nodes** — individual thinking steps of the agent
- **Fan-Out with `Send()`** — parallel execution
- **Orchestrator-Worker (Map-Reduce)** — production-grade pattern
- **Pydantic models** — structured output enforcement
- **LangSmith tracing** — debugging and evaluating agents

---

## 🔗 Connect

- LinkedIn: [linkedin.com/in/ashumish](https://www.linkedin.com/in/ashumish/)
- GitHub: [github.com/ashumishra2104](https://github.com/ashumishra2104)
