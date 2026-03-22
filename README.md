# LangChain, LangGraph & LangSmith — Theory and Tutorials

> A progressive, hands-on journey from basic AI agents to production-grade,
> research-augmented systems — built with the **Vibe Coding** method.  
> **Author:** [Ashu Mishra](https://github.com/ashumishra2104) · AI PM Cohort

---

## 📁 Repository Structure

### 1. [Blog Agent](./Blog%20Agent/) — Three-Generation Evolution

An AI blog-writing agent that has evolved across three versions, each adding a
major new capability. Read them in order to trace the full architectural journey.

| Version | Folder | What's New |
|---|---|---|
| V1 — Basic | [Basic Agent](./Blog%20Agent/Basic%20Agent/) | LangGraph nodes, shared State, `Send()` parallel execution |
| V2 — Improved | [Improved Agent](./Blog%20Agent/Improved%20Agent/) | Orchestration, Pydantic models, map-reduce, engineering-grade prompts |
| V3 — Research | [Research Agent](./Blog%20Agent/Research%20Agent/) | **Tavily web search**, smart research router, grounding policy, citation rules |
| V4 — Images & UI | [V4 - Image & Research Agent](./Blog%20Agent/V4%20-%20Image%20&%20Research%20Agent/) | **AI Image Generation (Gemini)**, **Streamlit Web UI**, date-aware research, reducer subgraph |

---

#### 🟢 [V1 — Basic Agent](./Blog%20Agent/Basic%20Agent/)

The entry point. Introduces the core LangGraph primitives with zero prior experience required.

**You will learn:**
- What a `StateGraph` is and how nodes communicate through shared State
- How to run sections in parallel with `Send()` and collect results with `operator.add`
- How to save a blog post to disk as a `.md` file

**Stack:** `langgraph` · `langchain-openai` · `pydantic` · `python-dotenv`

---

#### 🟡 [V2 — Improved Agent](./Blog%20Agent/Improved%20Agent/)

Upgrades the V1 agent with production-level techniques.

**You will learn:**
- Orchestrator → parallel workers → compiler pattern (map-reduce)
- Engineering-grade prompt design: tone, audience, constraint injection
- Structured output with Pydantic (`Plan`, `Task`) — no manual JSON parsing
- How to prevent LLM verbosity with word-count budgets per section

**Stack:** V1 stack + advanced Pydantic models + structured LLM output

---

#### 🔴 [V3 — Research Agent](./Blog%20Agent/Research%20Agent/) ← **NEW**

The most advanced version. The agent can now **read the live web** before writing.

**You will learn:**
- How to build a smart **research router** that decides *when* web search is needed
- Three research modes: `closed_book` · `hybrid` · `open_book`
- How **Tavily** differs from standard search APIs — and why it's built for LLMs
- Dual deduplication: LLM semantic layer + Python URL-key layer
- **Grounding policy**: how to prevent LLMs from hallucinating specific facts in research-backed content
- Citation rules: forcing `([Source](URL))` inline links for every verifiable claim
- Mode-aware planning: the orchestrator shapes the *entire blog structure* differently per mode

**New nodes:** `router_node` → `research_node` → `orchestrator_node` → `worker_node×N` → `reducer_node`

**New tools:** `TavilySearchResults` — LLM-optimised web search API

**Stack:** V2 stack + `tavily-python` + `langchain-community`

---

#### 🔵 [V4 — Image & Research Agent](./Blog%20Agent/V4%20-%20Image%20&%20Research%20Agent/) ← **LATEST**

The most powerful version. The agent now **designs technical diagrams** and packages everything in a **Streamlit Web UI**.

**You will learn:**
- **AI Image Planning**: How an LLM acts as a technical editor to decide *where* visuals improve the blog
- **Gemini 2.5 Flash Integration**: Generating high-fidelity PNG bytes directly from prompts
- **3-Node Reducer Subgraph**: `merge_content` → `decide_images` → `generate_and_place_images`
- **Streamlit Frontend**: Building a 5-tab dashboard for non-technical users to generate and preview blogs
- **Date-Aware Research**: Filtering EvidenceItems by recency to prevent citing stale news in roundups
- **Graceful Fallbacks**: Ensuring the blog remains readable even if an image generation fails
- **Bundle Packaging**: Creating `.zip` downloads containing both the `.md` and the `images/` assets

**New nodes:** `reducer_subgraph` (subgraph-as-a-node)

**New tools:** `google-genai` · `streamlit` · `pandas`

**Stack:** V3 stack + Google Gemini AI SDK + Streamlit framework

---

### 2. [Theory](./Theory/)

Deep dives into the underlying mechanics that power every agent in this repo.

- **LangGraph Architecture** — cycles, persistence, checkpointing, and state management
- **Agentic Workflows** — moving beyond simple chains to autonomous multi-step agents
- **LangSmith Tracing** — debugging and evaluating agentic logic in production

📄 [LangGraph Teaching Guide (PDF)](./Theory/LangGraph_Teaching_Guide_new.pdf)

---

## 🛠️ Full Tech Stack

| Category | Tools |
|---|---|
| **Agent Framework** | LangGraph, LangChain |
| **LLM** | OpenAI GPT-4o-mini |
| **Image Model** | Google Gemini 2.5 Flash (V4 only) |
| **Web Search** | Tavily (V3+ only) |
| **UI Framework** | Streamlit (V4 only) |
| **Data Validation** | Pydantic v2 |
| **Environment** | Python 3.10+, python-dotenv |
| **Output** | Markdown (`.md`) files saved to disk |

---

## 🚦 Getting Started

Each folder has its own `README.md` and `requirements.txt`.

### Clone the repo

```bash
git clone https://github.com/ashumishra2104/Langchain_langraph_langsmith-Theory-Tutorials-.git
cd Langchain_langraph_langsmith-Theory-Tutorials-
```

### Install dependencies (for your chosen version)

```bash
# e.g. for V3
cd "Blog Agent/Research Agent"
pip install -r requirements.txt
```

### Add API keys

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and (for V3) TAVILY_API_KEY
```

Get your keys:
- **OpenAI** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Tavily** *(V3 only)* → [app.tavily.com](https://app.tavily.com) — free tier: 1,000 searches/month

### Run

```bash
python blog_agent_v3.py   # or blog_agent.py for V1/V2
```

---

## 📈 Learning Path

Follow this order to build up skills progressively:

```
1. Theory/          → Understand WHY graphs — read before coding
        ↓
2. Basic Agent      → Learn HOW: nodes, state, Send(), parallel execution
        ↓
3. Improved Agent   → Master prompts, Pydantic, orchestrator pattern
        ↓
4. Research Agent   → Go beyond training data: router, Tavily, grounding policy
        ↓
5. Image & UI Agent → Final form: AI diagrams, Streamlit UI, date-aware research
```

Each version is designed so you can read the Python file top-to-bottom and understand every decision. Comments explain **why**, not just what.

---

## 🔑 Key Concepts Covered

| Concept | Where you learn it |
|---|---|
| `StateGraph`, nodes, edges | V1 |
| `Send()` for parallel fan-out | V1 |
| `operator.add` reducer | V1 → V3 |
| `with_structured_output(Pydantic)` | V2 → V3 |
| Orchestrator → workers → compiler | V2 |
| Conditional edges | V3 (`route_next`, `fanout`) |
| Research routing (3 modes) | V3 |
| Tavily web search integration | V3 |
| Grounding policy & citation rules | V3 |
| Dual deduplication (LLM + Python) | V3 |
| Mode-aware planning | V3 |
| Reducer Subgraph (3-node) | V4 |
| AI Image Generation (Gemini) | V4 |
| Technical Image Planning | V4 |
| Streamlit Web UI | V4 |
| Date-aware Research Filtering | V4 |

---

*Created and maintained by [ashumishra2104](https://github.com/ashumishra2104) · AI PM Cohort*
