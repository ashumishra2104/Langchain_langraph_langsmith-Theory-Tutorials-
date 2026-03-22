# LangChain, LangGraph, & LangSmith: Theory and Tutorials

Welcome to the **LangChain, LangGraph, and LangSmith Theory-Tutorials** repository. This project is a comprehensive guide to building advanced, production-grade AI agents, starting from basic "Vibe Coding" to complex Graph-based architectures.

## 📁 Repository Structure

### 1. [Blog Agent](./Blog%20Agent/)
A multi-stage evolution of an AI blog writing agent:
- **[Basic Agent (V1)](./Blog%20Agent/Basic%20Agent/)**: Introduction to LangGraph nodes, shared state, and parallel execution using `Send()`.
- **[Improved Agent (V2)](./Blog%20Agent/Improved%20Agent/)**: Advanced implementation featuring orchestration, engineering-grade prompting, map-reduce patterns, and structured data validation with Pydantic.

### 2. [Theory](./Theory/)
Deep dives into the underlying mechanics:
- **LangGraph Architecture**: Understanding cycles, persistence, and state management.
- **Agentic Workflows**: Moving beyond simple chains to autonomous agents.
- **LangSmith Tracing**: Best practices for debugging and evaluating your agentic logic.

## 🛠️ Tech Stack
- **Frameworks**: LangGraph, LangChain
- **Models**: OpenAI (GPT-4o, GPT-4o-mini)
- **Data Modeling**: Pydantic
- **Environment**: Python 3.10+, Dotenv

## 🚦 Getting Started
Each folder contains its own `README.md` and `requirements.txt`. Generally, you can get started by:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashumishra2104/Langchain_langraph_langsmith-Theory-Tutorials-.git
   ```
2. **Setup your API Keys**:
   Create a `.env` file in the relevant sub-folder with your `OPENAI_API_KEY`.

## 📈 Learning Path
1. Start with the **[Theory](./Theory/)** folder to understand the "Why" behind graphs.
2. Build the **[Basic Agent](./Blog%20Agent/Basic%20Agent/)** to learn the "How."
3. Graduate to the **[Improved Agent](./Blog%20Agent/Improved%20Agent/)** to master "Production Optimization."

---
*Created and maintained by [ashumishra2104](https://github.com/ashumishra2104).*
