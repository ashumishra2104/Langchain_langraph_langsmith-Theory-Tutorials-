# Blog Agent V2: Improved Prompting & Parallel Writing

This project implements a production-grade **Blog Writer AI Agent** using **LangGraph** and **OpenAI's GPT-4o-mini**. It uses a sophisticated **Orchestrator-Worker (Map-Reduce)** pattern to generate high-quality, technically rigorous blog posts in parallel.

## 🚀 Key Improvements in V2
- **Rich Data Modeling**: Uses Pydantic `Task` and `Plan` objects instead of simple strings for high-precision content control.
- **Engineering-Grade Prompting**: Implementing detailed system prompts (50+ lines) to enforce technical quality, specific structures (Problem → Intuition → Approach), and avoid "AI fluff."
- **Parallel Execution**: Uses LangGraph's `Send()` command to trigger multiple independent writers simultaneously, significantly reducing generation time.
- **Context-Aware Workers**: Each worker receives a full "editorial brief" (Goal, Bullets, Tone, Audience, Word Count) rather than just a title.
- **File Output**: Automatically compiles and saves the final post as a professionally formatted `.md` file to disk.

## 🛠️ Tech Stack
- **LangGraph**: Orchestration of the agent's state and workflow.
- **LangChain**: Interface with OpenAI and structured data parsing.
- **OpenAI (GPT-4o-mini)**: High-quality, cost-effective LLM for planning and writing.
- **Pydantic**: Data validation and schema enforcement.
- **Python-dotenv**: Secure environment variable management.

## 📂 Project Structure
- `blog_agent.py`: The core agent logic (Graph definition, Nodes, Logic).
- `.env`: API Key storage (see `.env.example`).
- `requirements.txt`: Project dependencies.
- `*.md`: Generated blog posts.

## 🚦 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

3. **Run the Agent**:
   ```bash
   python blog_agent.py
   ```

---
*Created as part of the LangGraph Advanced Agentic Coding series.*
