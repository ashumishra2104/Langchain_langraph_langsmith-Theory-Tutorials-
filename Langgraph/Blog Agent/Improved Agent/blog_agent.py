from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import TypedDict, List, Annotated, Literal, Dict
from pathlib import Path
import operator
from dotenv import load_dotenv

load_dotenv()

class Task(BaseModel):
    id: int = Field(description="Unique identifier for the section (used for ordering)")
    title: str = Field(description="The heading for this blog section")
    goal: str = Field(description="A one-sentence description of the intended reader takeaway")
    bullets: List[str] = Field(description="3 to 5 concrete, specific subpoints to cover in this section", min_length=3, max_length=5)
    target_words: int = Field(description="Target word count for this section (120-450)", ge=120, le=450)
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(description="Technically enforced type of content for this section")

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[Dict[int, str], operator.ior]
    final: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def orchestrator(state: State):
    print(f"--- ORCHESTRATOR: Planning for topic '{state['topic']}' ---")
    system_prompt = "You are a Senior Technical Writer. Plan a high-impact technical blog post. Produce 5-7 sections with unique IDs, goals, 3-5 bullets, and word counts."
    structured_llm = llm.with_structured_output(Plan)
    plan = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=f"Generate a detailed technical blog plan for: {state['topic']}")
    ])
    return {"plan": plan}

def worker(payload: dict):
    task: Task = payload["task"]
    topic: str = payload["topic"]
    print(f"--- WORKER: Writing section {task.id}: '{task.title}' ---")
    bullets_str = "\n".join([f"- {b}" for b in task.bullets])
    system_prompt = f"You are a Senior Technical Writer. Write ONE section in Markdown. Goal: {task.goal}. Word count: {task.target_words} ±15%. Start with '## [Title]'."
    human_message = f"Main Topic: {topic}\nSection Title: {task.title}\nGoal: {task.goal}\nRequired Bullets:\n{bullets_str}"
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_message)])
    return {"sections": {task.id: response.content}}

def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]}) for task in state["plan"].tasks]

def reducer(state: State):
    print("--- REDUCER: Compiling final blog post ---")
    title = state["plan"].blog_title
    sorted_ids = sorted(state.get("sections", {}).keys())
    ordered_content = [state["sections"][i] for i in sorted_ids]
    final_markdown = f"# {title}\n\n" + "\n\n".join(ordered_content)
    safe_name = "".join([c if c.isalnum() else "_" for c in title.lower()])
    Path(f"{safe_name}.md").write_text(final_markdown)
    print(f"--- DONE: Blog saved to '{safe_name}.md' ---")
    return {"final": final_markdown}

workflow = StateGraph(State)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("worker", worker)
workflow.add_node("reducer", reducer)
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", fanout, ["worker"])
workflow.add_edge("worker", "reducer")
workflow.add_edge("reducer", END)
app = workflow.compile()

if __name__ == "__main__":
    import os
    topic = "History of Sound and Music generation using AI"
    initial_state = {"topic": topic, "sections": {}}
    result = app.invoke(initial_state)
