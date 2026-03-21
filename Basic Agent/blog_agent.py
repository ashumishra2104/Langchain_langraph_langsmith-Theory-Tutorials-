import os
import operator
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

# 1. Load Environment Variables
load_dotenv()

# ==========================================
# 2. DEFINING OUR DATA MODELS (Pydantic)
# ==========================================

class SectionList(BaseModel):
    sections: List[str] = Field(description="A list of catchy titles for the blog sections.")

class SectionContent(BaseModel):
    title: str = Field(description="The title of the section.")
    content: str = Field(description="The written content for this specific section, formatted with a heading.")

class WriterInput(TypedDict):
    title: str
    topic: str

# ==========================================
# 3. DEFINING THE AGENT'S MEMORY (State)
# ==========================================

class AgentState(TypedDict):
    topic: str
    sections: List[str]
    completed_sections: Annotated[List[SectionContent], operator.add]
    final_blog: str

# 4. Initialize the LLM (OpenAI)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ==========================================
# 5. NODE FUNCTIONS (The Agent's Thinking Steps)
# ==========================================

def planner_node(state: AgentState):
    """
    Step 1: The Planner.
    Generates exactly 3 section titles.
    """
    print(f"--- PLANNING SECTIONS FOR: {state['topic']} ---")
    system_prompt = SystemMessage(content="You are a blog planning expert. Your job is to create exactly 3 clear, engaging section titles.")
    human_prompt = HumanMessage(content=f"Create 3 section titles for a blog post about: {state['topic']}")
    
    structured_llm = llm.with_structured_output(SectionList)
    response = structured_llm.invoke([system_prompt, human_prompt])
    
    return {"sections": response.sections}

def writer_node(state: WriterInput):
    """
    Node 1: Section Writer.
    Writes 2-3 paragraphs for one section.
    """
    print(f"--- WRITING SECTION: {state['title']} ---")
    
    system_prompt = SystemMessage(content="You are a professional blog writer. Write 2 to 3 engaging paragraphs for a specific section. Start with the section title as a Markdown heading (e.g., ## Title).")
    human_prompt = HumanMessage(content=f"Topic: {state['topic']}\nSection Title: {state['title']}")
    
    structured_llm = llm.with_structured_output(SectionContent)
    response = structured_llm.invoke([system_prompt, human_prompt])
    
    return {"completed_sections": [response]}

def compiler_node(state: AgentState):
    """
    Node 2: Blog Compiler.
    Combines all sections into the final blog post.
    """
    print("--- COMPILING FINAL BLOG ---")
    
    final_post = f"# {state['topic']}\n\n"
    # Note: Sections might be out of order due to parallel execution, 
    # but we'll focus on the fan-out logic for now.
    for section in state["completed_sections"]:
        final_post += f"{section.content}\n\n"
        
    return {"final_blog": final_post}

# ==========================================
# 6. FAN-OUT LOGIC (The Router)
# ==========================================

def fan_out_sections(state: AgentState):
    """
    Uses Send() to launch writers in parallel for each planned section.
    """
    print(f"--- FANNING OUT: Launching {len(state['sections'])} parallel writers ---")
    
    return [
        Send("writer", {"title": s, "topic": state["topic"]}) 
        for s in state["sections"]
    ]

# ==========================================
# 7. BUILDING THE GRAPH
# ==========================================

workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("planner", planner_node)
workflow.add_node("writer", writer_node)
workflow.add_node("compiler", compiler_node)

# Connect the nodes
workflow.set_entry_point("planner")

# Conditional Edge for Fan-Out
workflow.add_conditional_edges("planner", fan_out_sections, ["writer"])

# Return to compiler
workflow.add_edge("writer", "compiler")
workflow.add_edge("compiler", END)

# Compile the agent
app = workflow.compile()

# ==========================================
# 8. RUNNING THE AGENT
# ==========================================

if __name__ == "__main__":
    # Input topic as requested
    inputs = {"topic": "The Future of AI in Healthcare"}
    
    print("Starting the Blog Agent...")
    print("-" * 30)
    
    result = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("FINAL GENERATED BLOG POST")
    print("="*50 + "\n")
    print(result["final_blog"])
    print("\n" + "="*50)
