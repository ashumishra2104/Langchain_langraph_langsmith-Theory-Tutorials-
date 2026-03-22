from __future__ import annotations

import operator
import os
import re
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------
# 1) Schemas
# -----------------------------

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" only if explicitly known
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str = Field(..., description="Why this mode was chosen.")
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/ folder, e.g. qkv_flow.png")
    alt: str = Field(..., description="Accessibility text.")
    caption: str = Field(..., description="Shown under the image in the blog.")
    prompt: str = Field(..., description="The exact prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str = Field(..., description="The full blog markdown with [[IMAGE_N]] tags inserted.")
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str  # today's date
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

# -----------------------------
# 2) Router
# -----------------------------

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts, fundamental principles, no research needed.
- hybrid (needs_research=true): mostly evergreen but needs up-to-date examples, current tools, or specific version details.
- open_book (needs_research=true): volatile topics: news events, weekly roundups, latest rankings, or pricing.

If needs_research=true:
- Output 3–10 specific, high-signal, scoped queries.
- For open_book weekly topics, include queries reflecting the last 7 days.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 3) Research
# -----------------------------

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        raw = tool.invoke(query)
        
        normalized = []
        for r in raw:
            snippet = r.get("content") or r.get("snippet") or ""
            pub = r.get("published_date") or r.get("published_at") or None
            normalized.append({
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": str(snippet).strip(),
                "published_at": pub,
                "source": r.get("source", "")
            })
        return normalized
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        # Take first 10 chars (YYYY-MM-DD)
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer for a technical blog agent.

Goal: From raw search results, select the highest signal evidence for the blog.

Constraints:
1. ONLY include items with a non-empty URL.
2. Prefer authoritative, technical sources (docs, official blogs, arXiv).
3. Normalise published_at to YYYY-MM-DD if you can reliably infer it (e.g. from snippet/URL); else NULL. NEVER guess.
4. Keep snippets concise (150-300 chars).
5. Ensure evidence is relevant to the topic and the 'as_of' date.
"""

def research_node(state: State) -> dict:
    queries = state.get("queries", [])[:10]
    raw_results = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=6))
    
    if not raw_results:
        return {"evidence": []}
    
    synthesizer = llm.with_structured_output(EvidencePack)
    human_msg = f"As-of: {state['as_of']}\nRecency Days: {state['recency_days']}\nRaw Results:\n{raw_results}"
    
    pack = synthesizer.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=human_msg)
    ])
    
    # Python deduplication by URL
    by_url = {}
    for item in pack.evidence:
        if item.url not in by_url:
            by_url[item.url] = item
    
    final_evidence = list(by_url.values())
    
    # Mode-based date filtering for open_book
    if state["mode"] == "open_book":
        as_of_date = _iso_to_date(state["as_of"])
        if as_of_date:
            filtered = []
            for e in final_evidence:
                pub_date = _iso_to_date(e.published_at)
                if pub_date:
                    delta = (as_of_date - pub_date).days
                    if 0 <= delta <= state["recency_days"]:
                        filtered.append(e)
            final_evidence = filtered
            
    return {"evidence": final_evidence}

# -----------------------------
# 4) Orchestrator
# -----------------------------

ORCHESTRATOR_SYSTEM = """You are a master blog planner.

Plan the structure of a blog based on the topic and the mode.

Grounding Rules:
- closed_book: Keep content evergreen, focus on fundamentals, no dependence on external evidence.
- hybrid: Integrate specific evidence for examples/tools; tasks using evidence must have requires_research=true and requires_citations=true.
- open_book: This is a news roundup. Focus on the most recent facts; avoid tutorials or deep explains unless explicitly requested. If evidence is weak, be transparent about it in the plan.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "hybrid")
    evidence = state.get("evidence", []) or []
    
    # model_dump up to 16 evidence items
    evidence_dump = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence[:16]]

    human_msg = (
        f"Topic: {state['topic']}\n"
        f"Mode: {mode}\n"
        f"As-Of: {state['as_of']}\n"
        f"Recency Days: {state['recency_days']}\n"
        f"Evidence: {evidence_dump}"
    )

    plan = planner.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=human_msg)
    ])

    # Explicit override for open_book
    if mode == "open_book":
        plan.blog_kind = "news_roundup"

    return {"plan": plan}

def fanout_to_workers(state: State):
    plan = state.get("plan")
    assert plan is not None, "Plan must be created before fanout."
    
    topic = state["topic"]
    mode = state["mode"]
    as_of = state["as_of"]
    recency_days = state["recency_days"]
    evidence = [e.model_dump() for e in state["evidence"]] if state["evidence"] else []

    return [
        Send(
            "worker",
            {
                "task": t.model_dump(),
                "topic": topic,
                "mode": mode,
                "as_of": as_of,
                "recency_days": recency_days,
                "plan": plan.model_dump(),
                "evidence": evidence
            }
        )
        for t in plan.tasks
    ]

# -----------------------------
# 5) Worker
# -----------------------------

WORKER_SYSTEM = """You are a senior technical writer. Your task is to write ONE section of a blog post in Markdown.

Instructions:
1. Cover ALL bullets in the provided order.
2. Stay within ±15% of the target word count.
3. Output ONLY the section content, starting with a '##' heading. No preamble or post-amble.
4. If the blog_kind is 'news_roundup', focus on reporting events and their implications—do NOT write tutorials or 'how-to' guides.
5. If the mode is 'open_book', every factual claim (company, event, model detail) MUST have a citation to a provided source URL using the format ([Source](URL)). If a claim is not in the evidence, write 'Not found in provided sources'.
6. If the section requires citations (requires_citations=true), cite URLs from the provided evidence.
7. If the section requires code (requires_code=true), include at least one minimal, valid code snippet.
8. Style: Short paragraphs, use bullets where helpful for readability, use triple-backtick code fences. No fluff or flowery language.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload["evidence"]]
    
    topic = payload["topic"]
    mode = payload["mode"]
    as_of = payload["as_of"]
    recency_days = payload["recency_days"]

    bullets_str = "\n".join([f"- {b}" for b in task.bullets])
    
    evidence_lines = []
    for e in evidence:
        date_str = e.published_at or "date:unknown"
        evidence_lines.append(f"{e.title} | {e.url} | {date_str} | {e.snippet}")
    evidence_text = "\n".join(evidence_lines)

    human_msg = (
        f"Blog Title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Blog Kind: {plan.blog_kind}\n"
        f"Topic: {topic}\n"
        f"Mode: {mode}\n"
        f"As-Of Date: {as_of}\n"
        f"Recency Days: {recency_days}\n\n"
        f"SECTION DETAILS:\n"
        f"Title: {task.title}\n"
        f"Goal: {task.goal}\n"
        f"Target Words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"Flags: research={task.requires_research}, citations={task.requires_citations}, code={task.requires_code}\n"
        f"Bullets to Cover:\n{bullets_str}\n\n"
        f"PROVIDED EVIDENCE:\n{evidence_text}"
    )

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=human_msg)
    ]).content

    return {"sections": [(task.id, section_md)]}

# -----------------------------
# 6) Reducer (Subgraph)
# -----------------------------

def merge_content(state: State) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Plan is missing! Cannot merge content.")
    
    # Get sections and sort by task_id
    sections = state.get("sections", [])
    sections.sort(key=lambda x: x[0])
    
    # Extract markdown body
    body_parts = [s[1] for s in sections]
    joined_body = "\n\n".join(body_parts)
    
    merged_md = f"# {plan.blog_title}\n\n{joined_body}"
    return {"merged_md": merged_md}

# Initialize Subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)

# -----------------------------
# 7) Image Decision (Reducer Node)
# -----------------------------

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor deciding if images are needed for a blog post.

Constraints:
1. MAX 3 images total.
2. Each image must materially improve understanding — prefer diagrams, flowcharts, or table-like visuals.
3. NO decorative images (landscapes, stock photos, abstract art).
4. Insert placeholders in the blog text in this exact format: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
5. If no images are needed, the md_with_placeholders must equal the input text unchanged and the images list must be empty.

Output must strictly match the GlobalImagePlan schema.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    plan = state.get("plan")
    blog_kind = plan.blog_kind if plan else "explainer"
    
    human_msg = (
        f"Blog Kind: {blog_kind}\n"
        f"Topic: {state['topic']}\n"
        f"Instruction: Decide if any of the sections need a clarifying technical image. "
        f"If so, insert a placeholder in the markdown text and provide the prompt/specs for that image.\n\n"
        f"Merged Markdown Content:\n{state['merged_md']}"
    )

    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=human_msg)
    ])

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [i.model_dump() for i in image_plan.images]
    }

# Update Subgraph
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_edge("merge_content", "decide_images")

# -----------------------------
# 8) Image Generation & Placement (Reducer Node)
# -----------------------------

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH"
            )
        ]
    )
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image", # Corrected model name
            contents=prompt,
            config=config
        )
        
        parts = None
        if hasattr(resp, "parts") and resp.parts:
            parts = resp.parts
        elif hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            
        if not parts:
            raise RuntimeError("No parts found in Gemini response.")
            
        for p in parts:
            if hasattr(p, "inline_data") and p.inline_data and p.inline_data.data:
                return p.inline_data.data
                
        raise RuntimeError("No inline image bytes found in response parts.")
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {str(e)}")

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")
    md = state.get("md_with_placeholders") or state.get("merged_md") or ""
    image_specs = state.get("image_specs", []) or []
    blog_title = plan.blog_title if plan else "blog"

    if not image_specs:
        out_name = f"{_safe_slug(blog_title)}.md"
        Path(out_name).write_text(md, encoding="utf-8")
        return {"final": md}

    Path("images").mkdir(exist_ok=True)
    
    for spec in image_specs:
        placeholder = spec.get("placeholder", "")
        filename = spec.get("filename", "image.png")
        alt = spec.get("alt", "image")
        caption = spec.get("caption", "")
        prompt = spec.get("prompt", "")
        
        out_path = Path("images") / filename
        
        success = False
        error_msg = ""
        
        if out_path.exists():
            success = True
        else:
            try:
                img_bytes = _gemini_generate_image_bytes(prompt)
                out_path.write_bytes(img_bytes)
                success = True
            except Exception as e:
                error_msg = str(e)
                
        if success:
            replacement = f"![{alt}](images/{filename})\n*{caption}*"
        else:
            replacement = (
                f"\n> **Image Generation Failed**\n"
                f"> **Caption:** {caption}\n"
                f"> **Alt:** {alt}\n"
                f"> **Prompt:** {prompt}\n"
                f"> **Error:** {error_msg}\n"
            )
        
        md = md.replace(placeholder, replacement)

    out_name = f"{_safe_slug(blog_title)}.md"
    Path(out_name).write_text(md, encoding="utf-8")
    
    return {"final": md}

# Complete Subgraph
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.set_entry_point("merge_content")
reducer_graph.set_finish_point("generate_and_place_images")

reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Main Graph
# -----------------------------

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator"
    }
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout_to_workers, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

def run(topic: str, as_of: Optional[str] = None) -> dict:
    if as_of is None:
        as_of = date.today().isoformat()
        
    initial_state = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of,
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }
    
    return app.invoke(initial_state)

if __name__ == "__main__":
    import json
    print("🚀 Starting test run...")
    topic = "Image Generation Models - A brief history and the way ahead"
    out = run(topic)
    
    print("-" * 50)
    print(f"1. Mode chosen by router: {out.get('mode')}")
    
    images_dir = Path("images")
    if images_dir.exists():
        pngs = list(images_dir.glob("*.png"))
        print(f"2. Images generated: {len(pngs)} PNG files found in images/")
    else:
        print("2. No images generated (images/ folder not found).")
        
    final_md = out.get("final", "")
    has_img_syntax = "![" in final_md and "](images/" in final_md
    print(f"3. Contains Markdown image syntax: {has_img_syntax}")
    
    print("-" * 50)
    print("4. Final Blog Preview:")
    print(final_md[:1000] + "...") # Print first 1000 chars
