# ============================================================
# Blog Agent V3 — Web-Augmented Blog Writer with Tavily Research
# ============================================================
# Architecture:
#   plan → [parallel research + write per section] → compile
#
# New in V3: Each section is researched via Tavily before writing,
# grounding the content in real, up-to-date web sources.
# ============================================================

# ── LangGraph ──────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ── LangChain + OpenAI ─────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ── Pydantic for structured data models ────────────────────
from pydantic import BaseModel, Field

# ── Typing utilities ───────────────────────────────────────
from typing import TypedDict, List, Optional, Literal, Annotated

# ── Standard library ───────────────────────────────────────
from pathlib import Path
import operator

# ── Tavily — LLM-optimised web search tool ─────────────────
from langchain_community.tools.tavily_search import TavilySearchResults

# ── Environment / secrets ──────────────────────────────────
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()

# ============================================================
# Quick sanity-check: warn early if keys are missing
# ============================================================
_OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
_TAVILY_KEY  = os.getenv("TAVILY_API_KEY")

if not _OPENAI_KEY:
    raise EnvironmentError("OPENAI_API_KEY is missing — please set it in your .env file.")
if not _TAVILY_KEY:
    raise EnvironmentError("TAVILY_API_KEY is missing — please set it in your .env file.")

print("✅ API keys loaded successfully.")

# ============================================================
# ── MODEL 1: EvidenceItem ───────────────────────────────────
# Represents a single web search result returned by Tavily.
# published_at and snippet are Optional because Tavily does not
# always provide them; we never guess or fabricate these values.
# ============================================================
class EvidenceItem(BaseModel):
    title: str = Field(..., description="Title of the web page or article")
    url: str = Field(..., description="Full URL of the source")
    published_at: Optional[str] = Field(
        default=None,
        description="ISO date string if explicitly in the result — never inferred"
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Short extract or summary of the page content"
    )
    source: Optional[str] = Field(
        default=None,
        description="Domain or publisher name (e.g. 'arxiv.org')"
    )


# ============================================================
# ── MODEL 2: RouterDecision ─────────────────────────────────
# The LLM examines the topic and decides how much research is
# needed before writing. Three modes:
#   "closed_book"  — write from training knowledge only
#   "hybrid"       — a few targeted lookups to verify facts
#   "open_book"    — deep research on every section
# ============================================================
class RouterDecision(BaseModel):
    needs_research: bool = Field(
        ...,
        description="True if any web search is required for this topic"
    )
    mode: Literal["closed_book", "hybrid", "open_book"] = Field(
        ...,
        description="Research intensity level chosen by the router"
    )
    queries: List[str] = Field(
        default_factory=list,
        description="Search queries to execute; empty list when needs_research=False"
    )


# ============================================================
# ── MODEL 3: EvidencePack ───────────────────────────────────
# A cleaned container for all EvidenceItems resulting from the
# Tavily searches. Stored on State and passed into every writer
# node so each section can cite from the same shared pool.
# ============================================================
class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Deduplicated list of search results gathered by the researcher"
    )


# ============================================================
# ── MODEL 4: Task ───────────────────────────────────────────
# A single section brief produced by the planner. Each Task
# maps 1-to-1 to one parallel worker node in the graph.
# target_words is a soft budget kept intentionally tight (<=550)
# to prevent runaway LLM verbosity in any single section.
# ============================================================
class Task(BaseModel):
    id: int = Field(..., description="Unique sequential section index, used for ordering")
    title: str = Field(..., description="Section heading that appears in the final blog")
    goal: str = Field(..., description="One-sentence description of what this section achieves")
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="Key points the section must cover (3-6 bullets)"
    )
    target_words: int = Field(
        ...,
        ge=120,
        le=550,
        description="Soft word-count target for this section (120-550 words)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Topical tags for retrieval or filtering (e.g. ['python', 'async'])"
    )
    requires_research: bool = Field(
        default=False,
        description="If True, the worker will run a Tavily search before writing"
    )
    requires_citations: bool = Field(
        default=False,
        description="If True, the writer must inline at least one citation"
    )
    requires_code: bool = Field(
        default=False,
        description="If True, the writer should include a code snippet"
    )


# ============================================================
# ── MODEL 5: Plan ───────────────────────────────────────────
# The full editorial plan output by the planner node. Contains
# high-level blog metadata plus the ordered list of Tasks that
# the parallel writer nodes will consume via Send().
# ============================================================
class Plan(BaseModel):
    blog_title: str = Field(..., description="Final headline for the blog post")
    audience: str = Field(..., description="Target reader (e.g. 'senior Python engineers')")
    tone: str = Field(..., description="Writing style (e.g. 'technical but approachable')")
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = Field(
        default="explainer",
        description="Genre of blog post — controls structural conventions"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Global rules every section must follow (e.g. 'no marketing fluff')"
    )
    tasks: List[Task] = Field(
        ...,
        description="Ordered section briefs — each becomes one parallel worker node"
    )


# ============================================================
# ── MODEL 6: State (TypedDict) ──────────────────────────────
# The shared memory object that flows through the entire graph.
#
# KEY DESIGN — why sections uses Annotated[List[tuple], operator.add]:
#
#   LangGraph runs writer nodes in PARALLEL via Send(). Each worker
#   finishes at its own speed and appends ONE tuple to `sections`.
#   A plain List would cause a race condition — the second write
#   would overwrite the first.  operator.add is the "reducer":
#   instead of replacing the list, LangGraph *concatenates* each
#   incoming value onto the accumulated list safely.
#
#   Storing (id, content) tuples (not bare strings) means the
#   compiler can sort by id and rebuild the correct section order
#   regardless of which worker finished first.
# ============================================================
class State(TypedDict):
    topic: str                                      # User's blog topic
    mode: str                                       # Router's chosen mode
    needs_research: bool                            # Whether research phase runs
    queries: List[str]                              # Tavily search queries
    evidence: List[EvidenceItem]                    # Aggregated search results
    plan: Optional[Plan]                            # Planner output (None until set)
    sections: Annotated[                            # <- reducer prevents race conditions
        List[tuple],                                #    each tuple: (task_id, markdown_text)
        operator.add
    ]
    final: str                                      # Compiled, ordered blog post


# ============================================================
# ── LLM Initialization ──────────────────────────────────────
# gpt-4o-mini: fast, cost-efficient, and strong enough for
# structured output (JSON mode) and long-form content writing.
# temperature=0 for deterministic planner/router decisions;
# writer nodes will override this locally if creative variation
# is desired.
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("📦 All imports resolved — Blog Agent V3 is ready to build!")
print("🧠 LLM (gpt-4o-mini) initialised.")
print("📐 All 6 data models defined: EvidenceItem, RouterDecision, EvidencePack, Task, Plan, State")


# ============================================================
# ── NODE 1: router_node ─────────────────────────────────────
#
# The very first node in the graph. It reads the user's topic
# and decides the research strategy BEFORE any writing happens.
#
# Three modes the LLM can choose:
#
#   closed_book  (needs_research=False)
#     │  Topic is evergreen — correctness does not depend on recent
#     │  facts. Classic CS concepts, language fundamentals, design
#     │  patterns. Writing from training data is sufficient.
#     └─ queries: []   (no searches needed)
#
#   hybrid       (needs_research=True)
#     │  Mostly evergreen but enriched by current examples: "best
#     │  Python async libraries", "top LLM frameworks in 2025".
#     │  A handful of targeted queries prevents hallucinated tool
#     │  versions or benchmark numbers.
#     └─ queries: 3–5 specific lookups
#
#   open_book    (needs_research=True)
#     │  Primarily volatile content: weekly roundups, "this week
#     │  in AI", pricing tables, regulatory changes, rankings.
#     │  Writing without live data would produce stale misinformation.
#     └─ queries: 5–10 highly scoped, time-anchored queries
#
# Query quality rules enforced in the system prompt:
#   • Never use generic terms ("AI", "Python") alone
#   • Include exact version numbers, dates, or named entities
#   • If the topic contains "last week / this week / latest",
#     that time constraint MUST appear in every relevant query
# ============================================================

ROUTER_SYSTEM_PROMPT = """You are the Research Router for an AI blog-writing agent.

Your job is to read the user's blog topic and decide the optimal research strategy.

## Modes

**closed_book** (needs_research = false)
- Use when the topic is *evergreen*: correctness does not depend on recent facts.
- Examples: "what is recursion", "explain the CAP theorem", "how does TCP/IP work".
- Output an empty queries list.

**hybrid** (needs_research = true)
- Use when the topic is mostly evergreen but benefits from up-to-date examples,
  tool names, library versions, or benchmark numbers.
- Examples: "best Python async frameworks", "LangGraph vs LangChain in 2025".
- Output 3–5 targeted queries.

**open_book** (needs_research = true)
- Use when the topic is primarily *volatile*: weekly/monthly roundups, "this week",
  "latest", pricing, policy/regulation changes, leaderboard rankings.
- Examples: "AI news this week", "latest GPT-4o updates", "LLM pricing comparison March 2025".
- Output 5–10 highly specific, time-anchored queries.

## Query quality rules (STRICTLY enforce these)
1. Never output a bare generic term ("AI", "Python", "machine learning") as a query.
2. Every query must be scoped — include named entities, version numbers, or dates.
3. If the user's topic contains phrases like "last week", "this week", "latest",
   or "this month", that exact time constraint MUST appear in every relevant query.
4. Prefer queries that would retrieve primary sources: official docs, benchmarks,
   papers, changelogs — not tutorial blog posts.
5. Aim for diversity: each query should target a different facet of the topic.

Return a single RouterDecision JSON object. Nothing else.
"""


def router_node(state: State) -> dict:
    """
    NODE 1 — Research Router

    Reads state["topic"] and produces a RouterDecision using structured
    LLM output. The decision (mode, needs_research, queries) is written
    directly into State so the conditional edge can branch correctly.

    Flow after this node:
        needs_research=True  →  research node (Tavily searches)
        needs_research=False →  orchestrator node (plan immediately)
    """
    topic = state["topic"]

    # Bind the LLM to always return a valid RouterDecision object.
    # with_structured_output uses function-calling / JSON mode under
    # the hood — no manual JSON parsing needed.
    router_llm = llm.with_structured_output(RouterDecision)

    decision: RouterDecision = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Blog topic: {topic}"),
    ])

    print(f"\n🔀 Router decision:")
    print(f"   mode           : {decision.mode}")
    print(f"   needs_research : {decision.needs_research}")
    if decision.queries:
        print(f"   queries ({len(decision.queries)})    :")
        for i, q in enumerate(decision.queries, 1):
            print(f"     {i}. {q}")
    else:
        print("   queries        : [] (closed_book — no searches needed)")

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


# ============================================================
# ── CONDITIONAL EDGE: route_next ────────────────────────────
#
# LangGraph conditional edges call a function that returns a
# STRING — that string is the name of the next node to visit.
#
# This function is registered on the graph like this:
#   graph.add_conditional_edges("router", route_next,
#       {"research": "research", "orchestrator": "orchestrator"})
#
# The mapping dict tells LangGraph which node name corresponds
# to each return value of this function.
# ============================================================

def route_next(state: State) -> str:
    """
    CONDITIONAL EDGE — called immediately after router_node.

    Returns:
        "research"     -> route to the Tavily research node
        "orchestrator" -> skip research, go straight to planning
    """
    if state["needs_research"]:
        print("↪  Routing to: RESEARCH (Tavily searches queued)")
        return "research"
    else:
        print("↪  Routing to: ORCHESTRATOR (closed_book — no searches)")
        return "orchestrator"


# ============================================================
# ── HELPER: _tavily_search ──────────────────────────────────
#
# A thin, private wrapper around TavilySearchResults that:
#   1. Calls Tavily with a single query string
#   2. Normalises the raw result dicts into a consistent schema
#      (Tavily's field names vary across result types — e.g.
#      "content" vs "snippet", "published_date" vs "published_at")
#   3. Returns a plain list of dicts — NOT EvidenceItem objects yet,
#      because we want to pass the full raw batch to the LLM for
#      semantic deduplication before constructing Pydantic models.
#
# max_results=6 per query gives 6 x N results total (where N is
# the number of queries). The LLM deduplication step will trim
# this down to the highest-quality unique items.
# ============================================================

def _tavily_search(query: str, max_results: int = 5) -> list:
    """
    Private helper — executes one Tavily search and normalises results.

    Args:
        query:       The search string to send to Tavily.
        max_results: Maximum number of results to fetch (default 5).

    Returns:
        List of normalised dicts, each with keys:
            title, url, snippet, published_at, source
        An empty list is returned if Tavily returns nothing.
    """
    tool = TavilySearchResults(max_results=max_results)
    raw_results = tool.invoke(query)          # returns list[dict] or []

    if not raw_results:
        return []

    normalised = []
    for item in raw_results:
        # ── URL ────────────────────────────────────────────
        url = item.get("url", "").strip()
        if not url:
            continue                          # skip results with no URL

        # ── Title ──────────────────────────────────────────
        title = item.get("title", "Untitled").strip()

        # ── Snippet ────────────────────────────────────────
        # Tavily may use "content" or "snippet" depending on result type
        snippet = (
            item.get("content")
            or item.get("snippet")
            or ""
        ).strip()

        # ── Published date ─────────────────────────────────
        # Field name also varies; we never fabricate a date if absent
        published_at = (
            item.get("published_date")
            or item.get("published_at")
            or None
        )

        # ── Source / domain ────────────────────────────────
        source = item.get("source", "").strip() or None

        normalised.append({
            "title":        title,
            "url":          url,
            "snippet":      snippet or None,
            "published_at": published_at,
            "source":       source,
        })

    return normalised


# ============================================================
# ── NODE 2: research_node ───────────────────────────────────
#
# Runs AFTER the router when needs_research=True.
# Responsible for:
#   1. Firing off all router-generated queries via _tavily_search
#   2. Collecting the full raw evidence pool (potentially 30-60 items)
#   3. Passing the pool through LLM-based semantic deduplication
#      and quality filtering → EvidencePack
#   4. Applying a final Python-level URL-keyed deduplication as a
#      deterministic safety net
#   5. Writing the clean evidence list back into State
#
# DUAL DEDUPLICATION DESIGN:
#   Layer 1 (LLM)   : semantic / near-duplicate removal, quality gate
#   Layer 2 (Python): exact URL collision guarantee (O(n) dict trick)
# ============================================================

RESEARCH_SYSTEM_PROMPT = """You are a Research Synthesizer for an AI blog-writing agent.

You will receive a list of raw web search results in JSON format.
Your task is to clean, filter, and deduplicate them into a high-quality evidence pack.

## Rules — follow ALL of them strictly

**Inclusion**
- Only include items that have a non-empty URL.
- Prefer authoritative sources: official documentation, company engineering blogs,
  peer-reviewed papers, reputable news outlets (e.g. ArXiv, GitHub, TechCrunch,
  official vendor blogs). Deprioritise low-quality SEO content.

**Snippet quality**
- Keep snippets concise: 1-3 sentences maximum.
- If the raw snippet is very long, summarise it into the key factual claim only.
- Never add information that is not present in the raw input.

**Dates**
- If a published date is explicitly present in the raw data, preserve it as YYYY-MM-DD.
- If no date is present, set published_at to null. NEVER guess or infer a date.

**Deduplication**
- If two or more results point to the same URL (including near-duplicates like
  http vs https, trailing slashes), keep only ONE — the one with the richer snippet.
- If two results cover the exact same factual claim from different URLs, keep the
  one from the more authoritative source.

Return a single EvidencePack JSON object. Nothing else.
"""


def research_node(state: State) -> dict:
    """
    NODE 2 — Tavily Research

    Executes all search queries from the router, collects and cleans
    the results into a deduplicated EvidencePack, and writes the
    final evidence list back to State for downstream nodes to use.

    Returns:
        {"evidence": list[EvidenceItem]}  — empty list if no results found
    """
    queries = state["queries"]
    print(f"\n🔬 Research node — running {len(queries)} Tavily search(es)...")

    # ── Step 1: Fire all queries, collect raw results ──────
    all_raw = []
    for i, query in enumerate(queries, 1):
        print(f"   [{i}/{len(queries)}] Searching: \"{query}\"")
        results = _tavily_search(query, max_results=6)
        print(f"           → {len(results)} results returned")
        all_raw.extend(results)

    print(f"\n   📥 Total raw results collected: {len(all_raw)}")

    # ── Step 2: Guard — nothing came back ─────────────────
    if not all_raw:
        print("   ⚠️  No results from Tavily — proceeding with empty evidence.")
        return {"evidence": []}

    # ── Step 3: LLM semantic deduplication + quality gate ──
    # We pass ALL raw results at once. The LLM:
    #   - drops low-quality / duplicate items  (Layer 1 dedup)
    #   - validates date format (YYYY-MM-DD or null)
    #   - trims bloated snippets
    #   - returns a structured EvidencePack
    research_llm = llm.with_structured_output(EvidencePack)

    pack: EvidencePack = research_llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Here are {len(all_raw)} raw search results to clean and deduplicate:\n\n"
            + str(all_raw)
        )),
    ])

    print(f"   🧠 LLM kept {len(pack.evidence)} items after semantic dedup")

    # ── Step 4: Python-level URL deduplication (Layer 2) ───
    # dict preserves insertion order in Python 3.7+.
    # If two EvidenceItems share an identical URL, only the FIRST
    # (higher-ranked by LLM) is kept. This is O(n) and deterministic —
    # it cannot be fooled by LLM non-compliance.
    seen = {}
    for item in pack.evidence:
        canonical_url = item.url.rstrip("/").lower()   # normalise trailing slash + case
        if canonical_url not in seen:
            seen[canonical_url] = item

    deduped = list(seen.values())

    print(f"   ✅ Final evidence count after Python URL-dedup: {len(deduped)}")
    if deduped:
        print("   📑 Sources retained:")
        for ev in deduped:
            date_str = f" ({ev.published_at})" if ev.published_at else ""
            print(f"      • {ev.title[:60]}{date_str}")
            print(f"        {ev.url}")

    return {"evidence": deduped}


# ============================================================
# ── NODE 3: orchestrator_node ───────────────────────────────
#
# The editorial planner. Runs after research (or directly after
# the router for closed_book topics). Produces a fully structured
# Plan object that every parallel writer node will consume.
#
# Key responsibilities:
#   1. Receive mode + evidence from State
#   2. Apply mode-specific grounding rules (see system prompt)
#   3. Return a Plan with 5-9 Tasks, each containing a goal,
#      3-6 actionable bullets, and a word-count budget
#
# Why mode matters HERE (not just at write time) — see explanation
# at the bottom of this module.
# ============================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Editorial Planner for an AI blog-writing agent.

Your job is to read a blog topic, a research mode, and (optionally) a set of evidence items,
then produce a complete, structured Plan that parallel writer nodes will execute.

====================================================================
HARD REQUIREMENTS — apply to ALL modes
====================================================================

**Sections**
- Create between 5 and 9 sections (Task objects) appropriate for the topic and audience.
- Sections must flow logically: introduction → body sections → conclusion/summary.
- Each Task must have:
    • id         : sequential integer starting at 1
    • title      : clear, specific section heading (not generic like "Introduction")
    • goal        : exactly ONE sentence describing what this section achieves for the reader
    • bullets     : 3-6 ACTIONABLE content points (see bullet rules below)
    • target_words: integer between 120 and 550

**Bullet quality rules (strictly enforced)**
- Bullets must be ACTIONABLE directives, not passive descriptions.
  GOOD: "Build a minimal FastAPI endpoint and show the full request/response cycle"
  BAD:  "Explain what FastAPI is"
- Use verbs like: build, compare, measure, verify, debug, demonstrate, benchmark,
  contrast, walk through, show, outline the tradeoffs of, quantify.
- At least ONE bullet per section must be concrete and specific (named tool, metric,
  code pattern, failure scenario, or real-world example).

**Depth requirements — the overall plan must include at least 2 of:**
  [ ] A code sketch or worked example
  [ ] Edge cases or failure modes
  [ ] Performance or cost considerations
  [ ] Security considerations
  [ ] Debugging, observability, or testing tips

====================================================================
THREE-MODE GROUNDING RULES — apply based on the mode field
====================================================================

**MODE: closed_book**
- The agent has NO live web data. Write the plan as if the evidence list is empty.
- Keep all content evergreen: no version numbers, no "as of 2025", no named models.
- Focus on concepts, principles, and timeless patterns.
- Set requires_research=False and requires_citations=False on every Task.
- Set blog_kind to "explainer" or "tutorial" as appropriate.

**MODE: hybrid**
- The agent has SOME live evidence to supplement its training knowledge.
- Use the evidence items to populate bullets with up-to-date examples, tool names,
  benchmark numbers, or library versions where they strengthen the section.
- For any Task whose bullets rely on fresh evidence, set:
    requires_research=True AND requires_citations=True
- Tasks covering timeless concepts should remain requires_research=False.
- Set blog_kind to "explainer", "tutorial", or "comparison" as appropriate.

**MODE: open_book**
- The content is primarily volatile (news, rankings, pricing, policy, roundups).
- Set blog_kind to "news_roundup".
- Every section should summarise events and their implications — NOT teach concepts.
- Do NOT include tutorial-style sections unless the user explicitly requested them.
- If the evidence provided is empty or clearly insufficient to cover the topic,
  include a constraint in the plan's constraints list noting this transparently,
  e.g. "Evidence was limited — writers should flag uncertainty rather than speculate."
- Set requires_research=True and requires_citations=True on ALL Tasks.

====================================================================
OUTPUT FORMAT
====================================================================
Return a single Plan JSON object. Nothing else.
Ensure tasks is a non-empty list ordered by section appearance in the blog.
"""


def orchestrator_node(state: State) -> dict:
    """
    NODE 3 — Editorial Orchestrator (Planner)

    Reads the topic, mode, and evidence from State. Calls the LLM
    with a mode-aware system prompt to produce a fully structured Plan.
    The Plan is written back to State and consumed by spawn_writers,
    which fans it out into parallel writer nodes via Send().

    Returns:
        {"plan": Plan}
    """
    topic    = state["topic"]
    mode     = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    print(f"\n🗂️  Orchestrator node — planning blog for topic: \"{topic}\"")
    print(f"   mode     : {mode}")
    print(f"   evidence : {len(evidence)} item(s) available")

    # ── Serialise evidence for the LLM ─────────────────────
    # Cap at 16 items to keep the prompt within a safe token budget.
    # The LLM saw ALL results; we trust it already ranked the best ones
    # to the front during research_node deduplication.
    evidence_for_prompt = evidence[:16]
    evidence_dicts = [ev.model_dump() for ev in evidence_for_prompt]

    # ── Build the human message ─────────────────────────────
    human_content = (
        f"Topic  : {topic}\n"
        f"Mode   : {mode}\n\n"
        f"Evidence ({len(evidence_for_prompt)} items):\n"
        + (str(evidence_dicts) if evidence_dicts else "[]  (no evidence — closed_book run)")
    )

    # ── Invoke the planner LLM ─────────────────────────────
    planner_llm = llm.with_structured_output(Plan)

    plan: Plan = planner_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ])

    # ── Log the resulting plan ─────────────────────────────
    print(f"\n   ✅ Plan produced: \"{plan.blog_title}\"")
    print(f"   kind     : {plan.blog_kind}")
    print(f"   audience : {plan.audience}")
    print(f"   tone     : {plan.tone}")
    print(f"   sections : {len(plan.tasks)}")
    if plan.constraints:
        print(f"   constraints ({len(plan.constraints)}):")
        for c in plan.constraints:
            print(f"      • {c}")
    print()
    for task in plan.tasks:
        research_flags = []
        if task.requires_research:   research_flags.append("research")
        if task.requires_citations:  research_flags.append("citations")
        if task.requires_code:       research_flags.append("code")
        flags_str = f"  [{', '.join(research_flags)}]" if research_flags else ""
        print(f"   [{task.id:02d}] {task.title} (~{task.target_words}w){flags_str}")
        for bullet in task.bullets:
            print(f"         - {bullet}")

    return {"plan": plan}


# ============================================================
# ── NODE 4: worker_node ─────────────────────────────────────
#
# One instance of this node runs PER TASK, all in PARALLEL.
# Each instance receives a self-contained payload dict (not State)
# because Send() passes a custom dict directly to the node —
# it does NOT receive the full graph State.
#
# Payload keys:
#   task     : Task serialised as dict (model_dump())
#   plan     : Plan serialised as dict (model_dump())
#   evidence : list of EvidenceItem dicts (model_dump())
#   topic    : str
#   mode     : str
#
# Output:
#   {"sections": [(task.id, markdown_string)]}
#   The tuple carries the task ID so the compiler node can sort
#   results into the correct order after all workers finish.
#
# Three prompt-level guardrails enforced in every run:
#   1. Scope guard      — news_roundup != tutorial
#   2. Grounding policy — open_book claims must cite evidence URLs
#   3. Citation rule    — requires_citations forces inline Markdown links
# ============================================================

WORKER_SYSTEM_PROMPT = """You are a senior technical writer producing ONE section of a larger blog post.
You will receive a section brief and must write ONLY that section in clean Markdown.

====================================================================
HARD CONSTRAINTS — non-negotiable
====================================================================
1. Cover every bullet in the brief, in the order given. Do not skip any.
2. Stay within ±15% of the target word count specified in the brief.
3. Output ONLY the section content. Start directly with the ## heading.
   Do not add a preamble, do not repeat the topic title, do not add a conclusion
   for the overall blog — only the content of this one section.
4. Use short paragraphs (3-5 sentences max). Use bullet lists where they aid
   clarity. Use code fences (```language) for all code snippets.
5. No fluff, no marketing language, no filler phrases like "In today's fast-paced world".

====================================================================
SCOPE GUARD — applies when blog_kind is "news_roundup"
====================================================================
- Do NOT turn this section into a tutorial, how-to guide, or concept explainer.
- Your job is to SUMMARISE EVENTS and explain their IMPLICATIONS for the reader.
- Structure: what happened → why it matters → what to watch.
- Do not include step-by-step instructions or beginner definitions.

====================================================================
GROUNDING POLICY — applies when mode is "open_book"
====================================================================
This is the most important rule for open_book mode:

- Do NOT introduce any specific claim about an event, company, product, model,
  funding round, benchmark result, or policy change UNLESS that claim is directly
  supported by a URL in the Evidence section below.
- For every such claim, attach the source as a Markdown inline link:
  Format: claim text ([Source](URL))
  Example: "OpenAI released GPT-4o with native audio support ([Source](https://openai.com/blog/gpt-4o))"
- If you want to make a claim that is NOT found in the provided evidence, write:
  "Not found in provided sources." — do not speculate or hallucinate.
- You MAY use your general world knowledge for background context and framing,
  but ALL specific factual claims (names, numbers, dates, events) must be sourced.

====================================================================
CITATION RULE — applies when requires_citations is True
====================================================================
- For any claim about the outside world (library version, benchmark, research
  finding, company announcement), cite the supporting evidence URL inline using
  the same ([Source](URL)) format described above.
- Aim for at least one citation per major claim.

====================================================================
CODE RULE — applies when requires_code is True
====================================================================
- Include at least one code snippet in a fenced code block.
- The code must be minimal, correct, and directly illustrate the section's goal.
- Prefer runnable snippets over pseudocode. Add concise inline comments.

====================================================================
OUTPUT FORMAT
====================================================================
Start with: ## <Section Title>
Then write the section body.
Nothing before the ## heading. Nothing after the section ends.
"""


def worker_node(payload: dict) -> dict:
    """
    NODE 4 — Parallel Section Writer

    Receives a self-contained payload from Send() (not the full State).
    Reconstructs Task, Plan, and EvidenceItem objects, then calls the
    LLM to write exactly one section of the blog in Markdown.

    Args:
        payload: dict with keys task, plan, evidence, topic, mode

    Returns:
        {"sections": [(task.id, markdown_string)]}
        The tuple enables the compiler to sort sections by id after
        all parallel workers finish.
    """
    # ── Reconstruct typed objects from raw dicts ────────────
    task      = Task(**payload["task"])
    plan      = Plan(**payload["plan"])
    evidence  = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic     = payload["topic"]
    mode      = payload["mode"]

    print(f"   ✍️  Worker [{task.id:02d}] starting: \"{task.title}\"  "
          f"(~{task.target_words}w, mode={mode})")

    # ── Format bullets as a numbered list ──────────────────
    bullets_text = "\n".join(
        f"  {i}. {b}" for i, b in enumerate(task.bullets, 1)
    )

    # ── Format evidence as a compact reference block ────────
    # One line per item: title | url | date (or "date:unknown")
    if evidence:
        evidence_lines = []
        for ev in evidence:
            date_part = ev.published_at if ev.published_at else "date:unknown"
            evidence_lines.append(f"- {ev.title} | {ev.url} | {date_part}")
            if ev.snippet:
                evidence_lines.append(f"  Snippet: {ev.snippet[:200]}")
        evidence_text = "\n".join(evidence_lines)
    else:
        evidence_text = "(No evidence available — write from general knowledge only)"

    # ── Format plan-level constraints ──────────────────────
    constraints_text = (
        "\n".join(f"  • {c}" for c in plan.constraints)
        if plan.constraints else "  (none)"
    )

    # ── Build the human message ─────────────────────────────
    human_content = f"""
BLOG CONTEXT
============
Blog title   : {plan.blog_title}
Audience     : {plan.audience}
Tone         : {plan.tone}
Blog kind    : {plan.blog_kind}
Topic        : {topic}
Mode         : {mode}
Global constraints:
{constraints_text}

SECTION BRIEF
=============
Section id   : {task.id}
Section title: {task.title}
Goal         : {task.goal}
Target words : {task.target_words}
Tags         : {", ".join(task.tags) if task.tags else "none"}
Requires research  : {task.requires_research}
Requires citations : {task.requires_citations}
Requires code      : {task.requires_code}

Bullets to cover (IN ORDER):
{bullets_text}

EVIDENCE
========
{evidence_text}
""".strip()

    # ── Invoke the writer LLM ───────────────────────────────
    # Use temperature=0.3 for a slight touch of stylistic variation
    # while keeping factual claims deterministic.
    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    response = writer_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ])

    section_md = response.content.strip()

    # ── Log completion ──────────────────────────────────────
    word_count  = len(section_md.split())
    target      = task.target_words
    variance    = round(((word_count - target) / target) * 100, 1)
    variance_str = f"+{variance}%" if variance >= 0 else f"{variance}%"
    print(f"   ✅ Worker [{task.id:02d}] done: \"{task.title}\"  "
          f"({word_count}w, {variance_str} vs target)")

    # Return a list containing one (id, content) tuple.
    # operator.add reducer accumulates these across all parallel workers.
    return {"sections": [(task.id, section_md)]}


# ============================================================
# ── FAN-OUT: fanout ─────────────────────────────────────────
#
# This function is used as a conditional edge SOURCE — it is
# called by LangGraph after orchestrator_node finishes and
# returns a list of Send() commands.
#
# Send(node_name, payload) tells LangGraph to:
#   1. Spawn a new instance of "worker" node
#   2. Pass `payload` as the node's input argument (NOT State)
#   3. Run all spawned instances in PARALLEL
#
# Each Send() is fully self-contained: the worker gets everything
# it needs in the payload dict. It does NOT read from State at
# runtime (State is not passed to Send-receiver nodes directly).
#
# After ALL workers finish, LangGraph merges their outputs back
# into State using the reducer on State["sections"] (operator.add),
# producing the accumulated list of (id, content) tuples.
# ============================================================

def fanout(state: State) -> list:
    """
    FAN-OUT — generates one Send() per Task in the plan.

    Called as a conditional edge after orchestrator_node.
    Dispatches all section writers simultaneously using LangGraph's
    Send() API. Each Send carries a self-contained payload so workers
    never need to access shared State directly.

    Returns:
        list[Send] — one Send("worker", payload) per Task
    """
    plan     = state["plan"]
    topic    = state["topic"]
    mode     = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    print(f"\n🚀 Fan-out: dispatching {len(plan.tasks)} parallel worker(s)...")

    sends = []
    for task in plan.tasks:
        payload = {
            "task":     task.model_dump(),
            "plan":     plan.model_dump(),
            "topic":    topic,
            "mode":     mode,
            "evidence": [ev.model_dump() for ev in evidence],
        }
        sends.append(Send("worker", payload))

    return sends


# ============================================================
# ── NODE 5: reducer_node ─────────────────────────────────────
#
# The final node in the graph. Runs after ALL parallel worker
# nodes have finished and their outputs have been merged into
# State["sections"] by the operator.add reducer.
#
# Responsibilities:
#   1. Sort (id, markdown) tuples by id  →  correct reading order
#   2. Join sections with double newlines →  clean Markdown body
#   3. Prepend the blog title as an H1   →  complete document
#   4. Save to a .md file on disk
#   5. Write the final string to State["final"]
#
# Why sorting is necessary here is explained below in the docstring.
# ============================================================

def reducer_node(state: State) -> dict:
    """
    NODE 5 — Reducer / Compiler

    Collects all parallel worker outputs from State["sections"],
    sorts them into the correct reading order by task ID, assembles
    the final Markdown document, saves it to disk, and writes it
    to State["final"].

    State["sections"] is a list of (task_id, section_markdown) tuples
    accumulated by the operator.add reducer as workers finish.
    Because workers run in parallel, they can arrive in ANY order.
    Sorting by task_id is the only guarantee of correct sequence.

    Returns:
        {"final": full_markdown_string}
    """
    plan     = state["plan"]
    sections = state.get("sections", [])

    print(f"\n📚 Reducer node — compiling {len(sections)} section(s) into final blog...")

    # ── Step 1: Sort by task_id (first element of each tuple) ──
    # Workers finish in non-deterministic wall-clock order.
    # Sorting by the integer task.id restores the editorial sequence
    # that the orchestrator planned regardless of arrival order.
    sorted_sections = sorted(sections, key=lambda tup: tup[0])

    print("   Section assembly order:")
    for task_id, md in sorted_sections:
        word_count = len(md.split())
        # Grab the first line (the ## heading) for display
        heading = md.splitlines()[0] if md else f"[Section {task_id}]"
        print(f"      [{task_id:02d}] {heading}  ({word_count}w)")

    # ── Step 2: Extract markdown bodies, join with spacing ─────
    bodies = [md for _, md in sorted_sections]
    blog_body = "\n\n".join(bodies)

    # ── Step 3: Prepend the H1 title ───────────────────────────
    # The individual sections start with ## (H2) headings.
    # The H1 title lives only at document level — no worker adds it.
    final_markdown = f"# {plan.blog_title}\n\n{blog_body}"

    # ── Step 4: Save to disk ────────────────────────────────────
    # Sanitise the title for use as a filename:
    #   • lowercase, spaces to underscores
    #   • strip characters that are illegal on most filesystems
    safe_title = (
        plan.blog_title
        .lower()
        .replace(" ", "_")
    )
    # Remove anything that isn't a letter, digit, underscore, or hyphen
    safe_title = "".join(c for c in safe_title if c.isalnum() or c in ("_", "-"))
    safe_title = safe_title[:80]          # cap filename length
    filename   = f"{safe_title}.md"
    output_path = Path(filename)

    output_path.write_text(final_markdown, encoding="utf-8")

    # ── Step 5: Summary log ─────────────────────────────────────
    total_words = len(final_markdown.split())
    total_chars = len(final_markdown)
    print(f"\n   ✅ Blog compiled successfully!")
    print(f"   Title      : {plan.blog_title}")
    print(f"   Sections   : {len(sorted_sections)}")
    print(f"   Total words: ~{total_words}")
    print(f"   Total chars: {total_chars}")
    print(f"   Saved to   : {output_path.resolve()}")

    return {"final": final_markdown}


# ============================================================
# ── GRAPH ASSEMBLY ───────────────────────────────────────────
#
# Full V3 graph topology:
#
#                    ┌─────────────────────────────┐
#   START ──► router ┤ needs_research=True          │
#                    │    └──► research ──► orchestrator
#                    │ needs_research=False          │
#                    │    └──► orchestrator ◄────────┘
#                    └─────────────────────────────┘
#                              │
#                              │ fanout (conditional edge)
#                              │ spawns N parallel Send("worker")
#                              ▼
#                 ┌────────────────────────┐
#                 │  worker  worker  ...   │  (parallel)
#                 └────────────────────────┘
#                              │
#                              ▼
#                           reducer ──► END
#
# Key wiring decisions:
#   • router  → conditional edge  (route_next: str-based routing)
#   • orchestrator → conditional edge (fanout: returns list[Send])
#   • worker  → reducer is a DIRECT edge — LangGraph knows all
#     parallel workers must complete before reducer fires
# ============================================================

builder = StateGraph(State)

# ── Register nodes ─────────────────────────────────────────
builder.add_node("router",       router_node)
builder.add_node("research",     research_node)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("worker",       worker_node)
builder.add_node("reducer",      reducer_node)

# ── Entry point ────────────────────────────────────────────
builder.add_edge(START, "router")

# ── Conditional edge: router → research OR orchestrator ────
builder.add_conditional_edges(
    "router",
    route_next,
    {
        "research":     "research",
        "orchestrator": "orchestrator",
    },
)

# ── Direct edge: research always flows into orchestrator ───
builder.add_edge("research", "orchestrator")

# ── Conditional edge: orchestrator → parallel workers ──────
# fanout() returns list[Send("worker", payload)] — each Send
# spawns one independent worker instance simultaneously.
builder.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"],             # declare "worker" as a valid destination
)

# ── Direct edge: each worker feeds into reducer ─────────────
# LangGraph waits for ALL parallel workers before reducer runs.
builder.add_edge("worker", "reducer")

# ── Terminal edge ───────────────────────────────────────────
builder.add_edge("reducer", END)

# ── Compile ─────────────────────────────────────────────────
app = builder.compile()
print("🔗 LangGraph compiled — V3 agent ready.\n")


# ============================================================
# ── RUNNER ───────────────────────────────────────────────────
# ============================================================

def run(topic: str) -> dict:
    """
    Entry point for the V3 Blog Agent.

    Initialises all State fields to safe defaults and invokes the
    compiled graph. Returns the final State dict on completion.

    Args:
        topic: The blog topic as a plain English string.

    Returns:
        The complete final State dict, including "final" (the
        compiled Markdown blog) and all intermediate fields.
    """
    print("=" * 65)
    print(f"  V3 BLOG AGENT")
    print(f"  Topic: {topic}")
    print("=" * 65)

    initial_state: State = {
        "topic":          topic,
        "mode":           "",          # set by router_node
        "needs_research": False,       # set by router_node
        "queries":        [],          # set by router_node
        "evidence":       [],          # set by research_node (if run)
        "plan":           None,        # set by orchestrator_node
        "sections":       [],          # accumulated by worker_nodes
        "final":          "",          # set by reducer_node
    }

    result = app.invoke(initial_state)
    return result


# ============================================================
# ── TEST 1: Evergreen / closed_book topic ────────────────────
# Expected: router picks "closed_book", no Tavily calls fire,
# agent goes straight router → orchestrator → workers → reducer
# ============================================================

if __name__ == "__main__":

    print("\n" + "━" * 65)
    print("  TEST 1 — EVERGREEN TOPIC (expected: closed_book)")
    print("━" * 65 + "\n")

    result1 = run("Explain how attention mechanisms work in transformer models")

    print("\n" + "─" * 65)
    print("  TEST 1 SUMMARY")
    print("─" * 65)
    print(f"  Mode chosen    : {result1['mode']}")
    print(f"  Needs research : {result1['needs_research']}")
    print(f"  Queries fired  : {len(result1['queries'])}  (0 = no Tavily calls)")
    print(f"  Evidence items : {len(result1['evidence'])}")
    print(f"  Sections built : {len(result1['sections'])}")
    print(f"  Word count     : ~{len(result1['final'].split())}")

    # ============================================================
    # ── TEST 2: Current events / open_book topic ────────────────
    # Expected: router picks "open_book", Tavily fires N queries,
    # evidence is collected, writers cite URLs inline
    # ============================================================

    print("\n\n" + "━" * 65)
    print("  TEST 2 — CURRENT EVENTS TOPIC (expected: open_book)")
    print("━" * 65 + "\n")

    result2 = run("State of Multimodal LLMs in 2026")

    print("\n" + "─" * 65)
    print("  TEST 2 SUMMARY")
    print("─" * 65)
    print(f"  Mode chosen    : {result2['mode']}")
    print(f"  Queries fired  : {len(result2['queries'])}")
    for i, q in enumerate(result2["queries"], 1):
        print(f"    {i}. {q}")
    print(f"  Evidence items : {len(result2['evidence'])}")
    citation_count = result2["final"].count("([Source](")
    print(f"  Citation links : {citation_count}  (Markdown [Source](URL) occurrences)")
    print(f"  Sections built : {len(result2['sections'])}")
    print(f"  Word count     : ~{len(result2['final'].split())}")

    print("\n" + "─" * 65)
    print("  TEST 2 — FINAL BLOG OUTPUT")
    print("─" * 65 + "\n")
    print(result2["final"])
