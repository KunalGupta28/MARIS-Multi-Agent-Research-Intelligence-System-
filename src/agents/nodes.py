"""
MARIS Agent Nodes — Individual agent implementations for the LangGraph pipeline.

Each node is a function that takes ResearchState, performs its task, and returns
a partial state update. LangGraph merges these updates into the global state.

Nodes:
    planner_node      — Decomposes query into targeted sub-topics
    retriever_node    — Searches arXiv + Vector DB for relevant papers/chunks
    extractor_node    — Extracts structured facts from retrieved chunks
    synthesizer_node  — Generates a grounded literature review with citations
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import (
    ResearchState,
    RetrievedChunk,
    ExtractedFact,
    AgentStatus,
)
from src.agents.llm import get_llm
from src.ingestion.arxiv_client import ArxivClient
from src.storage.vector_store import VectorStore
from src.storage.database import MARISDatabase
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


# ── Shared Resources (initialized lazily) ──────────────────────────

_arxiv_client: ArxivClient | None = None
_vector_store: VectorStore | None = None
_db: MARISDatabase | None = None


def _get_db() -> MARISDatabase:
    global _db
    if _db is None:
        _db = MARISDatabase()
    return _db


def _get_arxiv_client() -> ArxivClient:
    global _arxiv_client
    if _arxiv_client is None:
        _arxiv_client = ArxivClient(db=_get_db())
    return _arxiv_client


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(db=_get_db())
    return _vector_store


# ═══════════════════════════════════════════════════════════════════
# PLANNER NODE
# ═══════════════════════════════════════════════════════════════════

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research planning agent. Given a research query, decompose it 
into 3-5 specific, searchable sub-queries that together would comprehensively 
cover the topic for a literature review on arXiv.

CRITICAL INSTRUCTIONS FOR ARXIV BOOLEAN SEARCH:
1. Every sub-query MUST be formatted as a structured Lucene-style boolean query using arXiv search fields ('all:', 'ti:', 'abs:') and boolean operators 'AND', 'OR' to ensure highly relevant search results.
2. Every sub-query MUST explicitly include the core keywords of the main query to prevent off-topic results. For example, if the query is about GANs and EEG, every sub-query must include 'all:GAN' and 'all:EEG'.
3. Use clean terms, parentheses, and boolean operators. Do NOT use double quotes inside the query to avoid JSON escaping issues; use single quotes or hyphenation if needed (e.g. all:state-space or all:'state space').
4. Each sub-query should target a different aspect:
   - Core methodology/architecture
   - Datasets, benchmarks, or data augmentation
   - Comparative approaches
   - Limitations and challenges
   - Recent state-of-the-art advancements

Return ONLY a JSON array of strings. No other text.

Example for 'GAN Based EEG-BCI's':
[
  "(all:GAN OR all:Generative-Adversarial) AND all:EEG AND all:BCI AND all:architecture",
  "(all:GAN OR all:Generative-Adversarial) AND all:EEG AND (all:dataset OR all:augmentation)",
  "(all:GAN OR all:Generative-Adversarial) AND all:EEG AND all:comparison AND all:classification",
  "(all:GAN OR all:Generative-Adversarial) AND all:EEG AND (all:limitations OR all:challenges)",
  "(all:GAN OR all:Generative-Adversarial) AND all:EEG AND all:recent AND all:advancements"
]

Example for 'transformer attention vs state space models':
[
  "(all:transformer OR all:attention) AND all:state-space AND all:comparison",
  "all:transformer AND all:attention AND all:architecture",
  "all:state-space AND (all:S4 OR all:Mamba) AND all:evaluation"
]""",
        ),
        ("human", "Research query: {query}"),
    ]
)


def planner_node(state: ResearchState) -> dict[str, Any]:
    """
    Decompose the research query into targeted sub-queries.

    Input: state.research_query
    Output: state.sub_queries, state.research_plan, state.agent_trace
    """
    logger.info(f"[Planner] Decomposing: '{state.research_query}'")

    trace = AgentStatus(
        node_name="Planner",
        status="started",
        message=f"Decomposing research query into sub-topics...",
    )

    llm = get_llm()
    chain = PLANNER_PROMPT | llm

    try:
        response = chain.invoke({"query": state.research_query})
        content = response.content.strip()

        # Parse JSON array from LLM response (robust extraction)
        # Try regex extraction first — handles code blocks, extra text, etc.
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        sub_queries = json.loads(content)

        if not isinstance(sub_queries, list):
            sub_queries = [state.research_query]

        plan = f"Generated {len(sub_queries)} sub-queries:\n"
        for i, sq in enumerate(sub_queries, 1):
            plan += f"  {i}. {sq}\n"

        logger.info(f"[Planner] Generated {len(sub_queries)} sub-queries")

        return {
            "sub_queries": sub_queries,
            "research_plan": plan,
            "agent_trace": [
                trace,
                AgentStatus(
                    node_name="Planner",
                    status="completed",
                    message=f"Created {len(sub_queries)} sub-queries",
                ),
            ],
            "current_step": "retriever",
        }

    except Exception as e:
        logger.error(f"[Planner] Error: {e}")
        return {
            "sub_queries": [state.research_query],
            "research_plan": "Fallback: using original query directly.",
            "agent_trace": [
                AgentStatus(
                    node_name="Planner",
                    status="error",
                    message=f"Error: {str(e)[:200]}. Using original query.",
                )
            ],
            "errors": [f"Planner error: {str(e)}"],
            "current_step": "retriever",
        }


# ═══════════════════════════════════════════════════════════════════
# RETRIEVER NODE
# ═══════════════════════════════════════════════════════════════════


def retriever_node(state: ResearchState) -> dict[str, Any]:
    """
    Search arXiv and the local vector store for relevant papers and chunks.

    Input: state.sub_queries
    Output: state.retrieved_chunks, state.papers_found, state.agent_trace
    """
    logger.info(f"[Retriever] Searching for {len(state.sub_queries)} sub-queries")

    traces = [
        AgentStatus(
            node_name="Retriever",
            status="started",
            message=f"Searching arXiv and local database for {len(state.sub_queries)} queries...",
        )
    ]

    arxiv_client = _get_arxiv_client()
    vector_store = _get_vector_store()
    all_chunks: list[RetrievedChunk] = []
    all_paper_ids: list[str] = []

    for i, query in enumerate(state.sub_queries):
        traces.append(
            AgentStatus(
                node_name="Retriever",
                status="started",
                message=f"[{i+1}/{len(state.sub_queries)}] Searching: '{query[:60]}...'",
            )
        )

        # Step 1: Search arXiv for new papers
        try:
            papers = arxiv_client.search_papers(query, max_results=5)
            for p in papers:
                if p.arxiv_id not in all_paper_ids:
                    all_paper_ids.append(p.arxiv_id)

            # Auto-ingestion of top 3 results to maximize grounding coverage
            if papers:
                db_ref = _get_db()
                pipeline = IngestionPipeline(vector_store=vector_store)
                for paper in papers[:3]:
                    paper_record = db_ref.get_paper(paper.arxiv_id)
                    if not paper_record or paper_record.get("chunk_count", 0) == 0:
                        traces.append(
                            AgentStatus(
                                node_name="Retriever",
                                status="started",
                                message=f"📥 Auto-ingesting: '{paper.title[:50]}...'",
                            )
                        )
                        indexed_chunks = pipeline.ingest_paper(paper)
                        traces.append(
                            AgentStatus(
                                node_name="Retriever",
                                status="started",
                                message=f"✅ Indexed {indexed_chunks} chunks for '{paper.arxiv_id}'",
                            )
                        )
        except Exception as e:
            logger.error(f"[Retriever] ArXiv search/ingestion failed for '{query}': {e}")

        # Step 2: Hybrid search on local vector store
        try:
            results = vector_store.hybrid_search(query, top_k=5)
            for r in results:
                # Enrich with verified DB metadata
                pid = r.get("paper_id", "")
                db_paper = _get_db().get_paper(pid) if pid else None
                chunk = RetrievedChunk(
                    chunk_id=r.get("chunk_id", ""),
                    paper_id=pid,
                    text=r.get("text", ""),
                    section=r.get("section", ""),
                    page_number=r.get("page_number", 0),
                    title=db_paper["title"] if db_paper else r.get("title", ""),
                    authors=db_paper["authors"] if db_paper else r.get("authors", []),
                    pdf_url=db_paper["pdf_url"] if db_paper else r.get("pdf_url", ""),
                    score=r.get("rrf_score", r.get("score", 0.0)),
                )
                all_chunks.append(chunk)
        except Exception as e:
            logger.error(f"[Retriever] Vector search failed for '{query}': {e}")

    # Deduplicate chunks by chunk_id
    seen_ids = set()
    unique_chunks = []
    for c in all_chunks:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            unique_chunks.append(c)

    traces.append(
        AgentStatus(
            node_name="Retriever",
            status="completed",
            message=f"Found {len(all_paper_ids)} papers, {len(unique_chunks)} relevant chunks",
        )
    )

    logger.info(
        f"[Retriever] Found {len(all_paper_ids)} papers, {len(unique_chunks)} chunks"
    )

    return {
        "retrieved_chunks": unique_chunks,
        "papers_found": all_paper_ids,
        "agent_trace": traces,
        "current_step": "extractor",
    }


# ═══════════════════════════════════════════════════════════════════
# EXTRACTOR NODE
# ═══════════════════════════════════════════════════════════════════

class BenchmarkMetric(BaseModel):
    """Structured benchmark metric parsed from the paper."""
    dataset: str = Field(description="Normalized name of the dataset/benchmark (e.g. 'LRA', 'GLUE', 'ImageNet').")
    metric_name: str = Field(description="Metric name (e.g. 'Accuracy', 'Perplexity', 'F1-score', 'Speedup').")
    value: str = Field(description="Proposed model's performance value (e.g. '84.2%', '1.24').")
    baseline: str = Field(default="", description="Baseline performance for comparison (e.g. '81.0%').")
    context: str = Field(default="", description="Context or model variant detail (e.g. 'Mamba-3B', '100k length').")


class StructuredCitation(BaseModel):
    """Structured citation representation."""
    citation_key: str = Field(description="Citation format key e.g. '[^arXiv:1905.04149]' or '[1]'.")
    title: str = Field(description="Title of the referenced paper.")
    authors: str = Field(default="", description="First author or et al. format.")
    year: str = Field(default="", description="Year of publication.")


class ScientificExtractionSchema(BaseModel):
    """Structured scientific information extracted from a research paper."""
    problem_statement: str = Field(description="The primary research problem or question addressed by the paper.")
    methods: list[str] = Field(default_factory=list, description="Specific models, methods, architectures, or algorithms proposed (e.g. ['Transformer', 'Mamba SSM', 'LSTM']).")
    datasets: list[str] = Field(default_factory=list, description="Datasets, benchmarks, or environments used to evaluate the methods (e.g. ['LRA', 'GLUE', 'MNIST']).")
    key_results: list[str] = Field(default_factory=list, description="Main results, performance gains, metrics, or comparisons (e.g. ['3x speedup', '92% accuracy']).")
    limitations: list[str] = Field(default_factory=list, description="Acknowledged limitations, scaling constraints, or failure modes.")
    benchmarks: list[BenchmarkMetric] = Field(default_factory=list, description="Granular benchmarks and metrics parsed from the text.")
    citations: list[StructuredCitation] = Field(default_factory=list, description="Key references, baselines, or foundational cited papers.")


EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research extraction agent. Given text chunks from a scientific paper,
extract the key scientific details into a structured representation.""",
        ),
        (
            "human",
            """Paper: {title} (by {authors})

Text chunks from this paper:
{chunks_text}

Extract the structured scientific elements:""",
        ),
    ]
)


def extractor_node(state: ResearchState) -> dict[str, Any]:
    """
    Extract structured facts from retrieved chunks, grouped by paper.

    Input: state.retrieved_chunks
    Output: state.extracted_facts, state.agent_trace
    """
    logger.info(f"[Extractor] Processing {len(state.retrieved_chunks)} chunks")

    traces = [
        AgentStatus(
            node_name="Extractor",
            status="started",
            message=f"Extracting structured facts from {len(state.retrieved_chunks)} chunks...",
        )
    ]

    # Group chunks by paper
    paper_chunks: dict[str, list[RetrievedChunk]] = {}
    for chunk in state.retrieved_chunks:
        if chunk.paper_id not in paper_chunks:
            paper_chunks[chunk.paper_id] = []
        paper_chunks[chunk.paper_id].append(chunk)

    llm = get_llm()
    # Configure LLM to force schema-guided structured extraction
    structured_llm = llm.with_structured_output(ScientificExtractionSchema)
    chain = EXTRACTOR_PROMPT | structured_llm
    facts: list[ExtractedFact] = []

    for paper_id, chunks in paper_chunks.items():
        # Always use verified DB metadata as source of truth
        db_paper = _get_db().get_paper(paper_id)
        if db_paper:
            title = db_paper["title"]
            authors_list = db_paper["authors"] if isinstance(db_paper["authors"], list) else []
            authors = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
        else:
            title = chunks[0].title if chunks else paper_id
            authors = ", ".join(chunks[0].authors[:3]) if chunks and chunks[0].authors else "Unknown"

        # Concatenate chunk texts (limit to avoid context overflow)
        chunks_text = "\n\n---\n\n".join(
            f"[Section: {c.section}, Page {c.page_number}]\n{c.text}"
            for c in chunks[:6]  # max 6 chunks per paper to stay within context
        )

        try:
            # chain.invoke directly returns a ScientificExtractionSchema instance!
            data = chain.invoke(
                {
                    "title": title,
                    "authors": authors,
                    "chunks_text": chunks_text,
                }
            )

            fact = ExtractedFact(
                paper_id=paper_id,
                title=title,
                problem_statement=data.problem_statement or "",
                methods=data.methods or [],
                datasets=data.datasets or [],
                key_results=data.key_results or [],
                limitations=data.limitations or [],
                benchmarks=[b.model_dump() for b in data.benchmarks] if getattr(data, "benchmarks", None) else [],
                citations=[c.model_dump() for c in data.citations] if getattr(data, "citations", None) else [],
            )
            facts.append(fact)

            traces.append(
                AgentStatus(
                    node_name="Extractor",
                    status="started",
                    message=f"Extracted facts from: {title[:60]}...",
                )
            )

        except Exception as e:
            logger.error(f"[Extractor] Failed for {paper_id}: {e}")

    traces.append(
        AgentStatus(
            node_name="Extractor",
            status="completed",
            message=f"Extracted facts from {len(facts)} papers",
        )
    )

    return {
        "extracted_facts": facts,
        "agent_trace": traces,
        "current_step": "synthesizer",
    }


# ═══════════════════════════════════════════════════════════════════
# SYNTHESIZER NODE
# ═══════════════════════════════════════════════════════════════════

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an academic literature review writer. Your task is to synthesize 
information from multiple research papers into a cohesive, well-structured 
literature review.

CRITICAL RULES — STRICT CITATION INTEGRITY:
1. You are provided with a CITATION POOL below. This is the ONLY set of papers you may cite.
2. EVERY claim MUST include a citation tag exactly matching one from the pool (e.g. [1], [2]).
3. You MUST NOT invent, fabricate, or hallucinate any citation that is not in the pool.
4. If you cannot support a claim with a paper from the pool, DO NOT make the claim.
5. Never use superscript symbols (¹, ²), markdown footnotes ([^1]), or arXiv-style keys ([^arXiv:...]). ONLY use [N] format.
6. Do NOT include a "References" section — it will be appended automatically by the system.
7. Group related findings thematically, not paper-by-paper. Compare and contrast approaches.
8. Identify gaps, contradictions, and open problems based ONLY on the provided facts.
9. Use academic writing style.
10. IMPORTANT: If the user's query specifically asks about a named methodology (e.g. 'CRAG', 'Self-RAG') and NONE of the papers in the CITATION POOL are the original/seminal paper for that methodology, you MUST explicitly state: 'Note: The seminal paper for [methodology name] was not retrieved in this search.' Do NOT falsely attribute the methodology to a tangentially related paper.

OUTPUT FORMAT (Markdown — NO References section):
## Literature Review: [Topic]

### Introduction to [Topic] and Challenges
General introduction citing relevant sources [1], [2] ...

### [Thematic Section 1]
Content with citations [1], [3] ...

### [Thematic Section 2]
Content with citations [2] ...

### Comparative Analysis
Comparison table or discussion...

### Research Gaps & Future Directions
Identified gaps and future work...

REMEMBER: Do NOT write a References/Bibliography section. It will be appended automatically.""",
        ),
        (
            "human",
            """Research Query: {query}

=== CITATION POOL (use ONLY these) ===
{citation_pool}

=== Extracted Facts from Papers ===
{facts_text}

=== Source Chunks for Grounding ===
{source_chunks}

Write a comprehensive literature review using ONLY the citations from the pool above:""",
        ),
    ]
)


def _build_citation_pool(state: ResearchState) -> list[dict]:
    """
    Build a verified citation pool from the database for all papers
    referenced in extracted_facts and retrieved_chunks.

    Returns a list of dicts with keys: index, paper_id, title, authors, year, url.
    """
    db = _get_db()
    seen_ids = []
    # Collect unique paper_ids preserving order (facts first, then chunks)
    for fact in state.extracted_facts:
        if fact.paper_id and fact.paper_id not in seen_ids:
            seen_ids.append(fact.paper_id)
    for chunk in state.retrieved_chunks:
        if chunk.paper_id and chunk.paper_id not in seen_ids:
            seen_ids.append(chunk.paper_id)

    pool = []
    for idx, paper_id in enumerate(seen_ids, start=1):
        paper = db.get_paper(paper_id)
        if paper:
            authors_list = paper.get("authors", [])
            if isinstance(authors_list, str):
                try:
                    authors_list = json.loads(authors_list)
                except (json.JSONDecodeError, TypeError):
                    authors_list = []
            # Format authors: "First Author et al." if > 3
            if len(authors_list) > 3:
                authors_str = f"{authors_list[0]} et al."
            elif authors_list:
                authors_str = ", ".join(authors_list)
            else:
                authors_str = "Unknown"
            # Extract year from published_date (YYYY-MM-DD)
            pub_date = paper.get("published_date", "")
            year = pub_date[:4] if pub_date and len(pub_date) >= 4 else "N/A"
            pool.append({
                "index": idx,
                "paper_id": paper_id,
                "title": paper.get("title", paper_id),
                "authors": authors_str,
                "year": year,
                "url": f"https://arxiv.org/abs/{paper_id}",
            })
        else:
            pool.append({
                "index": idx,
                "paper_id": paper_id,
                "title": paper_id,
                "authors": "Unknown",
                "year": "N/A",
                "url": f"https://arxiv.org/abs/{paper_id}",
            })
    return pool


def _format_citation_pool(pool: list[dict]) -> str:
    """Format the citation pool as a numbered list for the LLM prompt."""
    lines = []
    for entry in pool:
        lines.append(
            f"[{entry['index']}] {entry['authors']}, \"{entry['title']}\", "
            f"{entry['year']}. {entry['url']}"
        )
    return "\n".join(lines)


def _postprocess_review(review: str, pool: list[dict]) -> str:
    """
    Post-process the LLM's literature review to:
    1. Strip any LLM-generated References/Bibliography section.
    2. Validate that every [N] tag maps to a real pool entry.
    3. Re-index citation tags to be sequential with no gaps.
    4. Append a programmatically generated References section.
    """
    # Step 1: Strip any LLM-generated References section
    ref_pattern = re.compile(
        r"\n###?\s*(?:References|Bibliography|Works Cited).*",
        re.IGNORECASE | re.DOTALL,
    )
    review = ref_pattern.sub("", review).rstrip()

    # Step 2: Find all cited [N] tags in the review body
    cited_numbers = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", review)))

    # Step 3: Build a mapping from old index -> new sequential index
    pool_index_map = {entry["index"]: entry for entry in pool}
    valid_cited = [n for n in cited_numbers if n in pool_index_map]

    if not valid_cited:
        # No valid citations found — return review as-is with full pool
        ref_lines = ["\n\n### References"]
        for entry in pool:
            ref_lines.append(
                f"[{entry['index']}]: {entry['authors']}, \"{entry['title']}\", "
                f"{entry['year']}. {entry['url']}"
            )
        return review + "\n".join(ref_lines)

    # Re-map to sequential 1..N
    old_to_new = {old: new for new, old in enumerate(valid_cited, start=1)}

    # Replace tags in the review text (process largest numbers first to avoid
    # partial replacements like [1] matching inside [12])
    processed_review = review
    for old_num in sorted(old_to_new.keys(), reverse=True):
        new_num = old_to_new[old_num]
        processed_review = processed_review.replace(f"[{old_num}]", f"[__CITE_{new_num}__]")

    # Remove any [N] tags that don't map to a valid pool entry (hallucinated)
    processed_review = re.sub(r"\[\d+\]", "", processed_review)

    # Restore placeholder tags to final format
    for new_num in sorted(old_to_new.values()):
        processed_review = processed_review.replace(f"[__CITE_{new_num}__]", f"[{new_num}]")

    # Step 4: Append verified References section
    ref_lines = ["\n\n### References"]
    for old_num in valid_cited:
        new_num = old_to_new[old_num]
        entry = pool_index_map[old_num]
        ref_lines.append(
            f"[{new_num}]: {entry['authors']}, \"{entry['title']}\", "
            f"{entry['year']}. {entry['url']}"
        )

    return processed_review + "\n".join(ref_lines)


def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    """
    Generate a grounded literature review from extracted facts.

    Input: state.extracted_facts, state.retrieved_chunks, state.research_query
    Output: state.literature_review, state.agent_trace
    """
    logger.info("[Synthesizer] Generating literature review")

    traces = [
        AgentStatus(
            node_name="Synthesizer",
            status="started",
            message="Synthesizing literature review with source grounding...",
        )
    ]

    # Build verified citation pool from database
    citation_pool = _build_citation_pool(state)
    citation_pool_text = _format_citation_pool(citation_pool)

    # Format extracted facts for the prompt
    facts_parts = []
    for fact in state.extracted_facts:
        # Find the pool index for this paper
        pool_tag = ""
        for entry in citation_pool:
            if entry["paper_id"] == fact.paper_id:
                pool_tag = f" — cite as [{entry['index']}]"
                break
        parts = [f"### Paper: {fact.title} [{fact.paper_id}]{pool_tag}"]
        if fact.problem_statement:
            parts.append(f"**Problem:** {fact.problem_statement}")
        if fact.methods:
            parts.append(f"**Methods:** {', '.join(fact.methods)}")
        if fact.datasets:
            parts.append(f"**Datasets:** {', '.join(fact.datasets)}")
        if fact.key_results:
            parts.append(f"**Results:** {'; '.join(fact.key_results)}")
        if fact.limitations:
            parts.append(f"**Limitations:** {'; '.join(fact.limitations)}")
        
        # Format benchmarks if present
        if getattr(fact, "benchmarks", None):
            parts.append("**Granular Benchmarks:**")
            for bm in fact.benchmarks:
                parts.append(
                    f"  - Dataset: {bm.get('dataset', '')} | Metric: {bm.get('metric_name', '')} | "
                    f"Value: {bm.get('value', '')} (Baseline: {bm.get('baseline', 'N/A')}, Context: {bm.get('context', 'N/A')})"
                )
                
        facts_parts.append("\n".join(parts))

    facts_text = "\n\n---\n\n".join(facts_parts) if facts_parts else "No facts extracted."

    # Format source chunks for grounding
    source_parts = []
    for chunk in state.retrieved_chunks[:15]:  # limit context size
        source_parts.append(
            f"[Chunk {chunk.chunk_id[:8]}] Paper: {chunk.title} [{chunk.paper_id}] "
            f"Section: {chunk.section}, Page {chunk.page_number}\n"
            f"Text: {chunk.text[:500]}"
        )
    source_chunks = "\n\n".join(source_parts) if source_parts else "No source chunks."

    llm = get_llm()
    chain = SYNTHESIZER_PROMPT | llm

    try:
        response = chain.invoke(
            {
                "query": state.research_query,
                "citation_pool": citation_pool_text,
                "facts_text": facts_text,
                "source_chunks": source_chunks,
            }
        )

        raw_review = response.content.strip()
        tokens = response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") and response.usage_metadata else 0

        # Post-process: validate citations and generate verified references
        review = _postprocess_review(raw_review, citation_pool)

        traces.append(
            AgentStatus(
                node_name="Synthesizer",
                status="completed",
                message=f"Literature review generated ({len(review)} chars, {tokens} tokens)",
                tokens_used=tokens,
            )
        )

        logger.info(f"[Synthesizer] Generated review: {len(review)} chars")

        return {
            "literature_review": review,
            "agent_trace": traces,
            "current_step": "complete",
        }

    except Exception as e:
        logger.error(f"[Synthesizer] Error: {e}")
        return {
            "literature_review": f"Error generating review: {str(e)}",
            "agent_trace": [
                AgentStatus(
                    node_name="Synthesizer",
                    status="error",
                    message=f"Synthesis failed: {str(e)[:200]}",
                )
            ],
            "errors": [f"Synthesizer error: {str(e)}"],
            "current_step": "complete",
        }
