"""
MARIS — Streamlit Research Workspace

A premium, interactive UI for the Multi-Agent Research Intelligence System.
Features:
    - Animated search interface
    - Live streaming agent thoughts panel
    - Citation network visualization
    - Research session history
    - One-click literature review export
"""

import os

# ── Suppress noisy transformers warnings ──────────────────────────
# The transformers library (pulled in by sentence-transformers) emits
# advisory warnings when Streamlit's file watcher inspects __path__
# on vision submodules (vitmatte, vitpose, yolos, zoedepth) that try
# to import torchvision. These are harmless — MARIS only uses text
# embeddings — but flood the terminal with tracebacks.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
import time
import uuid
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Silence the transformers and streamlit watcher loggers to avoid residual import noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="MARIS — Research Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Premium CSS ───────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ────────────────────────────────────────────── */
    .maris-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .maris-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .maris-header p {
        color: #8892b0;
        font-size: 1rem;
        font-weight: 300;
    }

    /* ── Search Box ────────────────────────────────────────── */
    .search-container {
        max-width: 700px;
        margin: 1.5rem auto;
        position: relative;
    }

    /* ── Agent Trace Panel ─────────────────────────────────── */
    .trace-panel {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .trace-line {
        padding: 4px 0;
        color: #c9d1d9;
        border-bottom: 1px solid #161b22;
    }
    .trace-node {
        color: #58a6ff;
        font-weight: 600;
    }
    .trace-started {
        color: #f0883e;
    }
    .trace-completed {
        color: #3fb950;
    }
    .trace-error {
        color: #f85149;
    }

    /* ── Stats Cards ───────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* ── Session Card ──────────────────────────────────────── */
    .session-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .session-card:hover {
        border-color: #58a6ff;
    }
    .session-query {
        font-size: 0.85rem;
        font-weight: 500;
        color: #c9d1d9;
    }
    .session-meta {
        font-size: 0.7rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* ── Citation Tag ──────────────────────────────────────── */
    .citation-tag {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #93c5fd;
        cursor: help;
        margin: 0 2px;
    }

    /* ── Hide Streamlit defaults ───────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Initialize Session State ─────────────────────────────────────
def init_session_state():
    if "traces" not in st.session_state:
        st.session_state.traces = []
    if "result" not in st.session_state:
        st.session_state.result = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "history" not in st.session_state:
        st.session_state.history = []


# ── Render Functions ──────────────────────────────────────────────


def render_header():
    st.markdown(
        """
    <div class="maris-header">
        <h1>🔬 MARIS</h1>
        <p>Multi-Agent Research Intelligence System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_stats_bar():
    """Show system statistics in sleek cards."""
    try:
        from src.storage.database import MARISDatabase

        db = MARISDatabase()
        stats = db.get_stats()
    except Exception:
        stats = {"papers": 0, "chunks": 0, "citations": 0, "sessions": 0}

    cols = st.columns(4)
    metrics = [
        ("📄", stats["papers"], "Papers Indexed"),
        ("🧩", stats["chunks"], "Text Chunks"),
        ("🔗", stats["citations"], "Citations"),
        ("📋", stats["sessions"], "Sessions"),
    ]

    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_trace_panel(traces: list):
    """Render the live agent execution trace panel."""
    if not traces:
        return

    st.markdown("#### 🧠 Agent Execution Trace")

    trace_html = '<div class="trace-panel">'
    for t in traces:
        node = t.get("node", "System")
        msg = t.get("message", "")
        status = t.get("status", "started")

        status_class = f"trace-{status}"
        status_icon = {"started": "⏳", "completed": "✅", "error": "❌"}.get(
            status, "•"
        )

        trace_html += (
            f'<div class="trace-line">'
            f'{status_icon} <span class="trace-node">[{node}]</span> '
            f'<span class="{status_class}">{msg}</span>'
            f"</div>"
        )
    trace_html += "</div>"

    st.markdown(trace_html, unsafe_allow_html=True)


def render_literature_review(review: str):
    """Render the generated literature review with styled citations."""
    if not review:
        return

    st.markdown("---")
    st.markdown("#### 📝 Literature Review")
    st.markdown(review)


def render_sidebar():
    """Render the sidebar with session history and system info."""
    with st.sidebar:
        st.markdown("### 📚 Research Sessions")
        st.markdown("---")

        try:
            from src.storage.database import MARISDatabase

            db = MARISDatabase()
            sessions = db.get_recent_sessions(limit=10)

            if sessions:
                for session in sessions:
                    status_icon = (
                        "✅" if session["status"] == "completed" else "🔄"
                    )
                    with st.expander(
                        f"{status_icon} {session['query'][:50]}...", expanded=False
                    ):
                        st.caption(f"ID: {session['session_id']}")
                        st.caption(f"Created: {session['created_at']}")
                        st.caption(
                            f"Papers: {len(session.get('paper_ids', []))}"
                        )
                        if session.get("literature_review"):
                            if st.button(
                                "Load Review",
                                key=f"load_{session['session_id']}",
                            ):
                                st.session_state.result = session
                                st.rerun()
            else:
                st.info("No sessions yet. Run your first research query!")
        except Exception:
            st.info("Database not initialized. Run a query to get started.")

        st.markdown("---")
        st.markdown("### ⚙️ System Info")
        st.markdown(
            """
        - **LLM:** Groq (Llama 3.3 70B)
        - **Embeddings:** OpenAI `text-embedding-3-small`
        - **Vector DB:** Qdrant (Local Disk)
        - **Storage:** SQLite
        - **Observability:** LangSmith
        """
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size: 0.7rem; color: #8b949e; text-align: center;">'
            "MARIS v0.1.0 · Built with LangGraph</p>",
            unsafe_allow_html=True,
        )


# ── Main Application ─────────────────────────────────────────────


def main():
    inject_css()
    init_session_state()
    render_header()
    render_stats_bar()
    render_sidebar()

    st.markdown("")

    # ── Search Input ──────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Research Query",
            placeholder="e.g., Compare transformer attention mechanisms with state space models for long-range dependencies",
            label_visibility="collapsed",
        )
    with col2:
        run_btn = st.button("🔍 Research", type="primary", use_container_width=True)

    # ── Single Agent Trace Panel Container ────────────────────
    trace_container = st.empty()

    just_ran = False

    # ── Run Pipeline ──────────────────────────────────────────
    if run_btn and query:
        just_ran = True
        st.session_state.traces = []
        st.session_state.result = None
        st.session_state.running = True

        progress_bar = st.progress(0, text="Initializing research pipeline...")

        try:
            from src.agents.graph import ResearchRunner

            runner = ResearchRunner()
            session_id = str(uuid.uuid4())[:8]

            progress_steps = {
                "Planner": 0.2,
                "Retriever": 0.5,
                "Extractor": 0.75,
                "Synthesizer": 0.95,
            }

            final_result = None

            for node_name, message in runner.run_streaming(query, session_id):
                if node_name == "__result__":
                    final_result = message
                    break

                # Add trace
                st.session_state.traces.append(
                    {
                        "node": node_name,
                        "message": message,
                        "status": "completed"
                        if "completed" in message.lower() or "found" in message.lower()
                        else "started",
                    }
                )

                # Update progress
                progress = progress_steps.get(node_name, 0.1)
                progress_bar.progress(progress, text=f"[{node_name}] {message}")

                # Re-render trace panel
                with trace_container.container():
                    render_trace_panel(st.session_state.traces)

            progress_bar.progress(1.0, text="✅ Research complete!")
            time.sleep(0.5)
            progress_bar.empty()

            if final_result:
                st.session_state.result = {
                    "literature_review": final_result.literature_review,
                    "papers_found": final_result.papers_found,
                    "extracted_facts": [
                        f.model_dump() for f in final_result.extracted_facts
                    ],
                    "query": query,
                    "session_id": session_id,
                }

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            logging.exception("Pipeline failed")

        st.session_state.running = False

    # ── Display Results ───────────────────────────────────────
    if st.session_state.traces and not just_ran:
        with trace_container.container():
            render_trace_panel(st.session_state.traces)

    if st.session_state.result:
        result = st.session_state.result

        # Literature Review
        render_literature_review(result.get("literature_review", ""))

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            review_text = result.get("literature_review", "")
            if review_text:
                st.download_button(
                    "📥 Export Markdown",
                    data=review_text,
                    file_name=f"maris_review_{result.get('session_id', 'output')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        with col2:
            if st.button("📊 View Extracted Facts", use_container_width=True):
                st.session_state.show_facts = not st.session_state.get(
                    "show_facts", False
                )

        with col3:
            papers_found = result.get("papers_found", [])
            if papers_found:
                st.metric("Papers Analyzed", len(papers_found))

        # ── Extracted Facts (toggle persists across reruns) ────
        if st.session_state.get("show_facts", False):
            facts = result.get("extracted_facts", [])
            if facts:
                st.markdown("#### 📊 Extracted Facts")
                for fact in facts:
                    title_text = fact.get("title", "Unknown")
                    display_title = (
                        title_text[:60] + "..."
                        if len(title_text) > 60
                        else title_text
                    )
                    with st.expander(f"📄 {display_title}", expanded=False):
                        st.markdown(
                            f"**Problem:** {fact.get('problem_statement', 'N/A')}"
                        )
                        methods = fact.get("methods", [])
                        if methods:
                            st.markdown(f"**Methods:** {', '.join(methods)}")
                        datasets = fact.get("datasets", [])
                        if datasets:
                            st.markdown(f"**Datasets:** {', '.join(datasets)}")
                        results_list = fact.get("key_results", [])
                        if results_list:
                            st.markdown(f"**Results:** {'; '.join(results_list)}")
                        limitations = fact.get("limitations", [])
                        if limitations:
                            st.markdown(
                                f"**Limitations:** {'; '.join(limitations)}"
                            )
            else:
                st.info(
                    "No structured facts extracted yet. This happens when the "
                    "vector store has no indexed chunks. Try running the search "
                    "again — the system will now auto-ingest papers on each search."
                )

        # ── Citation Network ──────────────────────────────────
        if result.get("papers_found"):
            st.markdown("---")
            st.markdown("#### 🕸️ Citation Network")

            try:
                from streamlit_agraph import agraph, Node, Edge, Config

                from src.storage.database import MARISDatabase

                db = MARISDatabase()
                citation_data = db.get_citation_graph()
                nodes = []
                edges = []
                seen_nodes = set()

                for pid in result["papers_found"][:20]:
                    paper = db.get_paper(pid)
                    label = (
                        paper["title"][:40] + "..." if paper else pid[:15]
                    )
                    if pid not in seen_nodes:
                        nodes.append(
                            Node(
                                id=pid,
                                label=label,
                                size=25,
                                color="#58a6ff",
                            )
                        )
                        seen_nodes.add(pid)

                for cite in citation_data:
                    src = cite["source_paper_id"]
                    tgt = cite["target_paper_id"]
                    if src not in seen_nodes:
                        nodes.append(
                            Node(
                                id=src,
                                label=src[:15],
                                size=15,
                                color="#8b949e",
                            )
                        )
                        seen_nodes.add(src)
                    if tgt not in seen_nodes:
                        nodes.append(
                            Node(
                                id=tgt,
                                label=tgt[:15],
                                size=15,
                                color="#8b949e",
                            )
                        )
                        seen_nodes.add(tgt)
                    edges.append(
                        Edge(source=src, target=tgt, color="#30363d")
                    )

                if nodes:
                    config = Config(
                        width=800,
                        height=400,
                        directed=True,
                        physics=True,
                        hierarchical=False,
                    )
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info(
                        "No citation network data yet. "
                        "Ingest papers with PDFs to build the graph."
                    )

            except ImportError:
                _render_d3_citation_graph(result)

            except Exception as e:
                st.caption(f"Citation graph: {e}")


def _render_d3_citation_graph(result: dict):
    """Render an interactive D3.js force-directed citation graph.

    Uses a plain string template with string.replace() instead of f-strings
    so JavaScript variables (d, event, etc.) are NOT interpreted by Python.
    """
    import json as _json

    try:
        from src.storage.database import MARISDatabase

        db = MARISDatabase()
        citation_data = db.get_citation_graph()

        d3_nodes = []
        d3_links = []
        seen = set()

        for pid in result.get("papers_found", [])[:15]:
            paper = db.get_paper(pid)
            if paper:
                label = paper["title"][:50] + "..."
                auth = (
                    ", ".join(paper["authors"][:2])
                    if paper["authors"]
                    else "Unknown"
                )
            else:
                label = pid
                auth = "Unknown"
            if pid not in seen:
                d3_nodes.append(
                    {"id": pid, "label": label, "group": 1, "authors": auth}
                )
                seen.add(pid)

        for cite in citation_data:
            src = cite["source_paper_id"]
            tgt = cite["target_paper_id"]
            if src in seen or tgt in seen:
                for nid in (src, tgt):
                    if nid not in seen:
                        paper = db.get_paper(nid)
                        d3_nodes.append(
                            {
                                "id": nid,
                                "label": (
                                    paper["title"][:50] + "..."
                                    if paper
                                    else nid
                                ),
                                "group": 2,
                                "authors": (
                                    ", ".join(paper["authors"][:2])
                                    if paper and paper["authors"]
                                    else "Unknown"
                                ),
                            }
                        )
                        seen.add(nid)
                d3_links.append({"source": src, "target": tgt})

        if d3_nodes:
            # Plain string — NO f-string — so JS variables are safe
            _TPL = """<!DOCTYPE html>
<html><head>
<script src="https://d3js.org/d3.v6.min.js"></script>
<style>
body{background:#0d1117;margin:0;overflow:hidden;
     font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.node{stroke:#161b22;stroke-width:2;cursor:pointer}
.node:hover{stroke:#58a6ff;stroke-width:3}
.link{stroke:#30363d;stroke-opacity:.6;stroke-width:1.5}
.label{fill:#8b949e;font-size:10px;pointer-events:none;user-select:none}
.tip{position:absolute;background:rgba(22,27,34,.95);border:1px solid #30363d;
     padding:8px 12px;border-radius:6px;font-size:11px;color:#c9d1d9;
     pointer-events:none;opacity:0;transition:opacity .2s;max-width:280px;
     box-shadow:0 4px 12px rgba(0,0,0,.5);z-index:10}
.tip strong{color:#58a6ff}
</style></head><body>
<div class="tip" id="tip"></div>
<svg width="100%" height="400"></svg>
<script>
var data={nodes:__NODES__,links:__LINKS__};
var w=window.innerWidth||800,h=400;
var svg=d3.select("svg").attr("width",w).attr("height",h);

svg.append("defs").selectAll("marker").data(["c"]).enter().append("marker")
  .attr("id",function(d){return d}).attr("viewBox","0 -5 10 10")
  .attr("refX",18).attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto")
  .append("path").attr("fill","#8b949e").attr("d","M0,-5L10,0L0,5");

var sim=d3.forceSimulation(data.nodes)
  .force("link",d3.forceLink(data.links).id(function(d){return d.id}).distance(80))
  .force("charge",d3.forceManyBody().strength(-120))
  .force("center",d3.forceCenter(w/2,h/2))
  .force("x",d3.forceX(w/2).strength(.08))
  .force("y",d3.forceY(h/2).strength(.08));

var link=svg.append("g").selectAll("line").data(data.links).enter().append("line")
  .attr("class","link").attr("marker-end","url(#c)");

var tip=d3.select("#tip");

var node=svg.append("g").selectAll("circle").data(data.nodes).enter().append("circle")
  .attr("class","node")
  .attr("r",function(d){return d.group===1?8:5})
  .attr("fill",function(d){return d.group===1?"#58a6ff":"#30363d"})
  .call(d3.drag().on("start",ds).on("drag",dd).on("end",de))
  .on("mouseover",function(ev,d){
    tip.style("opacity",1)
       .html("<strong>"+d.label+"</strong><br><span style='color:#8b949e'>"+d.authors+"</span>");
  })
  .on("mousemove",function(ev){
    tip.style("left",(ev.pageX+12)+"px").style("top",(ev.pageY-12)+"px");
  })
  .on("mouseout",function(){tip.style("opacity",0)});

var lab=svg.append("g").selectAll("text").data(data.nodes).enter().append("text")
  .attr("class","label").attr("dx",10).attr("dy",4)
  .text(function(d){return d.id});

sim.on("tick",function(){
  link.attr("x1",function(d){return d.source.x}).attr("y1",function(d){return d.source.y})
      .attr("x2",function(d){return d.target.x}).attr("y2",function(d){return d.target.y});
  node.attr("cx",function(d){return d.x}).attr("cy",function(d){return d.y});
  lab.attr("x",function(d){return d.x}).attr("y",function(d){return d.y});
});

function ds(ev,d){if(!ev.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y}
function dd(ev,d){d.fx=ev.x;d.fy=ev.y}
function de(ev,d){if(!ev.active)sim.alphaTarget(0);d.fx=null;d.fy=null}
</script></body></html>"""

            html = _TPL.replace("__NODES__", _json.dumps(d3_nodes)).replace(
                "__LINKS__", _json.dumps(d3_links)
            )
            st.iframe(html, height=420)
        else:
            st.info(
                "No citation network data yet. "
                "Run a search to auto-ingest papers and build the graph."
            )
    except Exception as ex:
        st.caption(f"Citation graph: {ex}")


if __name__ == "__main__":
    main()
