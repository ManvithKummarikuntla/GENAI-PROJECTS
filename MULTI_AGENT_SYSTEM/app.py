import time
import streamlit as st
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


# ---------- helpers ----------
def to_text(output) -> str:
    """Chains/agents can return a str, an AIMessage, or a list of content blocks."""
    if hasattr(output, "content"):
        output = output.content
    if isinstance(output, list):
        output = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in output
        )
    return str(output)


# ---------- page setup ----------
st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research System")
st.caption("Search Agent → Reader Agent → Writer → Critic")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Search Agent** finds reliable sources\n"
        "2. **Reader Agent** scrapes the most relevant URL\n"
        "3. **Writer** drafts the research report\n"
        "4. **Critic** reviews the report and gives feedback"
    )
    show_raw = st.checkbox("Show intermediate results", value=True)
    if st.button("Clear results"):
        st.session_state.pop("state", None)
        st.rerun()

topic = st.text_input("Enter a research topic", placeholder="e.g. Latest advances in solid-state batteries")
run = st.button("🚀 Run Research", type="primary", disabled=not topic.strip())


# ---------- pipeline ----------
def run_pipeline_ui(topic: str) -> dict:
    state = {}
    start = time.time()
    progress = st.progress(0, text="Starting...")

    # Step 1 - Search
    with st.status("Step 1 – Search agent is working...", expanded=False) as s:
        search_agent = build_search_agent()
        result = search_agent.invoke(
            {"messages": [("user", f"Find reliable and detailed information about: {topic}")]}
        )
        state["search_results"] = to_text(result["messages"][-1])
        s.update(label="Step 1 – Search complete ✅", state="complete")
    progress.progress(25, text="Search done")

    # Step 2 - Reader
    with st.status("Step 2 – Reader agent is scraping...", expanded=False) as s:
        reader_agent = build_reader_agent()
        result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scrapped_content"] = to_text(result["messages"][-1])
        s.update(label="Step 2 – Reading complete ✅", state="complete")
    progress.progress(50, text="Reading done")

    # Step 3 - Writer
    with st.status("Step 3 – Writer is drafting the report...", expanded=False) as s:
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED RESULTS:\n{state['scrapped_content']}"
        )
        state["report"] = to_text(writer_chain.invoke({"topic": topic, "research": research_combined}))
        s.update(label="Step 3 – Report written ✅", state="complete")
    progress.progress(75, text="Report written")

    # Step 4 - Critic
    with st.status("Step 4 – Critic is reviewing...", expanded=False) as s:
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        s.update(label="Step 4 – Review complete ✅", state="complete")
    progress.progress(100, text=f"Done in {time.time() - start:.1f}s")

    state["topic"] = topic
    return state


if run:
    try:
        st.session_state["state"] = run_pipeline_ui(topic.strip())
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.exception(e)


# ---------- results ----------
state = st.session_state.get("state")
if state:
    st.divider()
    st.subheader(f"Results for: {state['topic']}")

    tabs = ["📄 Final Report", "🧐 Critic Feedback"]
    if show_raw:
        tabs += ["🔍 Search Results", "📖 Scraped Content"]
    t = st.tabs(tabs)

    with t[0]:
        st.markdown(state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=f"# {state['topic']}\n\n{state['report']}\n\n---\n## Critic Feedback\n\n{state['feedback']}",
            file_name=f"{state['topic'][:40].replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    with t[1]:
        st.markdown(state["feedback"])
    if show_raw:
        with t[2]:
            st.markdown(state["search_results"])
        with t[3]:
            st.markdown(state["scrapped_content"])