import os
import streamlit as st
import research_agent as ra

st.set_page_config(page_title="Autonomous Research Agent", page_icon="🔎", layout="wide")
st.title("🔎 Autonomous Research & Intelligence Agent")

with st.sidebar:
    st.header("Settings")
    # >>> GROQ API KEY goes here (or in env var GROQ_API_KEY / .streamlit/secrets.toml) <<<
    try:
        default_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = os.environ.get("GROQ_API_KEY", "")
    key = st.text_input("Groq API key", value=default_key, type="password",
                        help="Free key: https://console.groq.com/keys")
    model = st.text_input("Model", value=ra.DEFAULT_MODEL)
    max_sources = st.slider("Max sources", 6, 30, 10)
    min_cred = st.slider("Min credibility to keep a source", 0, 80, 40)
    n_contra = st.slider("Contradictions to investigate", 0, 5, 2)
    expand = st.checkbox("Dynamic research expansion", True)
    fast = st.checkbox("Use small model for extraction (saves quota)", False,
                       help="Uses llama-3.1-8b-instant for bulk extraction; separate Groq quota, slightly lower extraction quality")

query = st.text_area("Research question", "How will quantum computing impact cybersecurity over the next 10 years?")
run = st.button("Run research", type="primary", disabled=not (key and query))

if run:
    status = st.status("Researching...", expanded=True)
    try:
        agent = ra.ResearchAgent(llm=ra.LLM(api_key=key, model=model), max_sources=max_sources,
                                 min_credibility=min_cred, max_contradictions=n_contra, expand_topics=expand,
                                 fast_model="llama-3.1-8b-instant" if fast else None,
                                 log_fn=lambda m: status.write(m))
        st.session_state["agent"] = agent
        st.session_state["session"] = agent.run(query)
        st.session_state["chat"] = []
        status.update(label="Done", state="complete")
    except Exception as e:
        status.update(label="Failed", state="error")
        st.error(str(e))

s = st.session_state.get("session")
if s:
    agent = st.session_state["agent"]
    tabs = st.tabs(["📄 Report", "⚖️ Contradictions", "🕸️ Evidence graph", "📚 Sources", "🧩 Plan", "💬 Interactive"])
    with tabs[0]:
        st.markdown(s.report)
        st.download_button("Download report (.md)", s.report, file_name="research_report.md")
    with tabs[1]:
        df = ra.contradiction_table(s)
        if df.empty:
            st.info("No material contradictions detected.")
        else:
            st.dataframe(df, use_container_width=True)
            for c in s.contradictions:
                st.subheader(c.get("topic"))
                a, b = st.columns(2)
                a.metric("Claim A support", c["support_a"]); a.write(c["claim_a"]["text"])
                b.metric("Claim B support", c["support_b"]); b.write(c["claim_b"]["text"])
                st.success(f"Resolution ({c['verdict']}, {c['confidence']}% confidence): {c['resolution']}")
    with tabs[2]:
        st.pyplot(ra.draw_graph(s))
    with tabs[3]:
        st.dataframe(ra.sources_table(s), use_container_width=True)
    with tabs[4]:
        st.json({"understanding": s.understanding, "plan": s.plan})
    with tabs[5]:
        st.caption("Ask follow-ups or 'expand: <section name>' — no need to restart the research.")
        for role, msg in st.session_state.get("chat", []):
            st.chat_message(role).markdown(msg)
        q = st.chat_input("e.g. Explain the RSA section further")
        if q:
            st.chat_message("user").markdown(q)
            with st.spinner("Thinking..."):
                ans = agent.expand_section(q.split(":", 1)[1].strip(), s) if q.lower().startswith("expand:") else agent.ask(q, s)
            st.chat_message("assistant").markdown(ans)
            st.session_state["chat"] += [("user", q), ("assistant", ans)]
