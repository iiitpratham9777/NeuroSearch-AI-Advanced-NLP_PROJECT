import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

from src.pipeline import NeuroSearchPipeline
from src.analytics import analyze_corpus

st.set_page_config(page_title="NeuroSearch AI", page_icon="🧠", layout="wide")
st.title("🧠 NeuroSearch AI")
st.caption("Advanced NLP Document Intelligence • Semantic Search • Grounded Q&A • Corpus Analytics")

@st.cache_resource
def get_pipeline():
    return NeuroSearchPipeline()

pipeline = get_pipeline()

with st.sidebar:
    st.header("Workspace")
    uploaded = st.file_uploader("Upload PDF, TXT, or MD documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (words)", 100, 500, 220, 20)
    overlap = st.slider("Chunk overlap (words)", 0, 100, 40, 10)
    top_k = st.slider("Retrieved passages", 1, 10, 5)
    st.markdown("---")
    st.caption("Privacy note: files are processed by this app runtime. Avoid uploading sensitive documents to untrusted deployments.")

if uploaded:
    docs = []
    for f in uploaded:
        raw = f.read()
        docs.append({"name": f.name, "bytes": raw})
    if st.button("📚 Index documents", type="primary", use_container_width=True):
        with st.spinner("Extracting, chunking, and embedding documents…"):
            try:
                pipeline.ingest(docs, chunk_words=chunk_size, overlap_words=overlap)
                st.session_state["indexed"] = True
                st.success(f"Indexed {len(docs)} document(s), {len(pipeline.chunks)} passages.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

if st.session_state.get("indexed") and pipeline.chunks:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", len(pipeline.documents))
    m2.metric("Passages", len(pipeline.chunks))
    m3.metric("Embedding model", "MiniLM")
    m4.metric("Search mode", "Semantic")

    tab_qa, tab_search, tab_analytics, tab_about = st.tabs(["💬 Ask your documents", "🔎 Semantic search", "📊 NLP analytics", "ℹ️ About"])

    with tab_qa:
        st.subheader("Grounded question answering")
        question = st.text_input("Ask a question about your uploaded documents", placeholder="What are the key findings?")
        if st.button("Generate answer", type="primary", disabled=not question.strip()):
            with st.spinner("Retrieving evidence and composing a grounded response…"):
                result = pipeline.answer(question, top_k=top_k)
            st.markdown("### Answer")
            st.write(result["answer"])
            st.caption(f"Evidence coverage: {result['coverage']:.0%} • Retrieved {len(result['sources'])} passage(s)")
            st.markdown("### Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"[{i}] {source['document']} • relevance {source['score']:.3f}"):
                    st.write(source["text"])
            if result.get("warning"):
                st.warning(result["warning"])

    with tab_search:
        st.subheader("Meaning-based passage retrieval")
        query = st.text_input("Search query", key="search_query", placeholder="e.g., risks and limitations of the proposed method")
        if st.button("Search", disabled=not query.strip()):
            results = pipeline.search(query, top_k=top_k)
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['document']}** · score `{r['score']:.3f}`")
                st.write(r["text"])
                st.divider()

    with tab_analytics:
        st.subheader("Corpus-level NLP analytics")
        report = analyze_corpus(pipeline.documents)
        a, b, c = st.columns(3)
        a.metric("Total words", f"{report['total_words']:,}")
        b.metric("Unique terms", f"{report['unique_terms']:,}")
        c.metric("Average document length", f"{report['avg_words']:.0f} words")
        if report["top_terms"]:
            df = pd.DataFrame(report["top_terms"], columns=["Term", "Frequency"])
            fig = px.bar(df, x="Term", y="Frequency", title="Most frequent terms")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Document inventory")
        st.dataframe(pd.DataFrame(report["documents"]), use_container_width=True, hide_index=True)
        st.info("Frequency charts are descriptive, not a substitute for domain-specific evaluation or topic validation.")

    with tab_about:
        st.markdown("""
        **NeuroSearch AI** is a modular portfolio project demonstrating:
        - PDF/text ingestion and robust chunking
        - Sentence-transformer embeddings with vector similarity retrieval
        - Evidence-grounded extractive answers with source attribution
        - Corpus statistics and term-frequency analytics
        - Testable, modular Python architecture

        The default answer engine deliberately stays extractive: it does not invent facts when evidence is weak.
        A production deployment can add a hosted/local generative LLM behind a verified evidence-only prompt.
        """)
else:
    st.info("Upload one or more PDF, TXT, or Markdown files in the sidebar, then click **Index documents**.")
    st.markdown("### Architecture")
    st.code("Documents → Text extraction → Chunking → Sentence embeddings → FAISS vector index → Retrieval → Grounded answer + citations", language="text")
