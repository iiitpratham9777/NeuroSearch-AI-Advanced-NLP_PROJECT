# NeuroSearch AI — Intelligent Document Intelligence Platform

A portfolio-grade NLP application for semantic document search, evidence-grounded question answering, and corpus analytics.

## Features
- PDF, TXT, and Markdown ingestion
- Word-based chunking with configurable overlap
- Sentence Transformer embeddings (`all-MiniLM-L6-v2`)
- FAISS inner-product retrieval on normalized embeddings (cosine similarity)
- Grounded extractive Q&A with source passages and low-relevance abstention
- Corpus statistics and term-frequency visualizations
- Modular code and unit tests

## Quick start
Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first indexing run downloads the embedding model from Hugging Face, so internet access is required once unless the model is already cached.

## Usage
1. Run the app.
2. Upload PDF, TXT, or MD files in the sidebar.
3. Choose chunk size and overlap, then click **Index documents**.
4. Ask a question or run semantic search.
5. Inspect retrieved passages before relying on an answer.

## Architecture
`Files → Text extraction → Chunking → Embeddings → FAISS → Retrieval → Evidence-grounded answer + source passages`

## Evaluation plan (recommended for a serious portfolio)
Create a labeled test set with query, relevant document/chunk IDs, and expected answer facts. Report:
- Retrieval: Recall@K, Precision@K, MRR, nDCG@K
- Answering: factual correctness, evidence support, abstention quality
- System: indexing time, query latency, memory footprint
- Robustness: scanned PDFs, empty files, irrelevant questions, long documents

Do not claim model accuracy or performance metrics until measured on a documented test set.

## Design decisions
- Extractive QA is the default. This avoids presenting a generative answer as verified when it is not.
- FAISS is optional at runtime; a NumPy cosine-similarity fallback is included.
- Uploaded files are held in the app session/runtime and are not persisted by this starter implementation.

## Limitations and next steps
- Scanned PDFs need OCR (e.g., Tesseract) before extraction.
- Chunk-level source references currently identify document and passage; page-level PDF citations can be added by retaining page metadata during extraction.
- Production deployments should add authentication, upload limits, persistent vector storage, monitoring, and privacy controls.
- Optional upgrades: reranking with a cross-encoder, hybrid BM25 + vector retrieval, multilingual embeddings, OCR, and a locally hosted or API-based LLM with strict evidence-only prompting.

## Tests
```bash
pytest -q
```

## Suggested resume description (only after you have run and verified it)
**NeuroSearch AI — Intelligent Document Intelligence Platform**
- Built a document NLP pipeline with Sentence Transformers and FAISS for semantic retrieval over PDF and text corpora.
- Implemented evidence-grounded extractive question answering with source passage attribution and low-relevance abstention.
- Developed an interactive Streamlit dashboard for document ingestion, semantic search, and corpus-level text analytics.
