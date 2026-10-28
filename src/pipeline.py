from __future__ import annotations
from io import BytesIO
from typing import List, Dict
import re
import numpy as np

try:
    import fitz
except ImportError:
    fitz = None

class NeuroSearchPipeline:
    """Document ingestion, semantic retrieval, and evidence-grounded QA."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self._faiss = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _extract(self, name: str, data: bytes) -> str:
        suffix = name.lower().rsplit(".", 1)[-1] if "." in name else "txt"
        if suffix == "pdf":
            if fitz is None:
                raise RuntimeError("PDF support requires PyMuPDF. Install requirements.txt.")
            with fitz.open(stream=data, filetype="pdf") as pdf:
                return "\n".join(page.get_text("text") for page in pdf)
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _chunk(text: str, words: int, overlap: int) -> List[str]:
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return []
        step = max(1, words - overlap)
        chunks = []
        for start in range(0, len(tokens), step):
            part = " ".join(tokens[start:start + words]).strip()
            if part:
                chunks.append(part)
            if start + words >= len(tokens):
                break
        return chunks

    def ingest(self, files: List[Dict], chunk_words=220, overlap_words=40):
        if overlap_words >= chunk_words:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        documents, chunks = [], []
        for item in files:
            text = self._extract(item["name"], item["bytes"])
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if not text:
                continue
            documents.append({"name": item["name"], "text": text})
            for idx, chunk in enumerate(self._chunk(text, chunk_words, overlap_words)):
                chunks.append({"document": item["name"], "chunk_id": idx, "text": chunk})
        if not chunks:
            raise ValueError("No readable text found in uploaded documents.")
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.documents, self.chunks = documents, chunks
        self.embeddings = np.asarray(vectors, dtype="float32")
        try:
            import faiss
            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self._faiss = index
        except ImportError:
            self._faiss = None

    def search(self, query: str, top_k=5):
        if not self.chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, min(top_k, len(self.chunks)))
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            scores = (self.embeddings @ q[0]).tolist()
            pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in pairs:
            if idx < 0:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results

    def answer(self, question: str, top_k=5):
        sources = self.search(question, top_k=top_k)
        if not sources:
            return {"answer": "No indexed evidence is available.", "sources": [], "coverage": 0.0,
                    "warning": "Upload and index documents first."}
        best = sources[0]["score"]
        if best < 0.20:
            return {"answer": "I could not find sufficiently relevant evidence in the indexed documents. Try rephrasing the question or upload a more relevant source.",
                    "sources": sources, "coverage": max(0.0, min(1.0, best)), 
                    "warning": "Low retrieval similarity. The system abstained rather than guessing."}
        # Extractive fallback: show the strongest evidence passages instead of fabricating a generative answer.
        selected = [s for s in sources if s["score"] >= max(0.20, best - 0.12)][:3]
        answer = "\n\n".join(f"**Evidence {i}:** {s['text']}" for i, s in enumerate(selected, 1))
        return {"answer": answer, "sources": sources, "coverage": max(0.0, min(1.0, best)),
                "warning": "This is evidence-first extractive QA. Read the cited passages before treating them as a conclusion."}
