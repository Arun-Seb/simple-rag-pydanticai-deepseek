"""
rag.py — The query side of the RAG system.

Flow:  question  ->  embed  ->  retrieve top-k chunks from ChromaDB
       ->  build a context prompt  ->  PydanticAI + DeepSeek  ->  grounded answer

Exposes two functions:
    ask(question)             -> (answer_text, sources)
    ask_structured(question)  -> (Answer object with confidence, sources)

Both use `run_sync`, which is correct for a plain Python script. (In a Jupyter
notebook you would use `await agent.run(...)` instead, because the notebook is
already running an async event loop.)
"""
from __future__ import annotations

import os

import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.deepseek import DeepSeekProvider
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEEPSEEK_MODEL = "deepseek-flash"
TOP_K = 6

# --- DeepSeek model (via PydanticAI's OpenAI-compatible provider) ---
_model = OpenAIChatModel(
    DEEPSEEK_MODEL,
    provider=DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"]),
)

# Thinking mode is disabled for two reasons: it is required for forced tool use
# (structured output), and for grounded RAG we don't need the model's
# chain-of-thought — disabling it is also faster and cheaper.
_settings = OpenAIChatModelSettings(extra_body={"thinking": {"type": "disabled"}})

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided. If the answer is not in the context, say you don't have enough "
    "information — do not guess."
)

_text_agent = Agent(_model, model_settings=_settings, system_prompt=_SYSTEM_PROMPT)


class Answer(BaseModel):
    """Structured, validated output. `confidence` is guaranteed to be 0..1."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


_structured_agent = Agent(
    _model,
    output_type=Answer,
    model_settings=_settings,
    system_prompt=(
        "Answer using ONLY the provided context. Set confidence between 0 and 1 "
        "based on how well the context supports your answer. If the context does "
        "not contain the answer, say so and use a low confidence."
    ),
)

# --- embedder + existing ChromaDB (read-only here) ---
_embedder = SentenceTransformer(EMBEDDING_MODEL)
_collection = chromadb.PersistentClient(path=CHROMA_PATH).get_or_create_collection(COLLECTION_NAME)


def retrieve(question: str, top_k: int = TOP_K):
    """Embed the question and return the most similar chunks + their sources."""
    query_vector = _embedder.encode(question, normalize_embeddings=True).tolist()
    results = _collection.query(query_embeddings=[query_vector], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def _build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def ask(question: str, top_k: int = TOP_K):
    """Plain-text grounded answer + list of source files."""
    chunks, sources = retrieve(question, top_k)
    result = _text_agent.run_sync(_build_prompt(question, chunks))
    return result.output, sorted(set(sources))


def ask_structured(question: str, top_k: int = TOP_K):
    """Structured answer (answer + confidence) + list of source files."""
    chunks, sources = retrieve(question, top_k)
    result = _structured_agent.run_sync(_build_prompt(question, chunks))
    return result.output, sorted(set(sources))


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this document about?"
    answer, sources = ask_structured(q)
    print(answer.answer)
    print(f"\nConfidence: {answer.confidence:.0%}")
    print("Sources:", ", ".join(sources))
