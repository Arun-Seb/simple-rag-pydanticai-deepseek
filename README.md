# Simple RAG System — PydanticAI + DeepSeek

A small but complete **Retrieval-Augmented Generation (RAG)** application that answers
questions about your own documents. It reads a PDF, finds the passages most relevant
to a question, and asks a large language model (**DeepSeek**) to answer *using only
those passages* — so answers are grounded in your source material instead of the
model's general memory. The final answer comes back as a **validated, structured
object** (answer + confidence + sources) and is served through a small **web app**.

The project is written to be read. Every stage is a few lines of plain Python, and
this README explains not just *what* each stage does but *why* it's built that way and
*what the alternatives are* — so it works equally as a portfolio piece and as a
learn-RAG-from-scratch tutorial.

> **Tech stack:** Python · PydanticAI (agent + structured output) · DeepSeek (LLM API)
> · Sentence Transformers (embeddings) · ChromaDB (vector database) · Streamlit (UI)

---

## What is RAG, in one paragraph?

A language model only knows what it saw during training. It has never read *your*
PDF, and if you ask about it, it will either say it doesn't know or — worse —
confidently make something up. RAG fixes this by adding a **retrieval** step in front
of the model: we store your documents in a searchable database, fetch the few passages
most relevant to each question, and hand those passages to the model as *context*. The
model then answers from evidence you supplied. This is how you get a chatbot that
answers accurately about private documents, without retraining anything.

---

## Architecture

```
                 ┌─────────────────┐
                 │   Documents     │
                 │ PDF / TXT / MD  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Text Chunking  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Embeddings    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   ChromaDB      │
                 │ Vector Database │
                 └────────┬────────┘
                          │
             User Question│
                          ▼
                 ┌─────────────────┐
                 │    Retrieval    │
                 │  Top-K chunks   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   PydanticAI    │
                 │     Agent       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    DeepSeek     │
                 │      LLM        │
                 └────────┬────────┘
                          │
                          ▼
                       Answer
```

The system has two independent halves:

- **Ingestion (`ingest.py`)** — the "write" side. Run **once** per document set. It
  turns documents into a searchable vector database on disk (`chroma_db/`).
- **Query (`rag.py`, `app.py`)** — the "read" side. Runs every time you ask a
  question. It never re-processes the documents; it just reads the database built by
  ingestion.

This separation is deliberate: embedding a large document is the slow part, so you pay
that cost once and then answer thousands of questions cheaply.

---

## Project structure

```
simple-rag-pydanticai-deepseek/
├── ingest.py          # Stage 1–4: PDF → chunks → embeddings → ChromaDB (run once)
├── rag.py             # Stage 5–9: retrieve → ground → answer → cite → structure
├── app.py             # Streamlit web UI
├── requirements.txt
├── .env.example       # template for your DeepSeek API key
├── .gitignore
└── data/              # put your PDF(s) here
```

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/Arun-Seb/simple-rag-pydanticai-deepseek.git
cd simple-rag-pydanticai-deepseek

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your DeepSeek API key
cp .env.example .env               # Windows: copy .env.example .env
#   then edit .env and paste your key from https://platform.deepseek.com

# 5. Build the knowledge base from a PDF (run once)
python ingest.py data/your_document.pdf

# 6a. Ask from the command line
python rag.py "What is this document about?"

# 6b. …or launch the web app
streamlit run app.py
```

> **Note on the embedding model:** the first run downloads `all-MiniLM-L6-v2`
> (~80 MB) and caches it. Embeddings then run **locally on your CPU** — no GPU and no
> API cost. Only the final answer step calls the DeepSeek API.

---

## How it works — every stage explained

### Stage 1 — Load the document

**What:** read the PDF and extract its text into one big string (`load_pdf` in
`ingest.py`, using `pypdf`).

**Why this method:** `pypdf` is pure-Python, dependency-light, and reads the text
layer of a PDF directly — perfect for documents that were exported from Word or
LaTeX. We guard each page with `page.extract_text() or ""` because image-only pages
return `None`.

**Limitation & alternatives:** `pypdf` only reads *selectable* text. If your PDF is a
**scanned image**, extraction returns almost nothing and you need **OCR** (e.g.
`pytesseract`, `ocrmypdf`, or a cloud OCR service) to convert images to text first.
For richer layout handling (tables, columns, headings) consider `pymupdf` (fast),
`unstructured` (structure-aware), or `docling`/`marker` (modern PDF-to-markdown).

---

### Stage 2 — Chunk the text

**What:** split the text into overlapping windows of ~500 characters with ~50
characters of overlap (`chunk_text`).

**Why chunk at all:** two reasons. First, retrieval is more precise when each stored
unit is a small, focused idea — a whole document is too coarse to match against a
specific question. Second, LLMs have a context limit and cost scales with tokens, so
you want to send only the relevant slices, not the entire file.

**Why overlap:** a fixed-size cut can slice a sentence — or a key number — in half. The
overlap repeats the tail of one chunk at the head of the next, so the split idea
survives intact in at least one chunk.

**Why 500/50 specifically:** it's a sensible default that keeps chunks small enough for
precise retrieval while staying large enough to hold a coherent thought. Tune it to
your documents: larger chunks (800–1000) preserve more context per chunk but retrieve
less precisely; smaller chunks (200–300) are precise but can lose surrounding context.

**Alternatives (this is the biggest quality lever in RAG):**
- **Sentence / paragraph splitting** — break on natural boundaries so chunks never cut
  mid-sentence (e.g. LangChain's `RecursiveCharacterTextSplitter`).
- **Token-based chunking** — size by model tokens rather than characters for tighter
  cost control.
- **Semantic chunking** — group sentences by meaning so each chunk is one topic.
- **Structure-aware chunking** — split on headings/sections for structured docs.

We use the simple character method because it's transparent and easy to reason about;
"better chunking" is the first upgrade to try if answer quality disappoints.

---

### Stage 3 — Generate embeddings

**What:** convert each chunk into a 384-number vector that captures its meaning, using
the `all-MiniLM-L6-v2` Sentence Transformer.

**Why embeddings:** they turn "search by meaning" into simple geometry. Texts with
similar meaning get vectors that sit close together, so we can find relevant chunks
even when the question and the document use *different words* for the same idea
(e.g. "how much time off" vs "annual leave"). Keyword search can't do that.

**Why this model:** `all-MiniLM-L6-v2` is the popular default for good reason — it's
small (~80 MB), fast on CPU, free, and runs fully **locally**, so no data leaves your
machine and there's no per-call cost. `normalize_embeddings=True` scales every vector
to unit length, which makes similarity comparisons cleaner downstream.

**Alternatives:**
- **Stronger local models** — `bge-base-en-v1.5`, `bge-large`, `gte-large`, or
  `e5-large` for higher retrieval accuracy at the cost of size/speed.
- **Embedding APIs** — OpenAI `text-embedding-3-small/large`, Cohere, Voyage — often
  higher quality, but they cost money and send your text to a third party.
- **Multilingual models** — `paraphrase-multilingual-MiniLM` if your docs aren't
  English.

**Important rule:** whatever model you embed *documents* with, you must embed
*questions* with the **same** model — otherwise the vectors live in different spaces
and "closeness" is meaningless. If you change the embedding model, re-run ingestion.

---

### Stage 4 — Store vectors in ChromaDB

**What:** save each chunk's `id`, `text`, `embedding`, and `metadata` (the source
filename) into a persistent ChromaDB collection on disk.

**Why a vector database:** at query time we need the *nearest* vectors to the question
vector, fast. A vector DB indexes the vectors so nearest-neighbour search stays quick
as the collection grows, and it keeps the text and metadata alongside each vector so a
search returns everything we need in one call.

**Why ChromaDB:** it's the simplest way to get started — `pip install chromadb`, point
it at a folder, done. No server to run, no cloud account, and `PersistentClient` writes
to disk so the database survives between runs. Ideal for local and small-to-mid
projects. We store the source filename in `metadata` now because that's what powers
trustworthy **citations** later.

**Alternatives:**
- **FAISS** — a raw, very fast similarity-search library (no metadata store; you manage
  that yourself). Great when you want maximum speed and control.
- **Qdrant / Weaviate / Milvus** — full-featured vector databases for production, with
  filtering, scaling, and hybrid search.
- **pgvector** — add vector search to PostgreSQL, so your vectors live next to your
  relational data.
- **Pinecone** — a managed cloud vector database (no infra to run).

ChromaDB is the right choice here precisely because the project is meant to be simple
and local; the others earn their complexity at larger scale.

---

### Stage 5 — Retrieve relevant chunks

**What:** embed the user's question with the same model, then ask ChromaDB for the
`top_k` closest chunks (`retrieve` in `rag.py`; default `top_k=6`).

**Why top-k, and why 6:** you want enough context that the answer is likely present,
but not so much that you drown the model in noise and pay for wasted tokens. Small `k`
(2–3) is cheap but risks missing the relevant passage; large `k` (10+) is thorough but
noisier and pricier. Six is a good middle ground for short documents; tune it.

**A practical retrieval lesson:** results improve a lot when your question uses the
*document's own vocabulary*. Asking "what accuracy did the model achieve?" retrieves
results-section chunks better than a vague "what did it find?", because the words in
the query steer the semantic match.

**Alternatives / upgrades:**
- **Hybrid search** — combine vector similarity with keyword (BM25) matching to catch
  exact terms, names, and numbers that pure semantic search can miss.
- **Reranking** — retrieve a wide net (e.g. top 20), then use a cross-encoder reranker
  (e.g. `bge-reranker`, Cohere Rerank) to reorder and keep the best 3–5. This is often
  the single highest-impact quality upgrade.
- **Metadata filtering** — restrict search to a document, section, or date range.
- **Query rewriting** — have an LLM rephrase or expand the question before searching.

---

### Stage 6 — Ground the answer with PydanticAI + DeepSeek

**What:** stitch the retrieved chunks into a `CONTEXT` block, append the `QUESTION`,
and send it to a DeepSeek model wrapped in a PydanticAI `Agent` whose **system prompt**
instructs it to answer *only* from the context and to admit when the answer isn't
there.

**Why the system prompt matters:** that one instruction is the core of hallucination
control. Because the model is told to rely on the supplied evidence and to say "I don't
have enough information" otherwise, it stops inventing answers. In testing, asking about
a topic absent from the document correctly produces a low-confidence "not in the
context" response rather than a confident fabrication.

**Why PydanticAI:** it's a thin, typed agent layer over the model. It gives a clean
`Agent` abstraction, is model-agnostic (swap DeepSeek for another provider by changing
one line), and — most importantly — enforces **structured output** (Stage 8). You could
call the DeepSeek API directly, but you'd hand-roll everything PydanticAI gives you.

**Why DeepSeek:** strong quality at a very low price, with an OpenAI-compatible API so
it plugs straight into standard tooling. We disable "thinking mode" via
`extra_body={"thinking": {"type": "disabled"}}` because it's required for the forced
tool call that powers structured output, and because grounded RAG doesn't need the
model's chain-of-thought — turning it off is also faster and cheaper.

**Alternatives:** any OpenAI-compatible model works — OpenAI GPT, Anthropic Claude,
Google Gemini, or a **local** model via Ollama (fully offline, no API cost, at the
price of speed/quality on a laptop). The rest of the pipeline is unchanged.

---

### Stage 7 — Add citations

**What:** return the source filename(s) of the retrieved chunks alongside every answer.

**Why from retrieval, not the model:** we already know exactly which chunks (and
therefore which files) we fed the model, so those citations are **guaranteed accurate**.
If you instead asked the LLM to name its own sources, it could hallucinate them.
Trustworthy citations come from *what you retrieved*, never from *what the model
claims*. With multiple documents ingested, each answer correctly cites whichever files
the evidence actually came from.

---

### Stage 8 — Structured, validated output

**What:** instead of free text, return a typed `Answer` object with an `answer` string
and a `confidence` float constrained to 0–1 (`ask_structured` in `rag.py`).

**Why this is the payoff of PydanticAI:** you declare the exact shape you want as a
Pydantic model, pass it as `output_type=Answer`, and PydanticAI makes the model return
data matching that schema *and validates it*. `answer.confidence` is a real number you
can threshold on (`if confidence < 0.5: flag as unreliable`) — no fragile text parsing,
and a guarantee the output matches your schema or raises an error. This is what turns an
LLM from a text generator into a reliable component in a larger system.

**Alternatives:** you can extend the schema with anything useful — a list of extracted
`sources`, a boolean `answer_found`, key entities, categories, or a full nested object.
The same mechanism (schema + validation) scales to complex structured extraction tasks.

---

### Stage 9 — Web app

**What:** a small Streamlit UI (`app.py`) with a text box that shows the answer,
confidence, and sources.

**Why Streamlit:** it turns a Python script into a web app with no HTML or JavaScript —
ideal for demos and internal tools. Note the app is intentionally thin: all logic lives
in `rag.py`, and it simply *reads* the database built by ingestion, so it never
re-embeds the document.

**Alternatives:** **Gradio** (similar, ML-demo friendly), or **FastAPI** to expose the
RAG system as a JSON API that any frontend or service can call — the natural choice for
production.

> **One implementation note worth knowing:** in a plain script (like `app.py`) the
> model is called with `agent.run_sync(...)`. In a Jupyter notebook you'd use
> `await agent.run(...)` instead, because a notebook already runs an async event loop
> and `run_sync` would clash with it.

---

## Where to take it next

Roughly in order of impact:

1. **Ingest multiple documents** — loop ingestion over a folder; citations already
   track each file.
2. **Reranking** — retrieve wide, rerank, keep the best few. Usually the biggest
   quality jump.
3. **Better chunking** — sentence/semantic splitting instead of fixed windows.
4. **Hybrid search** — add keyword matching for names, IDs, and exact numbers.
5. **Evaluation** — measure faithfulness and answer-relevancy (e.g. with RAGAS or
   DeepEval) so you can prove quality, not just eyeball it.
6. **Show the exact snippet** used, not just the filename, so users can verify.
7. **Agentic RAG** — expose retrieval as a tool the agent calls on demand, enabling
   multi-step reasoning and follow-up searches.

---

## License

MIT — free to use, modify, and learn from.
