# Agentic RAG Document Assistant with LangChain

> Upload PDF, Markdown, or text files and ask grounded questions through a Streamlit chat interface powered by LangChain, OpenAI embeddings, and Deep Agents.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings%20%2B%20Chat-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-DE5FE9)](https://docs.astral.sh/uv/)

---

## What Is This?

This project is a learning-focused **agentic retrieval-augmented generation (RAG) application**. Instead of asking a language model to answer from memory, the app builds a temporary semantic index from the user's documents, retrieves the most relevant passages, delegates passage analysis to specialist subagents, and synthesizes a final answer.

The repository demonstrates two related workflows:

- **Streamlit application (`main.py`)** — upload your own PDF, TXT, or Markdown files, process them, and chat with the resulting knowledge base.
- **Fixed-corpus example (`agent.py`)** — run the same core ideas against the sample climate documents in `pdfs/` to study the pipeline as separate modules.

The project is intentionally compact so learners can trace the full path from file ingestion to a grounded response.

---

## Demo

![Document Q&A Streamlit application](<screenshots/Screenshot 2026-07-18 204134.png>)

The Streamlit workspace provides multi-file upload, explicit indexing, chat history, and document-grounded answers in a custom dark emerald interface.

---

## Features

- Upload multiple `.pdf`, `.txt`, and `.md` files.
- Extract text from PDFs with `pypdf`.
- Split documents into overlapping chunks for retrieval.
- Create embeddings with OpenAI's `text-embedding-3-large` model.
- Store and search vectors in LangChain's in-memory vector store.
- Retrieve up to four relevant chunks for each search.
- Save retrieved passages to the Deep Agents virtual filesystem.
- Delegate chunk analysis to specialist `chunk-analyst` subagents.
- Synthesize results through an OpenAI chat model configured in the code.
- Preserve chat messages during the active Streamlit session.
- Detect when the selected files change and need to be reprocessed.
- Clear the current conversation without rebuilding the interface.

---

## How It Works

1. **Ingest** — the user uploads one or more supported documents.
2. **Parse** — PDFs are processed with `PdfReader`; text and Markdown files are decoded as UTF-8.
3. **Chunk** — `RecursiveCharacterTextSplitter` creates 1,000-character chunks with 200 characters of overlap.
4. **Embed** — OpenAI embeddings convert each chunk into a vector representation.
5. **Index** — chunks are held in an `InMemoryVectorStore` for the current app session.
6. **Retrieve** — the agent calls `search_documentation` to find the most relevant chunks.
7. **Delegate** — retrieved chunks are written under `/retrieved/` and assigned to chunk-analysis subagents.
8. **Synthesize** — the coordinating agent combines the evidence into a final response.

```text
Uploaded PDF / TXT / Markdown files
                |
                v
        Text extraction
                |
                v
     Recursive text splitting
       (1,000 / 200 overlap)
                |
                v
       OpenAI embeddings
                |
                v
   InMemoryVectorStore index
                |
                v
       Semantic retrieval
                |
                v
  /retrieved/.../chunk_N.md
                |
                v
  Parallel chunk-analyst agents
                |
                v
       Grounded final answer
```

---

## Project Structure

```text
RAG_LangChain/
|
+-- main.py             # Primary Streamlit app and session-local RAG pipeline
+-- agent.py            # Fixed-corpus agent example and sample query
+-- filereader.py       # Reusable PDF text extraction helpers
+-- indexing.py         # Indexes the PDFs listed in DOC_PATHS
+-- tools.py            # Retrieval tool and Deep Agents state backend
+-- prompts.py          # RAG workflow and subagent instructions
+-- pdfs/               # Sample climate-change PDF corpus
+-- screenshots/        # Application screenshots used in this README
+-- pyproject.toml       # Project metadata and dependencies
+-- uv.lock              # Reproducible dependency lockfile
+-- .python-version      # Local Python version hint
`-- README.md
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Streamlit UI, uploaded-file parsing, chunking, indexing, agent creation, and chat state |
| `filereader.py` | Reads one or more PDFs and returns LangChain `Document` objects |
| `indexing.py` | Builds an in-memory index from the sample PDFs |
| `tools.py` | Exposes semantic search as an agent tool and stores retrieved chunks |
| `agent.py` | Configures the coordinator and chunk-analyst subagent for the sample corpus |
| `prompts.py` | Defines retrieval, delegation, synthesis, and prompt-injection safety instructions |

---

## Getting Started

### Prerequisites

| Requirement | Purpose |
|---|---|
| Python 3.12+ | Application runtime |
| `uv` | Dependency installation and command execution |
| OpenAI API key | Chat model and embedding access |

Check your installed versions:

```bash
python --version
uv --version
```

If `uv` is not installed, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) or install it with pip:

```bash
pip install uv
```

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG_LangChain.git
cd RAG_LangChain
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username.

### 2. Install dependencies

```bash
uv sync
```

This creates a local virtual environment and installs the versions resolved in `uv.lock`.

### 3. Configure the API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The repository's `.gitignore` excludes `.env`. Never commit a real API key.

### 4. Launch the Streamlit app

```bash
uv run streamlit run main.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### 5. Use the app

1. Upload one or more PDF, TXT, or Markdown files from the sidebar.
2. Select **Process documents** to extract, split, embed, and index their content.
3. Enter a question in the chat box.
4. Review the generated answer and continue with follow-up questions.
5. Use **Clear conversation** when you want to reset the visible chat history.

---

## Run the Fixed-Corpus Example

The repository also includes two sample PDFs under `pdfs/`. To run the modular example:

```bash
uv run python agent.py
```

This path imports `indexing.py`, indexes the files listed in its `DOC_PATHS`, and runs the example question defined near the bottom of `agent.py`.

To experiment with your own local corpus:

1. Add PDFs to `pdfs/`.
2. Update `DOC_PATHS` in `indexing.py`.
3. Change `example_query` in `agent.py`.
4. Run `uv run python agent.py` again.

For interactive use, the Streamlit workflow is the recommended entry point because it accepts files and questions without code changes.

---

## Key Learning Concepts

### Retrieval-augmented generation

RAG supplies the language model with relevant source passages at question time. This makes the response depend on the uploaded corpus rather than only on the model's pre-existing knowledge.

### Chunking and overlap

Large documents are divided into smaller units before embedding. The 200-character overlap helps preserve context when an idea crosses a chunk boundary.

### Semantic search

The query and document chunks are embedded into the same vector space. Similarity search can therefore retrieve conceptually related text even when the wording is not identical.

### Agentic retrieval

Retrieval is exposed as a tool instead of being called unconditionally by application code. The coordinating agent can search, assess the retrieved evidence, refine its query when necessary, and then synthesize an answer.

### Subagent delegation

Each retrieved passage can be assigned to a specialist chunk analyst. This keeps the coordinator focused on planning and synthesis while individual agents extract relevant facts from the evidence.

### Prompt-injection boundaries

The prompts explicitly instruct the agents to treat retrieved documents as reference data and ignore instructions embedded inside them. This is an important baseline defense when processing user-controlled content.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.12+ | Application runtime |
| UI | Streamlit | Upload and conversational interface |
| RAG framework | LangChain | Documents, splitting, vector search, tools, and model initialization |
| Agent framework | Deep Agents | Coordinator, subagents, task delegation, and virtual filesystem |
| LLM provider | OpenAI | Chat completion and vector embeddings |
| PDF parsing | pypdf | Text extraction from PDF files |
| Configuration | python-dotenv | Loads `OPENAI_API_KEY` from `.env` |
| Package management | uv | Environment and dependency management |

---

## Configuration

The main settings currently live in code and are easy to change while learning:

| Setting | Current value | Location |
|---|---:|---|
| Chat model | `openai:gpt-5.5` | `main.py`, `agent.py` |
| Embedding model | `text-embedding-3-large` | `main.py`, `indexing.py` |
| Chunk size | `1000` characters | `main.py`, `indexing.py` |
| Chunk overlap | `200` characters | `main.py`, `indexing.py` |
| Retrieved chunks | Up to `4` | `main.py`, `tools.py` |
| Concurrent analysts | `3` | `main.py`, `agent.py` |

Model availability depends on the models enabled for your OpenAI account. If necessary, replace the configured chat model with one available to your project.

---

## Design Decisions

| Choice | Why it is useful here |
|---|---|
| In-memory vector store | Keeps the demo simple and avoids requiring a database |
| Explicit **Process documents** action | Prevents expensive re-embedding on every Streamlit rerun |
| Content-based file-set hash | Warns users when uploaded files differ from the indexed set |
| Retrieved chunks in a virtual filesystem | Lets subagents read evidence by path instead of copying full passages into coordinator messages |
| One analyst role per chunk | Demonstrates delegation while keeping agent responsibilities focused |
| Separate fixed-corpus modules | Makes ingestion, indexing, retrieval, and orchestration easier to study independently |

---

## Limitations

- The vector index exists only in memory and is lost when the app session or process ends.
- Uploaded documents and chat history are not persisted to a database.
- PDF text extraction works best with text-based PDFs; scanned documents require OCR, which is not included.
- The app does not currently expose source passages or citations directly in the UI.
- Retrieval is limited to four chunks per search, which may be insufficient for broad questions or large corpora.
- Responses can still contain mistakes and should be checked against the original documents for high-stakes use cases.
- Processing and querying use paid OpenAI APIs and may incur costs.
- There is currently no automated test suite or user authentication.

---

## Ideas for Further Development

- Add persistent vector storage with Chroma, FAISS, pgvector, or another vector database.
- Display source filenames and retrieved passages alongside each answer.
- Add page-level metadata so PDF answers can cite page numbers.
- Support DOCX, HTML, CSV, and web-page ingestion.
- Add OCR for scanned PDFs and images.
- Stream agent progress and tokens to the interface.
- Cache embeddings for previously processed files.
- Add model and retrieval settings to the sidebar.
- Evaluate retrieval quality and groundedness with a small benchmark dataset.
- Add unit tests for parsing, hashing, retrieval, and response extraction.
- Add authentication and durable chat sessions for multi-user deployment.

---

## Security and Privacy

- Do not commit `.env` or any real credentials.
- Uploaded content is sent to the configured embedding and chat-model provider during processing and question answering.
- Review your provider's data-handling policy before uploading confidential or regulated documents.
- Treat generated answers as assistance, not authoritative advice.
- For production use, add file-size limits, malware scanning, access controls, logging safeguards, and stronger prompt-injection defenses.

---

## Contributing

Learning-focused improvements are welcome. A useful contribution might add tests, another document loader, persistent storage, better citations, or clearer educational documentation.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request describing what changed and why.

---

## License

MIT liscence 
