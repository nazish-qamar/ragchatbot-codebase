# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Uses ChromaDB for vector storage, Anthropic Claude for AI generation, and a FastAPI backend serving a vanilla JS frontend.

## Running the Application

```bash
# Install dependencies
uv sync

# Run the server (from project root, use Git Bash on Windows)
./run.sh

# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=<key>`.

## Architecture

The backend runs from the `backend/` directory. All Python imports are relative to `backend/`, not the project root.

### Query Flow

User question → `app.py` (FastAPI) → `RAGSystem.query()` → `AIGenerator` sends query to Claude with a `search_course_content` tool → Claude decides whether to search or answer directly → if searching, `CourseSearchTool` queries ChromaDB → results sent back to Claude in a 2nd API call (without tools) → response returned with sources.

### Key Design Decisions

- **Tool-use pattern**: Claude autonomously decides when to search via Anthropic's tool-use API, rather than always retrieving. The 2nd Claude call after tool execution intentionally omits tools to prevent recursive searching.
- **Two ChromaDB collections**: `course_catalog` stores one entry per course (title as document, metadata as JSON) for fuzzy course name resolution. `course_content` stores text chunks for semantic search. Both use `all-MiniLM-L6-v2` embeddings.
- **Document format**: Course files in `docs/` follow a structured format — first 3 lines are metadata (`Course Title:`, `Course Link:`, `Course Instructor:`), then `Lesson N: Title` markers delimit lesson content.
- **Session management**: In-memory only (no persistence). Sessions track conversation history as formatted strings appended to the system prompt.
- **Static files**: FastAPI serves the `frontend/` directory at `/` with `html=True`, so `index.html` is the default.

### Component Responsibilities

- `rag_system.py` — Orchestrator that wires together all components. Entry point for queries.
- `document_processor.py` — Parses course files, extracts metadata, chunks text into ~800-char sentence-aware segments with 100-char overlap.
- `vector_store.py` — ChromaDB wrapper. Handles course name resolution (catalog → exact title) and filtered content search.
- `ai_generator.py` — Claude API client. Manages the tool-use loop (1st call with tools, execute tool, 2nd call with results).
- `search_tools.py` — Abstract `Tool` base class + `ToolManager` registry. `CourseSearchTool` bridges Claude's tool calls to `VectorStore.search()`.
- `session_manager.py` — In-memory conversation history per session, capped at `MAX_HISTORY` exchanges.
- `config.py` — Single `Config` dataclass. Loads `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`.

## Code Quality

The project uses `black` for code formatting. All Python code must be formatted before committing.

```bash
# Check formatting (CI-safe, exits non-zero on failure)
bash quality.sh

# Auto-format all code
bash quality.sh --fix
```

Configuration is in `pyproject.toml` under `[tool.black]`.

## Dependencies

Managed with `uv`. Key packages: `fastapi`, `uvicorn`, `chromadb`, `anthropic`, `sentence-transformers`, `python-dotenv`. Python 3.13+.

**Always use `uv`** to run the server and manage dependencies. Never use `pip` directly.
