# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `./run.sh` (requires `chmod +x run.sh` first time)
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Web interface**: http://localhost:8000
- **API docs**: http://localhost:8000/docs

### Setup Commands
- **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh` (if not already installed)
- **Install dependencies**: `uv sync`
- **Environment setup**: Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`

### Python Package Management
- **IMPORTANT**: Always use `uv` for all dependency management (never use pip directly)
- Add dependencies: `uv add package_name`
- Add dev dependencies: `uv add package_name --group dev`
- Remove dependencies: `uv remove package_name`
- Install/sync dependencies: `uv sync`
- Run Python files: `uv run python script.py` (always use `uv run`, never `python` directly)
- Dependencies defined in `pyproject.toml`
- Lock file is `uv.lock`

### Code Quality Tools
- **Format code**: `./scripts/format-code.sh` or `uv run black . && uv run isort .`
- **Check quality**: `./scripts/quality-check.sh` (runs all quality checks)
- **Individual tools**:
  - Black formatting: `uv run black .` (format) or `uv run black --check .` (check only)
  - Import sorting: `uv run isort .` (format) or `uv run isort --check-only .` (check only)
  - Linting: `uv run flake8 .`
  - Type checking: `uv run mypy backend/ main.py`
- **IMPORTANT**: Always run quality checks before committing code

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system for querying course materials using semantic search and AI responses.

### Core Architecture Pattern
The system follows a modular RAG architecture with clear separation of concerns:

- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB interface for semantic search using sentence-transformers
- **DocumentProcessor** (`backend/document_processor.py`): Handles text chunking and course document parsing
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude API integration
- **SessionManager** (`backend/session_manager.py`): Conversation history management
- **ToolManager/CourseSearchTool** (`backend/search_tools.py`): Search functionality abstraction

### Data Flow
1. Course documents in `docs/` are processed into chunks via DocumentProcessor
2. Chunks are embedded and stored in ChromaDB via VectorStore 
3. User queries are semantically searched against the vector store
4. Relevant chunks are passed to Claude for RAG response generation
5. SessionManager maintains conversation context

### Key Configuration
- **Chunking**: 800 char chunks with 100 char overlap (configurable in `config.py`)
- **Embedding model**: all-MiniLM-L6-v2
- **Claude model**: claude-sonnet-4-20250514
- **Vector DB**: ChromaDB stored at `./chroma_db`

### Frontend
Simple HTML/CSS/JavaScript interface in `frontend/` that makes API calls to FastAPI backend.

### Data Model
- **Course**: Represents a course with title and lessons
- **Lesson**: Individual lesson within a course 
- **CourseChunk**: Text chunk from course content with metadata for vector storage

## API Structure

FastAPI application with main endpoints:
- `POST /query`: Submit questions with optional session_id
- `GET /courses`: List available courses and statistics
- `POST /courses/add`: Add new course documents (file upload)

The API serves the frontend static files and provides CORS-enabled endpoints for cross-origin requests.