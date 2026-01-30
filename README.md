# Local-AI-Agent
> A privacy-first, fully local document question-answering system built with FastAPI and Ollama.

A local RAG system that ingests documents at runtime and answers questions using a locally hosted LLM.

## Overview
Local AI Agent is a **local-first Retrieval-Augmented Generation (RAG) system** that allows you to upload documents and ask questions against them using a fully local AI stack.

### How it works
- Documents are uploaded at runtime and chunked safely
- Text is embedded using a local embedding model
- Relevant context is retrieved using vector similarity
- Answers are generated using a locally hosted LLM
- No external APIs or cloud services are required

This version (**v1.0**) provides a stable backend API and a lightweight HTML frontend.

---

## Requirements
- Python 3.10+
- Ollama installed and running
- Recommended: 8 GB RAM or more

---

## How to Run

1. Pull required Ollama models
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Start the backend
```bash
python -m uvicorn main:app --reload
```

5. Serve the frontend
```bash 
python3 -m http.server 3000
```

6. Open the app
``` bash
http://localhost:3000/universal_frontend.html
```
> Note: The backend must be running in the activated virtual environment.

## Notes
- Documents are uploaded at runtime via the web interface
- The backend exposes a FastAPI-based REST API
- All processing is performed locally

## Tech Stack 
- Python
- LangChain (retrieval and orchestration)
- Chroma (vector storage)
- Ollama (local LLM runtime)


