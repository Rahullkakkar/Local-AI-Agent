# Local-AI-Agent
A local RAG system that ingests documents at runtime and answers questions using a locally hosted LLM.

## Overview
This project demonstrates a minimal RAG pipeline:
- Documents are loaded and embedded
- Relevant context is retrieved using vector similarity
- Answers are generated using a local LLM

This version (**v0.1**) focuses on validating the core RAG workflow and runs as a terminal-based prototype.

## How to run
1. Install dependencies
   ```bash
   pip install -r requirements.txt

2. Run the prototype
   ```bash
   python main.py

Notes
	•	This version runs as a standalone script
	•	Documents are loaded at startup
	•	Future versions will introduce an API and interactive interface

Tech stack
	•	Python
	•	LangChain (retrieval and orchestration)
	•	Chroma (vector storage)
	•	Ollama (local LLM runtime)
