<div style="margin-bottom: 20px;">
  <img src="athena.jpg" alt="athena" width="100%" style="display: block;"/>
</div>

# rag chatbot

A functional RAG (Retrieval Augmented Generation) chatbot that processes PDF inputs to answer user questions.

## Features
- **PDF Ingestion:** Easily upload PDF documents for processing.
- **Question Answering:** Ask questions about the content of your PDFs and get accurate, context-aware answers.

## How it Works
The chatbot utilizes Retrieval Augmented Generation (RAG) to combine the power of large language models with external knowledge from your uploaded PDFs. When a question is asked, relevant information is first retrieved from the processed PDF, and then used to augment the language model's response for more precise and informed answers.

## Problems Solved (vs. Basic RAG)
Compared to basic retrieval pipelines, this system implements **Agentic/Corrective RAG** to solve common failures:
- **Lost in Context (Parsing Errors):** Uses **Contextual Retrieval** to prefix every chunk with an AI-generated summary, ensuring the model understands the global context of every snippet.
- **The "Black Box" of Hallucinations:** Implements a **Self-Grading Loop** via LangGraph that verifies if the generated answer is grounded in the documents and actually answers the user's question.
- **Missing Visual Information:** Features a **Multimodal Pipeline** that extracts images from PDFs and uses a vision model (`gemma3`) to generate text descriptions, allowing the system to "see" charts and diagrams.
- **Static Retrieval Failure:** Includes **Query Transformation** to automatically rewrite poorly phrased user questions into more effective search queries when initial retrieval fails.

## Key Optimizations
- **Streaming Response (SSE):** Uses Server-Sent Events to stream tokens in real-time, providing a snappy, modern chat experience.
- **Fully Local & Private:** Powered by **Ollama** and **FAISS**, ensuring all data stays on your machine with zero API costs.
- **Markdown-First Processing:** Converts PDFs into structured Markdown to preserve headers and tables, leading to better semantic chunking.
- **Modular Design:** Decouples the core RAG engine from the delivery layer, supporting both a FastAPI/React stack and a standalone Streamlit app.

## requirements

- **python 3.8+** with pip
- **node.js 16+** with npm
- **ollama** running on localhost:11434

## quick start

### 1. start ollama (required)
```bash
ollama serve
```

### 2. start backend (terminal 1)
```bash
chmod +x start_backend.sh
./start_backend.sh
```

### 3. start frontend (terminal 2)
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

### 4. open browser
- **frontend**: http://localhost:3000
- **backend api**: http://localhost:8000