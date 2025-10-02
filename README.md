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