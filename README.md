<div style="margin-bottom: 20px;">
  <img src="athena.jpg" alt="athena" width="100%" style="display: block;"/>
</div>

# Athena

A document-grounded AI agent. Upload PDFs, ask questions about them, and let the assistant search your documents and the web, do math, run Python, and research a topic in the background. It runs on OpenAI, Claude, or a local model through Ollama, so it can work with no API key and nothing leaving your machine.

## Demo

<video src="https://github.com/user-attachments/assets/b3714e3c-27b7-483c-bcfb-709182ac769b" controls muted width="100%"></video>

A 1½-minute walkthrough: upload a PDF, ask about it, and watch the agent use tools.

## What it does

- **Chats with their own documents.** Each chat has its own set of PDFs. A new chat can start empty or copy another chat's documents.
- **Cited answers.** Passages from your documents are highlighted in the reply, and the source chips open the document at the exact passage.
- **An agent, not a fixed pipeline.** For each message the model decides what it needs and the steps show up in the chat:
  - search, read and list your documents
  - search the web (including the text of the top pages) and read a web page
  - exact arithmetic, and today's date
  - run Python for analysis and coding
  - manage its own memory, only when you ask it to
- **Small documents are read whole.** If you say "this project" or "my homework" and what you uploaded is short (about 9,000 characters or less), it is handed to the model directly instead of searched.
- **Code runs with your permission.** Three modes per chat: *safe* (never runs code), *ask first* (shows the code and waits, the default) and *bypass*.
- **Memory you control.** The context the model sees is stored per chat, editable, and sized by a token budget. Edit it from the context screen or with slash commands.
- **Background research.** `/research <topic>` plans sub-questions, investigates each with your documents and the web, and writes a report you can download as Markdown. It keeps running if you leave the chat.
- **Chat niceties.** Markdown rendering (tables, code, lists), copy, regenerate and edit-last-message, stop, and chat history that survives a reload.
- **Three providers.** OpenAI, Claude (Anthropic) and Ollama, switchable in the app without losing your keys. Installed Ollama models can be picked from a dropdown.

## Slash commands

| Command | What it does |
|---|---|
| `/context` | Open the context screen: view, delete and tune what the model remembers |
| `/compact [focus]` | Summarize the whole context; an optional focus says what to keep detail on |
| `/forget <topic>` | Remove everything in the context that mentions a topic |
| `/context keep only info on <topic>` | The reverse: drop everything unrelated to a topic |
| `/context clear` | Empty the context |
| `/research <topic>` | Start a background research report |

## How it works

**Ingestion.** PDFs are converted to Markdown, with images described by a vision model (`gemma3:4b` through Ollama, only needed for PDFs that contain images). The text is split into chunks and each chunk is prefixed with a short AI-generated summary of where it sits in the document (contextual retrieval). Chunks are embedded locally with Nomic and stored in a per-chat Chroma database. Embeddings always run locally, whichever chat provider you use.

**Answering.** Each message runs a tool-calling loop (`app/agent.py`): the model either answers or asks for tools, the results go back to it, and this repeats up to a step limit you can set. If the model can't call tools, or you turn the agent off, a classic flow takes over: retrieve, drop chunks that aren't relevant to the question, then answer. A LangGraph workflow with answer grading and query rewriting (`app/qa_workflow.py`) backs the non-streaming endpoint.

**Small local models.** Models around 3B parameters are unreliable at choosing tools, so the code guards against the common mistakes: no tools for small talk, no code unless the message is about code, documents before the web, and a retry when a tool call comes back as broken JSON. A larger model such as `llama3.1:8b` follows instructions noticeably better but answers more slowly.

## Requirements

- Python 3.10+
- Node.js 18+
- One of:
  - an OpenAI API key, or an Anthropic (Claude) API key
  - [Ollama](https://ollama.com) running locally (free and private)

## Quick start

### 1. Pick a model provider

**Ollama (local).** Start Ollama and pull a chat model:
```bash
ollama pull llama3.1:8b      # recommended; llama3.2 is smaller and faster but less reliable with tools
```
No `.env` is needed. Athena uses Ollama when no API key is set.

**OpenAI or Claude.** Either copy `.env.example` to `.env` and add a key, or skip this and paste the key into the app's **ai provider** screen, which saves it to `.env` for you:
```bash
cp .env.example .env
```

### 2. Start the backend (terminal 1)
```bash
chmod +x start_backend.sh
./start_backend.sh
```

### 3. Start the frontend (terminal 2)
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

### 4. Open the app
- App: http://localhost:3000
- API: http://localhost:8000 (listens on this machine only)

Create a chat, drop in a PDF, and ask a question. Pick your provider and model under **ai provider**, and the tools and code permission under **agent**.

## Configuration

Settings live in `.env` (or are set from the app). All are optional.

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI key | unset |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) key | unset |
| `LLM_PROVIDER` | `openai`, `anthropic` or `ollama`: which one answers | the first of OpenAI, Claude with a key, otherwise Ollama |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Claude model | `claude-haiku-5-5` |
| `OLLAMA_MODEL` | Ollama chat model | `llama3.2:latest` |

Per-chat settings (tools on or off, code permission, step limit, web search, context size) are in the app and saved with each chat.

## Safety notes

- **Code execution.** `run_python` runs in a separate process inside the chat's workspace folder, with a time limit, a CPU and file-size cap, capped output and no API keys in its environment. It is not a sandbox: the code can read files and use the network like any program you run. Leave the permission on *ask first* unless you trust the model and your documents.
- **Untrusted content.** Web pages and documents can contain text aimed at the model. Tool results are treated as data, web fetching refuses private network addresses, and code needs your approval by default.
- **No login.** The backend has no authentication and listens only on `127.0.0.1`. Don't expose it to a network.
- **Your data.** Documents and chats stay in `backend/chats/` on your machine. With a cloud provider, the relevant excerpts and your messages go to that provider; web searches go to a public search engine (switchable off per chat).

## Tests

```bash
pip install -r backend/requirements-dev.txt
python3 -m pytest                                          # backend (no model or network needed)
cd frontend && CI=true npx react-scripts test --watchAll=false   # frontend
```

`tests/live/` has scripts that replay conversations against a real local model, for comparing models or checking a prompt change. They need Ollama running:
```bash
OLLAMA_MODEL=llama3.1:8b python3 tests/live/agent_scenarios.py
```

## Project layout

```
app/                  core logic, shared by the API
  agent.py              the tool-calling loop
  agent_tools.py        tools, web search, URL safety, Python sandbox
  context_manager.py    editable, token-budgeted conversation memory
  research.py           background research jobs
  llm_config.py         OpenAI / Claude / Ollama selection and keys
  llm_prompts.py        prompts
  document_processor.py chunking, contextual summaries, embeddings, retrieval
  pdf_converter.py      PDF to Markdown with image descriptions
  qa_workflow.py        LangGraph workflow (classic pipeline)
  chat_manager.py       per-chat storage: messages, context, config
backend/main.py       FastAPI server
frontend/             React app
tests/                backend tests; tests/live/ for real-model scenarios
docs/                 demo video
```

`app/main.py` is an older standalone Streamlit front end for the original single-document pipeline. It hasn't been updated for chats or the agent.

## Limitations

- PDFs only.
- Small local models make more mistakes with tools. Use a larger Ollama model or a cloud model for better results.
- Documents longer than about 9,000 characters are searched rather than read whole, so answers about the whole document depend more on the model.
- Token counts are estimated (about four characters per token), not exact.
