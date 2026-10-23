"""
Live check for document-grounded summaries: a homework PDF, and the prompt
"summarize this project, show the due date, and show a table of how long each
component should take". Prints what the agent did and said; needs Ollama.

    OLLAMA_MODEL=llama3.1:8b python tests/live/homework_scenario.py [--allow-code]

Any request to run code is denied unless --allow-code is given.
"""

import os
import sys

os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "ollama"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "app"))
os.chdir(os.path.join(ROOT, "backend"))

import agent_tools  # noqa: E402
import chat_manager  # noqa: E402
import context_manager  # noqa: E402
from agent import run_agent  # noqa: E402
from document_processor import index_documents  # noqa: E402
from llm_config import get_model_info  # noqa: E402

HOMEWORK = """# CS 340: Operating Systems - Project 2: A Multithreaded Web Server

**Due: Friday, October 30, 2026 at 11:59 PM**

## Overview
In this project you will build a small multithreaded HTTP server in C. The server accepts connections,
hands each request to a pool of worker threads, and serves static files from a directory.

## Components and suggested time
1. **Request parser** - parse the HTTP request line and headers. Suggested time: 3 hours.
2. **Thread pool** - a fixed pool of worker threads fed by a bounded queue with mutexes and condition
   variables. Suggested time: 6 hours.
3. **File server** - map URLs to files, set Content-Type, return 404 for missing files. Suggested time: 4 hours.
4. **Logging and stats** - thread-safe request log and a /stats endpoint. Suggested time: 2 hours.
5. **Testing and write-up** - a 2-page report with benchmark results for 1, 4 and 16 threads.
   Suggested time: 5 hours.

## Grading
Correctness 50%, concurrency design 25%, report 15%, code style 10%. Late submissions lose 10% per day.
"""

PROMPT = "summarize this project, show the due date, and show a table of how long each component should take"


def main():
    allow_code = "--allow-code" in sys.argv
    chat_id = chat_manager.create_chat("zz-homework")["id"]
    try:
        chroma, files = chat_manager.get_chat_paths(chat_id)
        os.makedirs(files, exist_ok=True)
        with open(os.path.join(files, "homework2.md"), "w") as f:
            f.write(HOMEWORK)
        index_documents(files, chroma)
        config = chat_manager.get_config(chat_id)
        print(f"\nmodel: {get_model_info()['model']}   code: {'allowed' if allow_code else 'denied'}\n" + "-" * 60)

        steps, answer = [], ""
        for event in run_agent(chat_id, PROMPT, context_manager.build_history(chat_id), config):
            if event["type"] == "step":
                arg = next(iter(event["args"].values()), "")
                steps.append(f"{event['tool']}({str(arg)[:70]!r})")
            elif event["type"] == "step_result":
                steps[-1] += f" -> {event['status']}"
            elif event["type"] == "permission_request":
                agent_tools.resolve_permission_request(chat_id, event["id"], allow_code)
            elif event["type"] == "chunk":
                answer += event["content"]
            elif event["type"] == "reset":
                answer = ""
        print("steps:", steps or "-")
        print("\n" + answer.strip())
    finally:
        chat_manager.delete_chat(chat_id)


if __name__ == "__main__":
    main()
