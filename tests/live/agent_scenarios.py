"""
Live agent scenarios against a real local model (needs Ollama running).

Not part of the pytest suite: model output varies run to run, so this prints
what the agent did and says rather than asserting. Use it to compare models
or to see whether a prompt change helped.

    python tests/live/agent_scenarios.py                     # default Ollama model
    OLLAMA_MODEL=llama3.1:8b python tests/live/agent_scenarios.py

Runs on a throwaway chat that's deleted afterwards. It forces Ollama for this
process only; the project's .env is never read for keys or written.
"""

import os
import sys
import time

os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "ollama"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "app"))
os.chdir(os.path.join(ROOT, "backend"))  # chats live in backend/chats

import chat_manager  # noqa: E402
import context_manager  # noqa: E402
from agent import run_agent  # noqa: E402
from document_processor import index_documents  # noqa: E402
from llm_config import get_model_info  # noqa: E402

RESUME = (
    "# Resume\n\nName: Bruno Inzunza\nEmail: bruno@example.com\n"
    "Experience: software engineer, 3 years, Python and React.\nEducation: B.S. Computer Science.\n"
)

# (message, python calls allowed: approve automatically)
CONVERSATION = [
    "hello!",
    "what is my name?",
    "my name is actually John",
    "what is my name?",
    "write a one sentence bio about me",
    "repeat the one sentence bio you wrote about me",
    "what is 17% of 4320?",
    "use python to compute the sum of the squares of the numbers 1 to 100",
    "what did I ask you about earlier?",
]


def main():
    chat_id = chat_manager.create_chat("zz-agent-scenarios")["id"]
    try:
        chroma, files = chat_manager.get_chat_paths(chat_id)
        os.makedirs(files, exist_ok=True)
        with open(os.path.join(files, "resume.md"), "w") as f:
            f.write(RESUME)
        index_documents(files, chroma)
        config = dict(chat_manager.get_config(chat_id), permission_mode="bypass")
        print(f"\nmodel: {get_model_info()['model']}\n" + "-" * 60)

        for question in CONVERSATION:
            tools, answer, started = [], "", time.time()
            for event in run_agent(chat_id, question, context_manager.build_history(chat_id), config):
                if event["type"] == "step":
                    tools.append(event["tool"])
                elif event["type"] == "chunk":
                    answer += event["content"]
                elif event["type"] == "reset":
                    answer = ""
            print(f"\nQ: {question}\n   tools: {tools or '-'}   ({time.time() - started:.1f}s)\n   A: {answer.strip()[:300]}")
    finally:
        chat_manager.delete_chat(chat_id)


if __name__ == "__main__":
    main()
