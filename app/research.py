"""
Long-term research: a background job that plans a topic into sub-questions,
investigates each with the agent's tools, and writes a report.

A job runs on its own thread and keeps its progress in a JSON file in the
chat's folder, so the UI can poll it, leave and come back, or cancel it.
"""

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

import chat_manager
import context_manager
from agent import agent_loop, today_string
from agent_tools import ToolBox
from llm_prompts import agent_tool_hints, get_research_planner, get_research_writer, research_note_prompt

MAX_SUBQUESTIONS = 5
STEPS_PER_SUBQUESTION = 8
MAX_NOTES_CHARS = 12000

_active: Dict[str, threading.Event] = {}   # job id -> cancel flag, while its thread runs
_active_lock = threading.Lock()


def _research_dir(chat_id: str) -> str:
    return os.path.join(chat_manager._chat_dir(chat_id), "research")


def _job_path(chat_id: str, job_id: str) -> str:
    return os.path.join(_research_dir(chat_id), f"{job_id}.json")


def _save(chat_id: str, job: dict) -> None:
    os.makedirs(_research_dir(chat_id), exist_ok=True)
    chat_manager._write_json_atomic(_job_path(chat_id, job["id"]), job)


def get_job(chat_id: str, job_id: str) -> Optional[dict]:
    """A job's current state, or None. A 'running' job with no live thread (backend restarted) is marked interrupted."""
    if not chat_manager.chat_exists(chat_id) or not re.fullmatch(r"[0-9a-f]{12}", job_id):
        return None
    path = _job_path(chat_id, job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        job = json.load(f)
    with _active_lock:
        alive = job_id in _active
    if job["status"] == "running" and not alive:
        job["status"] = "interrupted"
        job["error"] = "the backend restarted while this research was running"
        _save(chat_id, job)
    return job


def has_running_job(chat_id: str) -> bool:
    directory = _research_dir(chat_id)
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        job = get_job(chat_id, name[:-5]) if name.endswith(".json") else None
        if job and job["status"] == "running":
            return True
    return False


def cancel_job(chat_id: str, job_id: str) -> bool:
    job = get_job(chat_id, job_id)
    if not job or job["status"] != "running":
        return False
    with _active_lock:
        flag = _active.get(job_id)
    if flag:
        flag.set()
    return True


def start_research(chat_id: str, topic: str, classify_error: Callable) -> dict:
    """
    Start a research job on a background thread.

    Raises:
        ValueError: if the chat doesn't exist, the topic is empty, or this
            chat already has a job running
    """
    topic = topic.strip()
    if not chat_manager.chat_exists(chat_id):
        raise ValueError(f"Chat not found: {chat_id}")
    if not topic:
        raise ValueError("Give me a topic to research.")
    if has_running_job(chat_id):
        raise ValueError("A research job is already running in this chat. Cancel it or wait for it to finish.")

    job = {
        "id": uuid.uuid4().hex[:12], "topic": topic, "status": "running", "phase": "planning",
        "current": "", "plan": [], "notes": [], "report": "", "error": "", "error_code": "",
        "created_at": datetime.now().isoformat(), "finished_at": "",
    }
    _save(chat_id, job)
    cancel = threading.Event()
    with _active_lock:
        _active[job["id"]] = cancel
    threading.Thread(target=_run, args=(chat_id, job, cancel, classify_error), daemon=True).start()
    return job


def _parse_plan(text: str) -> list:
    questions = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if len(line) > 8:
            questions.append(line)
    return questions[:MAX_SUBQUESTIONS]


def _run(chat_id: str, job: dict, cancel: threading.Event, classify_error: Callable) -> None:
    try:
        config = chat_manager.get_config(chat_id)
        # No one is watching to approve code, so it runs only if the chat is set to bypass.
        toolbox = ToolBox(chat_id, config, allow_code=config["permission_mode"] == "bypass")

        job["plan"] = _parse_plan(get_research_planner().invoke({"topic": job["topic"]})) or [job["topic"]]
        job["phase"] = "researching"
        _save(chat_id, job)

        names = toolbox.document_names()
        offered = {tool.name for tool in toolbox.tools(include_context_tools=False)}
        tool_lines = "\n".join(f"- {hint}" for name, hint in agent_tool_hints.items() if name in offered)
        system = f"{research_note_prompt.replace('{today}', today_string())}\n\nTools:\n{tool_lines}"
        if names:
            system += "\n\nDocuments the user has uploaded: " + ", ".join(names)

        for question in job["plan"]:
            if cancel.is_set():
                break
            job["current"] = question
            _save(chat_id, job)
            prompt = f"Overall topic: {job['topic']}\nSub-question to research: {question}"
            note, used = "", []
            for attempt in range(2):
                messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
                loop = agent_loop(messages, toolbox, STEPS_PER_SUBQUESTION, include_context_tools=False)
                note, used = "", []
                try:
                    while True:
                        if cancel.is_set():
                            loop.close()
                            break
                        event = next(loop)
                        if event["type"] == "step":
                            used.append(event["tool"])
                            job["current"] = f"{question} - {event['tool']}"
                            _save(chat_id, job)
                    break
                except StopIteration as stop:
                    note = (stop.value or "").strip()
                    break
                except Exception as e:
                    # A dropped connection shouldn't cost the whole job. Retry once; if
                    # the sub-question still fails, note that and move on - unless the
                    # error is one every later step would hit too (no credits, bad key,
                    # Ollama down), in which case the job fails with that message.
                    print(f"research sub-question failed (attempt {attempt + 1}): {e}")
                    code, message = classify_error(e)
                    if code != "error":
                        raise
                    if attempt == 1:
                        note = f"(could not be researched: {message})"
            if note:
                job["notes"].append({"question": question, "text": note, "tools": used})
                _save(chat_id, job)

        if cancel.is_set():
            job["status"] = "cancelled"
        else:
            job["phase"] = "writing"
            job["current"] = ""
            _save(chat_id, job)
            notes_text = "\n\n".join(f"Q: {n['question']}\n{n['text']}" for n in job["notes"])[:MAX_NOTES_CHARS]
            writer = get_research_writer()
            try:
                report = writer.invoke({"topic": job["topic"], "notes": notes_text})
            except Exception as e:
                if classify_error(e)[0] != "error":
                    raise
                print(f"research report failed, retrying once: {e}")
                report = writer.invoke({"topic": job["topic"], "notes": notes_text})
            job["report"] = (report or "").strip()
            job["status"] = "done"
            try:
                context_manager.record_turn(chat_id, f"/research {job['topic']}", job["report"])
            except Exception as e:
                print(f"WARNING: failed to record research in context: {e}")
    except Exception as e:
        print(f"ERROR in research job {job['id']}: {e}")
        job["status"] = "error"
        job["error_code"], job["error"] = classify_error(e)
    finally:
        job["current"] = ""
        job["finished_at"] = datetime.now().isoformat()
        try:
            _save(chat_id, job)
        except Exception as e:
            print(f"WARNING: could not save research job: {e}")
        with _active_lock:
            _active.pop(job["id"], None)
