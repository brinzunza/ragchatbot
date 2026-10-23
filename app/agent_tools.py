"""
Tools the agent can call.

A ToolBox is built per request and bound to one chat: it knows that chat's
documents, workspace and config, and remembers which document chunks the
agent actually retrieved so the answer can cite them. Every tool returns a
string and reports failures as "error: ..." text rather than raising, so the
model sees what went wrong and can adapt instead of the whole run dying.
"""

import ast
import ipaddress
import math
import operator
import os
import re
import resource
import socket
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from langchain_core.tools import StructuredTool

import chat_manager
import context_manager
from document_processor import get_retriever, has_documents, list_documents

MAX_TOOL_OUTPUT_CHARS = 6000
# Uploaded documents up to this size (all of them together) can be handed to
# the model whole - about 2,000 tokens, comfortably inside a local model's window.
MAX_AUTO_READ_CHARS = 9000
MAX_CHUNK_CHARS = 1500
MAX_PYTHON_OUTPUT_CHARS = 8000
PYTHON_TIMEOUT_SECONDS = 30
MAX_PAGE_DOWNLOAD_BYTES = 400_000
PERMISSION_TIMEOUT_SECONDS = 120

# Tools that change something outside the conversation. Which of these the
# agent may use (and whether each use needs the user's OK) follows the
# chat's permission_mode.
CODE_TOOL = "run_python"


def _clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… [truncated, {len(text) - limit} more characters]"


# --- Permission requests (the "ask" mode) ---

_pending_permissions: Dict[str, dict] = {}
_permissions_lock = threading.Lock()


def create_permission_request(chat_id: str) -> str:
    """Register a pending approval and return its id."""
    request_id = uuid.uuid4().hex[:12]
    with _permissions_lock:
        _pending_permissions[request_id] = {
            "chat_id": chat_id,
            "event": threading.Event(),
            "approved": False,
        }
    return request_id


def resolve_permission_request(chat_id: str, request_id: str, approved: bool) -> bool:
    """Record the user's decision. False if the request is unknown, expired or another chat's."""
    with _permissions_lock:
        pending = _pending_permissions.get(request_id)
        if not pending or pending["chat_id"] != chat_id:
            return False
        pending["approved"] = approved
        pending["event"].set()
    return True


def wait_for_permission(request_id: str, timeout: float = PERMISSION_TIMEOUT_SECONDS) -> bool:
    """Block until the user decides. No answer in time counts as a denial."""
    with _permissions_lock:
        pending = _pending_permissions.get(request_id)
    if not pending:
        return False
    pending["event"].wait(timeout)
    with _permissions_lock:
        _pending_permissions.pop(request_id, None)
    return pending["approved"]


# --- Calculator ---

_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_FUNCTIONS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "min": min, "max": max,
}
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (limit {MAX_EXPONENT})")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _MATH_CONSTANTS:
        return _MATH_CONSTANTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _MATH_FUNCTIONS and not node.keywords):
        return _MATH_FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ValueError("only numbers, + - * / // % **, parentheses, pi, e and basic math functions are allowed")


def evaluate_expression(expression: str) -> str:
    """Safely evaluate an arithmetic expression (an AST whitelist, never eval)."""
    # "1,234" thousands separators and ^ for powers, which models often write
    expression = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.strip()).replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError) as e:
        return f"error: {e}"
    if isinstance(value, float):
        value = float(f"{value:.12g}")  # 0.17 * 4320 -> 734.4, not 734.4000000000001
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
    return str(value)


# --- Web search ---

# The engines behind the search library fail independently and at random - one
# says "No results found" for a query the next answers in a second - so a single
# attempt isn't a verdict. Try them in turn, "auto" first.
SEARCH_BACKENDS = ("auto", "duckduckgo", "yahoo", "bing")


def search_web(query: str, max_results: int = 5, backends=SEARCH_BACKENDS):
    """
    Search the web, falling back across engines until one returns results.

    Returns:
        (results, last_error) - results is [] only if every engine came up
        empty or failed
    """
    from ddgs import DDGS

    last_error = ""
    for backend in backends:
        try:
            results = DDGS().text(query, max_results=max_results, backend=backend)
        except Exception as e:
            last_error = str(e)
            continue
        if results:
            return results, ""
    return [], last_error


# Search snippets are a sentence or two, which is rarely the answer to "who
# were the top scorers" - that is a table on a page. A small model won't go and
# open a link by itself, so web_search reads the best pages for it.
PAGES_TO_READ = 2
PAGE_CANDIDATES = 3
PAGE_EXCERPT_CHARS = 2200


def read_top_pages(results, count: int = PAGES_TO_READ, candidates: int = PAGE_CANDIDATES):
    """
    Fetch the first few results in parallel and keep the first `count` that
    gave readable text, as [(url, text)]. Pages that fail or are empty are
    skipped; this never raises.
    """
    urls = [r["href"] for r in results[:candidates] if r.get("href")]
    if not urls:
        return []
    try:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            texts = list(pool.map(fetch_page, urls))
    except Exception:
        return []
    pages = [(url, text) for url, text in zip(urls, texts) if text and not text.startswith("error:")]
    return pages[:count]


# --- Web fetching ---

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip() + " ")


def extract_page_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def check_public_url(url: str) -> Optional[str]:
    """
    Return a reason the URL must not be fetched, or None if it's fine.

    Pages the agent reads are untrusted, and the backend itself (and
    anything else on the user's machine or network) answers on private
    addresses, so a page must never be able to point the agent at those.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "only http and https URLs can be fetched"
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror:
        return f"can't resolve {parsed.hostname}"
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            return "that address is on a private network, so it was not fetched"
    return None


def fetch_page(url: str) -> str:
    """Download a public web page and return its readable text."""
    try:
        with httpx.Client(timeout=15, follow_redirects=False,
                          headers={"User-Agent": "Mozilla/5.0 (compatible; AthenaAgent/1.0)"}) as client:
            for _ in range(4):
                problem = check_public_url(url)
                if problem:
                    return f"error: {problem}"
                with client.stream("GET", url) as response:
                    if response.is_redirect and response.headers.get("location"):
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        return f"error: the page returned HTTP {response.status_code}"
                    content_type = response.headers.get("content-type", "")
                    if content_type and not any(kind in content_type for kind in ("text/", "json", "xml")):
                        return f"error: can't read {content_type} content"
                    body = b""
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) > MAX_PAGE_DOWNLOAD_BYTES:
                            break
                    text = body.decode(response.encoding or "utf-8", errors="replace")
                    if "html" in content_type or text.lstrip().lower().startswith(("<!doctype", "<html")):
                        text = extract_page_text(text)
                    return _clip(text) or "error: the page had no readable text"
            return "error: too many redirects"
    except httpx.HTTPError as e:
        return f"error: couldn't fetch the page ({e})"


# --- Python sandbox ---

def _limit_child_resources():
    """Runs in the child before exec: cap CPU time and the size of any file it writes."""
    os.setsid()
    for limit, value in (
        (resource.RLIMIT_CPU, PYTHON_TIMEOUT_SECONDS),
        (resource.RLIMIT_FSIZE, 20 * 1024 * 1024),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def run_python_code(code: str, workspace: str) -> str:
    """
    Run code in a separate Python process inside the chat's workspace folder.

    Limits: a wall-clock timeout, a CPU-time cap, a cap on the size of files
    the code writes, a scrubbed environment (no API keys), and capped
    output. This is a guard against runaway code, not a security boundary:
    the process can still read files and use the network as the user can,
    which is why the "ask" permission mode exists.
    """
    script_path = os.path.join(workspace, "_athena_run.py")
    out_path = os.path.join(workspace, "_athena_out.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    before = {name: os.path.getmtime(os.path.join(workspace, name)) for name in os.listdir(workspace)}
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": workspace,
        "LANG": "en_US.UTF-8",
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
    }
    timed_out = False
    with open(out_path, "wb") as out:
        process = subprocess.Popen(
            [sys.executable, "-I", script_path],
            cwd=workspace, env=env, stdout=out, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, preexec_fn=_limit_child_resources,
        )
        try:
            process.wait(timeout=PYTHON_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            process.wait()

    with open(out_path, "rb") as f:
        output = f.read(MAX_PYTHON_OUTPUT_CHARS * 2).decode("utf-8", errors="replace")
    for path in (out_path, script_path):
        try:
            os.remove(path)
        except OSError:
            pass

    result = _clip(output.strip(), MAX_PYTHON_OUTPUT_CHARS) or "(no output)"
    if timed_out:
        result += f"\n[stopped: ran longer than {PYTHON_TIMEOUT_SECONDS} seconds]"
    elif process.returncode != 0:
        result += f"\n[exit code {process.returncode}]"
    changed = sorted(
        name for name in os.listdir(workspace)
        if name not in before or os.path.getmtime(os.path.join(workspace, name)) != before[name]
    )
    if changed:
        result += "\n[files in workspace: " + ", ".join(changed) + "]"
    return result


class ToolBox:
    """The tools available to one agent run in one chat."""

    def __init__(self, chat_id: str, config: dict, allow_code: Optional[bool] = None):
        self.chat_id = chat_id
        self.config = config
        self.chroma_db_path, self.files_path = chat_manager.get_chat_paths(chat_id)
        self.retrieved_docs: List = []
        # Context tools the model may use this run. Empty unless the caller
        # opts in: a tool that rewrites the conversation's memory must not
        # be one a small model can wander into on its own.
        self.context_tools_allowed: set = set()
        self._seen_chunks = set()
        # Code runs only if the chat's permission mode allows it; callers
        # with no one to ask (background research) can force it off.
        self.code_enabled = config["permission_mode"] != "safe" if allow_code is None else allow_code

    @property
    def has_documents(self) -> bool:
        return has_documents(self.chroma_db_path)

    def needs_approval(self, tool_name: str) -> bool:
        return tool_name == CODE_TOOL and self.config["permission_mode"] == "ask"

    # -- document tools --

    def search_documents(self, query: str) -> str:
        """Search the user's own uploaded documents (their files). Use this FIRST for any question about facts, policies, numbers or content that could be in their documents. Returns the most relevant excerpts."""
        if not self.has_documents:
            return "no documents have been uploaded to this chat yet"
        docs = get_retriever(self.chroma_db_path, top_k=4).invoke(query)
        if not docs:
            return "no matching excerpts found"
        lines = []
        for i, doc in enumerate(docs, 1):
            key = (doc.metadata.get("file_name"), doc.metadata.get("start_index"))
            if key not in self._seen_chunks:
                self._seen_chunks.add(key)
                self.retrieved_docs.append(doc)
            name = os.path.basename(doc.metadata.get("file_name", "unknown"))
            name = name[:-3] if name.endswith(".md") else name
            lines.append(f"[{i}] from \"{name}\":\n{_clip(doc.page_content, MAX_CHUNK_CHARS)}")
        return "\n\n".join(lines)

    def read_document(self, name: str, offset: int = 0) -> str:
        """Read a whole uploaded document in order, 4000 characters at a time starting at `offset`."""
        name = (name or "").strip().strip("*\"' ")
        if not name or "/" in name or "\\" in name or ".." in name:
            return "error: invalid document name"
        # Models guess names ("resume.pdf", "Resume", "the document"): match
        # loosely, and with a single document there's nothing to confuse.
        names = self.document_names()
        stem = name.removesuffix(".md").removesuffix(".pdf")
        match = next((n for n in names if n.lower() == stem.lower()), None)
        if match is None and len(names) == 1:
            match = names[0]
        if match is None:
            return f"error: no document named {name!r}. use list_documents to see the names."
        path = os.path.join(self.files_path, f"{match}.md")
        if not os.path.exists(path):
            return f"error: no document named {name!r}. use list_documents to see the names."
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        offset = max(0, int(offset))
        piece = content[offset:offset + 4000]
        end = offset + len(piece)
        note = f"\n[characters {offset}-{end} of {len(content)}" + (f"; call again with offset={end} for more]" if end < len(content) else "; end of document]")
        return piece + note

    def document_names(self) -> List[str]:
        if not self.has_documents:
            return []
        return [d["name"] for d in list_documents(self.chroma_db_path, self.files_path)]

    def small_documents(self, limit: int = MAX_AUTO_READ_CHARS) -> List[Tuple[str, str]]:
        """
        The full text of every uploaded document as [(name, text)], if they are
        together small enough to give the model whole - otherwise [].
        """
        documents, total = [], 0
        for name in self.document_names():
            path = os.path.join(self.files_path, f"{name}.md")
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            total += len(text)
            if total > limit:
                return []
            documents.append((name, text))
        return documents

    def list_documents_tool(self) -> str:
        """List the uploaded documents."""
        if not self.has_documents:
            return "no documents have been uploaded to this chat yet"
        docs = list_documents(self.chroma_db_path, self.files_path)
        return "\n".join(f"- {d['name']} ({d['chunk_count']} chunks)" for d in docs)

    # -- web tools --

    def web_search(self, query: str) -> str:
        """Search the public web. Use for current events, recent facts, and anything the user's documents would not contain. Returns titles, links and snippets."""
        results, last_error = search_web(query)
        if not results:
            return ("error: the search engines returned nothing for that query"
                    + (f" ({last_error})" if last_error else "")
                    + ". try a shorter or differently worded query before giving up.")
        snippets = "\n\n".join(f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}" for r in results)
        pages = read_top_pages(results)
        if not pages:
            return snippets
        excerpts = "\n\n".join(f"[{url}]\n{_clip(text, PAGE_EXCERPT_CHARS)}" for url, text in pages)
        return f"{snippets}\n\n--- text from the top pages ---\n\n{excerpts}"

    def fetch_url(self, url: str) -> str:
        """Read the text of a public web page."""
        return fetch_page(url)

    # -- utilities --

    def calculator(self, expression: str) -> str:
        """Evaluate an arithmetic expression exactly. Supports + - * / // % ** ( ), pi, e, sqrt, log, round, min, max."""
        return evaluate_expression(expression)

    def current_datetime(self) -> str:
        """The current local date and time. Only call this when the answer depends on today's date or time."""
        return datetime.now().astimezone().strftime("%A, %B %d, %Y, %I:%M %p %Z")

    def run_python(self, code: str) -> str:
        """Run Python code and return what it printed. pandas and numpy are available. Files saved in the working directory persist across runs. Print results you want to see."""
        return run_python_code(code, chat_manager.get_workspace_path(self.chat_id))

    # -- context tools --

    def compact_context(self) -> str:
        """Summarize the conversation so far into a short summary to free up context space."""
        try:
            result = context_manager.compact(self.chat_id)
        except ValueError as e:
            return f"error: {e}"
        return f"compacted context: {result['before']} entries -> {result['after']}"

    def keep_only_context(self, topic: str) -> str:
        """Drop everything from the conversation context except what concerns the given topic."""
        try:
            result = context_manager.keep_only(self.chat_id, topic)
        except ValueError as e:
            return f"error: {e}"
        return f"kept context about {topic!r}: {result['before']} entries -> {result['after']}"

    def forget_context(self, topic: str) -> str:
        """Drop everything from the conversation context that concerns the given topic."""
        try:
            result = context_manager.forget(self.chat_id, topic)
        except ValueError as e:
            return f"error: {e}"
        return f"removed context about {topic!r}: {result['before']} entries -> {result['after']}"

    def remember(self, fact: str) -> str:
        """Save a fact or preference the user asked you to remember, so it stays in the conversation context."""
        context_manager.record_turn(self.chat_id, f"Remember: {fact}", "Noted.")
        return "saved"

    def tools(self, include_context_tools: bool = True) -> List[StructuredTool]:
        """The tools to offer the model, per this chat's config."""
        def make(func: Callable, name: str) -> StructuredTool:
            return StructuredTool.from_function(func=func, name=name, description=func.__doc__)

        offered = [
            make(self.search_documents, "search_documents"),
            make(self.read_document, "read_document"),
            make(self.list_documents_tool, "list_documents"),
            make(self.calculator, "calculator"),
            make(self.current_datetime, "current_datetime"),
        ]
        if self.config["web_search_enabled"]:
            offered += [make(self.web_search, "web_search"), make(self.fetch_url, "fetch_url")]
        if self.code_enabled:
            offered.append(make(self.run_python, CODE_TOOL))
        if include_context_tools:
            context_tools = [
                make(self.compact_context, "compact_context"),
                make(self.keep_only_context, "keep_only_context"),
                make(self.forget_context, "forget_context"),
                make(self.remember, "remember"),
            ]
            offered += [tool for tool in context_tools if tool.name in self.context_tools_allowed]
        return offered
