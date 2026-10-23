import socket
import threading
import time

import httpx
import pytest

import agent_tools as tools


# --- calculator ---

@pytest.mark.parametrize("expression, expected", [
    ("2+2", "4"),
    ("17/100*4320", "734.4"),
    ("0.1+0.2", "0.3"),
    ("2^10", "1024"),
    ("1,234*2", "2468"),
    ("sqrt(144)+max(1,2)", "14"),
    ("-(3-5)", "2"),
    ("10//3", "3"),
    ("7%4", "3"),
    ("round(2.567, 1)", "2.6"),
    ("pi", "3.14159265359"),
])
def test_calculator_evaluates(expression, expected):
    assert tools.evaluate_expression(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",
    "(1).__class__",
    "open('x')",
    "[1, 2]",
    "lambda: 1",
    "1 if 1 else 2",
    "True + 1",
    "'a' * 3",
    "x + 1",
    "",
])
def test_calculator_rejects_anything_but_arithmetic(expression):
    assert tools.evaluate_expression(expression).startswith("error:")


def test_calculator_limits_exponents_and_handles_division_by_zero():
    assert "exponent too large" in tools.evaluate_expression("9**9**9")
    assert tools.evaluate_expression("1/0") == "error: division by zero"


# --- URL safety ---

def resolve_to(monkeypatch, *addresses):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in addresses
    ])


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.9",
                                "169.254.169.254", "0.0.0.0", "224.0.0.1"])
def test_private_and_special_addresses_are_blocked(monkeypatch, ip):
    resolve_to(monkeypatch, ip)
    assert "private network" in tools.check_public_url("http://example.test/")


def test_a_public_address_is_allowed(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    assert tools.check_public_url("https://example.test/page") is None


def test_one_private_address_among_public_ones_blocks_the_host(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34", "10.0.0.1")
    assert tools.check_public_url("http://example.test/") is not None


def test_only_http_and_https_are_allowed():
    assert "http and https" in tools.check_public_url("file:///etc/passwd")
    assert "http and https" in tools.check_public_url("ftp://example.test/x")
    assert "http and https" in tools.check_public_url("http:///nohost")


def test_unresolvable_host_is_reported(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("nope")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert "can't resolve" in tools.check_public_url("http://nowhere.test/")


# --- page fetching ---

REAL_CLIENT = httpx.Client


def serve(monkeypatch, handler, hosts=None):
    """Route fetch_page's HTTP client to `handler`; hosts maps name -> ip for the safety check."""
    hosts = hosts or {}
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (hosts.get(host, "93.184.216.34"), 443))
    ])
    monkeypatch.setattr(tools.httpx, "Client",
                        lambda **kwargs: REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs))


def test_fetch_page_extracts_readable_text(monkeypatch):
    html = ("<html><head><title>t</title><script>evil()</script></head><body><nav>menu</nav>"
            "<h1>Heading</h1><p>First paragraph.</p><style>p{}</style><p>Second.</p></body></html>")
    serve(monkeypatch, lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"}))
    text = tools.fetch_page("https://example.test/")
    assert "Heading" in text and "First paragraph." in text and "Second." in text
    assert "evil" not in text and "menu" not in text and "p{}" not in text


def test_fetch_page_refuses_a_redirect_to_a_private_address(monkeypatch):
    def handler(request):
        if request.url.host == "good.example":
            return httpx.Response(302, headers={"location": "http://internal.example/secret"})
        return httpx.Response(200, text="secret data")
    serve(monkeypatch, handler, hosts={"good.example": "93.184.216.34", "internal.example": "10.0.0.1"})
    result = tools.fetch_page("http://good.example/")
    assert result.startswith("error:") and "private network" in result
    assert "secret data" not in result


def test_fetch_page_reports_http_errors_binary_content_and_redirect_loops(monkeypatch):
    serve(monkeypatch, lambda request: httpx.Response(404, text="gone"))
    assert "HTTP 404" in tools.fetch_page("https://example.test/")

    serve(monkeypatch, lambda request: httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"}))
    assert "image/png" in tools.fetch_page("https://example.test/")

    serve(monkeypatch, lambda request: httpx.Response(302, headers={"location": "https://example.test/again"}))
    assert "too many redirects" in tools.fetch_page("https://example.test/")


# --- Python sandbox ---

@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return str(path)


def test_python_runs_and_returns_output(workspace):
    assert tools.run_python_code("print(sum(i * i for i in range(1, 101)))", workspace) == "338350"


def test_python_has_pandas_and_numpy(workspace):
    result = tools.run_python_code("import pandas as pd, numpy as np\nprint(pd.Series(np.arange(4)).sum())", workspace)
    assert result == "6"


def test_python_reports_errors_and_exit_code(workspace):
    result = tools.run_python_code("raise ValueError('boom')", workspace)
    assert "ValueError: boom" in result and "[exit code 1]" in result


def test_python_is_stopped_at_the_time_limit(workspace, monkeypatch):
    monkeypatch.setattr(tools, "PYTHON_TIMEOUT_SECONDS", 2)
    started = time.time()
    result = tools.run_python_code("while True: pass", workspace)
    assert "stopped: ran longer than 2 seconds" in result
    assert time.time() - started < 10


def test_python_output_is_capped(workspace):
    result = tools.run_python_code("print('x' * 200000)", workspace)
    assert len(result) < tools.MAX_PYTHON_OUTPUT_CHARS + 200
    assert "truncated" in result


def test_python_does_not_see_api_keys(workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    result = tools.run_python_code(
        "import os\nprint(os.environ.get('OPENAI_API_KEY'), os.environ.get('ANTHROPIC_API_KEY'))", workspace)
    assert result == "None None"


def test_python_has_no_stdin(workspace):
    assert "EOFError" in tools.run_python_code("input()", workspace)


def test_python_files_persist_across_runs_and_are_listed(workspace):
    first = tools.run_python_code("open('data.csv', 'w').write('a,b\\n1,2\\n')", workspace)
    assert "data.csv" in first
    second = tools.run_python_code("print(open('data.csv').read().strip().splitlines()[1])", workspace)
    assert second.startswith("1,2")
    # the script and output scratch files don't linger in the workspace
    assert sorted(__import__("os").listdir(workspace)) == ["data.csv"]


# --- permission requests ---

def test_permission_approval_and_denial(chat_id):
    request = tools.create_permission_request(chat_id)
    threading.Timer(0.05, tools.resolve_permission_request, (chat_id, request, True)).start()
    assert tools.wait_for_permission(request, timeout=5) is True

    request = tools.create_permission_request(chat_id)
    tools.resolve_permission_request(chat_id, request, False)
    assert tools.wait_for_permission(request, timeout=5) is False


def test_no_answer_in_time_counts_as_denied(chat_id):
    request = tools.create_permission_request(chat_id)
    assert tools.wait_for_permission(request, timeout=0.1) is False


def test_a_request_can_only_be_answered_by_its_own_chat_and_once(chat_id):
    request = tools.create_permission_request(chat_id)
    assert tools.resolve_permission_request("some-other-chat", request, True) is False
    assert tools.resolve_permission_request(chat_id, request, True) is True
    assert tools.wait_for_permission(request, timeout=1) is True
    assert tools.resolve_permission_request(chat_id, request, True) is False  # already consumed
    assert tools.resolve_permission_request(chat_id, "unknown", True) is False


# --- ToolBox ---

def make_box(chat_id, monkeypatch, **config):
    import chat_manager
    monkeypatch.setattr(tools, "has_documents", lambda path: False)
    return tools.ToolBox(chat_id, {**chat_manager.get_config(chat_id), **config})


def names(box, **kwargs):
    return {tool.name for tool in box.tools(**kwargs)}


def test_toolbox_offers_tools_per_config(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    assert {"search_documents", "calculator", "web_search", "fetch_url", "run_python"} <= names(box)

    assert "run_python" not in names(make_box(chat_id, monkeypatch, permission_mode="safe"))
    assert "run_python" in names(make_box(chat_id, monkeypatch, permission_mode="bypass"))
    assert not {"web_search", "fetch_url"} & names(make_box(chat_id, monkeypatch, web_search_enabled=False))
    # callers with no one to ask can force code off whatever the mode says
    forced_off = tools.ToolBox(chat_id, {**make_box(chat_id, monkeypatch).config, "permission_mode": "bypass"}, allow_code=False)
    assert "run_python" not in names(forced_off)


def test_memory_tools_are_offered_only_when_allowed(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    memory = {"compact_context", "keep_only_context", "forget_context", "remember"}
    assert not memory & names(box)
    box.context_tools_allowed = {"remember"}
    assert names(box) & memory == {"remember"}
    box.context_tools_allowed = set(memory)
    assert memory <= names(box)
    assert not memory & names(box, include_context_tools=False)


def test_only_ask_mode_needs_approval_and_only_for_code(chat_id, monkeypatch):
    assert make_box(chat_id, monkeypatch, permission_mode="ask").needs_approval("run_python")
    assert not make_box(chat_id, monkeypatch, permission_mode="ask").needs_approval("calculator")
    assert not make_box(chat_id, monkeypatch, permission_mode="bypass").needs_approval("run_python")
    assert not make_box(chat_id, monkeypatch, permission_mode="safe").needs_approval("run_python")


def test_read_document_matches_names_loosely(chat_id, monkeypatch):
    import os
    box = make_box(chat_id, monkeypatch)
    os.makedirs(box.files_path, exist_ok=True)
    with open(os.path.join(box.files_path, "resume.md"), "w") as f:
        f.write("Name: Bruno\n" * 500)
    monkeypatch.setattr(box, "document_names", lambda: ["resume"])

    for name in ("resume", "Resume", "resume.md", "resume.pdf", "**resume**", "the document"):
        assert box.read_document(name).startswith("Name: Bruno"), name
    page = box.read_document("resume")
    assert "characters 0-4000 of 6000" in page and "offset=4000" in page
    assert "end of document" in box.read_document("resume", 4000)


def test_read_document_rejects_bad_names_and_unknown_documents(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    monkeypatch.setattr(box, "document_names", lambda: ["resume", "taxes"])
    assert box.read_document("../etc/passwd") == "error: invalid document name"
    assert box.read_document("a/b") == "error: invalid document name"
    assert box.read_document("") == "error: invalid document name"
    assert "no document named" in box.read_document("unknown")  # two documents: no guessing


def test_search_documents_without_documents_says_so(chat_id, monkeypatch):
    assert "no documents" in make_box(chat_id, monkeypatch).search_documents("anything")


def test_remember_records_a_context_entry(chat_id, monkeypatch):
    import chat_manager
    make_box(chat_id, monkeypatch).remember("my dog is Rex")
    entries = chat_manager.get_context(chat_id)
    assert [(e["question"], e["answer"]) for e in entries] == [("Remember: my dog is Rex", "Noted.")]


# --- web search ---

def fake_ddgs(monkeypatch, outcomes):
    """Replace the search library: `outcomes` maps backend name -> results list, or an Exception to raise."""
    import ddgs
    asked = []

    class FakeDDGS:
        def text(self, query, max_results=5, backend="auto"):
            asked.append(backend)
            outcome = outcomes[backend]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    return asked


HIT = [{"title": "Golden Boot", "href": "https://example.test/gb", "body": "Top scorers..."}]


def test_search_returns_the_first_engine_that_has_results(monkeypatch):
    asked = fake_ddgs(monkeypatch, {"auto": HIT, "duckduckgo": HIT, "yahoo": HIT, "bing": HIT})
    assert tools.search_web("q") == (HIT, "")
    assert asked == ["auto"]


def test_search_falls_back_across_engines_when_one_finds_nothing(monkeypatch):
    nothing = Exception("No results found.")
    asked = fake_ddgs(monkeypatch, {"auto": nothing, "duckduckgo": [], "yahoo": HIT, "bing": nothing})
    results, error = tools.search_web("q")
    assert results == HIT and error == "" and asked == ["auto", "duckduckgo", "yahoo"]


def test_search_reports_the_last_error_when_every_engine_fails(monkeypatch):
    fake_ddgs(monkeypatch, {name: Exception(f"{name} blocked") for name in tools.SEARCH_BACKENDS})
    results, error = tools.search_web("q")
    assert results == [] and error == "bing blocked"


def test_the_search_tool_formats_results_and_tells_the_model_to_retry_on_failure(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    fake_ddgs(monkeypatch, {"auto": HIT, "duckduckgo": HIT, "yahoo": HIT, "bing": HIT})
    monkeypatch.setattr(tools, "fetch_page", lambda url: "error: the page had no readable text")
    assert box.web_search("q") == "Golden Boot\nhttps://example.test/gb\nTop scorers..."

    fake_ddgs(monkeypatch, {name: Exception("No results found.") for name in tools.SEARCH_BACKENDS})
    failed = box.web_search("q")
    assert failed.startswith("error:") and "No results found." in failed and "differently worded query" in failed


def three_hits():
    return [{"title": f"T{i}", "href": f"https://example.test/{i}", "body": f"snippet {i}"} for i in range(1, 6)]


def test_web_search_includes_text_from_the_top_pages(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    fake_ddgs(monkeypatch, {name: three_hits() for name in tools.SEARCH_BACKENDS})
    monkeypatch.setattr(tools, "fetch_page", lambda url: f"1. Mbappe 10 goals (from {url})\n" + "filler " * 1000)
    result = box.web_search("top scorers")
    assert "snippet 1" in result and "snippet 5" in result
    assert "--- text from the top pages ---" in result
    assert "[https://example.test/1]" in result and "[https://example.test/2]" in result
    assert "https://example.test/3]" not in result                 # only the best two pages are read
    assert "1. Mbappe 10 goals" in result
    assert len(result) < 3500 + 2 * tools.PAGE_EXCERPT_CHARS       # page text is trimmed


def test_unreadable_pages_are_skipped_in_favour_of_the_next_result(monkeypatch):
    pages = {"https://example.test/1": "error: the page had no readable text",
             "https://example.test/2": "good page two", "https://example.test/3": "good page three"}
    monkeypatch.setattr(tools, "fetch_page", lambda url: pages[url])
    assert tools.read_top_pages(three_hits()) == [
        ("https://example.test/2", "good page two"), ("https://example.test/3", "good page three")]


def test_reading_pages_never_raises_and_the_snippets_still_come_back(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    fake_ddgs(monkeypatch, {name: three_hits() for name in tools.SEARCH_BACKENDS})

    def explode(url):
        raise RuntimeError("network down")
    monkeypatch.setattr(tools, "fetch_page", explode)
    result = box.web_search("q")
    assert "snippet 1" in result and "text from the top pages" not in result
    assert tools.read_top_pages([]) == [] and tools.read_top_pages([{"title": "no link"}]) == []


# --- handing documents over whole ---

def write_docs(box, **docs):
    import os
    os.makedirs(box.files_path, exist_ok=True)
    for name, text in docs.items():
        with open(os.path.join(box.files_path, f"{name}.md"), "w") as f:
            f.write(text)
    box.document_names = lambda: list(docs)


def test_small_documents_come_back_whole_in_upload_order(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    write_docs(box, hw="homework text", rubric="rubric text")
    assert box.small_documents() == [("hw", "homework text"), ("rubric", "rubric text")]


def test_documents_that_do_not_fit_together_are_not_handed_over(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    write_docs(box, a="x" * 60, b="y" * 60)
    assert box.small_documents(limit=100) == []
    assert len(box.small_documents(limit=120)) == 2      # exactly at the limit still fits


def test_no_documents_or_a_missing_file_means_nothing_to_hand_over(chat_id, monkeypatch):
    box = make_box(chat_id, monkeypatch)
    assert box.small_documents() == []
    write_docs(box, hw="text")
    box.document_names = lambda: ["hw", "ghost"]          # indexed, but its file is gone
    assert box.small_documents() == []
