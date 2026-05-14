import asyncio
import warnings as _warnings_module
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_response(content="answer", tool_calls=None, total_duration=1_000_000_000, eval_count=10):
    r = MagicMock()
    r.message.content    = content
    r.message.tool_calls = tool_calls
    r.total_duration     = total_duration
    r.eval_count         = eval_count
    return r


def _make_tool_call(name, arguments):
    tc = MagicMock()
    tc.function.name      = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Fixture — patches both Client and AsyncClient at import time
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    with (
        patch("pyutils.ollama.ollama_collector.Client")      as mock_client_cls,
        patch("pyutils.ollama.ollama_collector.AsyncClient") as mock_async_cls,
    ):
        mock_sync  = MagicMock()
        mock_async = MagicMock()
        mock_client_cls.return_value = mock_sync
        mock_async_cls.return_value  = mock_async

        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(model="llama3.2", embedder="nomic-embed-text")
        c._client       = mock_sync
        c._async_client = mock_async
        yield c


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_defaults():
    with (
        patch("pyutils.ollama.ollama_collector.Client"),
        patch("pyutils.ollama.ollama_collector.AsyncClient"),
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector()
        assert c.model    == "llama3.2"
        assert c.embedder == "nomic-embed-text"
        assert c.host     == "http://localhost:11434"


def test_custom_values():
    with (
        patch("pyutils.ollama.ollama_collector.Client"),
        patch("pyutils.ollama.ollama_collector.AsyncClient"),
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(model="mistral", embedder="mxbai-embed-large", host="http://remote:11434")
        assert c.model    == "mistral"
        assert c.embedder == "mxbai-embed-large"
        assert c.host     == "http://remote:11434"


def test_repr_shows_host_and_models(collector):
    r = repr(collector)
    assert "OllamaCollector" in r
    assert "host="  in r
    assert "model=" in r


def test_client_receives_host_and_timeout():
    with (
        patch("pyutils.ollama.ollama_collector.Client")      as mock_c,
        patch("pyutils.ollama.ollama_collector.AsyncClient") as mock_a,
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        OllamaCollector(host="http://myhost:11434", timeout=30.0)
        mock_c.assert_called_once_with(host="http://myhost:11434", timeout=30.0)
        mock_a.assert_called_once_with(host="http://myhost:11434", timeout=30.0)


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------

def test_ping_returns_true_when_reachable(collector):
    collector._client.list.return_value = MagicMock()
    assert collector.ping() is True


def test_ping_returns_false_on_error(collector):
    collector._client.list.side_effect = ConnectionError("refused")
    assert collector.ping() is False


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_models_returns_dataframe(collector):
    import pandas as pd

    m = MagicMock()
    m.model        = "llama3.2:latest"
    m.size         = 4_000_000_000
    m.modified_at  = "2024-01-01"
    m.details.parameter_size     = "3B"
    m.details.quantization_level = "Q4_K_M"
    m.details.family             = "llama"

    collector._client.list.return_value.models = [m]

    df = collector.models()
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["model_name"] == "llama3.2:latest"
    assert df.iloc[0]["size_gb"]    == 4.0


def test_models_empty_when_no_models(collector):
    import pandas as pd
    collector._client.list.return_value.models = []
    df = collector.models()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_running_returns_list(collector):
    m = MagicMock()
    m.model      = "llama3.2:latest"
    m.size_vram  = 3_500_000_000
    m.expires_at = "2024-01-01T00:05:00"
    collector._client.ps.return_value.models = [m]

    result = collector.running()
    assert isinstance(result, list)
    assert result[0]["model"] == "llama3.2:latest"


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

def test_show_returns_dict(collector):
    resp = MagicMock()
    resp.modelfile  = "FROM llama3.2"
    resp.parameters = "temperature 0.7"
    resp.template   = "{{ .Prompt }}"
    resp.details    = MagicMock()
    collector._client.show.return_value = resp

    result = collector.show("llama3.2")
    assert result["modelfile"] == "FROM llama3.2"
    collector._client.show.assert_called_once_with("llama3.2")


def test_show_uses_default_model(collector):
    collector._client.show.return_value = MagicMock()
    collector.show()
    collector._client.show.assert_called_once_with("llama3.2")


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

def test_embed_returns_embeddings(collector):
    collector._client.embed.return_value.embeddings = [[0.1, 0.2, 0.3]]
    result = collector.embed("hello world")
    assert result == [[0.1, 0.2, 0.3]]
    collector._client.embed.assert_called_once_with(
        model    = "nomic-embed-text",
        input    = "hello world",
        truncate = True,
    )


def test_embed_uses_override_model(collector):
    collector._client.embed.return_value.embeddings = [[0.5]]
    collector.embed(["a", "b"], model="mxbai-embed-large")
    collector._client.embed.assert_called_once_with(
        model    = "mxbai-embed-large",
        input    = ["a", "b"],
        truncate = True,
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def test_ask_returns_string(collector):
    collector._client.chat.return_value = _make_chat_response("42 is the answer")
    result = collector.ask("What is the answer?")
    assert result == "42 is the answer"


def test_ask_sends_system_and_user_messages(collector):
    collector._client.chat.return_value = _make_chat_response()
    collector.content = "You are a pirate."
    collector.ask("Hello")

    _, kwargs = collector._client.chat.call_args
    msgs = kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "You are a pirate."}
    assert msgs[1]["role"] == "user"


def test_ask_uses_override_model(collector):
    collector._client.chat.return_value = _make_chat_response()
    collector.ask("Hi", model="mistral")
    _, kwargs = collector._client.chat.call_args
    assert kwargs["model"] == "mistral"


def test_ask_passes_think_param(collector):
    collector._client.chat.return_value = _make_chat_response()
    collector.ask("Reason about this", think="high")
    _, kwargs = collector._client.chat.call_args
    assert kwargs["think"] == "high"


def test_ask_timer_prints(collector, capsys):
    collector._client.chat.return_value = _make_chat_response(
        total_duration=2_000_000_000, eval_count=25
    )
    collector.ask("Hi", timer=True)
    out = capsys.readouterr().out
    assert "2.0s" in out
    assert "25" in out


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def test_chat_returns_string(collector):
    collector._client.chat.return_value = _make_chat_response("Hello!")
    msgs = [{"role": "user", "content": "Hi"}]
    assert collector.chat(msgs) == "Hello!"


def test_chat_passes_full_history(collector):
    collector._client.chat.return_value = _make_chat_response()
    msgs = [
        {"role": "user",      "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user",      "content": "Third"},
    ]
    collector.chat(msgs)
    _, kwargs = collector._client.chat.call_args
    assert kwargs["messages"] == msgs


def test_chat_returns_message_on_tool_call(collector):
    tool_call   = _make_tool_call("add", {"a": 1, "b": 2})
    response    = _make_chat_response(tool_calls=[tool_call])
    collector._client.chat.return_value = response

    tools  = [MagicMock()]
    result = collector.chat([{"role": "user", "content": "1+2"}], tools=tools)
    assert result is response.message


def test_chat_passes_format(collector):
    collector._client.chat.return_value = _make_chat_response('{"name":"Alice"}')
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    collector.chat([{"role": "user", "content": "Who?"}], format=schema)
    _, kwargs = collector._client.chat.call_args
    assert kwargs["format"] == schema


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------

def test_stream_chat_yields_chunks(collector):
    def _chunks():
        for text in ["Hello", " ", "world"]:
            c = MagicMock()
            c.message.content = text
            yield c

    collector._client.chat.return_value = _chunks()
    tokens = list(collector.stream_chat([{"role": "user", "content": "Hi"}]))
    assert tokens == ["Hello", " ", "world"]


def test_stream_chat_skips_empty_chunks(collector):
    def _chunks():
        for text in ["", "Hi", "", "!"]:
            c = MagicMock()
            c.message.content = text
            yield c

    collector._client.chat.return_value = _chunks()
    tokens = list(collector.stream_chat([{"role": "user", "content": "Hi"}]))
    assert tokens == ["Hi", "!"]


# ---------------------------------------------------------------------------
# run_with_tools
# ---------------------------------------------------------------------------

def test_run_with_tools_no_tool_calls(collector):
    collector._client.chat.return_value = _make_chat_response("Final answer")

    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    result = collector.run_with_tools("What is 2+2?", tools=[add])
    assert result == "Final answer"


def test_run_with_tools_dispatches_tool_and_continues(collector):
    tool_call     = _make_tool_call("add", {"a": 3, "b": 4})
    first_resp    = _make_chat_response(tool_calls=[tool_call])
    second_resp   = _make_chat_response("The answer is 7")
    collector._client.chat.side_effect = [first_resp, second_resp]

    def add(a: int, b: int) -> int:
        """Add two integers"""
        return a + b

    result = collector.run_with_tools("Add 3 and 4", tools=[add])
    assert result == "The answer is 7"
    assert collector._client.chat.call_count == 2


def test_run_with_tools_raises_on_unknown_tool(collector):
    tool_call  = _make_tool_call("nonexistent", {})
    response   = _make_chat_response(tool_calls=[tool_call])
    collector._client.chat.return_value = response

    def real_tool():
        """A tool"""
        return "ok"

    with pytest.raises(ValueError, match="unknown tool"):
        collector.run_with_tools("Do something", tools=[real_tool])


def test_run_with_tools_raises_after_max_turns(collector):
    def _infinite_tool_call(**_kwargs):
        tc       = _make_tool_call("add", {"a": 1, "b": 1})
        response = _make_chat_response(tool_calls=[tc])
        return response

    collector._client.chat.side_effect = _infinite_tool_call

    def add(a: int, b: int) -> int:
        """Add numbers"""
        return a + b

    with pytest.raises(RuntimeError, match="max_turns"):
        collector.run_with_tools("Loop forever", tools=[add], max_turns=3)


# ---------------------------------------------------------------------------
# ask_structured
# ---------------------------------------------------------------------------

def test_ask_structured_with_dict_schema(collector):
    collector._client.chat.return_value = _make_chat_response('{"name": "Alice"}')
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = collector.ask_structured("Who?", schema)
    assert result == '{"name": "Alice"}'
    _, kwargs = collector._client.chat.call_args
    assert kwargs["format"] == schema


def test_ask_structured_with_pydantic_schema(collector):
    collector._client.chat.return_value = _make_chat_response('{"age": 30}')

    class FakeModel:
        @staticmethod
        def model_json_schema():
            return {"type": "object", "properties": {"age": {"type": "integer"}}}

    result = collector.ask_structured("How old?", FakeModel)
    assert result == '{"age": 30}'
    _, kwargs = collector._client.chat.call_args
    assert kwargs["format"]["properties"]["age"]["type"] == "integer"


def test_ask_structured_uses_temperature_zero(collector):
    collector._client.chat.return_value = _make_chat_response("{}")
    collector.ask_structured("test", {})
    _, kwargs = collector._client.chat.call_args
    assert kwargs["options"] == {"temperature": 0}


# ---------------------------------------------------------------------------
# ask_with_image
# ---------------------------------------------------------------------------

def test_ask_with_image_sends_encoded_image(collector, tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    collector._client.chat.return_value = _make_chat_response("A sunset")
    result = collector.ask_with_image("What is this?", str(img))
    assert result == "A sunset"

    _, kwargs = collector._client.chat.call_args
    user_msg = kwargs["messages"][1]
    assert user_msg["role"] == "user"
    assert "images" in user_msg
    assert len(user_msg["images"]) == 1


def test_ask_with_image_raises_on_missing_file(collector):
    with pytest.raises(OSError):
        collector.ask_with_image("What?", "/no/such/file.jpg")


# ---------------------------------------------------------------------------
# encode_image
# ---------------------------------------------------------------------------

def test_encode_image_returns_base64(tmp_path):
    from pyutils.ollama.ollama_collector import OllamaCollector
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = OllamaCollector.encode_image(str(img))
    assert isinstance(result, str)
    assert len(result) > 0


def test_encode_image_raises_on_missing_file():
    from pyutils.ollama.ollama_collector import OllamaCollector
    with pytest.raises(OSError):
        OllamaCollector.encode_image("/nonexistent/image.png")


# ---------------------------------------------------------------------------
# Async — async_ask
# ---------------------------------------------------------------------------

def test_async_ask_returns_string(collector):
    async def _run():
        collector._async_client.chat = AsyncMock(
            return_value=_make_chat_response("async answer")
        )
        return await collector.async_ask("Hello")

    result = asyncio.run(_run())
    assert result == "async answer"


def test_async_ask_passes_think_param(collector):
    async def _run():
        collector._async_client.chat = AsyncMock(
            return_value=_make_chat_response("ok")
        )
        await collector.async_ask("Reason", think=True)
        _, kwargs = collector._async_client.chat.call_args
        return kwargs

    kwargs = asyncio.run(_run())
    assert kwargs["think"] is True


# ---------------------------------------------------------------------------
# Async — async_chat
# ---------------------------------------------------------------------------

def test_async_chat_returns_string(collector):
    async def _run():
        collector._async_client.chat = AsyncMock(
            return_value=_make_chat_response("done")
        )
        return await collector.async_chat([{"role": "user", "content": "hi"}])

    assert asyncio.run(_run()) == "done"


def test_async_chat_returns_message_on_tool_call(collector):
    async def _run():
        tool_call = _make_tool_call("search", {"q": "test"})
        response  = _make_chat_response(tool_calls=[tool_call])
        collector._async_client.chat = AsyncMock(return_value=response)
        tools  = [MagicMock()]
        return await collector.async_chat(
            [{"role": "user", "content": "search"}],
            tools=tools,
        )

    result = asyncio.run(_run())
    assert hasattr(result, "tool_calls")


# ---------------------------------------------------------------------------
# Async — async_stream_chat
# ---------------------------------------------------------------------------

def test_async_stream_chat_yields_chunks(collector):
    async def fake_stream():
        for text in ["chunk1", "chunk2", "chunk3"]:
            c = MagicMock()
            c.message.content = text
            yield c

    async def _run():
        collector._async_client.chat = AsyncMock(return_value=fake_stream())
        tokens = []
        async for token in collector.async_stream_chat([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        return tokens

    assert asyncio.run(_run()) == ["chunk1", "chunk2", "chunk3"]


# ---------------------------------------------------------------------------
# Async — async_run_with_tools
# ---------------------------------------------------------------------------

def test_async_run_with_tools_dispatches_and_returns(collector):
    async def _run():
        tool_call   = _make_tool_call("multiply", {"a": 3, "b": 5})
        first_resp  = _make_chat_response(tool_calls=[tool_call])
        second_resp = _make_chat_response("The result is 15")
        collector._async_client.chat = AsyncMock(side_effect=[first_resp, second_resp])

        def multiply(a: int, b: int) -> int:
            """Multiply two integers"""
            return a * b

        return await collector.async_run_with_tools("3 times 5", tools=[multiply])

    assert asyncio.run(_run()) == "The result is 15"


def test_async_run_with_tools_supports_async_tools(collector):
    async def _run():
        tool_call   = _make_tool_call("fetch", {"url": "http://example.com"})
        first_resp  = _make_chat_response(tool_calls=[tool_call])
        second_resp = _make_chat_response("Got the page")
        collector._async_client.chat = AsyncMock(side_effect=[first_resp, second_resp])

        async def fetch(url: str) -> str:
            """Fetch a URL"""
            return f"content from {url}"

        return await collector.async_run_with_tools("Fetch example", tools=[fetch])

    assert asyncio.run(_run()) == "Got the page"


def test_async_run_with_tools_raises_after_max_turns(collector):
    async def _run():
        tc       = _make_tool_call("loop", {})
        response = _make_chat_response(tool_calls=[tc])
        collector._async_client.chat = AsyncMock(return_value=response)

        def loop() -> str:
            """Loop forever"""
            return "again"

        await collector.async_run_with_tools("go", tools=[loop], max_turns=2)

    with pytest.raises(RuntimeError, match="max_turns"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# web_search / web_fetch — module-level functions
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_returns_error_without_api_key(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_search
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        result = web_search("test query")
        assert "OLLAMA_API_KEY" in result

    def test_formats_results(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_search
        import json, urllib.request

        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        payload = json.dumps({
            "results": [
                {"title": "Page A", "url": "https://a.com", "content": "snippet a"},
                {"title": "Page B", "url": "https://b.com", "content": "snippet b"},
            ]
        }).encode()

        class FakeResp:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *_): pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: FakeResp())
        result = web_search("test")
        assert "[1]" in result
        assert "Page A" in result
        assert "https://a.com" in result
        assert "[2]" in result

    def test_no_results_message(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_search
        import json, urllib.request

        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        payload = json.dumps({"results": []}).encode()

        class FakeResp:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *_): pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: FakeResp())
        assert web_search("nothing") == "No results found."

    def test_http_error_returns_message(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_search
        import urllib.error, urllib.request

        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        def _raise(*_a, **_k):
            raise urllib.error.HTTPError(None, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        result = web_search("test")
        assert "401" in result


class TestWebFetch:
    def test_returns_error_without_api_key(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_fetch
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        result = web_fetch("https://example.com")
        assert "OLLAMA_API_KEY" in result

    def test_formats_title_and_content(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_fetch
        import json, urllib.request

        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        payload = json.dumps({"title": "Example", "content": "Hello world"}).encode()

        class FakeResp:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *_): pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: FakeResp())
        result = web_fetch("https://example.com")
        assert "Title: Example" in result
        assert "Hello world" in result

    def test_http_error_returns_message(self, monkeypatch):
        from pyutils.ollama.ollama_collector import web_fetch
        import urllib.error, urllib.request

        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

        def _raise(*_a, **_k):
            raise urllib.error.HTTPError(None, 403, "Forbidden", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        result = web_fetch("https://example.com")
        assert "403" in result


# ---------------------------------------------------------------------------
# ask / async_ask — web_search=True routing
# ---------------------------------------------------------------------------


def test_ask_web_search_true_uses_web_tools(collector):
    """ask(web_search=True) passes WEB_TOOLS to the chat call."""
    collector._client.chat.return_value = _make_chat_response("web answer")

    from unittest.mock import patch as _patch
    import pyutils.ollama.ollama_collector as _mod

    fake_search = MagicMock(return_value="result", __name__="web_search")
    fake_search.__doc__ = "Search the internet."
    fake_fetch  = MagicMock(return_value="page",   __name__="web_fetch")
    fake_fetch.__doc__ = "Fetch a page."

    original = _mod.OllamaCollector.WEB_TOOLS
    collector.WEB_TOOLS = [fake_search, fake_fetch]
    try:
        result = collector.ask("current events", web_search=True)
        assert result == "web answer"
        _, kwargs = collector._client.chat.call_args
        assert fake_search in kwargs["tools"]
        assert fake_fetch  in kwargs["tools"]
    finally:
        collector.WEB_TOOLS = original


def test_async_ask_web_search_true_uses_web_tools(collector):
    async def _run():
        collector._async_client.chat = AsyncMock(
            return_value=_make_chat_response("async web answer")
        )
        import pyutils.ollama.ollama_collector as _mod

        fake_search = MagicMock(return_value="r", __name__="web_search")
        fake_search.__doc__ = "Search."
        fake_fetch  = MagicMock(return_value="p", __name__="web_fetch")
        fake_fetch.__doc__ = "Fetch."

        original = collector.WEB_TOOLS
        collector.WEB_TOOLS = [fake_search, fake_fetch]
        try:
            result = await collector.async_ask("news", web_search=True)
        finally:
            collector.WEB_TOOLS = original
        return result

    assert asyncio.run(_run()) == "async web answer"


# ---------------------------------------------------------------------------
# run_with_tools — web_search=True prepends WEB_TOOLS
# ---------------------------------------------------------------------------

def test_run_with_tools_web_search_prepends_web_tools(collector):
    collector._client.chat.return_value = _make_chat_response("done")

    def my_tool(x: int) -> int:
        """My custom tool."""
        return x * 2

    fake_search = MagicMock(return_value="results", __name__="web_search")
    fake_search.__doc__ = "Search."
    fake_fetch  = MagicMock(return_value="page",    __name__="web_fetch")
    fake_fetch.__doc__ = "Fetch."

    original = collector.WEB_TOOLS
    collector.WEB_TOOLS = [fake_search, fake_fetch]
    try:
        result = collector.run_with_tools("help", tools=[my_tool], web_search=True)
        assert result == "done"
        _, kwargs = collector._client.chat.call_args
        tool_names = [t.__name__ for t in kwargs["tools"]]
        assert tool_names[0] == "web_search"
        assert tool_names[1] == "web_fetch"
        assert "my_tool" in tool_names
    finally:
        collector.WEB_TOOLS = original


# ---------------------------------------------------------------------------
# chat — web_search=True prepends WEB_TOOLS
# ---------------------------------------------------------------------------

def test_chat_web_search_prepends_web_tools(collector):
    collector._client.chat.return_value = _make_chat_response("chat result")

    fake_search = MagicMock(return_value="r", __name__="web_search")
    fake_search.__doc__ = "Search."
    fake_fetch  = MagicMock(return_value="p", __name__="web_fetch")
    fake_fetch.__doc__ = "Fetch."

    original = collector.WEB_TOOLS
    collector.WEB_TOOLS = [fake_search, fake_fetch]
    try:
        msgs = [{"role": "user", "content": "hi"}]
        collector.chat(msgs, web_search=True)
        _, kwargs = collector._client.chat.call_args
        assert fake_search in kwargs["tools"]
        assert fake_fetch  in kwargs["tools"]
    finally:
        collector.WEB_TOOLS = original


# ---------------------------------------------------------------------------
# __init__ — new agentic params
# ---------------------------------------------------------------------------

def test_init_new_defaults():
    with (
        patch("pyutils.ollama.ollama_collector.Client"),
        patch("pyutils.ollama.ollama_collector.AsyncClient"),
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector()
        assert c.max_retries            == 3
        assert c.retry_base_delay       == 1.0
        assert c.retry_max_delay        == 8.0
        assert c.context_limit          == 4096
        assert c.context_warn_threshold == 0.8
        assert c.tool_concurrency       == 0
        assert c.on_tool_call           is None
        assert c.on_tool_result         is None
        assert c.confirm_tool_call      is None


def test_init_tool_semaphore_created_when_concurrency_set():
    import asyncio
    with (
        patch("pyutils.ollama.ollama_collector.Client"),
        patch("pyutils.ollama.ollama_collector.AsyncClient"),
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(tool_concurrency=3)
        assert isinstance(c._tool_semaphore, asyncio.Semaphore)


def test_init_no_semaphore_when_concurrency_zero():
    with (
        patch("pyutils.ollama.ollama_collector.Client"),
        patch("pyutils.ollama.ollama_collector.AsyncClient"),
    ):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(tool_concurrency=0)
        assert c._tool_semaphore is None


# ---------------------------------------------------------------------------
# _chat_with_retry / _async_chat_with_retry
# ---------------------------------------------------------------------------

def test_chat_with_retry_succeeds_after_transient_failure(collector):
    collector._client.chat.side_effect = [
        ConnectionError("refused"),
        _make_chat_response("ok"),
    ]
    with patch("pyutils.ollama.ollama_collector._time.sleep"):
        result = collector._chat_with_retry(model="llama3.2", messages=[])
    assert result.message.content == "ok"
    assert collector._client.chat.call_count == 2


def test_chat_with_retry_reraises_after_max_retries(collector):
    collector._client.chat.side_effect = ConnectionError("refused")
    collector.max_retries = 2
    with patch("pyutils.ollama.ollama_collector._time.sleep"):
        with pytest.raises(ConnectionError):
            collector._chat_with_retry(model="llama3.2", messages=[])
    assert collector._client.chat.call_count == 3  # initial + 2 retries


def test_chat_with_retry_does_not_retry_non_transient(collector):
    collector._client.chat.side_effect = ValueError("bad model")
    with pytest.raises(ValueError):
        collector._chat_with_retry(model="bad", messages=[])
    assert collector._client.chat.call_count == 1


def test_async_chat_with_retry_succeeds_after_transient_failure(collector):
    async def _run():
        collector._async_client.chat = AsyncMock(
            side_effect=[ConnectionError("refused"), _make_chat_response("ok")]
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            return await collector._async_chat_with_retry(model="llama3.2", messages=[])
    result = asyncio.run(_run())
    assert result.message.content == "ok"


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------

def test_package_exports_web_tools():
    from pyutils.ollama import web_search, web_fetch, OllamaCollector
    assert callable(web_search)
    assert callable(web_fetch)
    assert OllamaCollector.WEB_TOOLS[0] is web_search
    assert OllamaCollector.WEB_TOOLS[1] is web_fetch


# ---------------------------------------------------------------------------
# _check_context
# ---------------------------------------------------------------------------

def test_check_context_warns_at_threshold(collector):
    response = _make_chat_response()
    response.prompt_eval_count = 3400  # 83% of default 4096 limit
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        collector._check_context(response)
    assert len(caught) == 1
    assert "Context at" in str(caught[0].message)
    assert "3400/4096" in str(caught[0].message)


def test_check_context_no_warn_below_threshold(collector):
    response = _make_chat_response()
    response.prompt_eval_count = 1000  # 24% — below 80% threshold
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        collector._check_context(response)
    assert len(caught) == 0


def test_check_context_no_warn_when_prompt_eval_count_missing(collector):
    response = MagicMock(spec=[])  # no attributes
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        collector._check_context(response)
    assert len(caught) == 0
