import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    with patch("pyutils.ollama.ollama_collector.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(model="llama3.2", embedder="nomic-embed-text")
        c.client = mock_client
        yield c


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def test_default_host_stripped():
    with patch("pyutils.ollama.ollama_collector.OpenAI"):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(host="http://localhost:11434/")
        assert c.host == "http://localhost:11434"


def test_default_model_and_embedder():
    with patch("pyutils.ollama.ollama_collector.OpenAI"):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector()
        assert c.model == "llama3.2"
        assert c.embedder == "nomic-embed-text"


def test_custom_model_and_embedder():
    with patch("pyutils.ollama.ollama_collector.OpenAI"):
        from pyutils.ollama.ollama_collector import OllamaCollector
        c = OllamaCollector(model="mistral", embedder="mxbai-embed-large")
        assert c.model == "mistral"
        assert c.embedder == "mxbai-embed-large"


def test_client_created_with_correct_base_url():
    with patch("pyutils.ollama.ollama_collector.OpenAI") as mock_openai:
        from pyutils.ollama.ollama_collector import OllamaCollector
        OllamaCollector(host="http://localhost:11434")
        mock_openai.assert_called_once_with(
            base_url    = "http://localhost:11434/v1",
            api_key     = "ollama",
            max_retries = 5,
            timeout     = 125.0,
        )


def test_repr_masks_nothing_but_shows_host(collector):
    r = repr(collector)
    assert "OllamaCollector" in r
    assert "host=" in r
    assert "model=" in r


# ---------------------------------------------------------------------------
# get_available_models_dataframe
# ---------------------------------------------------------------------------

def test_get_available_models_dataframe_returns_dataframe(collector):
    import pandas as pd
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 4_000_000_000,
                "modified_at": "2024-01-01T00:00:00Z",
                "details": {
                    "parameter_size": "3B",
                    "quantization_level": "Q4_K_M",
                    "family": "llama",
                },
            }
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("pyutils.ollama.ollama_collector.requests.get", return_value=fake_response):
        df = collector.get_available_models_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "llama3.2:latest"
    assert df.iloc[0]["parameter_size"] == "3B"


def test_get_available_models_dataframe_raises_on_connection_error(collector):
    import requests as req
    with patch("pyutils.ollama.ollama_collector.requests.get",
               side_effect=req.exceptions.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="Could not reach Ollama"):
            collector.get_available_models_dataframe()


# ---------------------------------------------------------------------------
# get_tokens_count
# ---------------------------------------------------------------------------

def test_get_tokens_count_returns_int(collector):
    fake_response = MagicMock()
    fake_response.json.return_value = {"tokens": [1, 2, 3, 4, 5]}
    fake_response.raise_for_status = MagicMock()

    with patch("pyutils.ollama.ollama_collector.requests.post", return_value=fake_response):
        count = collector.get_tokens_count("Hello world")

    assert count == 5


def test_get_tokens_count_uses_default_model(collector):
    fake_response = MagicMock()
    fake_response.json.return_value = {"tokens": [1]}
    fake_response.raise_for_status = MagicMock()

    with patch("pyutils.ollama.ollama_collector.requests.post", return_value=fake_response) as mock_post:
        collector.get_tokens_count("test")
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama3.2"


def test_get_tokens_count_uses_override_model(collector):
    fake_response = MagicMock()
    fake_response.json.return_value = {"tokens": [1, 2]}
    fake_response.raise_for_status = MagicMock()

    with patch("pyutils.ollama.ollama_collector.requests.post", return_value=fake_response) as mock_post:
        collector.get_tokens_count("test", model="mistral")
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "mistral"


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------

def test_get_embeddings_uses_default_embedder(collector):
    mock_result = MagicMock()
    mock_result.data = [MagicMock()]
    collector.client.embeddings.create.return_value = mock_result

    collector.get_embeddings("some text")

    collector.client.embeddings.create.assert_called_once_with(
        model = "nomic-embed-text",
        input = "some text",
    )


def test_get_embeddings_uses_override_embedder(collector):
    mock_result = MagicMock()
    mock_result.data = [MagicMock()]
    collector.client.embeddings.create.return_value = mock_result

    collector.get_embeddings(["a", "b"], embedder="mxbai-embed-large")

    collector.client.embeddings.create.assert_called_once_with(
        model = "mxbai-embed-large",
        input = ["a", "b"],
    )


def test_get_embeddings_return_full_object(collector):
    mock_result = MagicMock()
    collector.client.embeddings.create.return_value = mock_result

    result = collector.get_embeddings("text", return_full_object=True)
    assert result is mock_result


def test_get_embeddings_return_data_by_default(collector):
    mock_result = MagicMock()
    mock_result.data = ["vec1"]
    collector.client.embeddings.create.return_value = mock_result

    result = collector.get_embeddings("text")
    assert result == ["vec1"]


# ---------------------------------------------------------------------------
# get_answer_given_query
# ---------------------------------------------------------------------------

def test_get_answer_given_query_returns_string(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "42 is the answer."
    collector.client.chat.completions.create.return_value = mock_response

    result = collector.get_answer_given_query("What is the answer?")
    assert result == "42 is the answer."


def test_get_answer_given_query_uses_system_content(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    collector.client.chat.completions.create.return_value = mock_response

    collector.content = "You are a pirate."
    collector.get_answer_given_query("Hello")

    _, kwargs = collector.client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a pirate."}


def test_get_answer_given_query_uses_override_model(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    collector.client.chat.completions.create.return_value = mock_response

    collector.get_answer_given_query("Hi", model="mistral")

    _, kwargs = collector.client.chat.completions.create.call_args
    assert kwargs["model"] == "mistral"


# ---------------------------------------------------------------------------
# chat  (multi-turn / agentic)
# ---------------------------------------------------------------------------

def test_chat_returns_string_when_no_tools(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello there!"
    mock_response.choices[0].finish_reason = "stop"
    collector.client.chat.completions.create.return_value = mock_response

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hi"},
    ]
    result = collector.chat(messages)
    assert result == "Hello there!"


def test_chat_passes_full_message_history(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "done"
    mock_response.choices[0].finish_reason = "stop"
    collector.client.chat.completions.create.return_value = mock_response

    messages = [
        {"role": "user",      "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user",      "content": "Third"},
    ]
    collector.chat(messages)

    _, kwargs = collector.client.chat.completions.create.call_args
    assert kwargs["messages"] == messages


def test_chat_with_tools_returns_message_on_tool_call(collector):
    mock_message = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message = mock_message
    mock_response.choices[0].finish_reason = "tool_calls"
    collector.client.chat.completions.create.return_value = mock_response

    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    result = collector.chat([{"role": "user", "content": "Search X"}], tools=tools)
    assert result is mock_message


def test_chat_stream_returns_generator(collector):
    chunk1, chunk2 = MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "Hello"
    chunk2.choices[0].delta.content = " world"
    collector.client.chat.completions.create.return_value = iter([chunk1, chunk2])

    gen = collector.chat([{"role": "user", "content": "Hi"}], stream=True)
    tokens = list(gen)
    assert tokens == ["Hello", " world"]


# ---------------------------------------------------------------------------
# get_structured_answer
# ---------------------------------------------------------------------------

def test_get_structured_answer_passes_schema(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"name": "Alice"}'
    collector.client.chat.completions.create.return_value = mock_response

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = collector.get_structured_answer("Who?", schema=schema)

    assert result == '{"name": "Alice"}'
    _, kwargs = collector.client.chat.completions.create.call_args
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == schema


# ---------------------------------------------------------------------------
# get_answer_with_image
# ---------------------------------------------------------------------------

def test_get_answer_with_image_encodes_and_sends(collector, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header bytes

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "A dog."
    collector.client.chat.completions.create.return_value = mock_response

    result = collector.get_answer_with_image("What is this?", str(img))
    assert result == "A dog."

    _, kwargs = collector.client.chat.completions.create.call_args
    user_msg = kwargs["messages"][1]
    assert user_msg["role"] == "user"
    assert any(p["type"] == "image_url" for p in user_msg["content"])


def test_get_answer_with_image_raises_on_missing_file(collector):
    with pytest.raises(OSError):
        collector.get_answer_with_image("What?", "/nonexistent/image.jpg")


# ---------------------------------------------------------------------------
# encode_image
# ---------------------------------------------------------------------------

def test_encode_image_returns_base64_string(tmp_path):
    from pyutils.ollama.ollama_collector import OllamaCollector
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n")
    result = OllamaCollector.encode_image(str(img))
    assert isinstance(result, str)
    assert len(result) > 0


def test_encode_image_raises_on_missing_file():
    from pyutils.ollama.ollama_collector import OllamaCollector
    with pytest.raises(OSError):
        OllamaCollector.encode_image("/no/such/file.png")
