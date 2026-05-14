import pytest
from unittest.mock import MagicMock, patch
from pyutils.openai.openai_collector import OpenAICollector


@pytest.fixture
def collector():
    with patch('pyutils.openai.openai_collector.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        c = OpenAICollector(api_key='sk-test', model='gpt-4.1')
        c.client = mock_client
        yield c


def test_get_embeddings_uses_provided_embedder(collector):
    # Bug: if len(embedder): embedder = self.embedder  ← always overwrites!
    # Fix: if not embedder: embedder = self.embedder
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    collector.client.embeddings.create.return_value = mock_response
    collector.embedder = 'text-embedding-3-large'

    collector.get_embeddings(['test'], embedder='text-embedding-ada-002')

    collector.client.embeddings.create.assert_called_once_with(
        model='text-embedding-ada-002',
        input=['test']
    )


def test_get_embeddings_uses_default_embedder_when_none_given(collector):
    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    collector.client.embeddings.create.return_value = mock_response
    collector.embedder = 'text-embedding-3-large'

    collector.get_embeddings(['test'])

    collector.client.embeddings.create.assert_called_once_with(
        model='text-embedding-3-large',
        input=['test']
    )


def test_get_answer_given_query(collector):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = 'The answer is 42'
    collector.client.chat.completions.create.return_value = mock_response

    result = collector.get_answer_given_query('What is the answer?')
    assert result == 'The answer is 42'


def test_get_tokens_in_string(collector):
    result = collector.get_tokens_in_string('Hello world')
    assert isinstance(result, int)
    assert result > 0
