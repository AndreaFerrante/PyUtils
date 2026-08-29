"""Tests for LMStudio.extract_from_pdf.

No live LM Studio server: LMStudio.chat is monkeypatched to return a canned
reply. The PDF path is exercised for real — a one-page PDF is written with
pdf_generator_from_text and read back with the real scrape_pdf_content.
"""

from __future__ import annotations

import pytest

from pyutils.lmstudio import LMStudio, LMStudioError
from pyutils.service_factory.pdf import pdf_generator_from_text

# scrape_pdf_content via PyPDF2 preserves whole words but can be lossy about
# inter-word spacing, so tests assert on single tokens, never on phrases.
PDF_TEXT = "ALPHA BRAVO CHARLIE 42"


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "doc.pdf"
    pdf_generator_from_text(str(path), PDF_TEXT)
    return str(path)


def make_client(monkeypatch, reply, sink=None):
    """An LMStudio whose .chat returns `reply` and (optionally) records its args."""
    lm = LMStudio()

    def fake_chat(prompt, model="default", temperature=0.7, max_tokens=-1):
        if sink is not None:
            sink["prompt"] = prompt
            sink["model"] = model
            sink["temperature"] = temperature
            sink["max_tokens"] = max_tokens
        return reply

    monkeypatch.setattr(lm, "chat", fake_chat)
    return lm


def make_client_chat_forbidden(monkeypatch):
    """An LMStudio whose .chat fails the test if it is called at all."""
    lm = LMStudio()

    def fail_chat(*args, **kwargs):
        raise AssertionError("chat() must not be called")

    monkeypatch.setattr(lm, "chat", fail_chat)
    return lm


def test_bad_output_format_raises_valueerror_before_chat(monkeypatch, pdf_path):
    lm = make_client_chat_forbidden(monkeypatch)
    with pytest.raises(ValueError):
        lm.extract_from_pdf(pdf_path, "x", output_format="yaml")


def test_missing_pdf_raises_filenotfound_before_chat(monkeypatch, tmp_path):
    lm = make_client_chat_forbidden(monkeypatch)
    with pytest.raises(FileNotFoundError):
        lm.extract_from_pdf(str(tmp_path / "nope.pdf"), "x")


def test_empty_pdf_text_raises_lmstudioerror_before_chat(monkeypatch, pdf_path):
    lm = make_client_chat_forbidden(monkeypatch)
    monkeypatch.setattr(
        "pyutils.service_factory.pdf.scrape_pdf_content", lambda p: "   \n  "
    )
    with pytest.raises(LMStudioError):
        lm.extract_from_pdf(pdf_path, "x")


def test_txt_reply_returned_stripped(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "  hello world  ")
    assert lm.extract_from_pdf(pdf_path, "x", output_format="txt") == "hello world"


def test_instruction_and_pdf_text_reach_chat(monkeypatch, pdf_path):
    sink = {}
    lm = make_client(monkeypatch, "ok", sink)
    lm.extract_from_pdf(pdf_path, "EXTRACT_THE_TOTAL", output_format="txt")
    system_msg, user_msg = sink["prompt"]
    assert system_msg["role"] == "system"
    assert user_msg["role"] == "user"
    assert "EXTRACT_THE_TOTAL" in user_msg["content"]
    assert "ALPHA" in user_msg["content"]


def test_temperature_defaults_to_zero(monkeypatch, pdf_path):
    sink = {}
    lm = make_client(monkeypatch, "ok", sink)
    lm.extract_from_pdf(pdf_path, "x", output_format="txt")
    assert sink["temperature"] == 0.0


def test_model_and_max_tokens_pass_through(monkeypatch, pdf_path):
    sink = {}
    lm = make_client(monkeypatch, "ok", sink)
    lm.extract_from_pdf(
        pdf_path, "x", model="my-model", output_format="txt", max_tokens=256
    )
    assert sink["model"] == "my-model"
    assert sink["max_tokens"] == 256


from pyutils.lmstudio.lmstudio import _strip_code_fence


def test_strip_code_fence_passes_through_plain_text():
    assert _strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


def test_strip_code_fence_removes_bare_fence():
    assert _strip_code_fence("```\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_code_fence_removes_language_tagged_fence():
    assert _strip_code_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'


def test_strip_code_fence_leaves_unterminated_fence_alone():
    # No closing fence -> return as-is (json.loads will then reject it).
    assert _strip_code_fence("```json\n{\"a\": 1}") == '```json\n{"a": 1}'


def test_json_object_reply_parsed(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, '{"total": 42}')
    assert lm.extract_from_pdf(pdf_path, "x", output_format="json") == {"total": 42}


def test_json_list_reply_parsed(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "[1, 2, 3]")
    assert lm.extract_from_pdf(pdf_path, "x", output_format="json") == [1, 2, 3]


def test_json_reply_wrapped_in_fence_parsed(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, '```json\n{"a": 1}\n```')
    assert lm.extract_from_pdf(pdf_path, "x", output_format="json") == {"a": 1}


def test_json_branch_uses_json_system_prompt(monkeypatch, pdf_path):
    sink = {}
    lm = make_client(monkeypatch, "{}", sink)
    lm.extract_from_pdf(pdf_path, "x", output_format="json")
    assert "JSON" in sink["prompt"][0]["content"]


def test_invalid_json_reply_raises_lmstudioerror(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "here is your answer: none")
    with pytest.raises(LMStudioError):
        lm.extract_from_pdf(pdf_path, "x", output_format="json")


def test_csv_reply_returned_as_string(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "name,amount\nwidget,10\ngadget,20")
    out = lm.extract_from_pdf(pdf_path, "x", output_format="csv")
    assert out == "name,amount\nwidget,10\ngadget,20"
    assert isinstance(out, str)


def test_csv_reply_wrapped_in_fence_stripped(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "```csv\nname,amount\nwidget,10\n```")
    assert (
        lm.extract_from_pdf(pdf_path, "x", output_format="csv")
        == "name,amount\nwidget,10"
    )


def test_csv_branch_uses_csv_system_prompt(monkeypatch, pdf_path):
    sink = {}
    lm = make_client(monkeypatch, "a,b\n1,2", sink)
    lm.extract_from_pdf(pdf_path, "x", output_format="csv")
    assert "CSV" in sink["prompt"][0]["content"]


def test_unparseable_csv_reply_raises_lmstudioerror(monkeypatch, pdf_path):
    # Unterminated quoted field running to EOF -> csv.Error (strict reader).
    lm = make_client(monkeypatch, '"unterminated field')
    with pytest.raises(LMStudioError):
        lm.extract_from_pdf(pdf_path, "x", output_format="csv")


def test_empty_csv_reply_raises_lmstudioerror(monkeypatch, pdf_path):
    lm = make_client(monkeypatch, "   ")
    with pytest.raises(LMStudioError):
        lm.extract_from_pdf(pdf_path, "x", output_format="csv")
