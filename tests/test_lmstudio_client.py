"""Tests for LMStudio client-level behaviour (no live server: `_request` is stubbed)."""

from __future__ import annotations

import pytest

from pyutils.lmstudio import LMStudio


def _capture_body(lm: LMStudio, result: object) -> dict:
    """Replace lm._request with a stub that records the request body and returns `result`."""
    seen: dict = {}

    def fake_request(method, url, body=None, stream=False):
        seen["method"] = method
        seen["url"] = url
        seen["body"] = body
        return result

    lm._request = fake_request  # type: ignore[method-assign]
    return seen


CHAT_OK = {"choices": [{"message": {"content": "ok"}}]}
COMPLETE_OK = {"choices": [{"text": "ok"}]}
EMBED_OK = {"data": [{"embedding": [0.0]}]}


def test_default_instance_model_is_backwards_compatible():
    """LMStudio() with no model arg still sends model='default', as before."""
    lm = LMStudio()
    seen = _capture_body(lm, CHAT_OK)
    lm.chat("hi")
    assert seen["body"]["model"] == "default"


def test_constructor_model_is_used_when_method_model_omitted():
    lm = LMStudio(model="configured-model")

    seen = _capture_body(lm, CHAT_OK)
    lm.chat("hi")
    assert seen["body"]["model"] == "configured-model"

    seen = _capture_body(lm, COMPLETE_OK)
    lm.complete("hi")
    assert seen["body"]["model"] == "configured-model"

    seen = _capture_body(lm, EMBED_OK)
    lm.embed("hi")
    assert seen["body"]["model"] == "configured-model"


def test_method_model_overrides_constructor_model():
    lm = LMStudio(model="configured-model")
    seen = _capture_body(lm, CHAT_OK)

    lm.chat("hi", model="explicit-override")
    assert seen["body"]["model"] == "explicit-override"


def test_chat_stateful_uses_instance_model():
    lm = LMStudio(model="configured-model")
    seen = _capture_body(lm, {"output": [], "response_id": "r1"})
    lm.chat_stateful("hi")
    assert seen["body"]["model"] == "configured-model"


def test_model_attribute_is_reassignable_after_construction():
    lm = LMStudio()
    seen = _capture_body(lm, CHAT_OK)
    lm.model = "swapped-in"
    lm.chat("hi")
    assert seen["body"]["model"] == "swapped-in"


def test_embed_uses_instance_model():
    lm = LMStudio(model="embed-model")
    seen = _capture_body(lm, EMBED_OK)
    lm.embed(["a"])
    assert seen["body"]["model"] == "embed-model"


def test_chat_hits_v0_endpoint_and_captures_stats():
    lm = LMStudio()
    stats = {"tokens_per_second": 42.0, "time_to_first_token": 0.1}
    seen = _capture_body(lm, {**CHAT_OK, "stats": stats})
    lm.chat("hi")
    assert seen["url"] == f"http://{lm.host}/api/v0/chat/completions"
    assert lm.last_stats == stats


def test_complete_hits_v0_endpoint_and_captures_stats():
    lm = LMStudio()
    stats = {"tokens_per_second": 10.0}
    seen = _capture_body(lm, {**COMPLETE_OK, "stats": stats})
    lm.complete("hi")
    assert seen["url"] == f"http://{lm.host}/api/v0/completions"
    assert lm.last_stats == stats


def test_last_stats_defaults_to_empty_dict_when_absent():
    lm = LMStudio()
    _capture_body(lm, CHAT_OK)
    lm.chat("hi")
    assert lm.last_stats == {}
